#!/usr/bin/env python3
"""Close the M=64 projection occupancy gap.

Diagnosis (note.txt / FINDINGS §20): at batch 64 the four projections are
memory-bound but their efficiency tracks *block count*, not shape volume --
qkv (20 blocks) 43%, o_proj (16) 38%, down (16) 43%, gate_up (96) 84%.

Two ways to raise the block count:

  A. smaller BLOCK_N          blocks = N / BLOCK_N, single kernel, no reduction
  B. split-K                  blocks = (N / BLOCK_N) * SPLIT_K, needs a reduction

A is much cheaper: decode makes 3 such GEMMs x 42 layers = 126 calls per step,
so every extra kernel launch (zero-fill, reduce) is multiplied by 126.  Try A
first and only fall back to B if A cannot reach the bandwidth.

Compares against mcblas (F.linear) at the exact shapes the model uses.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
import triton
import triton.language as tl

OUT = Path(__file__).resolve().parent
M = 64
PEAK_GBS = 1416.0
NLAYERS = 42
ITL_MS = 24.45
SHAPES = [
    ("qkv", 2560, 2048),
    ("o_proj", 2048, 2048),
    ("gate_up", 12288, 2048),
    ("down", 2048, 6144),
]


def dev_us(fn, warmup=10, iters=30):
    import torch
    from torch.profiler import ProfilerActivity, profile

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    return sum(e.self_device_time_total or 0 for e in p.key_averages()) / iters


def build_kernels():
    @triton.jit
    def gemm_nk(
        A, B, C, M, N, K,
        sam, sak, sbn, sbk, scm, scn,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        """C[M,N] = A[M,K] @ B[N,K]^T.  Grid = (cdiv(N, BLOCK_N),)."""
        pid_n = tl.program_id(0)
        rm = tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        mask_n = rn < N
        for k0 in range(0, K, BLOCK_K):
            rk = k0 + tl.arange(0, BLOCK_K)
            mask_k = rk < K
            a = tl.load(A + rm[:, None] * sam + rk[None, :] * sak,
                        mask=(rm[:, None] < M) & mask_k[None, :], other=0.0)
            b = tl.load(B + rn[:, None] * sbn + rk[None, :] * sbk,
                        mask=mask_n[:, None] & mask_k[None, :], other=0.0)
            acc += tl.dot(a, tl.trans(b))
        tl.store(C + rm[:, None] * scm + rn[None, :] * scn,
                 acc.to(C.dtype.element_ty),
                 mask=(rm[:, None] < M) & mask_n[None, :])

    @triton.jit
    def gemm_splitk(
        A, B, C, M, N, K,
        sam, sak, sbn, sbk,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
        SPLIT_K: tl.constexpr,
    ):
        """Same product, K split across program_id(1); fp32 atomic reduction
        into a caller-zeroed [M, N] float32 workspace."""
        pid_n = tl.program_id(0)
        pid_k = tl.program_id(1)
        rm = tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        k_per = tl.cdiv(K, SPLIT_K)
        k_lo = pid_k * k_per
        k_hi = tl.minimum(K, k_lo + k_per)
        mask_n = rn < N
        for k0 in range(k_lo, k_hi, BLOCK_K):
            rk = k0 + tl.arange(0, BLOCK_K)
            mask_k = rk < k_hi
            a = tl.load(A + rm[:, None] * sam + rk[None, :] * sak,
                        mask=(rm[:, None] < M) & mask_k[None, :], other=0.0)
            b = tl.load(B + rn[:, None] * sbn + rk[None, :] * sbk,
                        mask=mask_n[:, None] & mask_k[None, :], other=0.0)
            acc += tl.dot(a, tl.trans(b))
        tl.atomic_add(C + rm[:, None] * N + rn[None, :], acc,
                      mask=(rm[:, None] < M) & mask_n[None, :])

    return gemm_nk, gemm_splitk


def main() -> int:
    gemm_nk, gemm_splitk = build_kernels()
    rows = {}
    tot_native = 0.0
    tot_best = 0.0

    for name, N, K in SHAPES:
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
        ref = torch.nn.functional.linear(x, w).float()

        us_native = dev_us(lambda: torch.nn.functional.linear(x, w))
        tot_native += us_native
        gbs_native = (N * K * 2 / 1e9) / (us_native / 1e6)
        print(
            f"\n=== {name} (N={N}, K={K})  mcblas {us_native:.1f} us "
            f"= {100*gbs_native/PEAK_GBS:.0f}% peak, {triton.cdiv(N,128)} blocks ===",
            flush=True,
        )

        best = None
        _errs = set()
        # ---- A: plain, smaller BLOCK_N ----
        for BLOCK_N in (16, 32, 64, 128, 256):
            for BLOCK_K in (32, 64, 128):
                c = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
                grid = (triton.cdiv(N, BLOCK_N),)

                def run(c=c, grid=grid, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K):
                    gemm_nk[grid](
                        x, w, c, M, N, K,
                        x.stride(0), x.stride(1), w.stride(0), w.stride(1),
                        c.stride(0), c.stride(1),
                        BLOCK_M=M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                    )
                    return c

                try:
                    got = run()
                    torch.cuda.synchronize()
                    err = (got.float() - ref).abs().max().item()
                    if err > 2.0:
                        continue
                    us = dev_us(run)
                    tag = f"plain N={BLOCK_N} K={BLOCK_K} ({grid[0]} blk)"
                    if best is None or us < best[0]:
                        best = (us, tag, err)
                except Exception as exc:
                    _errs.add(f"{BLOCK_N}/{BLOCK_K}: {type(exc).__name__} {str(exc)[:200]}")
                    continue

        # ---- B: split-K (only if A struggles) ----
        if best is None or best[0] > 0.85 * us_native:
            for BLOCK_N in (64, 128, 256):
                for BLOCK_K in (64, 128):
                    for SPLIT_K in (2, 4, 8):
                        ws = torch.zeros(M, N, device="cuda", dtype=torch.float32)
                        grid = (triton.cdiv(N, BLOCK_N), SPLIT_K)

                        def run(ws=ws, grid=grid, BLOCK_N=BLOCK_N,
                                BLOCK_K=BLOCK_K, SPLIT_K=SPLIT_K):
                            ws.zero_()
                            gemm_splitk[grid](
                                x, w, ws, M, N, K,
                                x.stride(0), x.stride(1), w.stride(0), w.stride(1),
                                BLOCK_M=M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                                SPLIT_K=SPLIT_K,
                            )
                            return ws

                        try:
                            got = run()
                            torch.cuda.synchronize()
                            err = (got - ref).abs().max().item()
                            if err > 4.0:
                                continue
                            us = dev_us(run, warmup=5, iters=20)
                            tag = f"splitk N={BLOCK_N} K={BLOCK_K} sk={SPLIT_K} ({grid[0]*SPLIT_K} blk)"
                            if best is None or us < best[0]:
                                best = (us, tag, err)
                        except Exception:
                            continue

        for _e in list(_errs)[:2]:
            print(f"    ERR {_e}", flush=True)
        if best:
            us_b, tag, err = best
            tot_best += us_b
            gbs = (N * K * 2 / 1e9) / (us_b / 1e6)
            print(
                f"  best {us_b:6.1f} us  {us_b/us_native:5.2f}x  "
                f"{100*gbs/PEAK_GBS:3.0f}% peak  [{tag}]  err={err:.3g}",
                flush=True,
            )
            rows[name] = {
                "N": N, "K": K,
                "native_us": round(us_native, 1),
                "native_pct_peak": round(100 * gbs_native / PEAK_GBS),
                "best_us": round(us_b, 1),
                "best_speedup": round(us_native / us_b, 3),
                "best_pct_peak": round(100 * gbs / PEAK_GBS),
                "cfg": tag,
                "max_abs_err": err,
            }
        else:
            tot_best += us_native
            rows[name] = {"N": N, "K": K, "native_us": round(us_native, 1), "best": None}

    save_ms = (tot_native - tot_best) * NLAYERS / 1000
    print(f"\nper layer {tot_native:.1f} -> {tot_best:.1f} us ({tot_native/tot_best:.2f}x)")
    print(f"per decode step (42 layers): -{save_ms:.2f} ms = "
          f"{100*save_ms/ITL_MS:.1f}% of a {ITL_MS} ms step")
    rows["per_step"] = {
        "native_us_per_layer": round(tot_native, 1),
        "best_us_per_layer": round(tot_best, 1),
        "saving_ms": round(save_ms, 2),
        "saving_pct_of_itl": round(100 * save_ms / ITL_MS, 1),
    }
    (OUT / "proj_gemm_probe.json").write_text(json.dumps(rows, indent=2))
    print("wrote", OUT / "proj_gemm_probe.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
