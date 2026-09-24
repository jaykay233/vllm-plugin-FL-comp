#!/usr/bin/env python3
"""Kernel shoot-out for the decode-time (M=1) `linear` / GEMV on MetaX C500.

Diagnosis
---------
The generic ``linear_kernel`` uses ``grid = (cdiv(M, BLOCK_M), cdiv(N, BLOCK_N))``.
At M=1 the first grid axis is always 1, so the program count collapses to
``cdiv(N, BLOCK_N)``. For qkv_proj (N=2560) with BLOCK_N=128 that is 20
programs on a 104-SM device: ~20% occupancy, and the GEMV runs at ~125 GB/s
against an ~818 GB/s copy ceiling.

Candidates
----------
  A. ``linear_kernel`` tile sweep (BLOCK_M fixed at 16, tiny BLOCK_N) -- cheapest
     possible fix: keep the existing kernel, just stop under-parallelising it.
  B. ``gemv_v1``: grid over N, BLOCK_N outputs per program, 2D accumulate then
     reduce. No ``tl.dot`` (nothing to gain at M=1).
  C. ``gemv_v2``: one output row per program, 1D K-loop. Maximum parallelism.

Each candidate is checked for correctness before it is timed.

Run:  conda activate mx && python /root/src/bench_gemv_kernels.py
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import torch
import triton
import triton.language as tl

MODEL = "/root/models/MiniCPM5-2B"
OUT = Path("/root/bench_results/metax_gemv")
OUT.mkdir(parents=True, exist_ok=True)
DEV = "cuda"
DT = torch.bfloat16

PER_LAYER = ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"]


# --------------------------------------------------------------------------
# A. the generic kernel, without the autotuner so tiles can be forced
# --------------------------------------------------------------------------
@triton.jit
def linear_kernel_raw(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    M, N, K,
    stride_im, stride_ik, stride_wn, stride_wk,
    stride_om, stride_on, stride_bn,
    BIAS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    input_ptrs = input_ptr + (offs_m[:, None] * stride_im + offs_k[None, :] * stride_ik)
    weight_ptrs = weight_ptr + (offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        input_mask = (offs_m < M)[:, None] & (offs_k < (K - k * BLOCK_SIZE_K))[None, :]
        a = tl.load(input_ptrs, mask=input_mask, other=0.0)
        weight_mask = (offs_k < (K - k * BLOCK_SIZE_K))[:, None] & (offs_n < N)[None, :]
        b = tl.load(weight_ptrs, mask=weight_mask, other=0.0)
        accumulator += tl.dot(a, b, allow_tf32=False)
        input_ptrs += BLOCK_SIZE_K * stride_ik
        weight_ptrs += BLOCK_SIZE_K * stride_wk

    output_ptrs = output_ptr + (offs_m[:, None] * stride_om + offs_n[None, :] * stride_on)
    output_mask = (offs_m < M)[:, None] & (offs_n < N)[None, :]
    if BIAS:
        accumulator = accumulator + tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    tl.store(output_ptrs, accumulator.to(output_ptr.dtype.element_ty), mask=output_mask)


# --------------------------------------------------------------------------
# B. gemv_v1 -- BLOCK_N outputs per program, tile accumulate, reduce at the end
# --------------------------------------------------------------------------
@triton.jit
def gemv_v1(
    a_ptr, w_ptr, out_ptr, N, K, stride_wn,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    n_mask = offs_n < N
    acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
    for kb in range(0, tl.cdiv(K, BLOCK_K)):
        k_offs = kb * BLOCK_K + offs_k
        k_mask = k_offs < K
        a_vec = tl.load(a_ptr + k_offs, mask=k_mask, other=0.0)
        w_blk = tl.load(
            w_ptr + offs_n[:, None] * stride_wn + k_offs[None, :],
            mask=n_mask[:, None] & k_mask[None, :], other=0.0,
        )
        acc += w_blk.to(tl.float32) * a_vec.to(tl.float32)[None, :]
    res = tl.sum(acc, axis=1)
    tl.store(out_ptr + offs_n, res.to(out_ptr.dtype.element_ty), mask=n_mask)


# --------------------------------------------------------------------------
# C. gemv_v2 -- one output row per program
# --------------------------------------------------------------------------
@triton.jit
def gemv_v2(
    a_ptr, w_ptr, out_ptr, N, K, stride_wn,
    BLOCK_K: tl.constexpr,
):
    n = tl.program_id(0)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_K,), dtype=tl.float32)
    for kb in range(0, tl.cdiv(K, BLOCK_K)):
        k_offs = kb * BLOCK_K + offs_k
        k_mask = k_offs < K
        a_vec = tl.load(a_ptr + k_offs, mask=k_mask, other=0.0)
        w_vec = tl.load(w_ptr + n * stride_wn + k_offs, mask=k_mask, other=0.0)
        acc += w_vec.to(tl.float32) * a_vec.to(tl.float32)
    tl.store(out_ptr + n, tl.sum(acc, axis=0).to(out_ptr.dtype.element_ty))


# --------------------------------------------------------------------------
def load_shapes() -> list[tuple[str, int, int]]:
    cfg = json.loads((Path(MODEL) / "config.json").read_text())
    h, inter = cfg["hidden_size"], cfg["intermediate_size"]
    nq = cfg["num_attention_heads"]
    nkv = cfg.get("num_key_value_heads", nq)
    hd = cfg.get("head_dim") or h // nq
    shapes = [
        ("qkv_proj", nq * hd + 2 * nkv * hd, h),
        ("o_proj", nq * hd, h),
        ("gate_up_proj", 2 * inter, h),
        ("down_proj", h, inter),
    ]
    if not cfg.get("tie_word_embeddings", False):
        shapes.append(("lm_head", cfg["vocab_size"], h))
    return shapes


def bench(fn, n_warmup=10, n_iters=40) -> float:
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n_iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts)


def frob(got, ref) -> float:
    return ((got.float() - ref).norm() / ref.norm()).item()


def main() -> None:
    from flag_gems.ops.linear import linear as generic_linear

    shapes = load_shapes()
    sm = torch.cuda.get_device_properties(0).multi_processor_count
    print(f"MetaX C500: {sm} SMs")
    print(f"M=1 decode shapes: {[(s[0], s[1], s[2]) for s in shapes]}")
    print()

    # ---- build the candidate list -----------------------------------------
    cands = []  # (label, callable_factory)

    # A: forced-tile generic kernel. BLOCK_M must be >= 16 for tl.dot.
    for bm, bn, bk, nw, ns in [(16, 16, 64, 4, 2), (16, 32, 128, 4, 2),
                               (16, 64, 128, 4, 2), (16, 128, 128, 4, 2),
                               (16, 16, 256, 4, 1), (16, 32, 64, 8, 2),
                               (16, 32, 32, 4, 1)]:
        def mk_A(bm=bm, bn=bn, bk=bk, nw=nw, ns=ns):
            def run(a, w, out):
                M, K = a.shape
                N = w.shape[0]
                grid = (triton.cdiv(M, bm), triton.cdiv(N, bn))
                linear_kernel_raw[grid](
                    a, w, w, out, M, N, K,
                    a.stride(0), a.stride(1), w.stride(0), w.stride(1),
                    out.stride(0), out.stride(1), 0,
                    BIAS=False, BLOCK_SIZE_M=bm, BLOCK_SIZE_N=bn,
                    BLOCK_SIZE_K=bk, num_warps=nw, num_stages=ns,
                )
            return run
        cands.append((f"A lin_kern BM{bm} BN{bn} BK{bk} w{nw} s{ns}", mk_A()))

    # B: gemv_v1
    for bn, bk, nw in [(8, 256, 4), (16, 256, 4), (4, 512, 4), (8, 512, 4),
                       (32, 256, 4), (8, 1024, 4), (16, 128, 4), (8, 256, 2),
                       (8, 256, 8), (16, 512, 8), (64, 256, 4), (32, 512, 8)]:
        def mk_B(bn=bn, bk=bk, nw=nw):
            def run(a, w, out):
                K = a.shape[1]
                N = w.shape[0]
                grid = (triton.cdiv(N, bn),)
                gemv_v1[grid](a, w, out, N, K, w.stride(0),
                              BLOCK_N=bn, BLOCK_K=bk, num_warps=nw)
            return run
        cands.append((f"B gemv_v1 BN{bn} BK{bk} w{nw}", mk_B()))

    # C: gemv_v2
    for bk, nw in [(512, 4), (1024, 4), (2048, 4), (1024, 8), (512, 2)]:
        def mk_C(bk=bk, nw=nw):
            def run(a, w, out):
                K = a.shape[1]
                N = w.shape[0]
                grid = (N,)
                gemv_v2[grid](a, w, out, N, K, w.stride(0),
                              BLOCK_K=bk, num_warps=nw)
            return run
        cands.append((f"C gemv_v2 BK{bk} w{nw}", mk_C()))

    # ---- run -------------------------------------------------------------
    results = {}
    print("=" * 104)
    print(f"{'shape':14s} {'N':>7s} {'K':>5s} {'baseline':>9s} " + " ".join(
        f"{'cand':>8s}" for _ in range(0)) + "best candidates (ms)")
    print("=" * 104)

    per_shape = {}
    for name, N, K in shapes:
        torch.manual_seed(0)
        x = torch.randn(1, K, device=DEV, dtype=DT)
        w = torch.randn(N, K, device=DEV, dtype=DT)
        ref = (x.float() @ w.float().t())
        out = torch.empty(1, N, device=DEV, dtype=DT)

        tbase = bench(lambda: generic_linear(x, w))
        gb = N * K * 2 / (tbase * 1e-3) / 1e9
        per_shape[name] = {"baseline_ms": tbase, "baseline_gbs": gb, "cands": {}}

        rows = []
        for label, run in cands:
            try:
                out.zero_()
                run(x, w, out)
                torch.cuda.synchronize()
                f = frob(out, ref)
            except Exception as e:  # noqa: BLE001 - resource/spec failures
                per_shape[name]["cands"][label] = {
                    "ms": None, "frob": None, "bad": True,
                    "err": f"{type(e).__name__}: {str(e)[:60]}"}
                continue
            if f > 0.02:
                per_shape[name]["cands"][label] = {"ms": None, "frob": f, "bad": True}
                continue
            try:
                ms = bench(lambda: run(x, w, out))
            except Exception as e:  # noqa: BLE001
                per_shape[name]["cands"][label] = {
                    "ms": None, "frob": f, "bad": True,
                    "err": f"{type(e).__name__}: {str(e)[:60]}"}
                continue
            gbs = N * K * 2 / (ms * 1e-3) / 1e9
            per_shape[name]["cands"][label] = {"ms": ms, "gbs": gbs, "frob": f}
            rows.append((ms, label, gbs, f))

        rows.sort()
        print(f"\n  {name:14s} {N:7d} {K:5d}  baseline {tbase:7.4f}ms "
              f"({gb:6.1f} GB/s)")
        for ms, label, gbs, f in rows[:6]:
            print(f"      {label:34s} {ms:7.4f}ms ({gbs:6.1f} GB/s) "
                  f"{tbase / ms:5.2f}x  frob={f:.5f}")
        if not rows:
            print("      (all candidates failed the accuracy check)")

    # ---- 42-layer decode totals ------------------------------------------
    print()
    print("=" * 104)
    print("42-LAYER DECODE TOTAL (per-layer projections only, M=1)")
    print("=" * 104)
    base_total = sum(per_shape[s]["baseline_ms"] for s in PER_LAYER) * 42
    print(f"  baseline (generic linear) : {base_total:8.2f} ms")

    # best candidate per shape, and a single config that is good everywhere
    best_each = {}
    for s in PER_LAYER:
        ok = {k: v for k, v in per_shape[s]["cands"].items()
              if v.get("ms") and not v.get("bad")}
        if ok:
            best_each[s] = min(ok.items(), key=lambda kv: kv[1]["ms"])
            print(f"  best for {s:14s}      : {best_each[s][0]:34s} "
                  f"{best_each[s][1]['ms']:7.4f}ms")
    if best_each:
        tot = sum(v[1]["ms"] for v in best_each.values()) * 42
        print(f"  -> per-shape best total   : {tot:8.2f} ms  "
              f"({base_total / tot:.2f}x)")

    # a single global config ranked by summed per-layer time
    agg = {}
    for s in PER_LAYER:
        for label, v in per_shape[s]["cands"].items():
            if not v.get("ms") or v.get("bad"):
                agg[label] = None
                continue
            if label in agg and agg[label] is not None:
                agg[label] += v["ms"]
    agg = {k: v for k, v in agg.items() if v}
    print()
    print("  ranked by summed per-layer M=1 time (a single config for all shapes):")
    for label, tot_l in sorted(agg.items(), key=lambda kv: kv[1])[:8]:
        tot = tot_l * 42
        print(f"    {label:34s} {tot:8.2f} ms  ({base_total / tot:5.2f}x)")

    (OUT / "sweep.json").write_text(json.dumps(
        {"per_shape": per_shape, "baseline_42layer_ms": base_total}, indent=2,
        default=str))
    print(f"\n  Wrote {OUT / 'sweep.json'}")
    print("GEMV_SWEEP_DONE", flush=True)


if __name__ == "__main__":
    main()
