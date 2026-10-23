#!/usr/bin/env python3
"""Proton-style GEMM roofline + MetaX mm_kernel_nt tile/pipeline sweep.

MetaX Triton 3.0.0+metax does NOT ship ``triton.profiler`` / proton (no
``[proton]`` extra). This script mirrors proton's useful bits:

  * named scopes with ``flops16`` + ``bytes`` metrics
  * derived TFLOP/s and GB/s (achieved)
  * compare against native mcblas on the same tensors

Then sweeps ``mm_kernel_nt`` with fixed (BLOCK_*, num_stages, pipeline,
scenario) — the knobs MetaX's compiler actually consumes — to hunt a config
that closes the ~1.3× prefill gap vs mcblas.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

OUT = Path(__file__).resolve().parent

# Rough C500 peaks (HBM from mx-smi; BF16 tensor peak is soft — used for %).
PEAK_GBPS = float(os.environ.get("C500_PEAK_GBPS", "1600"))  # override if known
PEAK_TFLOPS_BF16 = float(os.environ.get("C500_PEAK_TFLOPS", "300"))

SHAPES = [
    # name, N, K  (F.linear weight is (N,K); we do x @ W.t())
    ("qkv", 2560, 2048),
    ("o_proj", 2048, 2048),
    ("gate_up", 12288, 2048),
    ("down", 2048, 6144),
    ("lm_head", 130560, 2048),
]
MS = [int(x) for x in os.environ.get("GEMM_MS", "64,2048,8192").split(",") if x.strip()]


@dataclass
class Row:
    scope: str
    name: str
    M: int
    N: int
    K: int
    impl: str
    device_us: float
    host_us: float
    tflops: float
    gbps: float
    pct_peak_flops: float
    pct_peak_bw: float
    config: str
    vs_native: float | None = None


def flops(M, N, K) -> float:
    return 2.0 * M * N * K


def nbytes(M, N, K) -> int:
    return 2 * (M * K + K * N + M * N)  # bf16


def device_us(fn, iters=30) -> float:
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    tot = sum((e.self_device_time_total or 0.0) for e in prof.key_averages())
    return tot / iters


def host_us(fn, iters=30) -> float:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    h = (time.perf_counter() - t0) / iters * 1e6
    torch.cuda.synchronize()
    return h


def metrics(M, N, K, dus: float) -> tuple[float, float, float, float]:
    if dus <= 0:
        return 0, 0, 0, 0
    tflops = flops(M, N, K) / (dus * 1e-6) / 1e12
    gbps = nbytes(M, N, K) / (dus * 1e-6) / 1e9
    return tflops, gbps, 100 * tflops / PEAK_TFLOPS_BF16, 100 * gbps / PEAK_GBPS


def fit_smem(bm, bn, bk, stages=2) -> bool:
    # bf16 A+B tiles per stage; MetaX C500 64KB/block
    return (bm * bk + bk * bn) * 2 * max(stages - 1, 1) <= 65536


# Candidate configs inspired by mcblas winners + existing yaml.
CONFIGS = []
for bm, bn, bk, stages, warps, pipe, scen in [
    # decode-ish
    (32, 64, 128, 2, 8, "basic", ""),
    (64, 32, 128, 2, 8, "basic", ""),
    (64, 128, 64, 2, 8, "basic", ""),
    (64, 128, 64, 3, 8, "basic", ""),
    # prefill-ish / mcblas-like
    (128, 128, 64, 2, 8, "basic", ""),
    (128, 128, 64, 2, 8, "basic", "fullstage"),
    (128, 128, 128, 2, 8, "basic", ""),
    (128, 128, 128, 3, 8, "basic", "fullstage"),
    (128, 128, 128, 4, 8, "basic", "fullstage"),
    (256, 256, 32, 2, 8, "basic", ""),
    (256, 256, 32, 2, 8, "basic", "fullstage"),
    (256, 256, 64, 2, 8, "basic", "fullstage"),  # may OOM smem
    (256, 128, 64, 2, 8, "basic", "fullstage"),
    (128, 256, 64, 2, 8, "basic", "fullstage"),
    # cpasync variants
    (64, 128, 64, 2, 8, "cpasync", ""),
    (128, 128, 64, 2, 8, "cpasync", ""),
    (128, 128, 64, 2, 8, "cpasync", "fullstage"),
    (256, 256, 32, 2, 8, "cpasync", "fullstage"),
    (128, 128, 128, 2, 8, "cpasync", "fullstage"),
]:
    if fit_smem(bm, bn, bk, stages) or pipe == "cpasync":
        CONFIGS.append((bm, bn, bk, stages, warps, pipe, scen))


def launch_nt(a, b_t, c, M, N, K, cfg, kern=None):
    """Launch mm_kernel_nt with fixed MetaX compiler options."""
    import triton

    bm, bn, bk, stages, warps, pipe, scen = cfg
    grid = (triton.cdiv(M, bm) * triton.cdiv(N, bn),)
    # b_t is (K, N) contiguous — mm_kernel_nt B layout: stride (1, K).
    kern[grid](
        a,
        b_t,
        c,
        M,
        N,
        K,
        GROUP_M=8,
        BLOCK_M=bm,
        BLOCK_N=bn,
        BLOCK_K=bk,
        EVEN_M=(M % bm == 0),
        EVEN_N=(N % bn == 0),
        EVEN_K=(K % bk == 0),
        num_warps=warps,
        num_stages=stages,
        pipeline=pipe,
        scenario=scen,
    )
    return c


def cfg_str(cfg) -> str:
    bm, bn, bk, st, w, p, s = cfg
    return f"{bm}x{bn}x{bk}/s{st}/w{w}/{p}/{s or '-'}"


def main() -> int:
    torch.manual_seed(0)
    rows: list[Row] = []

    print(
        f"peaks: {PEAK_GBPS} GB/s, {PEAK_TFLOPS_BF16} TFLOPS (soft); "
        f"{len(CONFIGS)} configs; proton N/A on metax-triton\n",
        flush=True,
    )

    # ---- native baselines ----
    native = {}
    print("=== native mcblas (proton-scope style) ===", flush=True)
    for name, N, K in SHAPES:
        for M in MS:
            if nbytes(M, N, K) > 4 * 1024**3:
                continue
            a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
            w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)

            def run(a=a, w=w):
                return torch.nn.functional.linear(a, w)

            dus = device_us(run, iters=20 if M >= 2048 else 40)
            hus = host_us(run, iters=20 if M >= 2048 else 40)
            tf, gb, pf, pb = metrics(M, N, K, dus)
            native[(name, M)] = dus
            r = Row(
                scope=f"native/{name}/M{M}",
                name=name,
                M=M,
                N=N,
                K=K,
                impl="native",
                device_us=dus,
                host_us=hus,
                tflops=tf,
                gbps=gb,
                pct_peak_flops=pf,
                pct_peak_bw=pb,
                config="mcblas",
            )
            rows.append(r)
            print(
                f"  {name:8s} M={M:5d}  {dus:8.1f}us  {tf:5.1f}T  {gb:6.0f}GB/s  "
                f"({pb:4.1f}%BW)  flops={pf:4.1f}%",
                flush=True,
            )
            del a, w
            torch.cuda.empty_cache()

    # Import AFTER native so aten isn't patched for baseline.
    # Note: ``from ...ops import mm`` binds the *function*; load the submodule.
    import importlib

    mm_mod = importlib.import_module("flag_gems.runtime.backend._metax.ops.mm")
    kern = mm_mod.mm_kernel_nt
    # Unwrap libtuner / libentry layers to the raw JITFunction.
    for _ in range(4):
        nxt = getattr(kern, "fn", None)
        if nxt is None or nxt is kern:
            break
        kern = nxt
    print(f"kernel type={type(kern)} file={mm_mod.__file__}", flush=True)

    print("\n=== mm_kernel_nt fixed-config sweep (vs native) ===", flush=True)
    best: dict[tuple[str, int], tuple[float, str]] = {}

    for name, N, K in SHAPES:
        for M in MS:
            if nbytes(M, N, K) > 4 * 1024**3:
                continue
            # Skip huge lm_head × many configs (too slow); only try promising set
            cfgs = CONFIGS
            if name == "lm_head" and M >= 2048:
                cfgs = [c for c in CONFIGS if c[0] >= 128][:8]
            if name == "lm_head" and M == 64:
                cfgs = [c for c in CONFIGS if c[0] <= 64] + [
                    c for c in CONFIGS if c[0] == 128
                ][:4]

            a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
            w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
            # Contiguous (K,N) for nt kernel: W.t().contiguous()
            b_t = w.t().contiguous()
            c = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
            nat = native.get((name, M))

            for cfg in cfgs:
                label = cfg_str(cfg)
                try:
                    # compile/warmup
                    for _ in range(2):
                        launch_nt(a, b_t, c, M, N, K, cfg, kern=kern)
                    torch.cuda.synchronize()

                    def run(a=a, b_t=b_t, c=c, cfg=cfg, kern=kern):
                        return launch_nt(a, b_t, c, M, N, K, cfg, kern=kern)

                    dus = device_us(run, iters=12 if M >= 2048 else 25)
                except Exception as exc:  # noqa: BLE001
                    print(f"  FAIL {name} M={M} {label}: {type(exc).__name__}: {exc}", flush=True)
                    continue

                tf, gb, pf, pb = metrics(M, N, K, dus)
                ratio = dus / nat if nat else None
                rows.append(
                    Row(
                        scope=f"nt/{name}/M{M}/{label}",
                        name=name,
                        M=M,
                        N=N,
                        K=K,
                        impl="mm_kernel_nt",
                        device_us=dus,
                        host_us=float("nan"),
                        tflops=tf,
                        gbps=gb,
                        pct_peak_flops=pf,
                        pct_peak_bw=pb,
                        config=label,
                        vs_native=ratio,
                    )
                )
                key = (name, M)
                if key not in best or dus < best[key][0]:
                    best[key] = (dus, label)
                mark = ""
                if ratio is not None and ratio < 0.98:
                    mark = " WIN"
                elif ratio is not None and ratio > 1.15:
                    mark = ""
                print(
                    f"  {name:8s} M={M:5d} {label:40s} {dus:8.1f}us  "
                    f"vs_nat={ratio:.3f}x  {tf:5.1f}T  {gb:6.0f}GB/s{mark}",
                    flush=True,
                )

            del a, w, b_t, c
            torch.cuda.empty_cache()

    print("\n=== BEST per shape (vs native) ===", flush=True)
    wins = 0
    for (name, M), (dus, label) in sorted(best.items()):
        nat = native[(name, M)]
        r = dus / nat
        flag = "WIN" if r < 0.98 else ("tie" if r < 1.05 else "LOSE")
        if r < 0.98:
            wins += 1
        print(
            f"  {flag:4s} {name:8s} M={M:5d}  best={dus:8.1f}us  nat={nat:8.1f}us  "
            f"ratio={r:.3f}  cfg={label}",
            flush=True,
        )

    out = OUT / "gemm_proton_sweep.json"
    out.write_text(json.dumps([asdict(r) for r in rows], indent=2))
    print(f"\nWrote {out}  (wins={wins})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
