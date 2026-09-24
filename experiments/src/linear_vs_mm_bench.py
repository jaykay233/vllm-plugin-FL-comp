#!/usr/bin/env python3
"""Confirm the GEMM gap found by the mcTracer profile.

The trace showed the eager path running `linear_kernel` at ~49us/call while the
compile path runs `mm_kernel_nt` / `mm_kernel_splitk` at ~23us/call for the same
layer shapes. Reason (from source): the MetaX flag_gems backend exports `mm`,
`mm_out`, `matmul_bf16` but NOT `linear` -> `F.linear` falls back to the generic
`flag_gems/ops/linear.py::linear_kernel`, which has no metax splitk/nt scenarios.

This benchmarks, at MiniCPM5-2B's real layer shapes:
    F.linear(x, w)     -> generic linear_kernel
    torch.mm(x, w.t()) -> metax flag_gems.mm -> mm_kernel_nt / mm_kernel_splitk
and reports the kernel each path actually launches.

Run:  python /root/src/linear_vs_mm_bench.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import torch

OUT = Path("/root/bench_results/linear_vs_mm")
OUT.mkdir(parents=True, exist_ok=True)

# MiniCPM5-2B layer geometry
HIDDEN = 2048
INTER = 6144
Q_HEADS, KV_HEADS, HEAD_DIM = 16, 2, 128
QKV_OUT = Q_HEADS * HEAD_DIM + 2 * KV_HEADS * HEAD_DIM  # 2560

# (name, in_features, out_features) matching vLLM's fused GEMMs
LAYERS = [
    ("qkv_proj", HIDDEN, QKV_OUT),
    ("o_proj", Q_HEADS * HEAD_DIM, HIDDEN),
    ("gate_up_proj", HIDDEN, 2 * INTER),
    ("down_proj", INTER, HIDDEN),
]


def device_us(fn, iters=30):
    """Pure device time per call, via the profiler (excludes host launch gaps)."""
    from torch.profiler import ProfilerActivity, profile

    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    tot = sum(ev.self_device_time_total
              for ev in prof.key_averages() if ev.self_device_time_total > 0)
    return tot / iters * 1e3, {ev.key: ev.self_device_time_total / ev.count * 1e3
                               for ev in prof.key_averages() if ev.self_device_time_total > 0}


def bench(fn, n_warmup=25, n_iters=200):
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(n_iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / n_iters * 1e3  # us (host-bound at small shapes)


def kernels_of(fn, iters=3):
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    out = {}
    for ev in prof.key_averages():
        if ev.self_device_time_total > 0:
            out[ev.key] = {"ms": ev.self_device_time_total / 1e3, "n": ev.count}
    return out


def main() -> None:
    import flag_gems

    flag_gems.enable()
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"flag_gems: {getattr(flag_gems, '__version__', '?')}\n")

    # confirm what munches `mm`
    from flag_gems.runtime.backend._metax.ops import mm as metax_mm_mod

    print(f"metax mm module: {metax_mm_mod.__name__}\n")

    dt = torch.bfloat16
    results = {}
    for T in (1, 512):
        print("=" * 92)
        print(f"tokens T={T}")
        print("=" * 92)
        print(f"  {'layer':16s} {'shape':>18s} | {'linear dev':>10s} {'mm dev':>8s} "
              f"{'speedup':>8s} | {'linear wall':>11s} {'mm wall':>8s} | {'rel':>8s}")
        results[T] = {}
        for name, K, N in LAYERS:
            x = torch.randn(T, K, device="cuda", dtype=dt)
            w = torch.randn(N, K, device="cuda", dtype=dt)

            d_lin, k_lin_ = device_us(lambda: torch.nn.functional.linear(x, w))
            d_mm, k_mm_ = device_us(lambda: torch.mm(x, w.t()))
            w_lin = bench(lambda: torch.nn.functional.linear(x, w))
            w_mm = bench(lambda: torch.mm(x, w.t()))

            # correctness cross-check
            a = torch.nn.functional.linear(x, w).float()
            b = torch.mm(x, w.t()).float()
            rel = ((a - b).abs() / b.abs().clamp_min(1e-2)).max().item()

            results[T][name] = {
                "shape": f"({T},{K})x({N},{K})",
                "linear_dev_us": d_lin, "mm_dev_us": d_mm,
                "linear_wall_us": w_lin, "mm_wall_us": w_mm,
                "dev_speedup": d_lin / d_mm, "max_rel": rel,
                "linear_kernels": k_lin_, "mm_kernels": k_mm_,
            }
            print(f"  {name:16s} {f'({T},{K})x({N},{K})':>18s} | {d_lin:10.2f} "
                  f"{d_mm:8.2f} {d_lin / d_mm:7.2f}x | {w_lin:11.2f} {w_mm:8.2f} | "
                  f"{rel:8.2e}")

        # kernel identity for one representative layer
        x = torch.randn(T, HIDDEN, device="cuda", dtype=dt)
        w = torch.randn(QKV_OUT, HIDDEN, device="cuda", dtype=dt)
        _, k_lin = device_us(lambda: torch.nn.functional.linear(x, w))
        _, k_mm = device_us(lambda: torch.mm(x, w.t()))
        print(f"\n  F.linear device kernels: {json.dumps({k: round(v, 2) for k, v in k_lin.items()})}")
        print(f"  torch.mm  device kernels: {json.dumps({k: round(v, 2) for k, v in k_mm.items()})}")
        results[T]["_kernels"] = {"F_linear": k_lin, "mm": k_mm}
        print()

    (OUT / "linear_vs_mm.json").write_text(json.dumps(results, indent=2))
    print(f"wrote {OUT / 'linear_vs_mm.json'}")
    print("\nLINEAR_VS_MM_DONE", flush=True)


if __name__ == "__main__":
    main()
