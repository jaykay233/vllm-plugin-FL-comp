#!/usr/bin/env python3
"""GEMM micro-profile on MiniCPM5-2B shapes (MetaX C500 / mcProfiler via torch.profiler).

On MACA there is no standalone ``mcProfiler`` CLI; device-time attribution uses
torch.profiler (MCPTI/Kineto), which is what prior FINDINGS call ``mcProfiler``.

Reports per-shape:
  - native torch.mm / F.linear device_us (mcblas path)
  - FlagGems mm device_us (after warm autotune, geometric strategy)
  - top kernel names + achieved GB/s (weights streamed once)

Does NOT enable global flag_gems for the native arm.
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

OUT = Path(__file__).resolve().parent
OUT.mkdir(parents=True, exist_ok=True)

# MiniCPM5-2B projection (N, K) as in F.linear(x, W) with W:(N,K)
SHAPES = [
    ("qkv", 2560, 2048),
    ("o_proj", 2048, 2048),
    ("gate_up", 12288, 2048),
    ("down", 2048, 6144),
    ("lm_head", 130560, 2048),
]
# Official / serve-relevant M values (conc64 decode ~64; chunked prefill up to MBT).
# Override with GEMM_MS=64,256,2048,8192 for a shorter pass.
_MS_ENV = os.environ.get("GEMM_MS", "1,64,256,1024,2048,4096,8192")
MS = [int(x) for x in _MS_ENV.split(",") if x.strip()]


def bytes_moved(M: int, N: int, K: int) -> int:
    # bf16: A(MK) + B(KN) + C(MN); for bandwidth-bound GEMV/GEMM approx weight+act+out
    return 2 * (M * K + K * N + M * N)


def device_us(fn, iters: int = 40) -> tuple[float, list[dict]]:
    import torch
    from torch.profiler import ProfilerActivity, profile

    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    by_k: dict[str, float] = defaultdict(float)
    by_c: dict[str, int] = defaultdict(int)
    tot = 0.0
    for e in prof.key_averages():
        t = e.self_device_time_total or 0.0
        if t <= 0:
            continue
        by_k[e.key] += t
        by_c[e.key] += e.count
        tot += t
    per = tot / iters  # us
    top = sorted(
        ({"kernel": k, "us_total": v / iters, "calls_per_iter": by_c[k] / iters}
         for k, v in by_k.items()),
        key=lambda r: -r["us_total"],
    )[:8]
    return per, top


def host_us(fn, iters: int = 40) -> float:
    import torch

    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    # do not sync inside — launch cost
    host = (time.perf_counter() - t0) / iters * 1e6
    torch.cuda.synchronize()
    return host


def main() -> int:
    import torch

    torch.manual_seed(0)
    results = []

    print("=== ARM A: NATIVE (no flag_gems) ===", flush=True)
    native = {}
    for name, N, K in SHAPES:
        for M in MS:
            # bf16 bytes for A+W+C; skip if > ~4 GiB working set
            if bytes_moved(M, N, K) > 4 * 1024**3:
                print(f"  skip {name} M={M} (too large)", flush=True)
                continue
            a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
            w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
            # F.linear is what the model calls
            def run_lin(a=a, w=w):
                return torch.nn.functional.linear(a, w)

            def run_mm(a=a, w=w):
                return torch.mm(a, w.t())

            dus, top = device_us(run_lin, iters=20 if M >= 2048 else 40)
            hus = host_us(run_lin, iters=20 if M >= 2048 else 40)
            gbps = bytes_moved(M, N, K) / (dus * 1e-6) / 1e9 if dus > 0 else 0
            row = {
                "arm": "native_linear",
                "name": name,
                "M": M,
                "N": N,
                "K": K,
                "device_us": dus,
                "host_us": hus,
                "GBps": gbps,
                "top_kernels": top,
            }
            native[(name, M)] = row
            results.append(row)
            k0 = top[0]["kernel"][:55] if top else "-"
            print(
                f"  {name:8s} M={M:5d}  device={dus:8.1f}us  host={hus:7.1f}us  "
                f"{gbps:7.0f}GB/s  top={k0}",
                flush=True,
            )
            del a, w
            torch.cuda.empty_cache()

    print("\n=== ARM B: FlagGems mm (enable after native) ===", flush=True)
    # Point at FlagGems-comp source if present
    os.environ.setdefault("VLLM_PLUGINS", "")
    import flag_gems

    print(f"  flag_gems={flag_gems.__file__}", flush=True)
    # Prefer only mm — avoid polluting other ops; use enable API if available
    enable = getattr(flag_gems, "enable", None) or getattr(flag_gems, "only_enable", None)
    if enable is not None:
        try:
            enable(include=["mm"])  # type: ignore[call-arg]
        except TypeError:
            try:
                flag_gems.enable()
            except Exception as e:
                print(f"  enable warn: {e}", flush=True)
    else:
        flag_gems.enable()

    gems = {}
    for name, N, K in SHAPES:
        for M in MS:
            if bytes_moved(M, N, K) > 4 * 1024**3:
                print(f"  skip {name} M={M} (too large)", flush=True)
                continue
            a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
            w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
            # Warm autotune once per shape (geometric buckets)
            for _ in range(3):
                flag_gems.mm(a, w.t())
            torch.cuda.synchronize()

            def run(a=a, w=w):
                return flag_gems.mm(a, w.t())

            dus, top = device_us(run, iters=12 if M >= 2048 else 30)
            hus = host_us(run, iters=12 if M >= 2048 else 30)
            gbps = bytes_moved(M, N, K) / (dus * 1e-6) / 1e9 if dus > 0 else 0
            nat = native.get((name, M))
            ratio = dus / nat["device_us"] if nat and nat["device_us"] > 0 else None
            row = {
                "arm": "flaggems_mm",
                "name": name,
                "M": M,
                "N": N,
                "K": K,
                "device_us": dus,
                "host_us": hus,
                "GBps": gbps,
                "vs_native_device": ratio,
                "top_kernels": top,
            }
            gems[(name, M)] = row
            results.append(row)
            k0 = top[0]["kernel"][:55] if top else "-"
            rtxt = f"{ratio:.3f}x" if ratio else "?"
            print(
                f"  {name:8s} M={M:5d}  device={dus:8.1f}us  host={hus:7.1f}us  "
                f"{gbps:7.0f}GB/s  vs_nat={rtxt}  top={k0}",
                flush=True,
            )
            del a, w
            torch.cuda.empty_cache()

    # Summary: where FlagGems wins on device
    print("\n=== WHERE FlagGems mm FASTER on device (ratio < 0.98) ===", flush=True)
    wins = []
    for (name, M), g in gems.items():
        r = g.get("vs_native_device")
        if r is not None and r < 0.98:
            wins.append((r, name, M, g["device_us"], native[(name, M)]["device_us"]))
            print(
                f"  WIN {name} M={M}: gems={g['device_us']:.1f}us "
                f"nat={native[(name, M)]['device_us']:.1f}us  ratio={r:.3f}",
                flush=True,
            )
    if not wins:
        print("  (none)", flush=True)

    print("\n=== WHERE FlagGems mm SLOWER on device (ratio > 1.05) ===", flush=True)
    losses = []
    for (name, M), g in gems.items():
        r = g.get("vs_native_device")
        if r is not None and r > 1.05:
            losses.append((r, name, M))
            print(
                f"  LOSE {name} M={M}: ratio={r:.3f}  "
                f"gems_host={g['host_us']:.1f}us nat_host={native[(name, M)]['host_us']:.1f}us",
                flush=True,
            )

    out_json = OUT / "gemm_mcprofiler.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_json}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
