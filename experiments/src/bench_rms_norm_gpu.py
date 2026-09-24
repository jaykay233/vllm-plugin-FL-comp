#!/usr/bin/env python3
"""rms_norm: measure GPU time properly, and bound how much it can matter.

Two corrections over a naive wall-clock microbenchmark:

1. Calling `torch.cuda.synchronize()` around every iteration measures
   synchronise + host-dispatch latency (~100us here), not the kernel. For a
   (1, 2048) tensor that is orders of magnitude off.

2. Decode runs under CUDA Graph capture (`Capturing CUDA graphs (decode,
   FULL)` in the vLLM log), so the whole step is a single graph replay and
   per-op Python dispatch is eliminated entirely. Only GPU kernel time counts.

So: time a long loop with CUDA events (amortising host overhead) to get real
GPU duration, then compare that against the decode step budget and the byte
volume the op actually moves.

Run:  conda activate mx && python /root/src/bench_rms_norm_gpu.py
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path("/root/bench_results/rms_norm_cmp")
OUT.mkdir(parents=True, exist_ok=True)

HIDDEN = 2048
EPS = 1e-6
MS = [1, 8, 128, 512]
LAYERS = 42
NORMS_PER_LAYER = 2
DECODE_STEP_MS = 10.15  # measured TPOT p50 at concurrency 1


def gpu_ms_per_call(fn, n=200) -> float:
    """Median GPU milliseconds per call, host overhead amortised away."""
    import torch

    for _ in range(30):
        fn()
    torch.cuda.synchronize()

    runs = []
    for _ in range(5):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        # queue n launches, then time the whole batch with events
        torch.cuda.synchronize()
        start.record()
        for _ in range(n):
            fn()
        end.record()
        torch.cuda.synchronize()
        runs.append(start.elapsed_time(end) / n)
    runs.sort()
    return runs[len(runs) // 2]


def peak_bw_gbs() -> float:
    """Copy bandwidth, as a practical ceiling for a read-dominated kernel."""
    import torch

    n = 64 * 1024 * 1024
    src = torch.empty(n, device="cuda", dtype=torch.bfloat16)
    dst = torch.empty_like(src)
    for _ in range(5):
        dst.copy_(src)
    torch.cuda.synchronize()
    best = 0.0
    for _ in range(5):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        s.record()
        dst.copy_(src)
        e.record()
        torch.cuda.synchronize()
        best = max(best, 2 * n * 2 / (s.elapsed_time(e) * 1e-3) / 1e9)
    return best


def main() -> None:
    import torch
    from vllm.ir import ops as irops

    op = irops.rms_norm
    dev = "cuda"
    dt = torch.bfloat16

    bw = peak_bw_gbs()
    print(f"  peak copy bandwidth: {bw:.0f} GB/s")
    print()
    print("=" * 94)
    print("rms_norm GPU kernel time (CUDA events, amortised over 200 launches)")
    print("=" * 94)
    print(f"  {'M':>5s} {'bytes':>9s} {'flagos us':>11s} {'native us':>11s} "
          f"{'flagos/native':>14s} {'flagos GB/s':>12s}")

    rows = []
    for M in MS:
        torch.manual_seed(0)
        x = torch.randn(M, HIDDEN, device=dev, dtype=dt)
        w = torch.randn(HIDDEN, device=dev, dtype=dt)
        nbytes = M * HIDDEN * 2 + HIDDEN * 2  # read x + w

        op.set_priority(["flagos"])
        t_flagos = gpu_ms_per_call(lambda: op(x, w, EPS))
        op.set_priority(["native"])
        t_native = gpu_ms_per_call(lambda: op(x, w, EPS))

        ratio = t_flagos / t_native
        bw_eff = nbytes / (t_flagos * 1e-3) / 1e9
        rows.append({"M": M, "nbytes": nbytes, "flagos_ms": t_flagos,
                     "native_ms": t_native, "ratio": ratio,
                     "flagos_gbs": bw_eff})
        print(f"  {M:5d} {nbytes:9d} {t_flagos * 1e3:11.3f} {t_native * 1e3:11.3f} "
              f"{ratio:13.3f}x {bw_eff:12.1f}")

    op.set_priority(["flagos", "native"])

    print()
    print("=" * 94)
    print(f"BOUND: can rms_norm matter at all? ({LAYERS} layers x "
          f"{NORMS_PER_LAYER} norms = {LAYERS * NORMS_PER_LAYER} calls at M=1)")
    print("=" * 94)
    r1 = rows[0]
    calls = LAYERS * NORMS_PER_LAYER
    total_flagos = r1["flagos_ms"] * calls
    total_native = r1["native_ms"] * calls
    print(f"  flagos kernel time  : {total_flagos:7.4f} ms  "
          f"({total_flagos / DECODE_STEP_MS * 100:.3f}% of a {DECODE_STEP_MS} ms step)")
    print(f"  native kernel time  : {total_native:7.4f} ms  "
          f"({total_native / DECODE_STEP_MS * 100:.3f}%)")
    print(f"  difference if flagos were free: {total_flagos:7.4f} ms "
          f"({total_flagos / DECODE_STEP_MS * 100:.3f}% ceiling)")
    print()
    print(f"  bytes moved by all {calls} calls at M=1: {r1['nbytes'] * calls / 1024:.1f} KiB")
    print(f"  vs ~3.7 GiB of weights read per decode step "
          f"({r1['nbytes'] * calls / (3.7 * 2**30) * 100:.5f}%)")
    print()
    if total_flagos / DECODE_STEP_MS < 0.01:
        print("  >> rms_norm is IRRELEVANT to decode latency. Even reducing its")
        print("     kernel time to zero would move the step by <1%. Optimising")
        print("     it is not a lever.")
    else:
        print("  >> rms_norm is a non-trivial share; worth attention.")
    print()
    print("  ratio flagos/native at each M (does the mandate cost anything?):")
    for r in rows:
        print(f"    M={r['M']:<4d} {r['ratio']:.3f}x")
    print("RMS_NORM_GPU_DONE", flush=True)

    (OUT / "gpu.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
