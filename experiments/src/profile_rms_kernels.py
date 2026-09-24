#!/usr/bin/env python3
"""Why does the ir op cost 15.6us/call more than a direct FlagGems call?

Graph replay is pure GPU execution, so the extra time must be additional GPU
kernels captured into the graph. Count them with the profiler for:

  * ir.ops.rms_norm  (flagos provider)  -- the engaged path
  * direct gems_rms_forward             -- FlagGems with no wrapper

and also report the per-kernel GPU durations, so it is clear whether the
path is launching more kernels or slower ones.

Run:  conda activate mx && python /root/src/profile_rms_kernels.py
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

OUT = Path("/root/bench_results/rms_norm_cmp")
OUT.mkdir(parents=True, exist_ok=True)

HIDDEN = 2048
EPS = 1e-6
ITERS = 20


def profile_calls(fn, iters=ITERS):
    """Return (kernel_counts, total_gpu_us, per_kernel_us)."""
    import torch

    for _ in range(5):
        fn()
    torch.cuda.synchronize()

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CUDA,
            torch.profiler.ProfilerActivity.CPU,
        ],
        record_shapes=False,
    ) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()

    counts: Counter[str] = Counter()
    dur: Counter[str] = Counter()
    for evt in prof.events():
        if evt.device_type == torch.autograd.DeviceType.CUDA and evt.name:
            counts[evt.name] += evt.count
            dur[evt.name] += evt.device_time
    total_us = sum(dur.values())
    return counts, total_us, dur


def report(tag, counts, total_us, iters, dur):
    print(f"\n  --- {tag} ---")
    print(f"    total GPU time over {iters} iters : {total_us:9.1f} us "
          f"({total_us / iters:7.3f} us/call)")
    n_kern = sum(counts.values())
    print(f"    kernel launches over {iters} iters: {n_kern} "
          f"({n_kern / iters:7.2f} per call)")
    print(f"    distinct kernels                  : {len(counts)}")
    for name, c in counts.most_common(8):
        short = name[:64]
        print(f"      {c:5d} launches  {dur[name]:9.1f}us  {short}")


def main() -> None:
    import torch
    from vllm.ir import ops as irops

    op = irops.rms_norm
    dev = "cuda"
    dt = torch.bfloat16
    M = 1

    torch.manual_seed(0)
    x = torch.randn(M, HIDDEN, device=dev, dtype=dt)
    w = torch.randn(HIDDEN, device=dev, dtype=dt)

    print("=" * 94)
    print(f"Kernel-level breakdown, M={M}, hidden={HIDDEN}, {ITERS} iterations")
    print("=" * 94)

    op.set_priority(["flagos"])
    counts, total_us, dur = profile_calls(lambda: op(x, w, EPS))
    report("ir.ops.rms_norm (flagos)", counts, total_us, ITERS, dur)
    ir_per_call = total_us / ITERS

    op.set_priority(["native"])
    counts, total_us, dur = profile_calls(lambda: op(x, w, EPS))
    report("ir.ops.rms_norm (native)", counts, total_us, ITERS, dur)
    op.set_priority(["flagos", "native"])

    try:
        from flag_gems.modules.normalization import gems_rms_forward

        counts, total_us, dur = profile_calls(
            lambda: gems_rms_forward(x, None, w, EPS))
        report("DIRECT gems_rms_forward", counts, total_us, ITERS, dur)
        gems_per_call = total_us / ITERS

        print()
        print("=" * 94)
        print("VERDICT")
        print("=" * 94)
        print(f"  ir op (flagos)        : {ir_per_call:8.3f} us/call GPU")
        print(f"  direct gems_rms_forward: {gems_per_call:8.3f} us/call GPU")
        print(f"  ir/dispatch overhead  : {ir_per_call - gems_per_call:8.3f} us/call")
        print()
        print(f"  over 84 calls that is "
              f"{(ir_per_call - gems_per_call) * 84 / 1000:.3f} ms "
              f"= {(ir_per_call - gems_per_call) * 84 / 10000 * 100:.2f}% "
              f"of a 10.15 ms decode step")
        print()
        print("  Kernel count per call is the tell: if the ir path launches")
        print("  noticeably more kernels, the overhead is extra GPU work rather")
        print("  than a slower kernel, and the fix is to avoid the wrapper.")
    except Exception as e:  # noqa: BLE001
        print(f"\n  direct gems_rms_forward unavailable: {e}")

    print("\nPROFILE_RMS_DONE", flush=True)


if __name__ == "__main__":
    main()
