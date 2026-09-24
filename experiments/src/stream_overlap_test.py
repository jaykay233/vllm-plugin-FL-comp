#!/usr/bin/env python3
"""Decide whether MetaX/MACA streams give real CONCURRENCY.

The first attempt (stream_support_test.py) compared two 4096^3 bf16 matmul
chains on 1 vs 2 streams and saw 1.00x.  That result is confounded: a single
stream already saturates all 104 SMs, so overlap is impossible by construction.

The proper test uses a kernel with a deliberately tiny grid, so most SMs sit
idle and a second stream *could* fill them:
  * concurrent -> 2 streams take about the same wall time as 1
  * serialized -> 2 streams take about twice as long
Plus a copy/compute overlap test, which is the case vLLM actually cares about
(the plugin's async output copy stream).

Usage:
    /opt/conda/envs/mx/bin/python /root/src/stream_overlap_test.py
"""

from __future__ import annotations

import time

import torch
import triton
import triton.language as tl

DEV = "cuda"


@triton.jit
def _spin(ptr, iters, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for _ in range(iters):
        acc += tl.load(ptr + offs) * 1.000001
    tl.store(ptr + offs, acc)


def spin(iters: int, grid: int, stream=None):
    buf = torch.ones(256, device=DEV, dtype=torch.float32)
    ctx = torch.cuda.stream(stream) if stream is not None else torch.cuda.stream(
        torch.cuda.current_stream()
    )
    with ctx:
        _spin[(grid,)](buf, iters, BLOCK=256)
    return buf


def calibrate(target_ms: float, grid: int) -> int:
    """Find an iteration count that takes roughly target_ms on the given grid.

    The first invocation of a Triton kernel pays JIT compilation (seconds), so
    it must be warmed up before any timing -- otherwise the calibration thinks
    a tiny kernel already meets the target and returns a useless iteration
    count.
    """
    spin(1000, grid)  # trigger JIT + warm up
    torch.cuda.synchronize()

    it = 2000
    for _ in range(14):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        spin(it, grid)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1e3
        if ms >= target_ms:
            return it
        it = max(int(it * 2), int(it * target_ms / max(ms, 1e-3)))
    return it


def main() -> None:
    p = torch.cuda.get_device_properties(0)
    sms = getattr(p, "multi_processor_count", 104)
    print(f"device {torch.cuda.get_device_name(0)}  SMs={sms}")
    print(f"\n{'=' * 78}\nA. concurrency with a HIGH-OCCUPANCY kernel (grid = {sms})\n{'=' * 78}")

    it = calibrate(25.0)
    print(f"  calibrated iters={it}")

    def timed(fn, reps=3):
        best = float("inf")
        for _ in range(reps):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            best = min(best, (time.perf_counter() - t0) * 1e3)
        return best

    for grid, label in ((sms, "saturating"), (1, "single-block"), (sms // 8, "low-occupancy")):
        one = timed(lambda: spin(it, grid))
        seq = timed(lambda: (spin(it, grid), spin(it, grid)))
        s1, s2 = torch.cuda.Stream(), torch.cuda.Stream()
        par = timed(lambda: (spin(it, grid, s1), spin(it, grid, s2)))
        print(f"\n  grid={grid:3d} ({label})   1x={one:7.2f} ms   2x-1stream={seq:7.2f}   "
              f"2x-2streams={par:7.2f}")
        print(f"     -> speedup = {seq / par:.2f}x   "
              f"({'CONCURRENT' if seq / par > 1.3 else 'SERIALIZED'})")

    print(f"\n{'=' * 78}\nB. copy / compute overlap (the vLLM async-output pattern)\n{'=' * 78}")
    try:
        mb = 256
        n = mb * 1024 * 1024 // 4
        host = torch.empty(n, dtype=torch.float32, pin_memory=True)
        host.fill_(1.0)
        gpu = torch.empty(n, dtype=torch.float32, device=DEV)

        copy_stream = torch.cuda.Stream()

        def copy_only():
            with torch.cuda.stream(copy_stream):
                gpu.copy_(host, non_blocking=True)
            torch.cuda.synchronize()

        def compute_only():
            spin(it, sms)
            torch.cuda.synchronize()

        t_copy = timed(copy_only)
        t_comp = timed(compute_only)

        def overlapped():
            with torch.cuda.stream(copy_stream):
                gpu.copy_(host, non_blocking=True)
            spin(it, sms)
            torch.cuda.synchronize()

        t_ovl = timed(overlapped)
        print(f"  {mb} MB pinned H2D alone   -> {t_copy:7.2f} ms")
        print(f"  compute alone            -> {t_comp:7.2f} ms")
        print(f"  overlapped (target=compute or copy alone) -> {t_ovl:7.2f} ms")
        print(f"  predicted if fully serial      -> {t_copy + t_comp:7.2f} ms")
        print(f"  predicted if fully concurrent  -> {max(t_copy, t_comp):7.2f} ms")
    except Exception as e:  # noqa: BLE001
        print(f"  FAILED: {type(e).__name__}: {e}")

    print("\nOVERLAP_TEST_DONE", flush=True)


if __name__ == "__main__":
    main()
