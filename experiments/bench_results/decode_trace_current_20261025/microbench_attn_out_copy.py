#!/usr/bin/env python3
"""Is the per-layer attention-output writeback a fixed copy-engine cost?

The decode trace shows, in every one of the 42 layers:

    flash_fwd_splitkv_combine_kernel   (attention done)
    Memcpy ... 262144 B  ~35 us        <- `output[:n] = attn_out`
    mcblas gemm 64x32x256              (o_proj)

262144 B is exactly 64 tokens x 2048 hidden x 2 B, and 35 us implies 7.5 GB/s,
which is slower than the PCIe H2D copy measured in the same trace (19 GB/s).
A device-to-device copy of that size should be a few microseconds.  If the cost
is copy-engine submission overhead rather than bandwidth, then replacing
`copy_` with any plain elementwise kernel that writes into the destination
should be several times faster.

This compares, on the exact shape used by decode:
  * flat_copy    -- `dst.copy_(src)`; contiguous same-dtype -> cudaMemcpyAsync
  * flat_copy_nb -- same, but with explicit non_blocking
  * add_zero     -- `torch.add(src, 0, out=dst)`; elementwise kernel
  * mul_one      -- `torch.mul(src, 1.0, out=dst)`; elementwise kernel
  * copy_kernel  -- `dst.copy_(src)` from a same-shape non-contiguous view so
                    ATen falls back from memcpy to ElementwiseCopy kernel

Timing uses CUDA events around 42 back-to-back ops (one per layer) and also
measures a single synchronised op, since in the real step these are separated
by kernels and each one's latency is what adds up.
"""
from __future__ import annotations

import torch

N_TOK, HID = 64, 2048
ITERS = 42        # one copy per layer
REPEATS = 200


def bench(fn, iters=ITERS, repeats=REPEATS):
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(repeats):
        for _ in range(iters):
            fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / (repeats * iters) * 1000  # us/op


def bench_latency(fn, repeats=200):
    """One op, full sync each: latency including submission."""
    tot = 0.0
    for _ in range(repeats):
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        tot += start.elapsed_time(end)
    return tot / repeats * 1000  # us


def main() -> None:
    dev = "cuda"
    src = torch.randn(N_TOK, HID, dtype=torch.bfloat16, device=dev)
    dst = torch.empty_like(src)
    # a non-contiguous same-dtype source holding the same values as src
    big = torch.randn(N_TOK, HID * 2, dtype=torch.bfloat16, device=dev)
    src_nc = big[:, ::2]
    src_nc.copy_(src)
    assert not src_nc.is_contiguous() and torch.equal(src_nc, src)

    cases = {
        "flat_copy": lambda: dst.copy_(src),
        "flat_copy_nb": lambda: dst.copy_(src, non_blocking=True),
        "add_zero": lambda: torch.add(src, 0.0, out=dst),
        "mul_one": lambda: torch.mul(src, 1.0, out=dst),
        "copy_kernel": lambda: dst.copy_(src_nc),
    }

    # correctness: every path must land the same bytes
    ref = src.clone()
    for name, fn in cases.items():
        dst.zero_()
        fn()
        torch.cuda.synchronize()
        ok = torch.equal(dst, ref)
        assert ok, f"{name} produced wrong bytes"
    print("all paths produce identical bytes\n")

    print(f"{'case':<14} {'us/op':>8} {'us/42':>10} {'GB/s':>7} {'us/op(sync)':>12}")
    print("-" * 56)
    nbytes = N_TOK * HID * 2
    results = {}
    for name, fn in cases.items():
        t = bench(fn)
        lat = bench_latency(fn)
        results[name] = t
        print(f"{name:<14} {t:>8.2f} {t * ITERS:>10.1f} "
              f"{nbytes / t / 1e3:>7.2f} {lat:>12.2f}")

    base = results["flat_copy"]
    print("\nsavings per decode step (42 copies)")
    for name, t in results.items():
        if name == "flat_copy":
            continue
        print(f"  {name:<14} {(base - t) * ITERS / 1000:>6.3f} ms/step")


if __name__ == "__main__":
    main()
