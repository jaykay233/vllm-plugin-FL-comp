#!/usr/bin/env python3
"""True per-kernel cost of the whole top-p chain, measured in isolation.

Trace attribution per decode step (rows=64, vocab=130560, fp32):

    _top_p_rowmax_v2_kernel     24 us   33.4 MB read
    _top_p_mass_v2_kernel      385 us   33.4 MB read, 8.36M atomics
    _top_p_amax_v2_kernel       41 us   reads 8 KB of partials only
    _top_p_threshold_v2_kernel  12 us   reads 2 MB of hist
    _top_p_mask_v2_kernel       47 us   33.4 MB read + 33.4 MB write

amax at 41 us for 8 KB of input is wildly out of line -- a grid of 64 programs
reading 32 floats each should be a few microseconds. Either the trace number
includes launch/gap, or tiny kernels have a large fixed cost on this platform.
That distinction matters: if small kernels each cost ~40 us regardless of work,
the 5-kernel chain is paying ~200 us of pure launch overhead, and fusing the
chain would be worth more than optimising mass.

Measures each kernel alone (repeated back-to-back, so launch cost is amortised
exactly as it is in a step) and the fused alternative for amax.
"""
from __future__ import annotations

import math

import torch
from flag_gems.ops.top_p_threshold import (
    _top_p_amax_v2_kernel,
    _top_p_mask_v2_kernel,
    _top_p_mass_v2_kernel,
    _top_p_rowmax_v2_kernel,
    _top_p_threshold_v2_kernel,
)

LN2 = math.log(2.0)
N_BINS, LO, HI = 8192, -40.0, 0.0
STEP = (HI - LO) / N_BINS
BLOCK = 4096


def bench(fn, repeats=300):
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(repeats):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / repeats * 1000


def main() -> None:
    rows, n_cols = 64, 130560
    nblk = (n_cols + BLOCK - 1) // BLOCK
    g = torch.Generator(device="cpu").manual_seed(0)
    base = torch.randn(rows, n_cols, generator=g) * 3.0
    base[:, :32] += 12.0
    logits = base.to("cuda").float()
    del base

    pmax = torch.empty(rows, nblk, device="cuda", dtype=torch.float32)
    hist = torch.zeros(rows, N_BINS, device="cuda", dtype=torch.float32)
    amax = torch.empty(rows, device="cuda", dtype=torch.float32)
    thresh = torch.empty(rows, device="cuda", dtype=torch.float32)
    p = torch.full((rows,), 0.95, device="cuda", dtype=torch.float32)
    out = torch.empty_like(logits)

    grid2d = (rows, nblk)
    cases = [
        ("rowmax", lambda: _top_p_rowmax_v2_kernel[grid2d](logits, pmax, n_cols, logits.stride(0), nblk, BLOCK)),
        ("hist.zero_ (2 MB)", lambda: hist.zero_()),
        ("mass", lambda: _top_p_mass_v2_kernel[grid2d](
            logits, pmax, hist, n_cols, logits.stride(0), nblk,
            nblk, N_BINS, LO, STEP, 1.0 / LN2, BLOCK)),
        ("amax", lambda: _top_p_amax_v2_kernel[(rows,)](pmax, amax, nblk, nblk)),
        ("threshold", lambda: _top_p_threshold_v2_kernel[(rows,)](
            hist, amax, p, thresh, N_BINS, LO, STEP, LN2)),
        ("mask", lambda: _top_p_mask_v2_kernel[grid2d](
            logits, out, thresh, n_cols, logits.stride(0), BLOCK)),
        ("empty launch (grid 64,1)", lambda: _top_p_amax_v2_kernel[(rows,)](pmax, amax, nblk, nblk)),
    ]
    print(f"rows={rows} n_cols={n_cols} nblk={nblk}\n")
    print(f"{'kernel':<28} {'us':>8}")
    total = 0.0
    for name, fn in cases:
        t = bench(fn)
        if name != "empty launch (grid 64,1)":
            total += t
        print(f"{name:<28} {t:8.2f}")
    print(f"{'-' * 36}\n{'sum':<28} {total:8.2f}")

    # is a *tiny* kernel launch really ~40 us, or is the trace number inflated?
    tiny = torch.empty(1, device="cuda", dtype=torch.float32)
    t = bench(lambda: _top_p_amax_v2_kernel[(1,)](pmax[:1], tiny, nblk, nblk))
    print(f"\nlaunch cost, grid (1,): {t:.2f} us")
    t = bench(lambda: _top_p_amax_v2_kernel[(1024,)](pmax, amax, nblk, nblk))
    print(f"launch cost, grid (1024,): {t:.2f} us")


if __name__ == "__main__":
    main()
