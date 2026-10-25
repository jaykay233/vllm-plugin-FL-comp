#!/usr/bin/env python3
"""Replicated histograms: kill atomic contention without changing the algorithm.

Diagnosis from probe_mass_kernel.py (rows=64, n_cols=130560, fp32, real-shaped
logits), all costs per step:

    read-only floor                 23.9 us
    atomic, zero conflict          104.9 us   <- raw atomic throughput floor
    atomic, 64 bins (hot)          240.7 us
    real kernel (8192 bins)        270.1 us
    busiest single bucket          39.8% of the row's mass on average

Contention, not bandwidth, is what costs 165 us on top of the atomic floor: the
hottest bucket receives ~52k atomics per row. Replicating the histogram R times
and sending program p to replica (p % R) divides the per-address traffic by R
while keeping the algorithm identical -- the replicas are summed at the end, so
the result matches a single histogram up to fp32 add order.

This measures the achievable speedup as a function of R, and the extra memory.
If this plateaus near the 105 us atomic floor, the remaining gap to the 24 us
read floor means the scatter has to go entirely (a pivot-search rewrite, much
riskier for sampling correctness).
"""
from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

LN2 = math.log(2.0)
N_BINS = 8192
LO = -40.0
STEP = 40.0 / N_BINS


@triton.jit
def _mass_repl_kernel(
    logits, pmax, hist, n_cols, stride_r, NBLK,
    R: tl.constexpr, NBLK_P2: tl.constexpr, N_BINS: tl.constexpr,
    LO: tl.constexpr, STEP: tl.constexpr, INV_LN2: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    blk = tl.program_id(1)
    bi = tl.arange(0, NBLK_P2)
    partials = tl.load(pmax + row * NBLK + bi, mask=bi < NBLK, other=float("-inf"))
    row_max = tl.max(partials, axis=0)
    row_max = tl.where(row_max > -3.0e38, row_max, 0.0)

    cols = blk * BLOCK + tl.arange(0, BLOCK)
    m = cols < n_cols
    x = tl.load(logits + row * stride_r + cols, mask=m, other=float("-inf"))
    e = tl.exp(x - row_max)
    bucket = ((x - row_max) * INV_LN2 - LO) / STEP
    bucket = tl.minimum(tl.maximum(bucket, 0.0), N_BINS - 1.0).to(tl.int32)
    # each program owns one replica -> R-fold less traffic per address
    rep = (row * NBLK + blk) % R
    tl.atomic_add(hist + (row * R + rep) * N_BINS + bucket, e, mask=m)


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
    nblk = 32
    gbytes = rows * n_cols * 4 / 1e9
    g = torch.Generator(device="cpu").manual_seed(0)
    base = torch.randn(rows, n_cols, generator=g) * 3.0
    base[:, :32] += 12.0
    logits = base.to("cuda").float()
    del base
    pmax = logits.view(rows, nblk, -1).amax(dim=-1).contiguous()

    print(f"rows={rows} n_cols={n_cols}  read floor 23.9 us, atomic floor 104.9 us\n")
    print(f"{'R':>4} {'us':>8} {'GB/s':>7} {'eff vs R=1':>11} {'scratch':>9}  check")

    ref = None
    base_us = None
    for R in (1, 2, 4, 8, 16, 32):
        hist = torch.zeros(rows * R, N_BINS, device="cuda", dtype=torch.float32)

        def fn(R=R, hist=hist):
            _mass_repl_kernel[(rows, nblk)](
                logits, pmax, hist, n_cols, logits.stride(0), nblk,
                R, 32, N_BINS, LO, STEP, 1.0 / LN2, 4096)

        t = bench(fn)
        if base_us is None:
            base_us = t
        # correctness: summing replicas must reproduce the R=1 histogram
        hist.zero_()
        fn()
        torch.cuda.synchronize()
        summed = hist.view(rows, R, N_BINS).sum(dim=1)
        if ref is None:
            ref = summed.clone()
            note = "reference"
        else:
            delta = (summed - ref).abs().max().item() / ref.abs().max().item()
            note = f"rel.diff {delta:.2e}"
        print(f"{R:4d} {t:8.1f} {gbytes / t * 1e6:7.0f} {base_us / t:10.2f}x "
              f"{rows * R * N_BINS * 4 / 1e6:8.1f} MB  {note}")


if __name__ == "__main__":
    main()
