#!/usr/bin/env python3
"""Can masked reductions replace the scatter-histogram in the mass pass?

Established by probe_mass_kernel.py at rows=64, n_cols=130560, fp32:
    read-only floor                 23.9 us  (1399 GB/s)
    atomic, zero conflict          104.9 us  -> raw atomic throughput floor
    atomic, 64 bins (hot)          240.7 us  -> contention adds ~150 us
    real kernel (8192 bins)        270.1 us
and the distribution is brutally peaked: the single busiest bucket carries 39.8%
of the row's mass on average, top-16 buckets carry 49.8%.

So a per-element scatter cannot be made fast here: even conflict-free atomics cost
4x the read. The alternative is to never scatter -- compute the *suffix mass* at S
fixed levels with masked block reductions, which is pure streaming compute:

    for j in range(S): acc[j] = sum(where(bucket <= j, e, 0))

That yields the CDF at S points in one pass, and S is a compute/memory tradeoff
with no atomics and no scatter. This measures the cost as a function of S, using
per-element bucket indices derived from the real logit distribution, and checks
what resolution S buys.

Also measures the two refinement strategies:
  single pass, S levels   -> range/40 ln2 divided by S
  two passes, S levels    -> divided by S^2, at the cost of a second read (24 us)
"""
from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

LN2 = math.log(2.0)
LO, HI = -40.0, 0.0
STEP_FULL = (HI - LO) / 8192


@triton.jit
def _suffix_mass_kernel(
    logits, pmax, out, n_cols, stride_r, NBLK_P2: tl.constexpr,
    S: tl.constexpr, LO: tl.constexpr, SPAN: tl.constexpr, BLOCK: tl.constexpr,
):
    """out[row, blk, j] = sum of exp(x-max) over the block for bucket <= j."""
    row = tl.program_id(0)
    blk = tl.program_id(1)
    bi = tl.arange(0, NBLK_P2)
    partials = tl.load(pmax + row * 32 + bi, mask=bi < 32, other=float("-inf"))
    row_max = tl.max(partials, axis=0)
    row_max = tl.where(row_max > -3.0e38, row_max, 0.0)

    cols = blk * BLOCK + tl.arange(0, BLOCK)
    m = cols < n_cols
    x = tl.load(logits + row * stride_r + cols, mask=m, other=float("-inf"))
    e = tl.exp(x - row_max)
    # fractional position in [0, S) over the LO..LO+SPAN window, clamped
    pos = (x - row_max) / 0.6931471805599453 - LO
    pos = tl.minimum(tl.maximum(pos * (S / SPAN), 0.0), S - 1e-6)
    b = pos.to(tl.int32)

    for j in tl.static_range(S):
        acc = tl.sum(tl.where(b <= j, e, 0.0), axis=0)
        tl.store(out + (row * 32 + blk) * S + j, acc)


def bench(fn, repeats=200):
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
    gbytes = rows * n_cols * 4 / 1e9
    g = torch.Generator(device="cpu").manual_seed(0)
    base = torch.randn(rows, n_cols, generator=g) * 3.0
    base[:, :32] += 12.0
    logits = base.to("cuda").float()
    del base
    pmax = logits.view(rows, 32, -1).amax(dim=-1).contiguous()

    print(f"rows={rows} n_cols={n_cols}, read floor 23.9 us\n")
    print(f"{'levels S':>9} {'us':>8} {'GB/s':>7} {'resolution (ln2)':>18} {'vs 8192 bins':>13}")
    for S in (16, 32, 64, 128, 256):
        out = torch.empty(rows, 32, S, device="cuda", dtype=torch.float32)
        fn = lambda S=S, out=out: _suffix_mass_kernel[(rows, 32)](
            logits, pmax, out, n_cols, logits.stride(0), 32,
            S, LO, 40.0, 4096)
        t = bench(fn)
        res = 40.0 / S
        print(f"{S:9d} {t:8.1f} {gbytes / t * 1e6:7.0f} {res:18.5f} {8192 / S:12.0f}x coarser")

    print("\ntwo-pass equivalent resolution (vs 8192 bins = 0.004883 ln2):")
    for S in (16, 32, 64, 128):
        res = 40.0 / (S * S)
        print(f"  S={S:3d}: {res:.6f} ln2  "
              f"({'finer' if res < 0.004883 else 'coarser'} than 8192 bins)")


if __name__ == "__main__":
    main()
