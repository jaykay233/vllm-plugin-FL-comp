#!/usr/bin/env python3
"""Why is _top_p_mass_v2_kernel 87 GB/s when its siblings run at peak?

Per-step decode trace, rows=64, n_cols=130560, fp32:
    _top_p_rowmax_v2_kernel    24 us   33.4 MB read   -> 1392 GB/s  (peak)
    _top_p_mass_v2_kernel     385 us   33.4 MB read   ->   87 GB/s  <-- outlier
    _top_p_amax_v2_kernel      41 us   reads 8 KB partials, latency floor
    _top_p_threshold_v2_kernel 12 us   reads 2 MB hist
    _top_p_mask_v2_kernel      47 us   33.4 MB r+w    -> 1421 GB/s  (peak)

The mass kernel differs from its siblings in exactly one respect: it issues one
`tl.atomic_add` per element into a (rows, 8192) histogram, i.e. 8.36M atomics per
step. Everything else is the same streaming read.

Two candidate causes, distinguishable by experiment:
  (A) contention -- many elements hit the same bucket, so atomics serialise on
      those addresses;
  (B) raw atomic throughput -- atomics are simply ~5x more expensive than a
      streaming load on this hardware, regardless of conflicts.

Variants:
  v0  real kernel (atomic into 8192 bins)
  v1  same read, no atomic: accumulate a dummy value with tl.sum and store one
      scalar. Gives the pure-read floor. If v1 ~24 us, the read is not the issue.
  v2  atomic into 8192 *distinct* per-element addresses (no conflict at all).
      v0 ~ v2  => cause (B), contention is irrelevant.
  v2b atomic into 64 bins (heavy conflict). v2b >> v2 confirms sensitivity.
  v3  real distribution shape check: how much mass sits in the top few bins, and
      how many elements land in bin 0 (the clamp bucket).
"""
from __future__ import annotations

import math

import torch
import triton
import triton.language as tl
from flag_gems.ops.top_p_threshold import _top_p_mass_v2_kernel

LN2 = math.log(2.0)
N_BINS = 8192
LO, HI = -40.0, 0.0
STEP = (HI - LO) / N_BINS


@triton.jit
def _read_only_kernel(logits, out, n_cols, stride_r, NBLK_P2: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    blk = tl.program_id(1)
    cols = blk * BLOCK + tl.arange(0, BLOCK)
    m = cols < n_cols
    x = tl.load(logits + row * stride_r + cols, mask=m, other=float("-inf"))
    tl.store(out + row * 64 + blk, tl.sum(x, axis=0))


@triton.jit
def _atomic_distinct_kernel(logits, hist, n_cols, stride_r, NB: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    blk = tl.program_id(1)
    cols = blk * BLOCK + tl.arange(0, BLOCK)
    m = cols < n_cols
    x = tl.load(logits + row * stride_r + cols, mask=m, other=float("-inf"))
    # unique address per element -> no contention at all
    tl.atomic_add(hist + cols, tl.exp(x), mask=m)


@triton.jit
def _atomic_conflict_kernel(logits, hist, n_cols, stride_r, NB: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    blk = tl.program_id(1)
    cols = blk * BLOCK + tl.arange(0, BLOCK)
    m = cols < n_cols
    x = tl.load(logits + row * stride_r + cols, mask=m, other=float("-inf"))
    b = (cols % NB).to(tl.int32)
    tl.atomic_add(hist + row * NB + b, tl.exp(x), mask=m)


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
    gbytes = rows * n_cols * 4 / 1e9

    # A realistic, heavy-tailed logit row: gaussian core plus a few large outliers,
    # which is what an lm_head produces after bf16 rounding.
    g = torch.Generator(device="cpu").manual_seed(0)
    base = torch.randn(rows, n_cols, generator=g) * 3.0
    base[:, :32] += 12.0                      # a few dominant candidates
    logits = base.to("cuda").float()
    del base

    pmax = torch.empty(rows, 32, device="cuda", dtype=torch.float32)
    hist = torch.zeros(rows, N_BINS, device="cuda", dtype=torch.float32)
    out = torch.empty(rows, 64, device="cuda", dtype=torch.float32)

    def run_mass():
        _top_p_mass_v2_kernel[(rows, 32)](
            logits, pmax, hist, n_cols, logits.stride(0), 32,
            32, N_BINS, LO, STEP, 1.0 / LN2, 4096)

    pmax.copy_(logits.view(rows, 32, -1).amax(dim=-1))

    print(f"rows={rows} n_cols={n_cols} read={gbytes * 1e3:.1f} MB\n")
    print(f"{'variant':<34} {'us':>8} {'GB/s':>8}")
    t = bench(run_mass); print(f"{'v0 real kernel (8192 bins, atomic)':<34} {t:8.1f} {gbytes / t * 1e6:8.0f}")

    t = bench(lambda: _read_only_kernel[(rows, 32)](logits, out, n_cols, logits.stride(0), 32, 4096))
    print(f"{'v1 read-only floor':<34} {t:8.1f} {gbytes / t * 1e6:8.0f}")

    t = bench(lambda: _atomic_distinct_kernel[(rows, 32)](logits, hist.view(-1), n_cols, logits.stride(0), N_BINS, 4096))
    print(f"{'v2 atomic, zero conflict':<34} {t:8.1f} {gbytes / t * 1e6:8.0f}")

    t = bench(lambda: _atomic_conflict_kernel[(rows, 32)](logits, hist.view(-1), n_cols, logits.stride(0), 8192, 4096))
    print(f"{'v2b atomic, 8192 bins synthetic':<34} {t:8.1f} {gbytes / t * 1e6:8.0f}")

    t = bench(lambda: _atomic_conflict_kernel[(rows, 32)](logits, hist.view(-1), n_cols, logits.stride(0), 64, 4096))
    print(f"{'v2c atomic, 64 bins (max conflict)':<34} {t:8.1f} {gbytes / t * 1e6:8.0f}")

    # distribution shape of the real kernel's buckets
    hist.zero_()
    run_mass()
    torch.cuda.synchronize()
    h = hist.cpu()
    tot = h.sum(dim=-1)
    nonzero = (h > 0).sum(dim=-1).float()
    b0 = h[:, 0] / tot
    top16 = h[:, -16:].sum(dim=-1) / tot
    print(f"\nbucket occupancy per row: mean {nonzero.mean():.0f} of {N_BINS} bins populated")
    print(f"  mass in bin 0 (the clamped bucket): mean {b0.mean() * 100:.2f}%  max {b0.max() * 100:.2f}%")
    print(f"  mass in top 16 bins:                mean {top16.mean() * 100:.4f}%  max {top16.max() * 100:.4f}%")
    counts = (h > 0).sum(dim=-1)
    print(f"  busiest bin holds {h.max(dim=-1).values.mean() / tot.mean() * 100:.2f}% of mass on average")


if __name__ == "__main__":
    main()
