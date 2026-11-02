#!/usr/bin/env python3
"""Two cheap questions about the mass kernel's atomics.

1. Is it atomic-throughput-bound? Run the real distribution at 1/4, 1/2, 3/4 and
   full column counts. Linear scaling proves throughput bound (nothing but fewer
   atomics helps); sublinear would mean a fixed overhead dominates.

2. Does the atomic memory ordering matter? A histogram sum needs no ordering at
   all, so `sem="relaxed"` is semantically identical to the default (acq_rel on
   Triton 3.0). If the default pays for fence/ordering that a relaxed RMW avoids,
   this is a one-word change in flag_gems with zero numerical risk. Also tries
   `scope="gpu"` vs the default scope, and int32 fixed-point accumulation, since
   an integer RMW can be cheaper than a float RMW on some ISA.
"""
from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

LN2 = math.log(2.0)
N_BINS, LO, HI = 8192, -40.0, 0.0
STEP = (HI - LO) / N_BINS
BLOCK = 4096


@triton.jit
def _mass_sem(logits, pmax, hist, n_cols, stride_r, NBLK,
              SEM: tl.constexpr, NBLK_P2: tl.constexpr, N_BINS: tl.constexpr,
              LO: tl.constexpr, STEP: tl.constexpr, INV_LN2: tl.constexpr,
              BLOCK: tl.constexpr):
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
    b = ((x - row_max) * INV_LN2 - LO) / STEP
    b = tl.minimum(tl.maximum(b, 0.0), N_BINS - 1.0).to(tl.int32)
    tl.atomic_add(hist + row * N_BINS + b, e, mask=m, sem=SEM)


@triton.jit
def _mass_int(logits, pmax, hist, n_cols, stride_r, NBLK,
              NBLK_P2: tl.constexpr, N_BINS: tl.constexpr, LO: tl.constexpr,
              STEP: tl.constexpr, INV_LN2: tl.constexpr, SCALE: tl.constexpr,
              BLOCK: tl.constexpr):
    """Same accumulation in fixed point: int32 RMW, converted back later."""
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
    b = ((x - row_max) * INV_LN2 - LO) / STEP
    b = tl.minimum(tl.maximum(b, 0.0), N_BINS - 1.0).to(tl.int32)
    q = (e * SCALE).to(tl.int32)
    tl.atomic_add(hist + row * N_BINS + b, q, mask=m, sem="relaxed")


def bench(fn, repeats=200):
    for _ in range(15):
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
    rows, full = 64, 130560
    g = torch.Generator(device="cpu").manual_seed(0)
    base = torch.randn(rows, full, generator=g) * 3.0
    base[:, :32] += 12.0
    logits = base.to("cuda").float()
    del base

    print("--- 1. does mass scale linearly with element count? ---")
    print(f"{'cols':>8} {'us':>8} {'us/Mcol':>9}")
    prev = None
    for frac in (0.25, 0.5, 0.75, 1.0):
        n_cols = int(full * frac) // BLOCK * BLOCK
        nblk = n_cols // BLOCK
        pmax = logits[:, :n_cols].view(rows, nblk, -1).amax(dim=-1).contiguous()
        hist = torch.zeros(rows, N_BINS, device="cuda", dtype=torch.float32)
        np2 = triton.next_power_of_2(nblk)
        fn = lambda n_cols=n_cols, nblk=nblk, pmax=pmax, hist=hist, np2=np2: _mass_sem[(rows, nblk)](
            logits, pmax, hist, n_cols, logits.stride(0), nblk,
            "acq_rel", np2, N_BINS, LO, STEP, 1.0 / LN2, BLOCK)
        t = bench(fn)
        tag = "" if prev is None else f"  ({t / prev:.2f}x time for {frac:.2f}x/0.25x cols)"
        prev = prev or t
        print(f"{n_cols:8d} {t:8.1f} {t / (rows * n_cols / 1e6):9.2f}{tag}")

    print("\n--- 2. memory ordering / dtype of the RMW ---")
    full = full // BLOCK * BLOCK   # 130560 is not a multiple of BLOCK
    nblk = full // BLOCK
    pmax = logits[:, :full].view(rows, nblk, -1).amax(dim=-1).contiguous()
    hist = torch.zeros(rows, N_BINS, device="cuda", dtype=torch.float32)
    histi = torch.zeros(rows, N_BINS, device="cuda", dtype=torch.int32)
    print(f"{'variant':<28} {'us':>8}")
    np2 = triton.next_power_of_2(nblk)
    for sem in ("acq_rel", "relaxed"):
        t = bench(lambda sem=sem: _mass_sem[(rows, nblk)](
            logits, pmax, hist, full, logits.stride(0), nblk,
            sem, np2, N_BINS, LO, STEP, 1.0 / LN2, BLOCK))
        print(f"{'fp32 atomic sem=' + sem:<28} {t:8.1f}")
    for scale in (1 << 16,):
        t = bench(lambda scale=scale: _mass_int[(rows, nblk)](
            logits, pmax, histi, full, logits.stride(0), nblk,
            np2, N_BINS, LO, STEP, 1.0 / LN2, scale, BLOCK))
        print(f"{'int32 fixed point 2^16':<28} {t:8.1f}")

    # correctness of the relaxed path against the default
    hist.zero_()
    _mass_sem[(rows, nblk)](logits, pmax, hist, full, logits.stride(0), nblk,
                            "acq_rel", np2, N_BINS, LO, STEP, 1.0 / LN2, BLOCK)
    torch.cuda.synchronize()
    ref = hist.clone()
    hist.zero_()
    _mass_sem[(rows, nblk)](logits, pmax, hist, full, logits.stride(0), nblk,
                            "relaxed", np2, N_BINS, LO, STEP, 1.0 / LN2, BLOCK)
    torch.cuda.synchronize()
    d = (hist - ref).abs().max().item() / ref.abs().max().item()
    print(f"\nrelaxed vs acq_rel histogram rel.diff: {d:.2e}")


if __name__ == "__main__":
    main()
