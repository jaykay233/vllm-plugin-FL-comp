#!/usr/bin/env python3
"""Bound the block_table upload cost before writing any patch.

Per decode step the engine calls BlockTable.commit_block_table(num_reqs), which is
    self.block_table.copy_to_gpu(num_reqs)
    -> self.gpu[:n].copy_(self.cpu[:n], non_blocking=True)
i.e. a pinned H2D copy of the *whole* [n, max_num_blocks_per_req] rectangle, even
though a decode step only appends one new KV block per request.

Sizing: max_num_blocks_per_req = cdiv(max_model_len, block_size) = 131072/128 = 1024,
and n = max_num_seqs in the batch. So the copy is 64 x 1024 x 4 B = 256 KB every
step, while the *live* prefix at 4k context is only cdiv(4096,128) = 32 columns
(8 KB), 32x smaller.

Two questions to answer with numbers, not intuition:
  1. what does the 256 KB H2D copy actually cost on this box, GPU and wall?
  2. what would an extent-limited copy cost, and is the saving even measurable
     against a 17.84 ms step?

Also times the numpy append, since that is host-side work on the same thread.
"""
from __future__ import annotations

import time

import numpy as np
import torch

MAX_REQS = 64
MAX_BLOCKS = 1024          # cdiv(131072, 128)
COLS_4K = 32               # cdiv(4096, 128)
COLS_16K = 128             # cdiv(16384, 128)
REPEATS = 2000


def bench_gpu(fn, repeats=REPEATS):
    for _ in range(50):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(repeats):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / repeats * 1000  # us


def bench_host(fn, repeats=REPEATS):
    for _ in range(50):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    dt = time.perf_counter() - t0
    return dt / repeats * 1e6  # us


def main() -> None:
    cpu = torch.zeros(MAX_REQS, MAX_BLOCKS, dtype=torch.int32, pin_memory=True)
    gpu = torch.zeros_like(cpu, device="cuda")
    npview = cpu.numpy()
    for r in range(MAX_REQS):
        npview[r, :COLS_4K] = np.arange(COLS_4K) + r * MAX_BLOCKS

    print(f"table [ {MAX_REQS}, {MAX_BLOCKS} ] int32 = {MAX_REQS * MAX_BLOCKS * 4 / 1024:.0f} KB")
    print(f"live extent at 4k context  = {COLS_4K} cols = {MAX_REQS * COLS_4K * 4 / 1024:.1f} KB")
    print(f"live extent at 16k context = {COLS_16K} cols = {MAX_REQS * COLS_16K * 4 / 1024:.1f} KB\n")

    cases = [
        ("full table (today)", lambda: gpu[:MAX_REQS].copy_(cpu[:MAX_REQS], non_blocking=True)),
        ("extent 4k  (32 cols)", lambda: gpu[:MAX_REQS, :COLS_4K].copy_(cpu[:MAX_REQS, :COLS_4K], non_blocking=True)),
        ("extent 16k (128 cols)", lambda: gpu[:MAX_REQS, :COLS_16K].copy_(cpu[:MAX_REQS, :COLS_16K], non_blocking=True)),
        ("one column (strided)", lambda: gpu[:MAX_REQS, 32:33].copy_(cpu[:MAX_REQS, 32:33], non_blocking=True)),
    ]
    for name, fn in cases:
        print(f"{name:<24} gpu {bench_gpu(fn):7.2f} us   host {bench_host(fn):7.2f} us")

    # the numpy append that runs before the copy, for reference
    ids = np.arange(4, dtype=np.int32)
    def append():
        for r in range(MAX_REQS):
            npview[r, COLS_4K:COLS_4K + 4] = ids
    print(f"\n{'numpy append x64 rows':<24} host {bench_host(append):7.2f} us")

    full_gpu = bench_gpu(cases[0][1])
    full_host = bench_host(cases[0][1])
    ext_gpu = bench_gpu(cases[1][1])
    ext_host = bench_host(cases[1][1])
    step_ms = 17.84
    print(f"\nstep is {step_ms:.2f} ms of GPU")
    print(f"  full: gpu {full_gpu / 1000:.4f} ms = {full_gpu / 1000 / step_ms * 100:.3f}% of step")
    print(f"  extent-limited saves {(full_gpu - ext_gpu) / 1000:.4f} ms gpu "
          f"({(full_gpu - ext_gpu) / 1000 / step_ms * 100:.3f}% of step)")
    print(f"  host time saved {(full_host - ext_host):+.2f} us "
          f"({(full_host - ext_host) / 1000 / step_ms * 100:+.3f}% of step)")


if __name__ == "__main__":
    main()
