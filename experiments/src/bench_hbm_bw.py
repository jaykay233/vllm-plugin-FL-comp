#!/usr/bin/env python3
"""Measure achievable HBM bandwidth on this card with plain copies."""
from __future__ import annotations

import torch

torch.cuda.init()
n = 1 << 28  # 1 GiB of bf16 = 2 GiB traffic per copy
a = torch.empty(n, dtype=torch.uint8, device="cuda")
b = torch.empty(n, dtype=torch.uint8, device="cuda")
a.fill_(1)
torch.cuda.synchronize()


def timeit(fn, iters=10):
    fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(True); e = torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


ms = timeit(lambda: b.copy_(a))
print(f"copy 1 GiB: {ms:.2f} ms -> {2 * n / (ms / 1e3) / 1e9:.0f} GB/s (r+w)")
ms = timeit(lambda: a.zero_())
print(f"fill 1 GiB: {ms:.2f} ms -> {n / (ms / 1e3) / 1e9:.0f} GB/s (w)")
ms = timeit(lambda: a.sum())
print(f"sum  1 GiB: {ms:.2f} ms -> {n / (ms / 1e3) / 1e9:.0f} GB/s (r)")

# read-bandwidth with a reduction-free op: 4 GiB fit in memory? use 512 MiB
m = 1 << 27
x = torch.empty(m, dtype=torch.uint8, device="cuda")
y = torch.empty(m, dtype=torch.uint8, device="cuda")


def rw():
    y.copy_(x)


ms = timeit(rw, 20)
print(f"copy 512 MiB: {ms:.2f} ms -> {2 * m / (ms / 1e3) / 1e9:.0f} GB/s (r+w)")
