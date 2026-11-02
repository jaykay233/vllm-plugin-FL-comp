#!/usr/bin/env python3
"""Streaming-read bandwidth ceiling, bf16 vs fp8.

Decode attention already runs at ~1.21 TB/s of bf16 KV traffic, which is close
to the measured copy ceiling. Two questions this answers:

1. What is the actual read-only ceiling (copies are r+w and understate it)?
2. Can a Triton kernel reach that ceiling at all on this card? (If Triton cannot
   stream at ~1.2 TB/s, a custom fp8 decode-attention kernel cannot win.)

Run: python experiments/src/bench_stream_read.py
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def stream_read_kernel(x_ptr, out_ptr, n_elem, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    # 4 independent loads per program keeps enough requests in flight;
    # unmasked because the caller rounds n_elem to a multiple of 4*BLOCK
    for i in tl.static_range(4):
        v = tl.load(x_ptr + offs + i * BLOCK)
        acc += v.to(tl.float32)
    tl.store(out_ptr + pid, tl.sum(acc))


def bench(dtype, n_bytes=512 << 20, BLOCK=8192):
    n_elem = n_bytes // torch.tensor([], dtype=dtype).element_size()
    n_elem = (n_elem // (4 * BLOCK)) * (4 * BLOCK)
    x = torch.ones(n_elem, dtype=dtype, device="cuda")
    grid = (n_elem // (4 * BLOCK),)
    out = torch.empty(grid[0], dtype=torch.float32, device="cuda")
    fn = lambda: stream_read_kernel[grid](x, out, n_elem, BLOCK=BLOCK)
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(True); e = torch.cuda.Event(True)
    s.record()
    for _ in range(20):
        fn()
    e.record(); torch.cuda.synchronize()
    ms = s.elapsed_time(e) / 20
    return ms, n_elem * x.element_size() / (ms / 1e3) / 1e9


print("device", torch.cuda.get_device_name(0))
for dt, name in ((torch.bfloat16, "bf16"), (torch.float8_e4m3fn, "fp8 "),
                 (torch.uint8, "u8  ")):
    ms, gb = bench(dt)
    print(f"triton stream-read {name}: {ms:7.2f} ms -> {gb:7.0f} GB/s")

# plain torch read+write copy for reference
n = 256 << 20
a = torch.empty(n, dtype=torch.uint8, device="cuda")
b = torch.empty(n, dtype=torch.uint8, device="cuda")
fn = lambda: b.copy_(a)
for _ in range(3):
    fn()
torch.cuda.synchronize()
s = torch.cuda.Event(True); e = torch.cuda.Event(True)
s.record()
for _ in range(20):
    fn()
e.record(); torch.cuda.synchronize()
ms = s.elapsed_time(e) / 20
print(f"torch copy 256MiB      : {ms:7.2f} ms -> {2 * n / (ms / 1e3) / 1e9:7.0f} GB/s (r+w)")
