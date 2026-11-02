#!/usr/bin/env python3
"""How fast can this card actually run the prefill GEMM shapes?

The 4k prefill trace shows the MLP `gate_up` GEMM (M=8192, K=2048, N=12288)
taking 1.93 ms, i.e. ~213 TFLOP/s, while `down` (M=8192, K=6144, N=2048) reaches
~334 TFLOP/s on the same trace. That gap is either "the shape is bad" or "the
library picks a bad tile for it". This measures the ceiling with the vendor
library at the exact shapes and at some alternatives.

Run: python experiments/src/bench_prefill_gemm.py
"""
from __future__ import annotations

import torch

torch.manual_seed(0)
M_LIST = [8192, 4096, 2048, 1024]


def tflops(m, k, n, iters=10):
    a = torch.randn(m, k, dtype=torch.bfloat16, device="cuda") * 0.05
    b = torch.randn(n, k, dtype=torch.bfloat16, device="cuda") * 0.05
    fn = lambda: torch.nn.functional.linear(a, b)
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(True); e = torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record(); torch.cuda.synchronize()
    ms = s.elapsed_time(e) / iters
    tf = 2 * m * k * n / (ms / 1e3) / 1e12
    traffic = (m * k + k * n + m * n) * 2 / 1e9
    return ms, tf, traffic / (ms / 1e3)


print(f"device={torch.cuda.get_device_name(0)}  bf16")
print(f"{'shape (M,K,N)':26s} {'ms':>8s} {'TFLOP/s':>9s} {'GB/s':>8s}")
SHAPES = [
    ("gate_up  (M,2048,12288)", 2048, 12288),
    ("down     (M,6144,2048)", 6144, 2048),
    ("qkv      (M,2048,2560)", 2048, 2560),
    ("o_proj   (M,2048,2048)", 2048, 2048),
    ("lm_head  (M,2048,130560)", 2048, 130560),
]
for name, k, n in SHAPES:
    for m in M_LIST:
        ms, tf, bw = tflops(m, k, n)
        print(f"{name.replace('(M', f'({m}') if False else f'{name.split()[0]:9s} M={m:5d} K={k:5d} N={n:6d}'}"
              f"{ms:8.3f} {tf:9.1f} {bw:8.0f}")

print("\nlarge square reference (compute peak probe)")
for m, k, n in [(4096, 4096, 4096), (8192, 8192, 8192), (16384, 4096, 4096)]:
    ms, tf, bw = tflops(m, k, n, 20)
    print(f"{'square':9s} M={m:5d} K={k:5d} N={n:6d}{ms:8.3f} {tf:9.1f} {bw:8.0f}")
