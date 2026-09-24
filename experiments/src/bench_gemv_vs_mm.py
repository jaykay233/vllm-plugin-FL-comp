#!/usr/bin/env python3
"""Does the existing MetaX GEMV kernel beat FlagGems' `mm` for decode projections?

`_metax/ops/linear.py` already ships `_gemv_kernel` (used today for lm_head,
N=130560: 535 MB in 333 us = 1607 GB/s).  The per-layer projections instead go
through `mm_kernel_nt` / `mm_kernel_splitk` at 633-1274 GB/s.

If GEMV wins on these shapes, the fix is to route M==1 `mm` calls to it.

Run:  python /root/src/bench_gemv_vs_mm.py
"""

from __future__ import annotations

import importlib
from collections import Counter

import torch
from torch.profiler import ProfilerActivity, profile

M = importlib.import_module("flag_gems.runtime.backend._metax.ops.mm")
L = importlib.import_module("flag_gems.runtime.backend._metax.ops.linear")

SHAPES = [
    ("qkv", 2560, 2048),
    ("gate_up", 12288, 2048),
    ("o_proj", 2048, 2048),
    ("down", 2048, 6144),
    ("lm_head", 130560, 2048),
]
ITERS = 30


def measure(fn, iters=ITERS):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    CM = torch.autograd.DeviceType.CUDA
    tot = 0.0
    per = Counter()
    for ev in prof.events():
        if ev.device_type == CM and ev.name and ev.device_time > 0:
            tot += ev.device_time
            per[ev.name] += ev.device_time
    return tot / iters, per


def main() -> None:
    print(f"{'shape':8s} {'N':>7s} {'K':>6s} {'MB':>7s} {'mm_us':>8s} {'GB/s':>7s} "
          f"{'gemv_us':>8s} {'GB/s':>7s} {'speedup':>8s} {'max|d|':>9s}", flush=True)

    tot_mm = tot_g = 0.0
    for name, n_, k_ in SHAPES:
        w = torch.randn(n_, k_, dtype=torch.bfloat16, device="cuda")
        a = torch.randn(1, k_, dtype=torch.bfloat16, device="cuda")
        mb = n_ * k_ * 2 / 1e6

        ref = M.mm(a, w.t())
        got = L.linear(a, w)
        err = ((got.float() - ref.float()).abs().max()
               / (ref.float().abs().max() + 1e-6)).item()

        t_mm, per_mm = measure(lambda: M.mm(a, w.t()))
        t_g, per_g = measure(lambda: L.linear(a, w))
        tot_mm += t_mm
        tot_g += t_g

        print(f"{name:8s} {n_:7d} {k_:6d} {mb:7.1f} {t_mm:8.2f} "
              f"{mb / 1e3 / (t_mm / 1e6):7.0f} {t_g:8.2f} "
              f"{mb / 1e3 / (t_g / 1e6):7.0f} {t_mm / t_g:7.2f}x {err:9.2e}",
              flush=True)
        if err > 5e-2:
            print(f"   !! GEMV mismatch on {name}", flush=True)
        print(f"   mm kernels:   {[f'{k[:34]}:{v / ITERS:.1f}us' for k, v in per_mm.most_common(3)]}",
              flush=True)
        print(f"   gemv kernels: {[f'{k[:34]}:{v / ITERS:.1f}us' for k, v in per_g.most_common(3)]}",
              flush=True)

    print(f"\ntotal: mm={tot_mm:.1f} us  gemv={tot_g:.1f} us  "
          f"speedup={tot_mm / tot_g:.2f}x", flush=True)
    print("GEMV_VS_MM_DONE", flush=True)


if __name__ == "__main__":
    main()
