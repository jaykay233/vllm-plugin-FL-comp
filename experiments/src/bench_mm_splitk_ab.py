#!/usr/bin/env python3
"""A/B the FlagGems MetaX mm dispatch with and without the split-k path.

Layout matters: vLLM's Linear calls `torch.mm(x, w.t())`, so `b` has stride
(1, K) which selects `nt_mm_scenario`.  Driving the public `M.mm()` entry keeps
every layout correct, and the only thing we toggle is `splitk_mm_scenario`.

Timing uses the profiler's device_time (events measure host launch overhead for
kernels this small).

Run:  python /root/src/bench_mm_splitk_ab.py
"""

from __future__ import annotations

import importlib
from collections import Counter

import torch
from torch.profiler import ProfilerActivity, profile

M = importlib.import_module("flag_gems.runtime.backend._metax.ops.mm")

# (name, M, N, K) with per-layer decode shapes
SHAPES = [
    ("qkv", 1, 2560, 2048),
    ("gate_up", 1, 12288, 2048),
    ("o_proj", 1, 2048, 2048),
    ("down", 1, 2048, 6144),
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
    print(f"sm={M.get_sm_count()} budget={max(1, M.get_sm_count() * 3 // 4)}",
          flush=True)
    orig = M.splitk_mm_scenario

    results = {}
    for name, m_, n_, k_ in SHAPES:
        a = torch.randn(m_, k_, dtype=torch.bfloat16, device="cuda")
        w = torch.randn(n_, k_, dtype=torch.bfloat16, device="cuda")
        b = w.t()  # stride (1, K) -> nt layout, as vLLM passes it
        mb = n_ * k_ * 2 / 1e6

        on = orig(m_, n_, k_)
        got_on = M.mm(a, b).float()
        t_on, per_on = measure(lambda: M.mm(a, b))

        M.splitk_mm_scenario = lambda *_a, **_k: False
        got_off = M.mm(a, b).float()
        t_off, per_off = measure(lambda: M.mm(a, b))
        M.splitk_mm_scenario = orig

        err = ((got_off - got_on).abs().max()
               / (got_on.abs().max() + 1e-6)).item()
        results[name] = (mb, on, t_on, t_off, per_on, per_off, err)
        print(f"\n##### {name}  M={m_} N={n_} K={k_}  {mb:.1f} MB  "
              f"splitk_scenario={on}", flush=True)
        print(f"  as-is      {t_on:8.2f} us   {mb / 1e3 / (t_on / 1e6):7.0f} GB/s",
              flush=True)
        print(f"  no-splitk  {t_off:8.2f} us   {mb / 1e3 / (t_off / 1e6):7.0f} GB/s"
              f"   max|Δ|={err:.2e}", flush=True)
        print("  kernels as-is:", flush=True)
        for k, v in per_on.most_common(6):
            print(f"     {k[:44]:46s} {v / ITERS:8.2f} us", flush=True)
        print("  kernels no-splitk:", flush=True)
        for k, v in per_off.most_common(6):
            print(f"     {k[:44]:46s} {v / ITERS:8.2f} us", flush=True)

    on_tot = sum(r[2] for r in results.values())
    off_tot = sum(r[3] for r in results.values())
    print(f"\nper-layer total: as-is={on_tot:.1f} us  no-splitk={off_tot:.1f} us  "
          f"delta={on_tot - off_tot:+.1f} us", flush=True)
    print("MM_SPLITK_AB_DONE", flush=True)


if __name__ == "__main__":
    main()
