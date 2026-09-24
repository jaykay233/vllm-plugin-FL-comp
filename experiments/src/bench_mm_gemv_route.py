#!/usr/bin/env python3
"""A/B the new M==1 GEMV routing inside FlagGems' MetaX `mm`.

Toggles `flag_gems..._metax.ops.mm._ENABLE_MM_GEMV` in-process so both arms use
exactly the same call site (`M.mm(a, w.t())`, the layout vLLM's Linear uses).

Run:  python /root/src/bench_mm_gemv_route.py
"""

from __future__ import annotations

import importlib
from collections import Counter

import torch
from torch.profiler import ProfilerActivity, profile

M = importlib.import_module("flag_gems.runtime.backend._metax.ops.mm")

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
    print(f"{'shape':8s} {'N':>7s} {'K':>6s} {'MB':>7s} "
          f"{'off_us':>8s} {'GB/s':>7s} {'on_us':>8s} {'GB/s':>7s} "
          f"{'speedup':>8s} {'max|d|':>9s}", flush=True)

    tot_off = tot_on = 0.0
    for name, n_, k_ in SHAPES:
        w = torch.randn(n_, k_, dtype=torch.bfloat16, device="cuda")
        a = torch.randn(1, k_, dtype=torch.bfloat16, device="cuda")
        mb = n_ * k_ * 2 / 1e6

        M._ENABLE_MM_GEMV = False
        ref = M.mm(a, w.t()).clone()
        t_off, per_off = measure(lambda: M.mm(a, w.t()))

        M._ENABLE_MM_GEMV = True
        got = M.mm(a, w.t()).clone()
        t_on, per_on = measure(lambda: M.mm(a, w.t()))

        err = ((got.float() - ref.float()).abs().max()
               / (ref.float().abs().max() + 1e-6)).item()
        tot_off += t_off
        tot_on += t_on

        print(f"{name:8s} {n_:7d} {k_:6d} {mb:7.1f} "
              f"{t_off:8.2f} {mb / 1e3 / (t_off / 1e6):7.0f} "
              f"{t_on:8.2f} {mb / 1e3 / (t_on / 1e6):7.0f} "
              f"{t_off / t_on:7.2f}x {err:9.2e}", flush=True)
        if err > 5e-2:
            print(f"   !! mismatch on {name}", flush=True)
        print(f"   off: {[f'{k[:30]}:{v / ITERS:.1f}us' for k, v in per_off.most_common(3)]}",
              flush=True)
        print(f"   on : {[f'{k[:30]}:{v / ITERS:.1f}us' for k, v in per_on.most_common(3)]}",
              flush=True)

    print(f"\ntotal: off={tot_off:.1f} us  on={tot_on:.1f} us  "
          f"speedup={tot_off / tot_on:.3f}x  saved={tot_off - tot_on:.1f} us",
          flush=True)
    print("MM_GEMV_ROUTE_DONE", flush=True)


if __name__ == "__main__":
    main()
