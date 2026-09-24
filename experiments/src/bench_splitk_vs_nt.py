#!/usr/bin/env python3
"""Microbenchmark: split-k vs non-split-k for the decode M=1 projections.

Decode budget says (per step):
    mm_kernel_nt       2257 us  x84.7   26.7 us/call   2 per layer
    mm_kernel_splitk   1692 us  x84.7   20.0 us/call   2 per layer
    zero_ (from split-k) 223 us  x84.7
    add (residual)       232 us  x86.7

Shapes per layer (M=1):
    qkv      [1, 2048]  x [2048, 2560]      10.5 MB weights
    gate_up  [1, 2048]  x [2048, 12288]     50.3 MB
    o_proj   [1, 2048]  x [2048, 2048]      8.4 MB
    down     [1, 6144]  x [6144, 2048]      25.2 MB

qkv/gate_up take the `nt` path (1131 GB/s effective) while o_proj/down take
split-k (657 GB/s effective).  This measures both paths on all four shapes.

Run:  python /root/src/bench_splitk_vs_nt.py
"""

from __future__ import annotations

import importlib
import statistics
import sys

import torch

sys.path.insert(0, "/root/src")

M = importlib.import_module("flag_gems.runtime.backend._metax.ops.mm")

SHAPES = [
    ("qkv", 1, 2560, 2048),
    ("gate_up", 1, 12288, 2048),
    ("o_proj", 1, 2048, 2048),
    ("down", 1, 2048, 6144),
]


def gpu_us(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(3):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters):
            fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e) * 1000 / iters)
    return min(times)


def main() -> None:
    dt = torch.bfloat16
    print(f"sm_count={M.get_sm_count()}  l2={M.get_l2_cache_size() / 1e6:.1f} MB",
          flush=True)
    print(f"parallelism_budget={max(1, M.get_sm_count() * 3 // 4)}\n", flush=True)
    print(f"{'shape':8s} {'M':>2s} {'N':>6s} {'K':>6s} {'MB':>6s} "
          f"{'splitk?':>8s} {'general_us':>10s} {'splitk_us':>10s} "
          f"{'nt_us':>9s} {'best':>8s}", flush=True)

    rows = []
    for name, m_, n_, k_ in SHAPES:
        a = torch.randn(m_, k_, dtype=dt, device="cuda")
        b = torch.randn(k_, n_, dtype=dt, device="cuda")
        mb = k_ * n_ * 2 / 1e6

        def f_general():
            c = torch.empty((m_, n_), dtype=dt, device="cuda")
            return M.general_mm(a, b, c, m_, n_, k_)

        def f_splitk():
            c = torch.empty((m_, n_), dtype=dt, device="cuda")
            return M.splitk_mm(a, b, c, m_, n_, k_)

        def f_nt():
            c = torch.empty((m_, n_), dtype=dt, device="cuda")
            return M.general_mm_nt(a, b, c, m_, n_, k_)

        def f_mm_dispatch():
            return M.mm(a, b)

        g = gpu_us(f_general)
        s = gpu_us(f_splitk)
        try:
            d = gpu_us(f_mm_dispatch)
        except Exception as exc:  # pragma: no cover
            d = float("nan")
            print(f"  [dispatch failed for {name}: {exc}]", flush=True)

        # sanity
        ref = f_general().float()
        for nm, fn in (("splitk", f_splitk), ("nt", f_nt)):
            got = fn().float()
            err = ((got - ref).abs().max() / (ref.abs().max() + 1e-6)).item()
            if err > 5e-2:
                print(f"  !! {name} {nm} max rel err {err:.3e}", flush=True)

        sc = M.splitk_mm_scenario(m_, n_, k_)
        best = min(g, s, d)
        who = "general" if best == g else ("splitk" if best == s else "mm")
        rows.append((name, mb, d, s, who))
        print(f"{name:8s} {m_:2d} {n_:6d} {k_:6d} {mb:6.1f} {str(sc):>8s} "
              f"{g:10.2f} {s:10.2f} {d:10.2f} {who:>8s}", flush=True)

    print("\n=== bandwidth (GB/s) at M=1 ===", flush=True)
    for name, mb, d, s, who in rows:
        print(f"  {name:8s} dispatch {mb / 1e3 / (d / 1e6):7.0f} GB/s   "
              f"splitk {mb / 1e3 / (s / 1e6):7.0f} GB/s", flush=True)

    tot_now = sum(r[2] for r in rows)
    tot_split = sum(r[3] for r in rows)
    print(f"\nper-layer total: dispatch={tot_now:.1f} us  forced-splitk={tot_split:.1f} us",
          flush=True)
    print("\nBENCH_SPLITK_DONE", flush=True)


if __name__ == "__main__":
    main()
