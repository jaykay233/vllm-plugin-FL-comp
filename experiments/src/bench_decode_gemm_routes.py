#!/usr/bin/env python3
"""Per-layer decode GEMM: mcblas vs FlagGems mm, at the shapes that matter.

`lm_head` (M=64, K=2048, N=130560) runs on FlagGems `mm` at ~1081 GB/s in the
served decode trace, while the four per-layer projections (same M, N in
2048..12288) run on mcblas at ~777 GB/s. If the FlagGems route holds up at the
smaller N, the per-layer GEMMs can recover ~1.4 ms of a 23 ms decode step.

Run: python experiments/src/bench_decode_gemm_routes.py
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

torch.manual_seed(0)
DEV = "cuda"
M = 64
SHAPES = [
    ("qkv     ", 2048, 2560),
    ("o_proj  ", 2048, 2048),
    ("gate_up ", 2048, 12288),
    ("down    ", 6144, 2048),
    ("lm_head ", 2048, 130560),
]


def timeit(fn, iters=30):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(True); e = torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def main() -> None:
    import flag_gems

    print(f"device={torch.cuda.get_device_name(0)}  M={M}  bf16")
    print(f"{'gemm':9s} {'K':>6s} {'N':>7s} {'mcblas ms':>10s} {'gems ms':>9s} "
          f"{'ratio':>7s} {'mcblas GB/s':>12s} {'gems GB/s':>10s} {'max|d|':>9s}")
    tot_b, tot_g = 0.0, 0.0
    for name, k, n in SHAPES:
        x = torch.randn(M, k, dtype=torch.bfloat16, device=DEV) * 0.05
        w = torch.randn(n, k, dtype=torch.bfloat16, device=DEV) * 0.05
        ref = F.linear(x, w)
        got = flag_gems.mm(x, w.t())
        d = (ref.float() - got.float()).abs().max().item()
        t_b = timeit(lambda: F.linear(x, w))
        t_g = timeit(lambda: flag_gems.mm(x, w.t()))
        gb = (M * k + k * n + M * n) * 2 / 1e9
        print(f"{name} {k:6d} {n:7d} {t_b:10.4f} {t_g:9.4f} {t_g/t_b:7.3f} "
              f"{gb/(t_b/1e3):12.0f} {gb/(t_g/1e3):10.0f} {d:9.2e}")
        if name != "lm_head ":
            tot_b += t_b
            tot_g += t_g
    print(f"\nper-layer GEMM sum (4 projections): mcblas {tot_b:.4f} ms  "
          f"gems {tot_g:.4f} ms  ratio {tot_g/tot_b:.3f}")
    print(f"decode step is ~18 ms, so the difference is "
          f"{100*(tot_b-tot_g)/18.0:+.1f}% of a step")


if __name__ == "__main__":
    main()
