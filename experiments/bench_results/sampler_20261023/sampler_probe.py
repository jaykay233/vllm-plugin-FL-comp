#!/usr/bin/env python3
"""Where does flag_gems.top_p_threshold spend its 1.0 ms at rows=64?

The per-step trace shows the top-p chain costing ~1.02 ms/step (4.2% of a
24.45 ms step), which makes it the largest remaining item the plugin actually
owns.  Broken down by kernel, per step:

    _top_p_rowmax_v2_kernel            24.4 us
    _top_p_mass_v2_kernel             412.1 us   <- 40% of the chain
    _top_p_amax_v2_kernel              36.2 us   <- 2 KB of data
    _top_p_threshold_v2_kernel         ~?  us
    _top_p_mask_v2_kernel              47.0 us
    + torch softmax / multinomial     ~234  us

`rowmax` and `mass` read exactly the same tensor, but rowmax runs at HBM speed
(33 MB in 24 us) while mass takes 412 us -- so the histogram `atomic_add` is the
bottleneck, not bandwidth.  This script confirms that, then tests the knobs that
can reduce atomic pressure: bucket count, block size, and grid shape.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import torch
import triton
import triton.language as tl

sys.path.insert(0, "/workspace/FlagGems-comp/src")
import flag_gems  # noqa: E402

ROWS = 64
VOCAB = 130560
OUT = Path("/workspace/vllm-plugin-FL-comp/experiments/bench_results/sampler_20261023")
PEAK_GBS = 1416.0
ITL_MS = 24.45
STEPS = 1  # one sampler call per decode step


def dev_us_detail(fn, warmup=10, iters=30):
    from torch.profiler import ProfilerActivity, profile

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    by = defaultdict(float)
    cnt = defaultdict(int)
    for e in p.key_averages():
        t = e.self_device_time_total or 0
        if t > 0:
            by[e.key] += t / iters
            cnt[e.key] += e.count
    return by, cnt


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    logits = torch.randn(ROWS, VOCAB, device="cuda", dtype=torch.float32)

    for p_val in (1.0, 0.95):
        p = torch.full((ROWS,), p_val, device="cuda", dtype=torch.float32)
        by, cnt = dev_us_detail(lambda: flag_gems.top_p_threshold(logits, p))
        tot = sum(by.values())
        print(f"\n=== rows={ROWS} vocab={VOCAB} p={p_val}: {tot:.1f} us ===")
        for k, v in sorted(by.items(), key=lambda kv: -kv[1]):
            print(f"  {v:8.1f} us  {100*v/tot:5.1f}%  {k[:66]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
