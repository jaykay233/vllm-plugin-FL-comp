#!/usr/bin/env python3
"""Quantify decode headroom: is the M=1 linear actually bandwidth-bound?

At M=1 the GEMM reads the whole weight matrix and does very little maths, so
the roofline is memory bandwidth. Compare the achieved bandwidth of the real
linear against a pure read of the same bytes, to see whether FlagGems' GEMV
path is leaving anything on the table.
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

MODEL = "/root/models/MiniCPM5-2B"


def bench(fn, n_warmup=15, n_iters=60) -> float:
    import torch

    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n_iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts)


def main() -> None:
    import torch
    import flag_gems
    from flag_gems.ops.linear import linear as generic_linear

    dev = "cuda"
    dt = torch.bfloat16
    cfg = json.loads((Path(MODEL) / "config.json").read_text())
    h, inter = cfg["hidden_size"], cfg["intermediate_size"]
    nq = cfg["num_attention_heads"]
    nkv = cfg.get("num_key_value_heads", nq)
    hd = cfg.get("head_dim") or h // nq

    shapes = [
        ("qkv_proj", nq * hd + 2 * nkv * hd, h),
        ("o_proj", nq * hd, h),
        ("gate_up_proj", 2 * inter, h),
        ("down_proj", h, inter),
    ]

    print("device:", torch.cuda.get_device_name(0))
    props = torch.cuda.get_device_properties(0)
    print(f"  SMs={props.multi_processor_count} "
          f"mem={props.total_memory / 2**30:.1f}GiB")
    print()

    # ceiling: pure read of a buffer (x.clone() reads + writes)
    print("=" * 96)
    print("BANDWIDTH CEILING (pure elementwise copy of the same byte volume)")
    print("=" * 96)
    for name, N, K in shapes:
        nb = N * K * 2
        src = torch.empty(N * K, device=dev, dtype=dt)
        dst = torch.empty_like(src)
        ms = bench(lambda: dst.copy_(src))
        bw = 2 * nb / (ms * 1e-3) / 1e9  # read + write
        print(f"  {name:14s} weight={nb / 2**20:7.2f}MiB  copy={ms:7.4f}ms  "
              f"-> {bw:8.1f} GB/s (rw)")

    print()
    print("=" * 96)
    print("DECODE M=1: linear achieved bandwidth (weight bytes / time)")
    print("=" * 96)
    print(f"  {'shape':14s} {'N':>7s} {'K':>5s} {'weight':>9s} "
          f"{'generic ms':>11s} {'GB/s':>8s} {'metax ms':>10s} {'GB/s':>8s}")
    tot_g = tot_m = 0.0
    for name, N, K in shapes:
        torch.manual_seed(0)
        x = torch.randn(1, K, device=dev, dtype=dt)
        w = torch.randn(N, K, device=dev, dtype=dt)
        nb = N * K * 2
        tg = bench(lambda: generic_linear(x, w))
        tm = bench(lambda: flag_gems.mm(x, w.t()))
        tot_g += tg
        tot_m += tm
        print(f"  {name:14s} {N:7d} {K:5d} {nb / 2**20:6.2f}MiB "
              f"{tg:11.4f} {nb / (tg * 1e-3) / 1e9:8.1f} "
              f"{tm:10.4f} {nb / (tm * 1e-3) / 1e9:8.1f}")

    print()
    print(f"  per-layer linear (M=1): generic={tot_g:.3f}ms  metax={tot_m:.3f}ms")
    print(f"  42 layers            : generic={tot_g * 42:.2f}ms  "
          f"metax={tot_m * 42:.2f}ms")
    print("DECODE_HEADROOM_DONE", flush=True)


if __name__ == "__main__":
    main()
