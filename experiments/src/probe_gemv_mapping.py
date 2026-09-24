#!/usr/bin/env python3
"""Canonical `linear` -> MetaX tuned kernels: find a mapping that is actually fast.

Leverage A as a plain `mm(x, W.T)` mapping was a 0.75x REGRESSION, so test the
transposed orientation instead.

Key observation: the MetaX backend ships dedicated K-parallel GEMV kernels
(`gemv_mm`, `gemv_mm_k_parallel`) but `mm()` only reaches them when the second
operand has a single column (`N == 1`). At decode, `linear(x[1,K], W[N,K])` is
exactly that geometry transposed:

    linear(x, W)  ==  mm(W, x.T).T        for M == 1

    a = W     (N, K) contiguous, strides (K, 1)
    b = x.T   (K, 1) strides (1, K)  -> N == 1  -> GEMV path, grid over N

That gives thousands of programs instead of ~20, split along K when K >= 2048.

Also re-checks numerics, since the previous mapping degraded accuracy at M=1.

Run:  conda activate mx && python /root/src/probe_gemv_mapping.py
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

MODEL = "/root/models/MiniCPM5-2B"
OUT = Path("/root/bench_results/metax_linear")
EPS_BF16 = 2.0 ** -8


def shapes_from_config() -> list[tuple[str, int, int]]:
    cfg = json.loads((Path(MODEL) / "config.json").read_text())
    h, inter = cfg["hidden_size"], cfg["intermediate_size"]
    nq = cfg["num_attention_heads"]
    nkv = cfg.get("num_key_value_heads", nq)
    hd = cfg.get("head_dim") or h // nq
    vocab = 0 if cfg.get("tie_word_embeddings") else cfg["vocab_size"]
    out = [
        ("qkv_proj", nq * hd + 2 * nkv * hd, h),
        ("o_proj", nq * hd, h),
        ("gate_up_proj", 2 * inter, h),
        ("down_proj", h, inter),
    ]
    if vocab:
        out.append(("lm_head", vocab, h))
    return out


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


def frob(got, ref) -> float:
    return ((got.float() - ref).norm() / ref.norm()).item()


def main() -> None:
    import torch
    import flag_gems
    from flag_gems.ops.linear import linear as generic_linear
    import importlib

    mm_mod = importlib.import_module("flag_gems.runtime.backend._metax.ops.mm")

    dev = "cuda"
    dt = torch.bfloat16

    print("=" * 96)
    print("DECODE (M=1): generic linear vs mm(x,W.T) vs transposed GEMV mm(W,x.T)")
    print("=" * 96)
    print(f"  {'shape':14s} {'K':>5s} {'N':>7s} {'gemv_path':>10s} "
          f"{'generic ms':>11s} {'mm(x,Wt) ms':>12s} {'mm(W,xt) ms':>12s} "
          f"{'speedup':>8s} {'frob':>9s}")

    rows = []
    for name, N, K in shapes_from_config():
        torch.manual_seed(0)
        x = torch.randn(1, K, device=dev, dtype=dt)
        w = torch.randn(N, K, device=dev, dtype=dt)
        ref = x.float() @ w.float().t()

        a = w                      # (N, K) contiguous
        b = x.t()                  # (K, 1) strides (1, K)
        gemv_would = (b.shape[1] == 1)

        got_generic = generic_linear(x, w)
        got_mm_nt = flag_gems.mm(x, w.t())
        got_gemv = flag_gems.mm(a, b).t().contiguous()

        f_gemv = frob(got_gemv, ref)

        tg = bench(lambda: generic_linear(x, w))
        tm1 = bench(lambda: flag_gems.mm(x, w.t()))
        tm2 = bench(lambda: flag_gems.mm(a, b).t().contiguous())

        # correctness of all three variants
        ok = (frob(got_generic, ref) < 0.02 and frob(got_mm_nt, ref) < 0.02
              and f_gemv < 0.02)
        rows.append({"shape": name, "K": K, "N": N, "generic_ms": tg,
                     "mm_nt_ms": tm1, "gemv_ms": tm2, "gemv_frob": f_gemv,
                     "gemv_path": gemv_would, "ok": ok})
        print(f"  {name:14s} {K:5d} {N:7d} {str(gemv_would):>10s} "
              f"{tg:11.4f} {tm1:12.4f} {tm2:12.4f} "
              f"{tg / tm2:7.2f}x {f_gemv:9.5f}{'' if ok else '  <-- BAD'}")

    tot_g = sum(r["generic_ms"] for r in rows) * 42
    tot_nt = sum(r["mm_nt_ms"] for r in rows) * 42
    tot_gemv = sum(r["gemv_ms"] for r in rows) * 42
    print()
    print(f"  42-layer decode total: generic={tot_g:.2f}ms  "
          f"mm(x,Wt)={tot_nt:.2f}ms  mm(W,xt)={tot_gemv:.2f}ms")
    print(f"  transposed-GEMV speedup over generic: {tot_g / tot_gemv:.2f}x")

    print()
    print("=" * 96)
    print("SCENARIO CHECK at decode (M=1) -- does mm(W, x.T) really take the GEMV path?")
    print("=" * 96)
    for name, N, K in shapes_from_config():
        print(f"  {name:14s} N_out={N:7d} K={K:5d} -> "
              f"a.shape={tuple(w.shape) if False else ''}"
              f"M={N} N=1 gemv_k_parallel={mm_mod._gemv_k_parallel_scenario(N, K)}")

    (OUT / "gemv_mapping.json").write_text(json.dumps(rows, indent=2))
    print(f"\n  Wrote {OUT / 'gemv_mapping.json'}")
    print("GEMV_MAPPING_DONE", flush=True)


if __name__ == "__main__":
    main()
