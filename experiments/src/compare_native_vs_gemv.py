#!/usr/bin/env python3
"""Decisive check: is FlagGems+my GEMV faster than the NATIVE path at decode?

The GEMV kernel is 2x faster than FlagGems' generic linear_kernel -- but the
per-layer projections never use flag_gems.linear. Under inductor they resolve
to forward_native, i.e. F.linear -> torch.mm -> the MetaX ATen implementation.

So the only question that matters for adding a `linear` IR op is:

    is my GEMV actually faster than the NATIVE GEMM at M == 1?

If not, routing the projections through FlagGems would be a regression, and
all the IR-op plumbing would be wasted work.

Run:  conda activate mx && python /root/src/compare_native_vs_gemv.py
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

MODEL = "/root/models/MiniCPM5-2B"
OUT = Path("/root/bench_results/native_vs_gemv")
OUT.mkdir(parents=True, exist_ok=True)

SHAPES = [
    ("qkv_proj", 2560, 2048),
    ("o_proj", 2048, 2048),
    ("gate_up_proj", 12288, 2048),
    ("down_proj", 2048, 6144),
    ("lm_head", 130560, 2048),
]


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
    import torch.nn.functional as F
    import flag_gems
    from flag_gems.ops.linear import linear as generic_linear

    dev = "cuda"
    dt = torch.bfloat16

    print("=" * 100)
    print("M=1 decode: NATIVE (what the projections actually use) vs FlagGems")
    print("=" * 100)
    print(f"  {'shape':13s} {'N':>7s} | {'native F.linear':>16s} "
          f"{'flag_gems generic':>18s} {'flag_gems gemv':>16s} | "
          f"{'gemv/native':>12s}")

    rows = []
    for name, N, K in SHAPES:
        torch.manual_seed(0)
        x = torch.randn(1, K, device=dev, dtype=dt)
        w = torch.randn(N, K, device=dev, dtype=dt)
        b = torch.randn(N, device=dev, dtype=dt)
        ref = x.float() @ w.float().t() + b.float()

        t_native = bench(lambda: F.linear(x, w, b))
        t_generic = bench(lambda: generic_linear(x, w, b))
        t_gemv = bench(lambda: flag_gems.linear(x, w, b))

        # correctness of each against the fp32 reference
        f_native = frob(F.linear(x, w, b), ref)
        f_generic = frob(generic_linear(x, w, b), ref)
        f_gemv = frob(flag_gems.linear(x, w, b), ref)

        ratio = t_gemv / t_native
        rows.append({"shape": name, "N": N, "K": K,
                     "native_ms": t_native, "generic_ms": t_generic,
                     "gemv_ms": t_gemv, "gemv_over_native": ratio,
                     "frob_native": f_native, "frob_gemv": f_gemv})
        verdict = "GEMV FASTER" if ratio < 0.97 else (
            "native faster" if ratio > 1.03 else "about equal")
        print(f"  {name:13s} {N:7d} | {t_native:14.4f}ms "
              f"{t_generic:16.4f}ms {t_gemv:14.4f}ms | "
              f"{ratio:10.3f}x  {verdict}")

    print()
    print("  accuracy (Frobenius vs fp32 ref):")
    for r in rows:
        print(f"    {r['shape']:13s} native={r['frob_native']:.5f}  "
              f"gemv={r['frob_gemv']:.5f}")

    print()
    print("=" * 100)
    print("VERDICT")
    print("=" * 100)
    per_layer = [r for r in rows if r["shape"] != "lm_head"]
    nat_layer = sum(r["native_ms"] for r in per_layer)
    gv_layer = sum(r["gemv_ms"] for r in per_layer)
    print(f"  per-layer projection total (M=1):")
    print(f"    native            : {nat_layer:8.4f} ms")
    print(f"    flag_gems generic : {sum(r['generic_ms'] for r in per_layer):8.4f} ms")
    print(f"    flag_gems gemv    : {gv_layer:8.4f} ms")
    print(f"    -> gemv vs native : {gv_layer / nat_layer:6.3f}x  (=42 layers: "
          f"{gv_layer * 42:.2f}ms vs {nat_layer * 42:.2f}ms)")

    lm = [r for r in rows if r["shape"] == "lm_head"][0]
    print(f"  lm_head: native={lm['native_ms']:.4f} gemv={lm['gemv_ms']:.4f} "
          f"({lm['gemv_over_native']:.3f}x)")

    if gv_layer < nat_layer * 0.97:
        print("\n  >> Routing the projections through FlagGems WOULD help.")
        print("     The IR-op plumbing is worth building.")
    else:
        print("\n  >> The native GEMM is already at least as fast as the GEMV.")
        print("     Routing the projections through FlagGems would NOT help.")
        print("     Do NOT build the linear IR op -- it would be another Leverage A.")
    print("NATIVE_VS_GEMV_DONE", flush=True)

    (OUT / "result.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
