#!/usr/bin/env python3
"""rms_norm: what does the FlagGems mandate cost or earn on the hot path?

`RMSNorm.forward_native` is just `ir.ops.rms_norm(...)`, and on MetaX only two
providers are registered for that IR op:

    native  -- vLLM's pure-torch implementation
    flagos  -- the FL dispatch -> FlagGems gems_rms_forward

The platform default priority is ['flagos', 'vllm_c', 'native'], so `flagos`
wins and FlagGems serves every rms_norm call. This script switches between the
two providers on the same IR op to measure exactly what that buys or costs.

MiniCPM5-2B issues 2 norms per layer over its 42 layers, so this is 84 calls
per decode step -- the last place FlagGems is actually engaged on the fast
path.

Run:  conda activate mx && python /root/src/compare_rms_norm.py
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

OUT = Path("/root/bench_results/rms_norm_cmp")
OUT.mkdir(parents=True, exist_ok=True)

HIDDEN = 2048
EPS = 1e-6
MS = [1, 8, 128, 512]
LAYERS = 42
NORMS_PER_LAYER = 2


def bench(fn, n_warmup=25, n_iters=100) -> float:
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
    from vllm.ir import ops as irops

    op = irops.rms_norm
    dev = "cuda"
    dt = torch.bfloat16

    print("=" * 92)
    print("rms_norm via the vllm.ir seam: 'flagos' (FlagGems) vs 'native' (torch)")
    print("=" * 92)
    print(f"  providers registered : {sorted(op.impls.keys())}")
    print(f"  hidden={HIDDEN} eps={EPS}")
    print()

    def torch_ref(x, w):
        var = x.float().pow(2).mean(-1, keepdim=True)
        return ((x.float() * torch.rsqrt(var + EPS)) * w.float()).to(dt)

    rows = []
    print(f"  {'M':>5s} {'flagos ms':>11s} {'native ms':>11s} "
          f"{'flagos/native':>14s}  verdict")
    for M in MS:
        torch.manual_seed(0)
        x = torch.randn(M, HIDDEN, device=dev, dtype=dt)
        w = torch.randn(HIDDEN, device=dev, dtype=dt)
        ref = torch_ref(x, w)

        op.set_priority(["flagos"])
        t_flagos = bench(lambda: op(x, w, EPS))
        f_flagos = frob(op(x, w, EPS), ref)

        op.set_priority(["native"])
        t_native = bench(lambda: op(x, w, EPS))
        f_native = frob(op(x, w, EPS), ref)

        ratio = t_flagos / t_native
        verdict = ("FLAGOS FASTER" if ratio < 0.97 else
                   "native faster" if ratio > 1.03 else "about equal")
        rows.append({"M": M, "flagos_ms": t_flagos, "native_ms": t_native,
                     "ratio": ratio, "frob_flagos": f_flagos,
                     "frob_native": f_native})
        print(f"  {M:5d} {t_flagos:11.5f} {t_native:11.5f} {ratio:13.3f}x  {verdict}")

    # restore the platform default
    op.set_priority(["flagos", "native"])

    print()
    print("  accuracy (Frobenius vs torch reference):")
    for r in rows:
        print(f"    M={r['M']:<4d} flagos={r['frob_flagos']:.6f}  "
              f"native={r['frob_native']:.6f}")

    print()
    print("=" * 92)
    print(f"PER DECODE STEP ({LAYERS} layers x {NORMS_PER_LAYER} norms = "
          f"{LAYERS * NORMS_PER_LAYER} calls at M=1)")
    print("=" * 92)
    r1 = next(r for r in rows if r["M"] == 1)
    calls = LAYERS * NORMS_PER_LAYER
    print(f"  flagos (engaged) : {r1['flagos_ms'] * calls:8.3f} ms")
    print(f"  native           : {r1['native_ms'] * calls:8.3f} ms")
    delta = (r1["flagos_ms"] - r1["native_ms"]) * calls
    print(f"  difference       : {delta:+8.3f} ms  "
          f"({'COST' if delta > 0 else 'GAIN'} of the mandate)")
    print(f"  as share of a ~10.15 ms decode step: "
          f"{abs(delta) / 10.15 * 100:.2f}%")
    if r1["ratio"] < 0.97:
        print("\n  >> FlagGems rms_norm BEATS native. The mandate is an asset")
        print("     here, and there is headroom to widen FlagGems coverage.")
    elif r1["ratio"] > 1.03:
        print("\n  >> FlagGems rms_norm is SLOWER than native. Widening FlagGems")
        print("     coverage would cost performance; optimising the FlagGems")
        print("     rms_norm kernel is the way to turn this into a gain.")
    else:
        print("\n  >> On par with native.")

    (OUT / "result.json").write_text(json.dumps(rows, indent=2))
    print("\nRMS_NORM_CMP_DONE", flush=True)


if __name__ == "__main__":
    main()
