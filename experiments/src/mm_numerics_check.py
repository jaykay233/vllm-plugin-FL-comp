#!/usr/bin/env python3
"""Is the MetaX flag_gems `mm` (mm_kernel_nt, the ~2x faster decode GEMM)
numerically usable at T=1?

Compares both GEMM paths against a float64 reference, reporting a proper
relative error (not a per-element max against a clamped denominator).

Run:  python /root/src/mm_numerics_check.py
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

OUT = Path("/root/bench_results/linear_vs_mm")
OUT.mkdir(parents=True, exist_ok=True)

HIDDEN, INTER = 2048, 6144
Q_HEADS, KV_HEADS, HEAD_DIM = 16, 2, 128
QKV_OUT = Q_HEADS * HEAD_DIM + 2 * KV_HEADS * HEAD_DIM

LAYERS = [
    ("qkv_proj", HIDDEN, QKV_OUT),
    ("o_proj", Q_HEADS * HEAD_DIM, HIDDEN),
    ("gate_up_proj", HIDDEN, 2 * INTER),
    ("down_proj", INTER, HIDDEN),
]


def rel_err(got, ref):
    """got/ref are numpy float64 arrays. Returns (global ||rel||, max abs)."""
    import numpy as np

    diff = np.abs(got - ref)
    denom = np.linalg.norm(ref)
    glob = float(np.linalg.norm(diff) / denom) if denom > 0 else float("inf")
    return glob, float(diff.max())


def main() -> None:
    import flag_gems

    flag_gems.enable()
    print(f"device: {torch.cuda.get_device_name(0)}  flag_gems {flag_gems.__version__}\n")

    out = {}
    for T in (1, 2, 8, 64, 512):
        print("=" * 90)
        print(f"T={T}")
        print("=" * 90)
        print(f"  {'layer':14s} | {'linear ||rel||':>13s} {'abs':>9s} | "
              f"{'mm ||rel||':>11s} {'abs':>9s}")
        out[T] = {}
        for name, K, N in LAYERS:
            g = torch.Generator(device="cuda").manual_seed(0)
            x = torch.randn(T, K, device="cuda", dtype=torch.bfloat16, generator=g)
            w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16, generator=g)

            # references / comparisons in numpy float64 on CPU so that
            # flag_gems' aten interception does not touch them
            ref = (x.double().cpu().numpy() @ w.double().cpu().numpy().T)

            a = torch.nn.functional.linear(x, w).double().cpu().numpy()
            b = torch.mm(x, w.t()).double().cpu().numpy()

            la, lx = rel_err(a, ref)
            ma, mx = rel_err(b, ref)

            out[T][name] = {"linear_rel": la, "linear_abs": lx,
                            "mm_rel": ma, "mm_abs": mx}
            flag = ""
            if ma > 1e-2:
                flag = "   <-- mm WRONG"
            print(f"  {name:14s} | {la:13.3e} {lx:9.3e} | {ma:11.3e} {mx:9.3e}{flag}")
        print()

    (OUT / "mm_numerics.json").write_text(json.dumps(out, indent=2))
    print(f"wrote {OUT / 'mm_numerics.json'}")
    print("\nMM_NUMERICS_DONE", flush=True)


if __name__ == "__main__":
    main()
