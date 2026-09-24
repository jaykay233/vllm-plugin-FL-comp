#!/usr/bin/env python3
"""Isolate the bias handling difference between metax linear and generic linear."""

from __future__ import annotations

import torch

import flag_gems
from flag_gems.ops.linear import linear as generic_linear

dev = "cuda"
dt = torch.bfloat16


def stats(tag, got, ref):
    g = got.float()
    rel = ((g - ref).abs() / ref.abs().clamp_min(1e-3)).max().item()
    absd = (g - ref).abs().max().item()
    print(f"  {tag:34s} max_rel={rel:9.5f} max_abs={absd:9.5f}")


def main() -> None:
    M, K, N = 1, 2048, 2560
    torch.manual_seed(0)
    x = torch.randn(M, K, device=dev, dtype=dt)
    w = torch.randn(N, K, device=dev, dtype=dt)
    b = torch.randn(N, device=dev, dtype=dt)

    ref_nobias = x.float() @ w.float().t()
    ref_bias = ref_nobias + b.float()

    print("=" * 92)
    print(f"M={M} K={K} N={N}")
    print("=" * 92)

    print("-- metax flag_gems.linear: no bias vs bias --")
    stats("metax linear(x, w)        [nobias]",
          flag_gems.linear(x, w), ref_nobias)
    stats("metax linear(x, w, b)     [bias]  ",
          flag_gems.linear(x, w, b), ref_bias)
    stats("metax mm(x, w.T) + b      [manual]",
          flag_gems.mm(x, w.t()) + b, ref_bias)

    print("-- generic flag_gems.ops.linear --")
    stats("generic linear(x, w)      [nobias]",
          generic_linear(x, w), ref_nobias)
    stats("generic linear(x, w, b)   [bias]  ",
          generic_linear(x, w, b), ref_bias)

    print("-- torch reference --")
    stats("torch F.linear(x, w, b)",
          torch.nn.functional.linear(x, w, b), ref_bias)

    print("-- cross-check metax vs generic (the failing assertion) --")
    got = flag_gems.linear(x, w, b)
    gen = generic_linear(x, w, b)
    d = (got.float() - gen.float()).abs()
    print(f"  max|metax-generic| = {d.max().item():.5f}  "
          f"nonzero={int((d > 1e-2).sum())}/{d.numel()}")
    idx = torch.argmax(d)
    j = int(idx % N)
    print(f"  worst col {j}: metax={got.float().flatten()[idx].item():.4f} "
          f"generic={gen.float().flatten()[idx].item():.4f} "
          f"bias={b[j].item():.4f} ref={ref_bias.flatten()[idx].item():.4f}")

    print("\n-- does generic linear apply the SAME bias? --")
    d_nb = (generic_linear(x, w, b).float() - generic_linear(x, w).float())
    print(f"  max|gen(bias) - gen(nobias)| = {d_nb.max().item():.5f}")
    print(f"  bias max|b|                  = {b.abs().max().item():.5f}")
    d_ok = (d_nb - b.float()).abs().max().item()
    print(f"  max|(gen(b)-gen(nb)) - b|    = {d_ok:.5f} "
          f"{'-> bias applied consistently' if d_ok < 1e-2 else '-> BIAS MISAPPLIED'}")

    print("\nMETAX_BIAS_DEBUG_DONE", flush=True)


if __name__ == "__main__":
    main()
