#!/usr/bin/env python3
"""Debug why the MetaX `linear` override produces wrong numbers.

Isolates the failure: is the MetaX `mm` itself wrong, or is the nt-layout
mapping wrong? Compares against a float32 reference for both layouts.
"""

from __future__ import annotations

import torch

import flag_gems
from flag_gems.runtime.backend._metax.ops.mm import mm as metax_mm
from flag_gems.runtime.backend._metax.ops.linear import linear as metax_linear

dev = "cuda"
dt = torch.bfloat16


def ref_linear(x2d, w2d, b=None):
    """y = x @ w.T + b   (linear semantics, weight is (N, K))"""
    y = x2d.float() @ w2d.float().t()
    if b is not None:
        y = y + b.float()
    return y


def ref_mm(a, b):
    """y = a @ b   (mm semantics, operands taken literally)"""
    return a.float() @ b.float()


def report(tag, got, expect):
    g = got.float()
    e = expect
    denom = e.abs().clamp_min(1e-3)
    rel = ((g - e).abs() / denom).max().item()
    absd = (g - e).abs().max().item()
    zero = (g.abs().max().item() == 0.0)
    print(f"  {tag:34s} max_rel={rel:9.5f} max_abs={absd:9.5f} "
          f"out_max={g.abs().max().item():8.3f} {'<-- ALL ZERO' if zero else ''}")
    return rel


def main() -> None:
    torch.manual_seed(0)
    M, K, N = 8, 256, 512

    x = torch.randn(M, K, device=dev, dtype=dt)
    w = torch.randn(N, K, device=dev, dtype=dt)
    expect_lin = ref_linear(x, w)
    expect_mm_nn = ref_mm(x, w.t())  # x @ w.T

    # b for nn layout: (K, N) contiguous
    w_nn = w.t().contiguous()  # (K, N) contiguous
    assert w_nn.stride() == (N, 1), w_nn.stride()

    print("=" * 92)
    print(f"geometry: x({M},{K}) w({N},{K})  -> expect_lin = x @ w.T")
    print("=" * 92)
    print(f"  x.stride      = {x.stride()}")
    print(f"  w.stride      = {w.stride()}   (linear weight layout)")
    print(f"  w.t().stride  = {w.t().stride()}   (nt view passed to mm)")
    print(f"  w_nn.stride   = {w_nn.stride()}   (nn contiguous)")
    print()

    print("--- torch.mm baseline (bf16) ---")
    report("torch.mm(x, w.T)", torch.mm(x, w.t()), expect_mm_nn)
    report("torch.mm(x, w_nn)", torch.mm(x, w_nn), expect_mm_nn)
    print()

    print("--- flag_gems.mm  (nt view: b = w.t()) ---")
    report("flag_gems.mm(x, w.t())", flag_gems.mm(x, w.t()), expect_mm_nn)
    print()

    print("--- flag_gems.mm  (nn contiguous: b = w_nn) ---")
    report("flag_gems.mm(x, w_nn)", flag_gems.mm(x, w_nn), expect_mm_nn)
    print()

    print("--- _metax mm direct ---")
    report("metax_mm(x, w.t())", metax_mm(x, w.t()), expect_mm_nn)
    report("metax_mm(x, w_nn)", metax_mm(x, w_nn), expect_mm_nn)
    print()

    print("--- metax linear ---")
    report("metax_linear(x, w)", metax_linear(x, w), expect_lin)
    print()

    # which scenario does mm pick for each layout?
    from flag_gems.runtime.backend._metax.ops import mm as mm_mod

    print("--- scenario selection ---")
    for label, a, b in (("nt (b=w.t())", x, w.t()), ("nn (b=w_nn)", x, w_nn)):
        ann = mm_mod.nn_mm_scenario(a, b, None, M, N, K)
        ntt = mm_mod.nt_mm_scenario(a, b, None, M, N, K)
        sk = mm_mod.splitk_mm_scenario(M, N, K)
        ts = mm_mod._select_two_step_split_k(M, N, K)
        print(f"  {label:16s} nn={ann} nt={ntt} splitk={sk} two_step_splitk={ts}")
    print()

    # per-scenario correctness for the nt layout
    print("--- nt layout, forcing each scenario ---")
    c = torch.empty((M, N), device=dev, dtype=dt)
    report("general_mm_nt", mm_mod.general_mm_nt(x, w.t(), c, M, N, K), expect_mm_nn)
    c2 = torch.empty((M, N), device=dev, dtype=dt).zero_()
    report("splitk_mm", mm_mod.splitk_mm(x, w.t(), c2, M, N, K), expect_mm_nn)
    c3 = torch.empty((M, N), device=dev, dtype=dt)
    report("general_mm", mm_mod.general_mm(x, w.t(), c3, M, N, K), expect_mm_nn)
    print()

    # generic (non-metax) linear for reference
    from flag_gems.ops.linear import linear as generic_linear

    print("--- generic flag_gems.ops.linear ---")
    report("generic linear(x, w)", generic_linear(x, w), expect_lin)

    print("\nMETAX_LINEAR_DEBUG_DONE", flush=True)


if __name__ == "__main__":
    main()
