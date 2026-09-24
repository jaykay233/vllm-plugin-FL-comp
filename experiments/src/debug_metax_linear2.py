#!/usr/bin/env python3
"""Locate the exact MetaX mm scenario that breaks the `linear` mapping.

The override works at K<2048 and fails at K>=2048, which is precisely where
the split-K scenarios activate. Test every scenario directly, for both the
`nn` and `nt` operand layouts, against a float32 reference.
"""

from __future__ import annotations

import importlib

import torch

mm_mod = importlib.import_module("flag_gems.runtime.backend._metax.ops.mm")

dev = "cuda"
dt = torch.bfloat16


def rel(got, expect):
    return ((got.float() - expect).abs() / expect.abs().clamp_min(1e-3)).max().item()


def row(tag, got, expect):
    r = rel(got, expect)
    flag = "" if r < 0.05 else "   <-- WRONG"
    print(f"    {tag:26s} max_rel={r:9.5f}{flag}")


def check(M, K, N):
    torch.manual_seed(0)
    x = torch.randn(M, K, device=dev, dtype=dt)
    w = torch.randn(N, K, device=dev, dtype=dt)

    w_nt = w.t()              # (K, N) view, strides (1, K)
    w_nn = w.t().contiguous()  # (K, N) contiguous, strides (N, 1)

    exp = x.float() @ w.float().t()

    print("=" * 92)
    print(f"M={M} K={K} N={N}")
    print("=" * 92)
    c_probe = torch.empty((M, N), device=dev, dtype=dt)
    print("  scenarios: "
          f"nn={mm_mod.nn_mm_scenario(x, w_nt, c_probe, M, N, K)} "
          f"nt={mm_mod.nt_mm_scenario(x, w_nt, c_probe, M, N, K)} "
          f"splitk={mm_mod.splitk_mm_scenario(M, N, K)} "
          f"two_step={mm_mod._splitk_mm_two_step_scenario(M, N, K)} "
          f"gemv_kpar={(N == 1) and mm_mod._gemv_k_parallel_scenario(M, K)}")

    for lname, b in (("nt (b=w.T)", w_nt), ("nn (b=contig)", w_nn)):
        print(f"  -- {lname} --")
        c = torch.empty((M, N), device=dev, dtype=dt)
        row("general_mm_nn", mm_mod.general_mm_nn(x, b, c, M, N, K), exp)
        c = torch.empty((M, N), device=dev, dtype=dt)
        row("general_mm_nt", mm_mod.general_mm_nt(x, b, c, M, N, K), exp)
        c = torch.empty((M, N), device=dev, dtype=dt)
        row("general_mm", mm_mod.general_mm(x, b, c, M, N, K), exp)
        if mm_mod.splitk_mm_scenario(M, N, K):
            c = torch.empty((M, N), device=dev, dtype=dt).zero_()
            row("splitk_mm", mm_mod.splitk_mm(x, b, c, M, N, K), exp)
        ts = mm_mod._select_two_step_split_k(M, N, K)
        if ts is not None:
            c = torch.empty((M, N), device=dev, dtype=dt)
            row(f"splitk_mm_two_step(sk={ts})",
                mm_mod.splitk_mm_two_step(x, b, c, M, N, K, ts), exp)

    print("  -- public entry points (nt) --")
    row("metax mm(x, w.T)", mm_mod.mm(x, w_nt), exp)
    from flag_gems.runtime.backend._metax.ops.linear import linear as ml

    row("metax linear(x, w)", ml(x, w), exp)
    from flag_gems.ops.linear import linear as gl

    row("generic linear(x, w)", gl(x, w), exp)
    print()


def main() -> None:
    # K < 2048 (known good), then the K >= 2048 cases that failed
    check(8, 256, 512)
    check(1, 2048, 2560)    # qkv_proj, decode
    check(512, 2048, 2560)  # qkv_proj, prefill
    check(512, 6144, 2048)  # down_proj, prefill
    print("METAX_LINEAR_DEBUG2_DONE", flush=True)


if __name__ == "__main__":
    main()
