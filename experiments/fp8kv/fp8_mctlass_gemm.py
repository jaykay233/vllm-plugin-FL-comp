# Copyright (c) 2026 BAAI. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""
fp8 GEMM on MetaX C500 -- the ONE path that actually works.

Correcting the record: `_C::cutlass_scaled_mm` is present in mcoplib but is
broken (unit scales -> inf/nan, per-row scales -> 147% relative error, and
[K,N] vs [N,K] b layouts return identical garbage, so strides are ignored).
Triton cannot lower an fp8 MMA either ("PassManager::run failed"). A dtype-only
smoke test passes on all of these, which is why they need a numeric check.

What does work is MetaX's own tiled-GEMM library:

    mctlassEx.ScaledGEMM()(shapes, A, B, C, scale_a, scale_b, None,
                           is_blockwise=True, use_fp8=True,
                           is_scale_a_1d=True, is_scale_b_1d=<False|True>)

which dispatches to `mctlass_fp8_blockwise_gemm`. fp8 x fp8 -> bf16, with
DeepSeek-V3 style 128-element blockwise scaling along K:

    A        [M, K]  fp8_e4m3, k-major
    B        [N, K]  fp8_e4m3, k-major
    C        [M, N]  bf16,      n-major
    scale_a  [M, K/128]      float32
    scale_b  [N, K/128] 1D  or  [N/128, K/128] 2D

K must be divisible by 128 (all MiniCPM5-2B projections are).

This script validates numerics first, then measures bytes/s at decode shapes
against bf16 `F.linear` (which is mcblas, i.e. the real incumbent).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

import mcoplib._C  # noqa: F401  (torch.ops._C, not required but harmless)
from mctlassEx import ScaledGEMM

BF16 = torch.bfloat16
FP8 = torch.float8_e4m3fn
BLK = 128
FP8_MAX = 448.0


def blockwise_quant(x: torch.Tensor, chunk: int = 8192):
    """Per-row, 128-element-block fp8 quantization along the last dim.

    Chunked over rows so the transient float32 copy never gets large (the
    lm_head weight alone would be a 1.07 GB fp32 temp if done in one shot).
    Returns (fp8 tensor, scales [rows, K/128] float32).
    """
    rows, K = x.shape
    assert K % BLK == 0, "K must be a multiple of 128"
    q = torch.empty(x.shape, dtype=FP8, device=x.device)
    scales = torch.empty(rows, K // BLK, dtype=torch.float32, device=x.device)
    for r0 in range(0, rows, chunk):
        r1 = min(r0 + chunk, rows)
        blk = x[r0:r1].view(-1, K // BLK, BLK).float()
        amax = blk.abs().amax(-1).clamp(min=1e-12)
        s = amax / FP8_MAX
        scales[r0:r1] = s
        q[r0:r1] = (blk / s[:, :, None]).to(FP8).view(-1, K)
    return q, scales


def dequant_ref(aq, as_, bq, bs):
    """Exact reference of what blockwise fp8 matmul should produce."""
    a = aq.float().view(-1, aq.shape[1] // BLK, BLK) * as_[:, :, None]
    a = a.view(aq.shape)
    b = bq.float().view(-1, bq.shape[1] // BLK, BLK) * bs[:, :, None]
    b = b.view(bq.shape)
    return a @ b.t()


def bench(fn, iters=20, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def main():
    dev = "cuda"
    p = torch.cuda.get_device_properties(0)
    print("DEV %s  SMs=%d" % (p.name, p.multi_processor_count))

    gemm = ScaledGEMM()
    torch.manual_seed(0)

    # ---------------- correctness ----------------
    print("\n--- correctness (1d2d, 1d1d) ---")
    for M, N, K in ((64, 256, 256), (1, 512, 2048), (64, 2048, 2048)):
        x = torch.randn(M, K, device=dev, dtype=BF16)
        w = torch.randn(N, K, device=dev, dtype=BF16) * 0.02
        ref = F.linear(x, w)

        xq, sa = blockwise_quant(x)
        wq, sb = blockwise_quant(w)
        expect = dequant_ref(xq, sa, wq, sb)

        for tag, sb_arg, sb_1d in (("1d2d", sb, False), ("1d1d", sb, True)):
            if sb_1d:
                sb_arg = sb
            c = torch.empty(M, N, device=dev, dtype=BF16)
            try:
                gemm([M, N, K], xq, wq, c, sa, sb_arg, None,
                     is_blockwise=True, use_fp8=True,
                     is_scale_a_1d=True, is_scale_b_1d=sb_1d)
                torch.cuda.synchronize()
                bad = bool(torch.isnan(c).any() or torch.isinf(c).any())
                e_deq = (c.float() - expect).abs().max().item()
                e_ref = (c.float() - ref).abs().max().item()
                rel_ref = e_ref / ref.float().abs().max().clamp(min=1e-6).item()
                rel_deq = e_deq / expect.abs().max().clamp(min=1e-6).item()
                print("OK  M=%-3d N=%-5d K=%-5d %-5s nan/inf=%-5s rel_vs_deq=%.2e "
                      "rel_vs_bf16=%.4f" % (M, N, K, tag, bad, rel_deq, rel_ref))
            except Exception as exc:
                print("ERR M=%-3d N=%-5d K=%-5d %-5s %s: %s"
                      % (M, N, K, tag, type(exc).__name__,
                         str(exc).replace("\n", " ")[:110]))

    # ---------------- perf at decode shapes ----------------
    SHAPES = (("qkv", 2560, 2048), ("gate_up", 12288, 2048),
              ("o_proj", 2048, 2048), ("down", 2048, 6144),
              ("lm_head", 130560, 2048))
    LAYERS = 42

    for M in (1, 64, 256):
        print("\n=== M=%d ===" % M)
        print("%-8s %6s %10s %10s %9s %9s %8s"
              % ("proj", "N", "bf16 ms", "fp8 ms", "bf16 GB/s", "fp8 GB/s", "speedup"))
        tot_bf = tot_f8 = 0.0
        for name, N, K in SHAPES:
            x = torch.randn(M, K, device=dev, dtype=BF16)
            w = torch.randn(N, K, device=dev, dtype=BF16) * 0.02
            xq, sa = blockwise_quant(x)
            wq, sb = blockwise_quant(w)
            c = torch.empty(M, N, device=dev, dtype=BF16)

            def run_fp8():
                gemm([M, N, K], xq, wq, c, sa, sb, None,
                     is_blockwise=True, use_fp8=True,
                     is_scale_a_1d=True, is_scale_b_1d=False)

            try:
                t8 = bench(run_fp8)
            except Exception as exc:
                print("%-8s %6d FAIL %s" % (name, N, type(exc).__name__))
                continue
            tbf = bench(lambda: F.linear(x, w))
            wb_bf, wb_8 = N * K * 2, N * K
            print("%-8s %6d %10.4f %10.4f %9.0f %9.0f %7.2fx"
                  % (name, N, tbf, t8, wb_bf / (tbf * 1e6), wb_8 / (t8 * 1e6),
                     tbf / t8))
            if name != "lm_head":
                tot_bf += tbf * LAYERS
                tot_f8 += t8 * LAYERS
            else:
                lm_bf, lm_f8 = tbf, t8
            del w, wq, x, xq
            torch.cuda.empty_cache()

        step_bf = tot_bf + lm_bf
        step_f8 = tot_f8 + lm_f8
        print("STEP  x42 + lm_head : bf16 %7.3f ms   fp8 %7.3f ms   %.2fx   saved %+.3f ms"
              % (step_bf, step_f8, step_bf / step_f8, step_bf - step_f8))


if __name__ == "__main__":
    main()
