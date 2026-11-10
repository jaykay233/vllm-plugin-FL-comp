# Copyright (c) 2026 BAAI. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""
Is there a CORRECT fp8 GEMM on MetaX C500?

`_C::cutlass_scaled_mm` exists but is not usable: with unit scales it returns
inf, with per-tensor scales it returns values ~1e9 off, and even with
per-row/per-col scales the error is 2x the reference magnitude. It does not
raise -- it silently writes garbage -- so a dtype-only smoke test would pass.
That rules out the native W8A8 path.

This checks the remaining option: does Triton's `tl.dot` support fp8 x fp8
on this backend, and is it numerically sane and fast? Measured against a
dequantized fp32 reference, at the decode shapes that matter.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

import mcoplib._C  # noqa: F401  (registers torch.ops._C)

BF16 = torch.bfloat16
FP8 = torch.float8_e4m3fn


@triton.jit
def scaled_mm_kernel(
    A, B, C, a_scale, b_scale,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        # fp8 x fp8 -> fp32 accumulate. If the backend cannot lower this it
        # will either refuse to compile or fall back to a scalar loop.
        acc += tl.dot(a, b, out_dtype=tl.float32)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    acc = acc * a_scale * b_scale
    c = acc.to(tl.bfloat16)
    c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, c, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def fp8_mm(a_fp8, b_fp8_t, a_scale, b_scale, bm=64, bn=64, bk=64,
           warps=4, stages=3, group=8):
    M, K = a_fp8.shape
    N = b_fp8_t.shape[1]
    c = torch.empty(M, N, device=a_fp8.device, dtype=BF16)
    grid = (triton.cdiv(M, bm) * triton.cdiv(N, bn),)
    scaled_mm_kernel[grid](
        a_fp8, b_fp8_t, c, a_scale, b_scale, M, N, K,
        a_fp8.stride(0), a_fp8.stride(1),
        b_fp8_t.stride(0), b_fp8_t.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_M=group,
        num_warps=warps, num_stages=stages,
    )
    return c


def quant(t, scale=None):
    amax = t.abs().amax().clamp(min=1e-6)
    s = (amax / 448.0).to(torch.float32) if scale is None else scale
    out = torch.empty(t.shape, device=t.device, dtype=FP8)
    torch.ops._C.static_scaled_fp8_quant(out, t, s.reshape(1))
    return out, s


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
    print("DEV", torch.cuda.get_device_properties(0).name)

    for (name, N, K) in (("qkv", 2560, 2048), ("gate_up", 12288, 2048),
                         ("o_proj", 2048, 2048), ("down", 2048, 6144),
                         ("lm_head", 130560, 2048)):
        for M in (1, 64):
            x = torch.randn(M, K, device=dev, dtype=BF16)
            w = torch.randn(N, K, device=dev, dtype=BF16) * 0.02
            ref = F.linear(x, w)

            xq, sa = quant(x)
            wq, sb = quant(w)
            bT = wq.t().contiguous()

            # numeric check: dequantized-torch reference vs kernel
            exp = (xq.float() @ wq.float().t()) * float(sa) * float(sb)
            try:
                got = fp8_mm(xq, bT, float(sa), float(sb))
                torch.cuda.synchronize()
                e_deq = (got.float() - exp).abs().max().item()
                e_ref = (got.float() - ref).abs().max().item()
                rel = e_ref / ref.float().abs().max().clamp(min=1e-6).item()
                print("MM %-8s M=%-3d N=%-6d  err_vs_deq=%.3e  err_vs_bf16=%.3e  rel=%.4f"
                      % (name, M, N, e_deq, e_ref, rel))
            except Exception as exc:
                print("MM %-8s M=%-3d FAIL %s: %s" % (name, M, type(exc).__name__,
                                                      str(exc).replace("\n", " ")[:140]))
                continue

            t_bf = bench(lambda: F.linear(x, w))
            t_f8 = bench(lambda: fp8_mm(xq, bT, float(sa), float(sb)))
            wb_bf, wb_f8 = N * K * 2, N * K
            print("    bf16 %8.4f ms %6.0f GB/s | triton-fp8 %8.4f ms %6.0f GB/s | %.2fx"
                  % (t_bf, wb_bf / (t_bf * 1e6), t_f8, wb_f8 / (t_f8 * 1e6), t_bf / t_f8))


if __name__ == "__main__":
    main()
