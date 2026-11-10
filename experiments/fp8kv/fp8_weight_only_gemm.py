# Copyright (c) 2026 BAAI. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""
The only fp8 GEMM variant that can actually work on C500: weight-only storage.

Status of every other route (all measured on this box):

    torch._scaled_mm            blocked by a hardcoded cc >= 8.9/9.0 check;
                                the device reports 8.0, so it raises before
                                reaching a kernel
    _C::cutlass_scaled_mm       present but BROKEN -- unit scales -> inf/nan,
                                per-row scales -> 147% relative error, and
                                [K,N] vs [N,K] b layouts give identical garbage,
                                so it ignores strides. Silently writes junk.
    triton tl.dot(fp8, fp8)     "PassManager::run failed" -- the backend cannot
                                lower an fp8 MMA at all
    _C::static_scaled_fp8_quant works correctly (quantization is fine)

So there is no fp8 compute available. What remains is: keep the weight in fp8
to halve the bytes on the wire, widen it to bf16 in-kernel, and run a normal
bf16 MMA. That trades DRAM traffic (2x less) against conversion work (the
measured 283 GB/s software widening path), so the win is not guaranteed and
has to be measured.

Reported bandwidth is weight_bytes / time, so fp8 rows count 1 byte/elem.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

import mcoplib._C  # noqa: F401

BF16 = torch.bfloat16
FP8 = torch.float8_e4m3fn


@triton.jit
def w_gemm_kernel(
    A, B, C, b_scale,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    FP8_W: tl.constexpr,
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
        if FP8_W:
            # bf16 MMA after widening; no fp8 MMA exists on this backend.
            b = (b.to(tl.float32) * b_scale).to(tl.bfloat16)
        acc += tl.dot(a, b, out_dtype=tl.float32)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = acc.to(tl.bfloat16)
    c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, c, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def w_gemm(a, b, b_scale, fp8_w, bm=64, bn=64, bk=64, warps=4, stages=3, group=8):
    M, K = a.shape
    N = b.shape[1]
    c = torch.empty(M, N, device=a.device, dtype=BF16)
    grid = (triton.cdiv(M, bm) * triton.cdiv(N, bn),)
    w_gemm_kernel[grid](
        a, b, c, b_scale, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        FP8_W=fp8_w, BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_M=group,
        num_warps=warps, num_stages=stages,
    )
    return c


def quant_w(w):
    amax = w.abs().amax().clamp(min=1e-6)
    s = (amax / 448.0).to(torch.float32)
    o = torch.empty(w.shape, device=w.device, dtype=FP8)
    torch.ops._C.static_scaled_fp8_quant(o, w, s.reshape(1))
    return o, s


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

    try:
        import mctlassEx
        api = [n for n in dir(mctlassEx) if not n.startswith("_")]
        print("MCTLASSEX", getattr(mctlassEx, "__version__", "?"), api[:20])
    except Exception as exc:
        print("MCTLASSEX unavailable:", type(exc).__name__, str(exc)[:80])

    SHAPES = (("qkv", 2560, 2048), ("gate_up", 12288, 2048),
              ("o_proj", 2048, 2048), ("down", 2048, 6144),
              ("lm_head", 130560, 2048))

    for M in (1, 64):
        print("\n=== M=%d ===" % M)
        print("%-8s %6s %9s %9s %9s %9s %8s %8s"
              % ("proj", "N", "bf16 ms", "fp8w ms", "bf16 GB/s", "fp8 GB/s",
                 "speedup", "rel"))
        for name, N, K in SHAPES:
            x = torch.randn(M, K, device=dev, dtype=BF16)
            w = torch.randn(N, K, device=dev, dtype=BF16) * 0.02
            wq, s = quant_w(w)
            bT8 = wq.t().contiguous()
            bT16 = w.t().contiguous()

            t_bf = bench(lambda: F.linear(x, w))
            try:
                t_8 = bench(lambda: w_gemm(x, bT8, float(s), True))
            except Exception as exc:
                print("%-8s %6d FAIL %s" % (name, N, type(exc).__name__))
                continue

            ref = F.linear(x, w)
            got = w_gemm(x, bT8, float(s), True)
            torch.cuda.synchronize()
            rel = ((got.float() - ref.float()).abs().max()
                   / ref.float().abs().max().clamp(min=1e-6)).item()

            wb_bf, wb_8 = N * K * 2, N * K
            print("%-8s %6d %9.4f %9.4f %9.0f %9.0f %7.2fx %8.4f"
                  % (name, N, t_bf, t_8, wb_bf / (t_bf * 1e6),
                     wb_8 / (t_8 * 1e6), t_bf / t_8, rel))


if __name__ == "__main__":
    main()
