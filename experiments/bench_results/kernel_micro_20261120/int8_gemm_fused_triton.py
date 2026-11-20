"""Can a FUSED int8 GEMM beat mcblas BF16 at prefill shapes?

The separate-epilogue version was break-even at best, because _int_mm must write
int32 (2x the bf16 bytes) and then a second pass reads it back.  A real int8
GEMM fuses the per-token / per-channel scale into the accumulator and writes
bf16 straight out, removing both extra passes.  This measures whether Triton can
do that fast enough to matter.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _i8_gemm_fused(A, B, XS, WS, C, M, N, K,
                   BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    m_mask = rm < M
    n_mask = rn < N

    acc = tl.zeros((BM, BN), dtype=tl.int32)
    for k0 in range(0, K, BK):
        kk = k0 + rk
        a = tl.load(A + rm[:, None] * K + kk[None, :],
                    mask=m_mask[:, None] & (kk[None, :] < K), other=0)
        b = tl.load(B + kk[:, None] * N + rn[None, :],
                    mask=(kk[:, None] < K) & n_mask[None, :], other=0)
        acc = tl.dot(a, b, acc, out_dtype=tl.int32)

    xs = tl.load(XS + rm, mask=m_mask, other=0.0).to(tl.float32)
    ws = tl.load(WS + rn, mask=n_mask, other=0.0).to(tl.float32)
    out = acc.to(tl.float32) * xs[:, None] * ws[None, :]
    tl.store(C + rm[:, None] * N + rn[None, :], out.to(tl.bfloat16),
             mask=m_mask[:, None] & n_mask[None, :])


def bench(fn, iters=15, warmup=4):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1000


SHAPES = [(16384, 2048, 12288, "prefill gate_up"),
          (16384, 2048, 2560, "prefill qkv"),
          (16384, 6144, 2048, "prefill down")]
CFGS = [(128, 128, 64), (128, 256, 64), (256, 128, 64), (128, 128, 128), (64, 256, 64)]

for M, K, N, label in SHAPES:
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * 0.5
    b = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.05
    t_bf = bench(lambda: torch.nn.functional.linear(a, b))

    sw = b.abs().amax(dim=1).clamp(min=1e-8).float() / 127.0
    bq = (b.float() / sw[:, None]).round().clamp(-127, 127).to(torch.int8)
    bt_ = bq.t().contiguous()                       # [K, N] int8
    xs = a.abs().amax(dim=1).clamp(min=1e-8).float() / 127.0
    ai = (a.float() / xs[:, None]).round().clamp(-127, 127).to(torch.int8)
    c = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)

    ref = torch.nn.functional.linear(a, b).float()
    best = (1e9, None)
    for BM, BN, BK in CFGS:
        grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
        fn = lambda: _i8_gemm_fused[grid](ai, bt_, xs, sw, c, M, N, K,
                                          BM=BM, BN=BN, BK=BK)
        try:
            fn()
            torch.cuda.synchronize()
            t = bench(fn)
            if t < best[0]:
                best = (t, (BM, BN, BK))
        except Exception as ex:
            print("   cfg %s failed: %s" % ((BM, BN, BK), str(ex)[:60]))
    err = (c.float() - ref).abs().max().item() / (ref.abs().max().item() + 1e-9)
    flops = 2.0 * M * N * K
    print("%-18s bf16 %6.0f us | fused int8 best %6.0f us %s | %5.2fx | "
          "%5.0f TFLOPS | relerr %.4f"
          % (label, t_bf, best[0], best[1], t_bf / best[0],
             flops / best[0] / 1e6, err))
