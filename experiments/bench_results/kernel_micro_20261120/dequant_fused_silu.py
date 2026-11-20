"""Fuse the int8 GEMM's scale dequant into the elementwise op that already
follows it, so the int32 result is never read by a separate pass.

gate_up produces [M, 2*I] which SiLU-mults down to [M, I].  Baseline reads the
bf16 GEMM output; the int8 version reads the int32 GEMM output and dequantises
inline.  The delta is the cost of reading int32 instead of bf16 once.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def silu_bf16(G, XS, WS, OUT, N, I, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    c = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = c < I
    g = tl.load(G + row.to(tl.int64) * N + c, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(G + row.to(tl.int64) * N + (I + c), mask=mask, other=0.0).to(tl.float32)
    s = 1.0 / (1.0 + tl.exp(-g))
    tl.store(OUT + row.to(tl.int64) * I + c, (g * s * u).to(tl.bfloat16), mask=mask)


@triton.jit
def silu_int8(G32, XS, WS, OUT, N, I, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    c = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = c < I
    sc = tl.load(XS + row) * tl.load(WS + c, mask=mask, other=0.0)
    sc2 = tl.load(XS + row) * tl.load(WS + I + c, mask=mask, other=0.0)
    g = tl.load(G32 + row.to(tl.int64) * N + c, mask=mask, other=0).to(tl.float32) * sc
    u = tl.load(G32 + row.to(tl.int64) * N + (I + c), mask=mask, other=0).to(tl.float32) * sc2
    s = 1.0 / (1.0 + tl.exp(-g))
    tl.store(OUT + row.to(tl.int64) * I + c, (g * s * u).to(tl.bfloat16), mask=mask)


def bench(fn, iters=20, warmup=6):
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


M, I = 16384, 6144
N = 2 * I
g_bf = torch.randn(M, N, device="cuda", dtype=torch.bfloat16) * 0.5
g32 = torch.randint(-30000, 30000, (M, N), device="cuda", dtype=torch.int32)
xs = torch.full((M,), 0.004, device="cuda", dtype=torch.float32)
ws = torch.full((N,), 0.004, device="cuda", dtype=torch.float32)
out = torch.empty(M, I, device="cuda", dtype=torch.bfloat16)

BLK = 1024
grid = (M, triton.cdiv(I, BLK))
tb = bench(lambda: silu_bf16[grid](g_bf, xs, ws, out, N, I, BLOCK=BLK))
t8 = bench(lambda: silu_int8[grid](g32, xs, ws, out, N, I, BLOCK=BLK))
print("silu from bf16 GEMM out : %6.0f us  (%d MB read)" % (tb, M * N * 2 / 1e6))
print("silu from int32 GEMM out: %6.0f us  (%d MB read)  delta +%.0f us"
      % (t8, M * N * 4 / 1e6, t8 - tb))
print()
gemm_i8, gemm_bf = 2924.0, 3912.0
print("gate_up layer-chunk, M=16384, N=12288, I=6144:")
print("  bf16 GEMM + bf16 silu      = %.0f + %.0f = %.0f us"
      % (gemm_bf, tb, gemm_bf + tb))
print("  int8 GEMM + fused-deq silu = %.0f + %.0f = %.0f us   -> %.2fx"
      % (gemm_i8, t8, gemm_i8 + t8, (gemm_bf + tb) / (gemm_i8 + t8)))
