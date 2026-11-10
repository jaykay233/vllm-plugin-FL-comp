"""Run one fp8 GEMM config in its own process (isolates OOM / traps).

usage:  python3 tune_one.py <cfg> [N] [K] [bench]
Prints a single line:  RES cfg=<c> rel=<e> ms=<t> gbs=<g>
"""

import ctypes
import os
import sys

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
LIB = os.path.join(HERE, "libgemm_v2.so")
FP8 = torch.float8_e4m3fn
MAXV = 448.0

cfg = int(sys.argv[1])
N = int(sys.argv[2]) if len(sys.argv) > 2 else 2048
K = int(sys.argv[3]) if len(sys.argv) > 3 else 2048
do_bench = len(sys.argv) > 4

lib = ctypes.CDLL(LIB)
lib.launch_gemm_v2.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 4 + [ctypes.c_void_p]

dev = "cuda"
torch.manual_seed(0)
st = torch.cuda.current_stream().cuda_stream
M = 64


def quant(t, chunk=8192):
    s = (t.abs().amax().clamp(min=1e-12) / MAXV).float()
    o = torch.empty(t.shape, dtype=FP8, device=t.device)
    for r0 in range(0, t.shape[0], chunk):
        o[r0:r0 + chunk] = (t[r0:r0 + chunk].float() / s).to(FP8)
    return o, s


def bench(fn, iters=20, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(iters):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / iters


xb = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
wb = torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.02
xq, xs = quant(xb)
wq, ws = quant(wb)
c = torch.full((M, N), float("nan"), device=dev, dtype=torch.float32)

if lib.launch_gemm_v2(xq.data_ptr(), wq.data_ptr(), c.data_ptr(), M, N, K, cfg, st) != 0:
    print("RES cfg=%d UNKNOWN_CFG" % cfg)
    raise SystemExit(1)
torch.cuda.synchronize()

if bool(torch.isnan(c).all()):
    print("RES cfg=%d NOTWRITTEN" % cfg)
    raise SystemExit(0)

c = c * (xs * ws)   # kernel returns the raw fp8 accumulator; scale is applied by the caller
ref = (xq.float() @ wq.float().t()) * (xs * ws)
rel = (c - ref).abs().max().item() / ref.abs().max().item()
del ref
torch.cuda.empty_cache()

if do_bench:
    t = bench(lambda: lib.launch_gemm_v2(xq.data_ptr(), wq.data_ptr(),
                                         c.data_ptr(), M, N, K, cfg, st))
    tbf = bench(lambda: F.linear(xb, wb))
    gbs = (N * K) / (t * 1e6)
    print("RES cfg=%d rel=%.2e ms=%.5f gbs=%.0f bf16_ms=%.5f bf16_gbs=%.0f gain=%.2f"
          % (cfg, rel, t, gbs, tbf, (N * K * 2) / (tbf * 1e6), tbf / t))
else:
    print("RES cfg=%d rel=%.2e" % (cfg, rel))
