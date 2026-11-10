"""Benchmark all gemm_fp8_v2 configs for one shape, in a single process.

usage: python3 bench_all.py <N> <K>
Prints one line per config plus the bf16 (mcblas) reference.
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

N = int(sys.argv[1])
K = int(sys.argv[2])

CFGS = {
    0: "WPB4 KC32  PAD0 NT1",
    1: "WPB4 KC64  PAD0 NT1",
    2: "WPB4 KC64  PAD4 NT1",
    3: "WPB4 KC128 PAD4 NT1",
    4: "WPB4 KC256 PAD4 NT1",
    5: "WPB4 KC128 PAD4 NT2",
    6: "WPB8 KC128 PAD4 NT1",
    7: "WPB8 KC128 PAD4 NT2",
    8: "WPB2 KC128 PAD4 NT1",
    9: "WPB4 KC128 PAD8 NT1",
}

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


def bench(fn, iters=30, warmup=5):
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
c = torch.empty(M, N, device=dev, dtype=torch.float32)

tbf = bench(lambda: F.linear(xb, wb))
bf_gbs = (N * K * 2) / (tbf * 1e6)
print("REF N=%d K=%d  bf16 %.5f ms  %.0f GB/s" % (N, K, tbf, bf_gbs))

results = []
for cfg in sorted(CFGS):
    try:
        t = bench(lambda c_=cfg: lib.launch_gemm_v2(
            xq.data_ptr(), wq.data_ptr(), c.data_ptr(), M, N, K, c_, st))
    except Exception as exc:
        print("BENCH cfg%d %-24s FAIL %s" % (cfg, CFGS[cfg], type(exc).__name__))
        continue
    gbs = (N * K) / (t * 1e6)
    print("BENCH cfg%d %-24s %8.5f ms  %6.0f GB/s  %.2fx"
          % (cfg, CFGS[cfg], t, gbs, tbf / t))
    results.append((t, gbs, cfg))

results.sort()
if results:
    t, gbs, cfg = results[0]
    print("BEST  cfg%d %.5f ms  %.0f GB/s  %.2fx vs bf16" % (cfg, t, gbs, tbf / t))
