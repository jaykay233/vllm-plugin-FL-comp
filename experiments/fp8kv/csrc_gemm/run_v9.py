"""Validate / benchmark gemm_fp8_v9 (v8 + arithmetic fp8->bf16, no LUT smem reads).

cfg encoding: cfg = (conv * 10 + tile_cfg) * 100 + splits
    conv 0 = 256-entry LUT (exact, 4 smem reads per fragment)
    conv 1 = pure arithmetic (3.8e-06, zero smem reads)

usage: python3 run_v9.py check [N K] | python3 run_v9.py bench <N> <K>
"""

import ctypes
import os
import sys

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
LIB = os.path.join(HERE, "libgemm_v9.so")
FP8 = torch.float8_e4m3fn
MAXV = 448.0

TC = {
    0: "WPB4 NT2",
    1: "WPB2 NT2",
    2: "WPB4 NT1",
    3: "WPB4 NT4",
    4: "WPB8 NT1",
    5: "WPB2 NT4",
}

CFGS = [
    (0, 0, 1),   # LUT,  WPB4 NT2, no split
    (1, 0, 1),   # arith,WPB4 NT2, no split
    (0, 3, 1),   # LUT,  WPB4 NT4, no split   <- best on lm_head in v8
    (1, 3, 1),   # arith,WPB4 NT4, no split
    (0, 1, 4),   # LUT,  WPB2 NT2, split 4    <- best on qkv in v8
    (1, 1, 4),   # arith,WPB2 NT2, split 4
    (0, 0, 4),   # LUT,  WPB4 NT2, split 4    <- best on o_proj/down in v8
    (1, 0, 4),   # arith,WPB4 NT2, split 4
    (0, 5, 4),
    (1, 5, 4),
]

lib = ctypes.CDLL(LIB)
lib.launch_gemm_v9.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 4 + [ctypes.c_void_p]
lib.launch_gemm_v9.restype = ctypes.c_int

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


def name(conv, tc, spl):
    return "%-5s %-9s s%d" % ("arith" if conv else "LUT", TC[tc], spl)


def code(conv, tc, spl):
    return (conv * 10 + tc) * 100 + spl


mode = sys.argv[1]

if mode == "check":
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 2048
    K = int(sys.argv[3]) if len(sys.argv) > 3 else 2048
    xb = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
    wb = torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.02
    wq, ws = quant(wb)
    c = torch.zeros(M, N, device=dev, dtype=torch.float32)
    ref = (xb.float() @ wq.float().t()) * ws
    refmax = ref.abs().max().item()
    for conv, tc, spl in CFGS:
        cfg = code(conv, tc, spl)
        c.zero_()
        rc = lib.launch_gemm_v9(xb.data_ptr(), wq.data_ptr(), c.data_ptr(), M, N, K, cfg, st)
        if rc != 0:
            print("CHK %-24s rc=%d" % (name(conv, tc, spl), rc))
            continue
        torch.cuda.synchronize()
        rel = ((c * ws) - ref).abs().max().item() / refmax
        print("CHK %-24s rel=%.2e %s" % (name(conv, tc, spl), rel, "OK" if rel < 1e-3 else "BAD"))
else:
    N, K = int(sys.argv[2]), int(sys.argv[3])
    xb = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
    wb = torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.02
    wq, ws = quant(wb)
    c = torch.zeros(M, N, device=dev, dtype=torch.float32)
    tbf = bench(lambda: F.linear(xb, wb))
    print("REF N=%d K=%d bf16 %.5f ms %.0f GB/s" % (N, K, tbf, (N * K * 2) / (tbf * 1e6)))
    best = None
    for conv, tc, spl in CFGS:
        cfg = code(conv, tc, spl)

        def go(c_=cfg):
            c.zero_()
            lib.launch_gemm_v9(xb.data_ptr(), wq.data_ptr(), c.data_ptr(), M, N, K, c_, st)

        if lib.launch_gemm_v9(xb.data_ptr(), wq.data_ptr(), c.data_ptr(), M, N, K, cfg, st) != 0:
            continue
        t = bench(go)
        gbs = (N * K) / (t * 1e6)
        print("BENCH %-24s %8.5f ms %6.0f GB/s %.2fx"
              % (name(conv, tc, spl), t, gbs, tbf / t))
        if best is None or t < best[0]:
            best = (t, gbs, name(conv, tc, spl))
    if best:
        print("BEST %-24s %.5f ms %.0f GB/s %.2fx vs bf16" % (best[2], best[0], best[1], tbf / best[0]))
