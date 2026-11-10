"""Validate / benchmark gemm_fp8_v8 (v7 + split-K).

Split-K accumulates with atomicAdd, so C must be zeroed before every launch; the
zeroing is included in the timed region because it is required in practice.

usage: python3 run_v8.py check [N K] | python3 run_v8.py bench <N> <K>
"""

import ctypes
import os
import sys

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
LIB = os.path.join(HERE, "libgemm_v8.so")
FP8 = torch.float8_e4m3fn
MAXV = 448.0

# cfg = tile config * 100 + splits
CFGS = [
    (0, 1, "WPB4 NT2  s1"),
    (0, 2, "WPB4 NT2  s2"),
    (0, 4, "WPB4 NT2  s4"),
    (0, 8, "WPB4 NT2  s8"),
    (1, 2, "WPB2 NT2  s2"),
    (1, 4, "WPB2 NT2  s4"),
    (1, 8, "WPB2 NT2  s8"),
    (2, 4, "WPB4 NT1  s4"),
    (2, 8, "WPB4 NT1  s8"),
    (3, 1, "WPB4 NT4  s1"),
    (3, 4, "WPB4 NT4  s4"),
    (5, 1, "WPB2 NT4  s1"),
    (5, 8, "WPB2 NT4  s8"),
]

lib = ctypes.CDLL(LIB)
lib.launch_gemm_v8.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 4 + [ctypes.c_void_p]
lib.launch_gemm_v8.restype = ctypes.c_int

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
    for tc, spl, name in CFGS:
        cfg = tc * 100 + spl
        c.zero_()
        rc = lib.launch_gemm_v8(xb.data_ptr(), wq.data_ptr(), c.data_ptr(), M, N, K, cfg, st)
        if rc != 0:
            print("CHK cfg%-4d %-18s rc=%d (split invalid for this K)" % (cfg, name, rc))
            continue
        torch.cuda.synchronize()
        rel = ((c * ws) - ref).abs().max().item() / refmax
        print("CHK cfg%-4d %-18s rel=%.2e %s"
              % (cfg, name, rel, "OK" if rel < 1e-3 else "BAD"))
else:
    N, K = int(sys.argv[2]), int(sys.argv[3])
    xb = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
    wb = torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.02
    wq, ws = quant(wb)
    c = torch.zeros(M, N, device=dev, dtype=torch.float32)
    tbf = bench(lambda: F.linear(xb, wb))
    print("REF N=%d K=%d bf16 %.5f ms %.0f GB/s" % (N, K, tbf, (N * K * 2) / (tbf * 1e6)))
    best = None
    for tc, spl, name in CFGS:
        cfg = tc * 100 + spl
        rc = lib.launch_gemm_v8(xb.data_ptr(), wq.data_ptr(), c.data_ptr(), M, N, K, cfg, st)
        if rc != 0:
            print("  cfg%-4d %-18s split %d invalid for K=%d" % (cfg, name, spl, K))
            continue

        def go(c_=cfg):
            c.zero_()
            lib.launch_gemm_v8(xb.data_ptr(), wq.data_ptr(), c.data_ptr(), M, N, K, c_, st)

        t = bench(go)
        gbs = (N * K) / (t * 1e6)
        print("BENCH cfg%-4d %-18s %8.5f ms %6.0f GB/s %.2fx"
              % (cfg, name, t, gbs, tbf / t))
        if best is None or t < best[0]:
            best = (t, gbs, cfg, name)
    if best:
        print("BEST cfg%d %s %.5f ms %.0f GB/s %.2fx vs bf16"
              % (best[2], best[3], best[0], best[1], tbf / best[0]))
