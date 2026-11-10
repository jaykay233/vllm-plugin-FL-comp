"""Validate / benchmark gemm_fp8_v3 (fp8 weights -> bf16 MMA).

usage:
  python3 run_v3.py check
  python3 run_v3.py bench <N> <K>
"""

import ctypes
import os
import subprocess
import sys

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "gemm_fp8_v4.cu")
LIB = os.path.join(HERE, "libgemm_v4.so")
FP8 = torch.float8_e4m3fn
MAXV = 448.0

CFGS = {
    0: "WPB4 KC64  NT1",
    1: "WPB4 KC64  NT2",
    2: "WPB4 KC128 NT1",
    3: "WPB2 KC64  NT1",
    4: "WPB2 KC128 NT2",
    5: "WPB4 KC64  NT4",
}

if not os.path.exists(LIB):
    r = subprocess.run(["/opt/maca/mxgpu_llvm/bin/mxcc", "-shared", "-fPIC", "-O3",
                        "--offload-arch=xcore1000", "-I/opt/maca/include", SRC, "-o", LIB],
                       capture_output=True, text=True)
    if r.returncode != 0 or "error:" in r.stderr:
        print("BUILD FAIL")
        print(r.stderr[-2500:])
        raise SystemExit(1)

lib = ctypes.CDLL(LIB)
lib.launch_gemm_v4.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 4 + [ctypes.c_void_p]

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
    N, K = 2048, 2048
    xb = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
    wb = torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.02
    xq, xs = quant(xb)
    wq, ws = quant(wb)
    c = torch.full((M, N), float("nan"), device=dev, dtype=torch.float32)
    ref = (xq.float() @ wq.float().t()) * (xs * ws)
    refmax = ref.abs().max().item()
    for cfg in sorted(CFGS):
        c.fill_(float("nan"))
        if lib.launch_gemm_v4(xq.data_ptr(), wq.data_ptr(), c.data_ptr(), M, N, K, cfg, st) != 0:
            print("CHK cfg%d UNKNOWN" % cfg)
            continue
        torch.cuda.synchronize()
        if bool(torch.isnan(c).all()):
            print("CHK cfg%d %-20s NOTWRITTEN" % (cfg, CFGS[cfg]))
            continue
        rel = ((c * (xs * ws)) - ref).abs().max().item() / refmax
        print("CHK cfg%d %-20s rel=%.2e %s"
              % (cfg, CFGS[cfg], rel, "OK" if rel < 1e-3 else "BAD"))
else:
    N = int(sys.argv[2])
    K = int(sys.argv[3])
    xb = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
    wb = torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.02
    xq, xs = quant(xb)
    wq, ws = quant(wb)
    c = torch.empty(M, N, device=dev, dtype=torch.float32)
    tbf = bench(lambda: F.linear(xb, wb))
    print("REF N=%d K=%d bf16 %.5f ms %.0f GB/s" % (N, K, tbf, (N * K * 2) / (tbf * 1e6)))
    best = None
    for cfg in sorted(CFGS):
        try:
            t = bench(lambda c_=cfg: lib.launch_gemm_v4(
                xq.data_ptr(), wq.data_ptr(), c.data_ptr(), M, N, K, c_, st))
        except Exception as exc:
            print("BENCH cfg%d FAIL %s" % (cfg, type(exc).__name__))
            continue
        gbs = (N * K) / (t * 1e6)
        print("BENCH cfg%d %-20s %8.5f ms %6.0f GB/s %.2fx"
              % (cfg, CFGS[cfg], t, gbs, tbf / t))
        if best is None or t < best[0]:
            best = (t, gbs, cfg)
    if best:
        print("BEST cfg%d %.5f ms %.0f GB/s %.2fx vs bf16" % (best[2], best[0], best[1], tbf / best[0]))
