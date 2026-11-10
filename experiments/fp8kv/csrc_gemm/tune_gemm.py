"""Tune the hand-written fp8 GEMM (gemm_fp8_v2.cu) on the shapes that matter.

Sweeps the config table in gemm_fp8_v2.cu, validating every config numerically
before timing it, so a fast-but-wrong variant can never look like a win.

Reference is a dequantized torch matmul of the SAME fp8 bytes, so the residual
is the MMA's error alone (quantization error is common to both).
"""

import ctypes
import os
import subprocess

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "gemm_fp8_v2.cu")
LIB = os.path.join(HERE, "libgemm_v2.so")

r = subprocess.run(["/opt/maca/mxgpu_llvm/bin/mxcc", "-shared", "-fPIC", "-O3",
                    "--offload-arch=xcore1000", "-I/opt/maca/include", SRC, "-o", LIB],
                   capture_output=True, text=True)
print("BUILD", "OK" if r.returncode == 0 else "FAIL")
if r.returncode:
    print(r.stderr[-4000:])
    raise SystemExit(1)

lib = ctypes.CDLL(LIB)
lib.launch_gemm_v2.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 4 + [ctypes.c_void_p]
lib.launch_gemm_v2.restype = ctypes.c_int

dev = "cuda"
torch.manual_seed(0)
st = torch.cuda.current_stream().cuda_stream
FP8 = torch.float8_e4m3fn
MAXV = 448.0

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


M = 64
# ---- correctness on a shape every config can handle ----
N, K = 2048, 2048
xb = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
wb = torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.02
xq, xs = quant(xb)
wq, ws = quant(wb)
ref = (xq.float() @ wq.float().t()) * (xs * ws)
print("\n--- correctness (M=64 N=2048 K=2048, rel vs dequantized ref) ---")
ok_cfgs = []
for cfg in sorted(CFGS):
    c = torch.full((M, N), float("nan"), device=dev, dtype=torch.float32)
    lib.launch_gemm_v2(xq.data_ptr(), wq.data_ptr(), c.data_ptr(), M, N, K, cfg, st)
    torch.cuda.synchronize()
    if bool(torch.isnan(c).all()):
        print("  cfg%-2d %-24s NOT WRITTEN" % (cfg, CFGS[cfg]))
        continue
    rel = (c - ref).abs().max().item() / ref.abs().max().item()
    good = rel < 1e-3
    if good:
        ok_cfgs.append(cfg)
    print("  cfg%-2d %-24s rel=%.2e %s" % (cfg, CFGS[cfg], rel, "OK" if good else "BAD"))
del xb, wb, xq, wq, ref
torch.cuda.empty_cache()

# ---- throughput on the projections that dominate decode ----
SHAPES = (("gate_up", 12288, 2048), ("lm_head", 130560, 2048))
print("\n--- throughput M=64, fp8 weight bytes / time ---")
for name, N, K in SHAPES:
    xb = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
    wb = torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.02
    xq, xs = quant(xb)
    wq, ws = quant(wb)
    c = torch.empty(M, N, device=dev, dtype=torch.float32)
    tbf = bench(lambda: F.linear(xb, wb))
    bf_gbs = (N * K * 2) / (tbf * 1e6)
    print("\n%s N=%d K=%d   bf16(mcblas) %.4f ms  %.0f GB/s"
          % (name, N, K, tbf, bf_gbs))
    best = None
    for cfg in ok_cfgs:
        try:
            t = bench(lambda c_=cfg: lib.launch_gemm_v2(
                xq.data_ptr(), wq.data_ptr(), c.data_ptr(), M, N, K, c_, st))
        except Exception as exc:
            print("  cfg%-2d %-24s FAIL %s" % (cfg, CFGS[cfg], type(exc).__name__))
            continue
        gbs = (N * K) / (t * 1e6)
        print("  cfg%-2d %-24s %8.4f ms  %6.0f GB/s  %.2fx vs bf16"
              % (cfg, CFGS[cfg], t, gbs, tbf / t))
        if best is None or t < best[0]:
            best = (t, cfg)
    if best:
        print("  BEST cfg%d %.4f ms  -> %.2fx vs bf16"
              % (best[1], best[0], tbf / best[0]))
    del xb, wb, xq, wq, c
    torch.cuda.empty_cache()
