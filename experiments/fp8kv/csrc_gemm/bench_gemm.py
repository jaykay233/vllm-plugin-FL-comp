"""Hardware-fp8-MMA GEMM vs mcblas bf16 at MiniCPM5-2B decode shapes (M=64).

Memory notes: the container cgroup is 32 GB and the MACA runtime alone maps
~22 GB, so weights are kept in bf16 and quantized in chunks -- materializing a
float32 copy of lm_head (1.07 GB) is enough to get OOM-killed.
"""

import ctypes
import os
import subprocess

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "gemm_fp8_mma.cu")
LIB = os.path.join(HERE, "libgemm_fp8.so")

r = subprocess.run(["/opt/maca/mxgpu_llvm/bin/mxcc", "-shared", "-fPIC", "-O3",
                    "--offload-arch=xcore1000", "-I/opt/maca/include", SRC, "-o", LIB],
                   capture_output=True, text=True)
print("BUILD", "OK" if r.returncode == 0 else "FAIL")
if r.returncode:
    print(r.stderr[-2500:])
    raise SystemExit(1)

lib = ctypes.CDLL(LIB)
dev = "cuda"
torch.manual_seed(0)
st = torch.cuda.current_stream().cuda_stream
FP8 = torch.float8_e4m3fn
SIG = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 3 + [ctypes.c_void_p]
for n in ("launch_fp8_gemm", "launch_gemm64", "launch_gemm64_smem", "launch_gemm64_pipe"):
    getattr(lib, n).argtypes = SIG

MAXV = 448.0


def quant_chunked(w, chunk=8192):
    """Per-tensor fp8 quant with amax -> 448.

    Scaling matters for correctness, not just accuracy: the hardware MMA flushes
    e4m3 subnormals to zero, so weights left in physical units (e.g. ~0.02) lose
    half their values. See dbg_subnormal.py.
    """
    s = (w.abs().amax().clamp(min=1e-12) / MAXV).to(torch.float32)
    q = torch.empty(w.shape, dtype=FP8, device=w.device)
    for r0 in range(0, w.shape[0], chunk):
        r1 = min(r0 + chunk, w.shape[0])
        q[r0:r1] = (w[r0:r1].float() / s).to(FP8)
    return q, s


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


# ---------------- correctness ----------------
M, N, K = 64, 256, 256
xb = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
wb = (torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.02)
xq, xs = quant_chunked(xb)
wq, ws = quant_chunked(wb)
ref_true = F.linear(xb, wb)
ref_deq = (xq.float() @ wq.float().t()) * (xs * ws)
for fn in ("launch_gemm64_pipe",):
    c = torch.full((M, N), float("nan"), device=dev, dtype=torch.float32)
    getattr(lib, fn)(xq.data_ptr(), wq.data_ptr(), c.data_ptr(), M, N, K, st)
    torch.cuda.synchronize()
    if bool(torch.isnan(c).all()):
        print("CORR %-16s NOT WRITTEN" % fn)
        continue
    c = c * (xs * ws)
    print("CORR %-16s rel_vs_deq=%.2e  rel_vs_fp32=%.4f"
          % (fn, (c - ref_deq).abs().max().item() / ref_deq.abs().max().item(),
             (c - ref_true).abs().max().item() / ref_true.abs().max().item()))
del xb, wb, xq, wq, c
torch.cuda.empty_cache()

# ---------------- throughput ----------------
print("\n--- M=64: hardware fp8 MMA vs mcblas bf16 ---")
SHAPES = (("qkv", 2560, 2048), ("gate_up", 12288, 2048), ("o_proj", 2048, 2048),
          ("down", 2048, 6144), ("lm_head", 130560, 2048))
LAYERS = 42
print("%-8s %6s %11s %11s %9s %9s %8s"
      % ("proj", "N", "pipe ms", "bf16 ms",
         "fp8 GB/s", "bf16 GB/s", "speedup"))
tot = {"g64": 0.0, "gen": 0.0, "bf": 0.0}

for name, N, K in SHAPES:
    xb = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
    wb = (torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.02)
    xq, xs = quant_chunked(xb)
    wq, ws = quant_chunked(wb)
    c = torch.empty(M, N, device=dev, dtype=torch.float32)

    t64 = bench(lambda: lib.launch_gemm64_pipe(xq.data_ptr(), wq.data_ptr(), c.data_ptr(), M, N, K, st))
    tbf = bench(lambda: F.linear(xb, wb))

    wb8, wb16 = N * K, N * K * 2
    print("%-8s %6d %11.4f %11.4f %9.0f %9.0f %7.2fx"
          % (name, N, t64, tbf, wb8 / (t64 * 1e6), wb16 / (tbf * 1e6), tbf / t64))

    mult = 1 if name == "lm_head" else LAYERS
    tot["g64"] += t64 * mult
    tot["bf"] += tbf * mult
    del xb, wb, xq, wq, c
    torch.cuda.empty_cache()

print("STEP  fp8-pipe %7.3f ms | bf16 %7.3f ms | gain %.2fx (saved %+.3f ms)"
      % (tot["g64"], tot["bf"], tot["bf"] / tot["g64"],
         tot["bf"] - tot["g64"]))
