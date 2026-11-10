"""Final sweep: best fp8 config per MiniCPM5-2B decode projection on C500.

For each shape it validates every candidate numerically against a dequantized
reference, benchmarks it, then reports the per-shape winner and the end-to-end
decode step against bf16 mcblas.

usage: python3 final_sweep.py
"""

import ctypes
import gc
import os

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
LIB = os.path.join(HERE, "libgemm_v12.so")
FP8 = torch.float8_e4m3fn
MAXV = 448.0

TC = {
    0: "WPB4 NT2 KC64",
    1: "WPB2 NT2 KC64",
    3: "WPB4 NT4 KC64",
    5: "WPB2 NT4 KC64",
    6: "WPB4 NT2 KC32",
    9: "WPB2 NT4 KC32",
}
GRID = [(1, tc, s) for tc in TC for s in (1, 4, 8)]

SHAPES = [
    ("qkv", 2560, 2048),
    ("gate_up", 12288, 2048),
    ("o_proj", 2048, 2048),
    ("down", 2048, 6144),
    ("lm_head", 130560, 2048),
]

lib = ctypes.CDLL(LIB)
lib.launch_gemm_v12.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 4 + [ctypes.c_void_p]
lib.launch_gemm_v12.restype = ctypes.c_int

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


tot_fp8 = tot_bf16 = 0.0
rows = []
for name, N, K in SHAPES:
    xb = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
    wb = torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.02
    wq, ws = quant(wb)
    c = torch.zeros(M, N, device=dev, dtype=torch.float32)

    tbf = bench(lambda: F.linear(xb, wb))
    ref = (xb.float() @ wq.float().t()) * ws
    refmax = ref.abs().max().item()
    del ref
    torch.cuda.empty_cache()

    best = None
    for conv, tc, spl in GRID:
        cfg = conv * 10000 + tc * 100 + spl
        c.zero_()
        if lib.launch_gemm_v12(xb.data_ptr(), wq.data_ptr(), c.data_ptr(),
                               M, N, K, cfg, st) != 0:
            continue
        torch.cuda.synchronize()
        rel = ((c * ws) - (xb.float() @ wq.float().t()) * ws).abs().max().item() / refmax

        def go(c_=cfg):
            c.zero_()
            lib.launch_gemm_v12(xb.data_ptr(), wq.data_ptr(), c.data_ptr(),
                                M, N, K, c_, st)

        t = bench(go)
        if rel < 1e-3 and (best is None or t < best[0]):
            best = (t, rel, tc, spl)

    if best is None:
        print("%-8s NO CONFIG PASSED" % name)
    else:
        t, rel, tc, spl = best
        tot_fp8 += t
        tot_bf16 += tbf
        rows.append((name, N, K, t, tbf, tbf / t, TC[tc], spl, rel))
        print("%-8s N=%-6d K=%-5d fp8 %.5f ms  bf16 %.5f ms  %.2fx  %-14s s%d  rel=%.1e"
              % (name, N, K, t, tbf, tbf / t, TC[tc], spl, rel))

    del xb, wb, wq, c
    gc.collect()
    torch.cuda.empty_cache()

print("\n%-8s %10s %10s %7s  %-14s %s" % ("proj", "fp8 ms", "bf16 ms", "ratio", "config", "split"))
for r in rows:
    print("%-8s %10.5f %10.5f %6.2fx  %-14s s%d" % (r[0], r[3], r[4], r[5], r[6], r[7]))
print("\nDECODE STEP  fp8 %.4f ms | bf16 %.4f ms | ratio %.3fx  (%+.1f%%)"
      % (tot_fp8, tot_bf16, tot_bf16 / tot_fp8, (tot_bf16 / tot_fp8 - 1) * 100))
