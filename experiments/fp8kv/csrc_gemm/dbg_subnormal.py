import ctypes, os, torch
HERE = os.path.dirname(os.path.abspath(__file__))
lib = ctypes.CDLL(os.path.join(HERE, "libgemm_fp8.so"))
lib.launch_fp8_gemm_w1.argtypes = [ctypes.c_void_p]*3 + [ctypes.c_int]*3 + [ctypes.c_void_p]
dev = "cuda"; torch.manual_seed(0)
st = torch.cuda.current_stream().cuda_stream
FP8 = torch.float8_e4m3fn
M, N, K = 64, 256, 256

def check(tag, a, b):
    c = torch.full((M, N), float("nan"), device=dev, dtype=torch.float32)
    lib.launch_fp8_gemm_w1(a.data_ptr(), b.data_ptr(), c.data_ptr(), M, N, K, st)
    torch.cuda.synchronize()
    ref = a.float() @ b.float().t()
    rel = (c - ref).abs().max().item() / ref.abs().max().item()
    sub = (b.float().abs() < 2**-6).float().mean().item()
    print("SUB %-30s rel=%.2e   frac|b|<2^-6 = %.3f" % (tag, rel, sub))

# A) magnitudes in the e4m3 normal range
check("b ~ N(0,1)", torch.randn(M, K, device=dev).to(FP8),
                    torch.randn(N, K, device=dev).to(FP8))

# B) tiny weights, many subnormals (what the earlier run used)
check("b ~ N(0,0.02)", torch.randn(M, K, device=dev).to(FP8),
                       (torch.randn(N, K, device=dev) * 0.02).to(FP8))

# C) real inference: quantize with a scale so amax -> 448, then fold the scale back
w = torch.randn(N, K, device=dev) * 0.02
ws = w.abs().amax() / 448.0
wq = (w / ws).to(FP8)
x = torch.randn(M, K, device=dev)
xs = x.abs().amax() / 448.0
xq = (x / xs).to(FP8)
c = torch.full((M, N), float("nan"), device=dev, dtype=torch.float32)
lib.launch_fp8_gemm_w1(xq.data_ptr(), wq.data_ptr(), c.data_ptr(), M, N, K, st)
torch.cuda.synchronize()
ref_scaled = (xq.float() @ wq.float().t()) * (ws * xs)
ref_true = x @ w.t()
print("SUB scaled fp8 (real usage)     rel_vs_deq=%.2e  rel_vs_fp32=%.4f"
      % ((c * (ws * xs) - ref_scaled).abs().max().item() / ref_scaled.abs().max().item(),
         (c * (ws * xs) - ref_true).abs().max().item() / ref_true.abs().max().item()))
print("SUB frac |wq| < 2^-6 = %.4f" % (wq.float().abs() < 2**-6).float().mean().item())
