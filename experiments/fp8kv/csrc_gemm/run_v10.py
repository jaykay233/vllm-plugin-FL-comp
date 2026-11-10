import ctypes, os, sys, torch, torch.nn.functional as F
HERE=os.path.dirname(os.path.abspath(__file__))
LIB=os.path.join(HERE,"libgemm_v10.so")
FP8=torch.float8_e4m3fn; MAXV=448.0
lib=ctypes.CDLL(LIB)
lib.launch_gemm_v9.argtypes=[ctypes.c_void_p]*3+[ctypes.c_int]*4+[ctypes.c_void_p]
lib.launch_gemm_v9.restype=ctypes.c_int
dev="cuda"; torch.manual_seed(0); st=torch.cuda.current_stream().cuda_stream; M=64
def quant(t,chunk=8192):
    s=(t.abs().amax().clamp(min=1e-12)/MAXV).float()
    o=torch.empty(t.shape,dtype=FP8,device=t.device)
    for r0 in range(0,t.shape[0],chunk): o[r0:r0+chunk]=(t[r0:r0+chunk].float()/s).to(FP8)
    return o,s
def bench(fn,iters=30,warmup=5):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    a,b=torch.cuda.Event(True),torch.cuda.Event(True); a.record()
    for _ in range(iters): fn()
    b.record(); torch.cuda.synchronize(); return a.elapsed_time(b)/iters
# cfg = mode*100000 + (conv*10+tc)*100 + spl   ; tc3 = WPB4 NT4 arith
CFG=(1*10+3)*100+1
MODES={0:"full          ",1:"staging only  ",2:"compute only  "}
for N,K in ((130560,2048),(12288,2048)):
    xb=torch.randn(M,K,device=dev,dtype=torch.bfloat16)
    wb=torch.randn(N,K,device=dev,dtype=torch.bfloat16)*0.02
    wq,_=quant(wb); c=torch.zeros(M,N,device=dev,dtype=torch.float32)
    tbf=bench(lambda: F.linear(xb,wb))
    print("\nN=%d K=%d  bf16 %.5f ms (%.0f GB/s)  fp8 bytes=%.1f MB"%(N,K,tbf,N*K*2/(tbf*1e6),N*K/1e6))
    for m,nm in MODES.items():
        cfg=m*100000+CFG
        t=bench(lambda cf=cfg: lib.launch_gemm_v9(xb.data_ptr(),wq.data_ptr(),c.data_ptr(),M,N,K,cf,st))
        print("  %s %8.5f ms  %6.0f GB/s  %.2fx"%(nm,t,N*K/(t*1e6),tbf/t))
    del xb,wb,wq,c; torch.cuda.empty_cache()
