import ctypes, os, subprocess, torch
HERE=os.path.dirname(os.path.abspath(__file__))
SRC=os.path.join(HERE,"diag_split.cu"); LIB=os.path.join(HERE,"libdiag.so")
r=subprocess.run(["/opt/maca/mxgpu_llvm/bin/mxcc","-shared","-fPIC","-O3","--offload-arch=xcore1000","-I/opt/maca/include",SRC,"-o",LIB],capture_output=True,text=True)
print("BUILD","OK" if (r.returncode==0 and "error:" not in r.stderr) else "FAIL")
if "error:" in r.stderr: print(r.stderr[-2000:]); raise SystemExit(1)
lib=ctypes.CDLL(LIB)
lib.launch_diag.argtypes=[ctypes.c_void_p]*3+[ctypes.c_int]*4+[ctypes.c_void_p]
dev="cuda"; st=torch.cuda.current_stream().cuda_stream; M=64
FP8=torch.float8_e4m3fn
def quant(t,chunk=8192):
    s=(t.abs().amax().clamp(min=1e-12)/448.0).float()
    o=torch.empty(t.shape,dtype=FP8,device=t.device)
    for r0 in range(0,t.shape[0],chunk): o[r0:r0+chunk]=(t[r0:r0+chunk].float()/s).to(FP8)
    return o,s
def bench(fn,iters=30,warmup=5):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    a,b=torch.cuda.Event(True),torch.cuda.Event(True); a.record()
    for _ in range(iters): fn()
    b.record(); torch.cuda.synchronize(); return a.elapsed_time(b)/iters
for N,K in ((12288,2048),(130560,2048)):
    xb=torch.randn(M,K,device=dev,dtype=torch.bfloat16)
    wb=torch.randn(N,K,device=dev,dtype=torch.bfloat16)*0.02
    xq,_=quant(xb); wq,_=quant(wb)
    c=torch.empty(M,N,device=dev,dtype=torch.float32)
    print("\nN=%d K=%d  fp8 weight bytes=%.1f MB"%(N,K,N*K/1e6))
    for mode,name in ((0,"staging only "),(1,"compute x64  "),(2,"full         ")):
        t=bench(lambda m=mode: lib.launch_diag(xq.data_ptr(),wq.data_ptr(),c.data_ptr(),M,N,K,m,st))
        print("  %s %8.5f ms   %6.0f GB/s-equiv"%(name,t,(N*K)/(t*1e6)))
    del xb,wb,xq,wq,c; torch.cuda.empty_cache()
