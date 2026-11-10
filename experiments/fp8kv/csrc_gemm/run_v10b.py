import ctypes, os, sys, torch, torch.nn.functional as F
HERE=os.path.dirname(os.path.abspath(__file__))
lib=ctypes.CDLL(os.path.join(HERE,"libgemm_v10.so"))
lib.launch_gemm_v9.argtypes=[ctypes.c_void_p]*3+[ctypes.c_int]*4+[ctypes.c_void_p]
FP8=torch.float8_e4m3fn
dev="cuda"; torch.manual_seed(0); st=torch.cuda.current_stream().cuda_stream; M=64
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
TC={0:"WPB4 NT2 KC64",1:"WPB2 NT2 KC64",2:"WPB4 NT1 KC64",3:"WPB4 NT4 KC64",4:"WPB8 NT1 KC64",5:"WPB2 NT4 KC64",
    6:"WPB4 NT2 KC32",7:"WPB2 NT2 KC32",8:"WPB4 NT4 KC32",9:"WPB8 NT2 KC32"}
for N,K in ((130560,2048),):
    xb=torch.randn(M,K,device=dev,dtype=torch.bfloat16)
    wb=torch.randn(N,K,device=dev,dtype=torch.bfloat16)*0.02
    wq,_=quant(wb); c=torch.zeros(M,N,device=dev,dtype=torch.float32)
    tbf=bench(lambda: F.linear(xb,wb))
    print("N=%d K=%d bf16 %.5f ms %.0f GB/s"%(N,K,tbf,N*K*2/(tbf*1e6)))
    res=[]
    for tc,nm in TC.items():
        cfg=(1*10+tc)*100+1
        rc=lib.launch_gemm_v9(xb.data_ptr(),wq.data_ptr(),c.data_ptr(),M,N,K,cfg,st)
        if rc!=0: print("  %-16s rc=%d"%(nm,rc)); continue
        t=bench(lambda cf=cfg: lib.launch_gemm_v9(xb.data_ptr(),wq.data_ptr(),c.data_ptr(),M,N,K,cf,st))
        print("  %-16s %8.5f ms %6.0f GB/s %.2fx"%(nm,t,N*K/(t*1e6),tbf/t)); res.append((t,nm))
    res.sort(); print("BEST", res[0][1], "%.5f ms %.2fx"%(res[0][0],tbf/res[0][0]))
    del xb,wb,wq,c; torch.cuda.empty_cache()
