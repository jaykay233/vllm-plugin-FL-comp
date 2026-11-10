import ctypes, os, sys, torch, torch.nn.functional as F
HERE=os.path.dirname(os.path.abspath(__file__))
lib=ctypes.CDLL(os.path.join(HERE,"libgemm_v12.so"))
lib.launch_gemm_v12.argtypes=[ctypes.c_void_p]*3+[ctypes.c_int]*4+[ctypes.c_void_p]
lib.launch_gemm_v12.restype=ctypes.c_int
FP8=torch.float8_e4m3fn
dev="cuda"; torch.manual_seed(0); st=torch.cuda.current_stream().cuda_stream; M=64
TC={0:"WPB4 NT2 KC64",1:"WPB2 NT2 KC64",3:"WPB4 NT4 KC64",5:"WPB2 NT4 KC64",
    6:"WPB4 NT2 KC32",9:"WPB2 NT4 KC32",10:"WPB2 NT8 KC64",11:"WPB2 NT8 KC32"}
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
CONV={0:"LUT  ",1:"arith"}
mode=sys.argv[1]
if mode=="check":
    N,K=2048,2048
    xb=torch.randn(M,K,device=dev,dtype=torch.bfloat16)
    wb=torch.randn(N,K,device=dev,dtype=torch.bfloat16)*0.02
    wq,ws=quant(wb); c=torch.zeros(M,N,device=dev,dtype=torch.float32)
    ref=(xb.float()@wq.float().t())*ws; rm=ref.abs().max().item()
    for cv in (0,1):
        for tc in sorted(TC):
            cfg=cv*10000+tc*100+1
            c.zero_()
            if lib.launch_gemm_v12(xb.data_ptr(),wq.data_ptr(),c.data_ptr(),M,N,K,cfg,st)!=0: continue
            torch.cuda.synchronize()
            rel=((c*ws)-ref).abs().max().item()/rm
            print("CHK %s %-15s rel=%.2e %s"%(CONV[cv],TC[tc],rel,"OK" if rel<1e-3 else "BAD"))
else:
    N,K=int(sys.argv[2]),int(sys.argv[3])
    xb=torch.randn(M,K,device=dev,dtype=torch.bfloat16)
    wb=torch.randn(N,K,device=dev,dtype=torch.bfloat16)*0.02
    wq,_=quant(wb); c=torch.zeros(M,N,device=dev,dtype=torch.float32)
    tbf=bench(lambda: F.linear(xb,wb))
    print("REF N=%d K=%d bf16 %.5f ms %.0f GB/s"%(N,K,tbf,N*K*2/(tbf*1e6)))
    res=[]
    for cv in (0,1):
        for tc in sorted(TC):
            cfg=cv*10000+tc*100+1
            t=bench(lambda cf=cfg: lib.launch_gemm_v12(xb.data_ptr(),wq.data_ptr(),c.data_ptr(),M,N,K,cf,st))
            print("  %s %-15s %8.5f ms %6.0f GB/s %.2fx"%(CONV[cv],TC[tc],t,N*K/(t*1e6),tbf/t))
            res.append((t,CONV[cv]+TC[tc]))
    res.sort(); print("BEST %-20s %.5f ms %.2fx"%(res[0][1],res[0][0],tbf/res[0][0]))
