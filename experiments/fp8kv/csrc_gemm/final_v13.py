import ctypes, os, sys, torch, torch.nn.functional as F
HERE=os.path.dirname(os.path.abspath(__file__))
lib=ctypes.CDLL(os.path.join(HERE,"libgemm_v13.so"))
lib.launch_gemm_v13.argtypes=[ctypes.c_void_p]*3+[ctypes.c_int]*4+[ctypes.c_void_p]
lib.launch_gemm_v13.restype=ctypes.c_int
FP8=torch.float8_e4m3fn
M=64
name,N,K,tc,spl=sys.argv[1],int(sys.argv[2]),int(sys.argv[3]),int(sys.argv[4]),int(sys.argv[5])
dev="cuda"; torch.manual_seed(0); st=torch.cuda.current_stream().cuda_stream
def bench(fn,iters=50,warmup=10):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    a,b=torch.cuda.Event(True),torch.cuda.Event(True); a.record()
    for _ in range(iters): fn()
    b.record(); torch.cuda.synchronize(); return a.elapsed_time(b)/iters
xb=torch.randn(M,K,device=dev,dtype=torch.bfloat16)
wb=(torch.randn(N,K,device=dev,dtype=torch.bfloat16)*0.02)
tbf=bench(lambda: F.linear(xb,wb))
s=(wb.abs().amax().float()/448.0); wq=(wb.float()/s).to(FP8)
ref=(xb.float()@wq.float().t())*s; rm=ref.abs().max().item(); del wb
c=torch.zeros(M,N,device=dev,dtype=torch.float32)
cfg=tc*100+spl
rc=lib.launch_gemm_v13(xb.data_ptr(),wq.data_ptr(),c.data_ptr(),M,N,K,cfg,st)
torch.cuda.synchronize()
rel=((c*s)-ref).abs().max().item()/rm
def go():
    c.zero_(); lib.launch_gemm_v13(xb.data_ptr(),wq.data_ptr(),c.data_ptr(),M,N,K,cfg,st)
t=bench(go)
print("%-9s N=%6d K=%5d tc=%d s%d  fp8 %8.5f  bf16 %8.5f  %.2fx  rel=%.1e %s"%(name,N,K,tc,spl,t,tbf,tbf/t,rel,"OK" if rel<1e-3 else "BAD"))
