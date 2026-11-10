import ctypes, os, sys, torch, torch.nn.functional as F
HERE=os.path.dirname(os.path.abspath(__file__))
lib=ctypes.CDLL(os.path.join(HERE,"libgemm_v15.so"))
lib.launch_gemm_v15.argtypes=[ctypes.c_void_p]*3+[ctypes.c_int]*4+[ctypes.c_void_p]
lib.launch_gemm_v15.restype=ctypes.c_int
lib.v15_occ_blocks.restype=ctypes.c_int
FP8=torch.float8_e4m3fn
M=64; N=int(sys.argv[1]); K=int(sys.argv[2]); tc=int(sys.argv[3]); spl=int(sys.argv[4]); name=sys.argv[5]
dev="cuda"; torch.manual_seed(0); st=torch.cuda.current_stream().cuda_stream
def bench(fn,iters=30,warmup=5):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    a,b=torch.cuda.Event(True),torch.cuda.Event(True); a.record()
    for _ in range(iters): fn()
    b.record(); torch.cuda.synchronize(); return a.elapsed_time(b)/iters
xb=torch.randn(M,K,device=dev,dtype=torch.bfloat16)
wb=(torch.randn(N,K,device=dev,dtype=torch.bfloat16)*0.02)
tbf=bench(lambda: F.linear(xb,wb))
s=(wb.abs().amax().float()/448.0); wq=(wb.float()/s).to(FP8)
del wb; torch.cuda.empty_cache()
c=torch.zeros(M,N,device=dev,dtype=torch.float32)
cfg=tc*100+spl
if lib.launch_gemm_v15(xb.data_ptr(),wq.data_ptr(),c.data_ptr(),M,N,K,cfg,st)!=0:
    print("%-26s s%d INVALID"%(name,spl)); sys.exit(0)
torch.cuda.synchronize()
# correctness: fp8 accumulate, then scale
ref=(xb.float()@wq.float().t())*s; rm=ref.abs().max().item()
rel=((c*s)-ref).abs().max().item()/rm
def go():
    c.zero_(); lib.launch_gemm_v15(xb.data_ptr(),wq.data_ptr(),c.data_ptr(),M,N,K,cfg,st)
t=bench(go)
print("%-26s s%d blk/SM=%d %8.5f ms %6.0f GB/s %.2fx rel=%.1e %s"%(
    name,spl,lib.v15_occ_blocks(tc),t,N*K/(t*1e6),tbf/t,rel,"OK" if rel<1e-3 else "BAD"))
