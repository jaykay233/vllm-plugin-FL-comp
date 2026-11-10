import ctypes, os, torch, torch.nn.functional as F
HERE=os.path.dirname(os.path.abspath(__file__))
lib=ctypes.CDLL(os.path.join(HERE,"libgemm_v13.so"))
lib.launch_gemm_v13.argtypes=[ctypes.c_void_p]*3+[ctypes.c_int]*4+[ctypes.c_void_p]
lib.launch_gemm_v13.restype=ctypes.c_int
FP8=torch.float8_e4m3fn; M=64; dev="cuda"; torch.manual_seed(0)
st=torch.cuda.current_stream().cuda_stream
# MiniCPM5-2B decode shapes: (name, N, K, tilecfg, splits)
SH=[("qkv",2560,2048,23,4),("gate_up",12288,2048,23,1),("o_proj",2048,2048,23,4),
    ("down",2048,2048,23,4),("lm_head",130560,2048,20,1)]
def bench(fn,iters=50,warmup=10):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    a,b=torch.cuda.Event(True),torch.cuda.Event(True); a.record()
    for _ in range(iters): fn()
    b.record(); torch.cuda.synchronize(); return a.elapsed_time(b)/iters
W=[];C=[]
for nm,N,K,tc,spl in SH:
    x=torch.randn(M,K,device=dev,dtype=torch.bfloat16)
    w=torch.randn(N,K,device=dev,dtype=torch.bfloat16)*0.02
    s=(w.abs().amax().float()/448.0); wq=(w.float()/s).to(FP8)
    c=torch.zeros(M,N,device=dev,dtype=torch.float32)
    W.append((nm,N,K,tc,spl,x,w,wq,s,c))
def bf16_chain():
    for nm,N,K,tc,spl,x,w,wq,s,c in W: F.linear(x,w)
def fp8_chain():
    for nm,N,K,tc,spl,x,w,wq,s,c in W:
        c.zero_()
        lib.launch_gemm_v13(x.data_ptr(),wq.data_ptr(),c.data_ptr(),M,N,K,tc*100+spl,st)
tb=bench(bf16_chain); tf=bench(fp8_chain)
print("\n%-9s %10s %10s %8s"%("proj","fp8 ms","bf16 ms","ratio"))
tot_f=tot_b=0.0
for nm,N,K,tc,spl,x,w,wq,s,c in W:
    t1=bench(lambda x=x,w=w: F.linear(x,w))
    t2=bench(lambda x=x,wq=wq,c=c,tc=tc,spl=spl: (c.zero_(),
        lib.launch_gemm_v13(x.data_ptr(),wq.data_ptr(),c.data_ptr(),M,N,K,tc*100+spl,st)))
    tot_f+=t2; tot_b+=t1
    print("%-9s %10.5f %10.5f %7.2fx"%(nm,t2,t1,t1/t2))
print("%-9s %10.5f %10.5f %7.3fx"%("sum",tot_f,tot_b,tot_b/tot_f))
print("DECODE STEP  fp8 %.4f ms | bf16 %.4f ms | ratio %.3fx  (%+.1f%%)"%(tf,tb,tb/tf,(tb/tf-1)*100))
