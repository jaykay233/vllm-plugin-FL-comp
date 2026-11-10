import ctypes, os, torch, statistics, torch.nn.functional as F
H=os.path.dirname(os.path.abspath(__file__))
lib=ctypes.CDLL(os.path.join(H,"libgemm_v16.so"))
lib.launch_gemm_v16.argtypes=[ctypes.c_void_p]*3+[ctypes.c_int]*4+[ctypes.c_void_p]
lib.launch_gemm_v16.restype=ctypes.c_int
FP8=torch.float8_e4m3fn; M=64; dev="cuda"; torch.manual_seed(0)
st=torch.cuda.current_stream().cuda_stream
SH=[("qkv",2560,2048,23,4),("gate_up",12288,2048,23,1),("o_proj",2048,2048,23,4),
    ("down",2048,2048,23,4),("lm_head",130560,2048,20,1)]
W=[]
for nm,N,K,tc,spl in SH:
    x=torch.randn(M,K,device=dev,dtype=torch.bfloat16)
    w=torch.randn(N,K,device=dev,dtype=torch.bfloat16)*0.02
    s=(w.abs().amax().float()/448.0); wq=(w.float()/s).to(FP8)
    c=torch.zeros(M,N,device=dev,dtype=torch.float32)
    W.append(dict(nm=nm,N=N,K=K,tc=tc,spl=spl,x=x,w=w,wq=wq,s=s,c=c))
def bench(fn,it=30):
    for _ in range(10): fn()
    torch.cuda.synchronize()
    v=[]
    for _ in range(9):
        a,b=torch.cuda.Event(True),torch.cuda.Event(True); a.record()
        for _ in range(it): fn()
        b.record(); torch.cuda.synchronize(); v.append(a.elapsed_time(b)/it)
    return statistics.median(v)
def call(d,use_fp8):
    d["c"].zero_()
    lib.launch_gemm_v16(d["x"].data_ptr(),d["wq"].data_ptr(),d["c"].data_ptr(),
                        M,d["N"],d["K"],d["tc"]*100+d["spl"],st)
def all_bf16():
    for d in W: F.linear(d["x"],d["w"])
def all_fp8():
    for d in W: call(d,True)
def hybrid():
    # bf16 for the four small serial projections, fp8 only for lm_head
    for d in W[:-1]: F.linear(d["x"],d["w"])
    call(W[-1],True)
r={}
for nm,fn in (("all bf16",all_bf16),("all fp8",all_fp8),("hybrid",hybrid)):
    r[nm]=bench(fn)
    print("%-10s %.4f ms"%(nm,r[nm]))
print()
print("all fp8 vs bf16 : %.3fx  (%+.1f%%)"%(r["all bf16"]/r["all fp8"],(r["all bf16"]/r["all fp8"]-1)*100))
print("hybrid  vs bf16 : %.3fx  (%+.1f%%)"%(r["all bf16"]/r["hybrid"],(r["all bf16"]/r["hybrid"]-1)*100))
