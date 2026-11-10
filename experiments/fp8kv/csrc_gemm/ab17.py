import ctypes, os, torch, statistics, torch.nn.functional as F
H=os.path.dirname(os.path.abspath(__file__))
def load(n):
    l=ctypes.CDLL(os.path.join(H,"libgemm_%s.so"%n))
    f=getattr(l,"launch_gemm_"+n)
    f.argtypes=[ctypes.c_void_p]*3+[ctypes.c_int]*4+[ctypes.c_void_p]; f.restype=ctypes.c_int
    return f
v13=load("v13"); v16=load("v16"); v17=load("v17")
M=64; N=130560; K=2048; dev="cuda"; torch.manual_seed(0)
st=torch.cuda.current_stream().cuda_stream
x=torch.randn(M,K,device=dev,dtype=torch.bfloat16)
w=torch.randn(N,K,device=dev,dtype=torch.bfloat16)*0.02
s=(w.abs().amax().float()/448.0); wq=(w.float()/s).to(torch.float8_e4m3fn)
del w; torch.cuda.empty_cache()
c=torch.zeros(M,N,device=dev,dtype=torch.float32)
# (label, fn, cfg)
K3=[("v13 tc20",v13,2001),("v16 tc20",v16,2001),("v17 tc20 nonpersist",v17,2001),
    ("v17 tc21 persist",v17,2101),("v17 tc27 NT2 PERSIST",v17,2701)]
def run(f,cfg):
    c.zero_(); f(x.data_ptr(),wq.data_ptr(),c.data_ptr(),M,N,K,cfg,st)
for lbl,f,cfg in K3: run(f,cfg)
torch.cuda.synchronize()
samples={lbl:[] for lbl,_,_ in K3}
IT=20
for rep in range(9):
    for lbl,f,cfg in K3:
        a,b=torch.cuda.Event(True),torch.cuda.Event(True); a.record()
        for _ in range(IT): run(f,cfg)
        b.record(); torch.cuda.synchronize(); samples[lbl].append(a.elapsed_time(b)/IT)
print("%-24s %8s %8s %8s"%("variant","median","min","spread"))
for lbl,_,_ in K3:
    v=samples[lbl]
    print("%-24s %8.4f %8.4f %8.4f"%(lbl,statistics.median(v),min(v),max(v)-min(v)))
