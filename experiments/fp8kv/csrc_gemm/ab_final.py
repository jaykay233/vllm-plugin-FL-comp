import ctypes, os, torch, statistics
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
K3=[("v13 tc20",v13,2001),("v16 tc20",v16,2001),("v16 tc24 NT2",v16,2401),
    ("v17 tc21 persist",v17,2101)]
def run(f,cfg):
    c.zero_(); f(x.data_ptr(),wq.data_ptr(),c.data_ptr(),M,N,K,cfg,st)
for lbl,f,cfg in K3: run(f,cfg)
torch.cuda.synchronize()
smp={lbl:[] for lbl,_,_ in K3}; IT=25
for rep in range(14):
    for lbl,f,cfg in K3:
        a,b=torch.cuda.Event(True),torch.cuda.Event(True); a.record()
        for _ in range(IT): run(f,cfg)
        b.record(); torch.cuda.synchronize(); smp[lbl].append(a.elapsed_time(b)/IT)
print("%-20s %8s %8s"%("variant","median","min"))
base=None
for lbl,_,_ in K3:
    v=smp[lbl]; m=statistics.median(v)
    if base is None: base=m
    print("%-20s %8.4f %8.4f   %+.1f%%"%(lbl,m,min(v),(base/m-1)*100))
