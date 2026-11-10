import ctypes, os, statistics, torch
H=os.path.dirname(os.path.abspath(__file__))
lib=ctypes.CDLL(os.path.join(H,"librows.so"))
for nm in ("launch_rows","launch_rows_kc64"):
    f=getattr(lib,nm); f.argtypes=[ctypes.c_void_p,ctypes.c_void_p]+[ctypes.c_int]*3+[ctypes.c_void_p]; f.restype=ctypes.c_int
dev="cuda"; st=torch.cuda.current_stream().cuda_stream
N,K=130560,2048
B=torch.zeros(N*K,device=dev,dtype=torch.uint8)
sink=torch.zeros(4096,device=dev,dtype=torch.float32)
mb=N*K/1e6
def bench(fn,arg,it=10):
    def go(): fn(B.data_ptr(),sink.data_ptr(),N,K,arg,st)
    for _ in range(5): go()
    torch.cuda.synchronize()
    v=[]
    for _ in range(5):
        a,b=torch.cuda.Event(True),torch.cuda.Event(True); a.record()
        for _ in range(it): go()
        b.record(); torch.cuda.synchronize(); v.append(a.elapsed_time(b)/it*1000.0)
    return statistics.median(v)
print("lm_head weights %.1f MB, contiguous floor 177.7 us / 1505 GB/s\n"%(mb))
print("%-28s %9s %8s"%("per-row run length","us","GB/s"))
for kc,label in ((4,"KC=64   (current)"),(8,"KC=128"),(16,"KC=256"),(32,"KC=512"),
                 (64,"KC=1024"),(128,"KC=2048 (contiguous)")):
    t=bench(lib.launch_rows,kc)
    print("%-28s %9.1f %8.0f"%(label,t,mb/t*1000))
print("\nKC=64 held fixed, varying rows per block:")
print("%-28s %9s %8s"%("rows/block","us","GB/s"))
for rows,nthr in ((256,256),(128,256),(64,256),(32,256),(16,64)):
    t=bench(lib.launch_rows_kc64,rows)
    print("%-28s %9.1f %8.0f"%("%d rows (%d thr)"%(rows,nthr),t,mb/t*1000))
