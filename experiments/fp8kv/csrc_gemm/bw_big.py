import ctypes, os, statistics, torch
H=os.path.dirname(os.path.abspath(__file__))
lib=ctypes.CDLL(os.path.join(H,"libfloor.so"))
lib.launch_floor.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p]+[ctypes.c_int]*5+[ctypes.c_void_p]
lib.launch_floor.restype=ctypes.c_int
dev="cuda"; st=torch.cuda.current_stream().cuda_stream
LM=130560*2048  # lm_head fp8 bytes
B=torch.zeros(LM,device=dev,dtype=torch.uint8)
sink=torch.zeros(16384,device=dev,dtype=torch.float32)
C=torch.zeros(64,2048,device=dev,dtype=torch.float32)
def bench(nbytes,nblk,it=10):
    def go(): lib.launch_floor(B.data_ptr(),C.data_ptr(),sink.data_ptr(),1,nbytes,1,256,nblk,st)
    for _ in range(5): go()
    torch.cuda.synchronize()
    v=[]
    for _ in range(5):
        a,b=torch.cuda.Event(True),torch.cuda.Event(True); a.record()
        for _ in range(it): go()
        b.record(); torch.cuda.synchronize(); v.append(a.elapsed_time(b)/it*1000.0)
    return statistics.median(v)
mb=LM/1e6
print("lm_head weight stream: %.1f MB"%(mb))
print("%8s %10s %10s"%("blk","us","GB/s"))
best=(0,0,0)
for nblk in (104,208,416,510,832,1024,1664,2048,3328):
    t=bench(LM,nblk); g=mb/t*1000
    print("%8d %10.2f %10.0f"%(nblk,t,g))
    if g>best[0]: best=(g,nblk,t)
print("\nbest contiguous: %.0f GB/s at %d blk (%.1f us)"%best)
print("current fp8 lm_head staging-only: 444 us = 602 GB/s -> %.2fx headroom"%(best[0]/602))
print("current fp8 lm_head full:         523 us = 511 GB/s -> %.2fx headroom"%(best[0]/511))
