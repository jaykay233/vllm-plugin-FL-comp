import ctypes, os, statistics, torch
H=os.path.dirname(os.path.abspath(__file__))
lib=ctypes.CDLL(os.path.join(H,"libfloor.so"))
lib.launch_floor.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p]+[ctypes.c_int]*5+[ctypes.c_void_p]
lib.launch_floor.restype=ctypes.c_int
dev="cuda"; st=torch.cuda.current_stream().cuda_stream
POOL=64*1024*1024
B=torch.zeros(POOL,device=dev,dtype=torch.uint8)
sink=torch.zeros(8192,device=dev,dtype=torch.float32)
C=torch.zeros(64,2048,device=dev,dtype=torch.float32)
def bench(nbytes,nblk,it=30):
    def go(): lib.launch_floor(B.data_ptr(),C.data_ptr(),sink.data_ptr(),1,nbytes,1,256,nblk,st)
    for _ in range(10): go()
    torch.cuda.synchronize()
    v=[]
    for _ in range(5):
        a,b=torch.cuda.Event(True),torch.cuda.Event(True); a.record()
        for _ in range(it): go()
        b.record(); torch.cuda.synchronize(); v.append(a.elapsed_time(b)/it*1000.0)
    return statistics.median(v)
MB=1024*1024
print("%9s %6s %9s %9s"%("size","blk","us","GB/s"))
best={}
for mb in (1,2,4,8,16,32,64):
    nb=mb*MB
    for nblk in (104,256,512,1024,2048,4096):
        if nb//16//(nblk*256) < 1: continue
        t=bench(nb,nblk)
        gbs=mb*1.0/t*1000
        print("%7d MB %6d %9.2f %9.0f"%(mb,nblk,t,gbs))
        if mb not in best or gbs>best[mb][0]: best[mb]=(gbs,nblk,t)
print()
print("%9s %10s"%("size","best GB/s"))
for mb in sorted(best): print("%7d MB %10.0f  (blk=%d, %.2f us)"%(mb,best[mb][0],best[mb][1],best[mb][2]))
