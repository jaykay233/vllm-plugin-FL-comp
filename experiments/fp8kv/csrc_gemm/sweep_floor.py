import ctypes, os, statistics, torch
H=os.path.dirname(os.path.abspath(__file__))
lib=ctypes.CDLL(os.path.join(H,"libfloor.so"))
lib.launch_floor.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p]+[ctypes.c_int]*5+[ctypes.c_void_p]
lib.launch_floor.restype=ctypes.c_int
dev="cuda"; st=torch.cuda.current_stream().cuda_stream
BIG=torch.zeros(64*1024*1024,device=dev,dtype=torch.uint8)  # 64 MB pool
sink=torch.zeros(4096,device=dev,dtype=torch.float32)
C=torch.zeros(64,2048,device=dev,dtype=torch.float32)
def bench(nbytes,mode=1,nblk=512,it=50):
    def go(): lib.launch_floor(BIG.data_ptr(),C.data_ptr(),sink.data_ptr(),2048,2048,mode,256,nblk,st)
    # for flat mode, nbytes is controlled by N*K passed; use ksweep variant instead
    for _ in range(15): go()
    torch.cuda.synchronize()
    v=[]
    for _ in range(7):
        a,b=torch.cuda.Event(True),torch.cuda.Event(True); a.record()
        for _ in range(it): go()
        b.record(); torch.cuda.synchronize(); v.append(a.elapsed_time(b)/it*1000.0)
    return statistics.median(v)
# empty-kernel cost vs grid size
print("%-22s %9s"%("grid","us"))
for nb in (26,52,104,128,256,512,1024,2048):
    print("%-22s %9.2f"%("empty %d blk"%nb, bench(0,0,nb)))
