import ctypes, os, statistics, torch
H=os.path.dirname(os.path.abspath(__file__))
lib=ctypes.CDLL(os.path.join(H,"libfloor.so"))
lib.launch_floor.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p]+[ctypes.c_int]*5+[ctypes.c_void_p]
lib.launch_floor.restype=ctypes.c_int
dev="cuda"; torch.manual_seed(0); st=torch.cuda.current_stream().cuda_stream
N,K=2048,2048
B=torch.randint(0,255,(N,K),device=dev,dtype=torch.uint8)
C=torch.zeros(64,N,device=dev,dtype=torch.float32)
sink=torch.zeros(4096,device=dev,dtype=torch.float32)
MB=N*K/1e6
def bench(mode,nthreads=256,nblk=512,it=50):
    def go(): lib.launch_floor(B.data_ptr(),C.data_ptr(),sink.data_ptr(),N,K,mode,nthreads,nblk,st)
    for _ in range(15): go()
    torch.cuda.synchronize()
    v=[]
    for _ in range(9):
        a,b=torch.cuda.Event(True),torch.cuda.Event(True); a.record()
        for _ in range(it): go()
        b.record(); torch.cuda.synchronize(); v.append(a.elapsed_time(b)/it*1000.0)  # us
    return statistics.median(v)
print("o_proj shape: read %.2f MB weights, write %.2f MB fp32 output"%(MB,64*N*4/1e6))
print("DRAM peak at 1416 GB/s -> %.2f us for the reads alone\n"%(MB/1416*1000))
rows=[]
t=bench(0,256,128); print("empty (128 blk)      %8.2f us"%t); rows.append(("empty",t))
for nb in (128,512,1024):
    t=bench(1,256,nb); print("flat  nb=%-5d        %8.2f us  %6.0f GB/s"%(nb,t,MB/t*1000)); rows.append(("flat%d"%nb,t))
t=bench(2); print("tiled (GEMM pattern) %8.2f us  %6.0f GB/s"%(t,MB/t*1000)); rows.append(("tiled",t))
t=bench(3); print("tiled + output write %8.2f us  %6.0f GB/s"%(t,(MB+64*N*4/1e6)/t*1000)); rows.append(("tiled_w",t))
print("\nreference: gemm_fp8_v16 o_proj = 38.3 us (105 GB/s)")
