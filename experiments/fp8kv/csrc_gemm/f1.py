import ctypes, os, torch, sys
H=os.path.dirname(os.path.abspath(__file__))
lib=ctypes.CDLL(os.path.join(H,"libfloor.so"))
lib.launch_floor.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p]+[ctypes.c_int]*5+[ctypes.c_void_p]
lib.launch_floor.restype=ctypes.c_int
st=torch.cuda.current_stream().cuda_stream
N,K=2048,2048
B=torch.zeros(N,K,device="cuda",dtype=torch.uint8)
C=torch.zeros(64,N,device="cuda",dtype=torch.float32)
sink=torch.zeros(4096,device="cuda",dtype=torch.float32)
print("alloc ok", flush=True)
rc=lib.launch_floor(B.data_ptr(),C.data_ptr(),sink.data_ptr(),N,K,0,256,128,st)
print("empty rc",rc, flush=True); torch.cuda.synchronize(); print("sync ok", flush=True)
rc=lib.launch_floor(B.data_ptr(),C.data_ptr(),sink.data_ptr(),N,K,1,256,128,st)
print("flat rc",rc, flush=True); torch.cuda.synchronize(); print("flat done", flush=True)
rc=lib.launch_floor(B.data_ptr(),C.data_ptr(),sink.data_ptr(),N,K,2,256,128,st)
print("tiled rc",rc, flush=True); torch.cuda.synchronize(); print("tiled done", flush=True)
rc=lib.launch_floor(B.data_ptr(),C.data_ptr(),sink.data_ptr(),N,K,3,256,128,st)
print("tiled_w rc",rc, flush=True); torch.cuda.synchronize(); print("tiled_w done", flush=True)
