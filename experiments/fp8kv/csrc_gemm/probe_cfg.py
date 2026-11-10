import ctypes, os, torch
HERE=os.path.dirname(os.path.abspath(__file__))
lib=ctypes.CDLL(os.path.join(HERE,"libgemm_v10.so"))
lib.launch_gemm_v9.argtypes=[ctypes.c_void_p]*3+[ctypes.c_int]*4+[ctypes.c_void_p]
dev="cuda"; st=torch.cuda.current_stream().cuda_stream; M=64
K=2048; N=2048
x=torch.zeros(M,K,device=dev,dtype=torch.bfloat16)
w=torch.zeros(N,K,device=dev,dtype=torch.uint8)
c=torch.zeros(M,N,device=dev,dtype=torch.float32)
for tc in range(10):
    cfg=(10+tc)*100+1
    rc=lib.launch_gemm_v9(x.data_ptr(),w.data_ptr(),c.data_ptr(),M,N,K,cfg,st)
    print("tc=%d cfg=%d rc=%d"%(tc,cfg,rc))
