import ctypes, os, subprocess, torch
HERE=os.path.dirname(os.path.abspath(__file__))
LIB=os.path.join(HERE,"libperm.so")
r=subprocess.run(["/opt/maca/mxgpu_llvm/bin/mxcc","-shared","-fPIC","-O3","--offload-arch=xcore1000","-I/opt/maca/include",os.path.join(HERE,"perm_test.cu"),"-o",LIB],capture_output=True,text=True)
print("BUILD","OK" if "error:" not in r.stderr else "FAIL")
lib=ctypes.CDLL(LIB)
lib.launch_perm_probe.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p]
dev="cuda"; st=torch.cuda.current_stream().cuda_stream
bq=torch.tensor([0x04030201],dtype=torch.int64).to(torch.uint32).to(dev)
out=torch.zeros(4,dtype=torch.int64).to(torch.uint32).to(dev)
lib.launch_perm_probe(bq.data_ptr(),out.data_ptr(),st); torch.cuda.synchronize()
names=["0x4140","0x0414","0x0011","0x1100"]
for n,v in zip(names,out.tolist()):
    bs=[(v>>(8*i))&0xFF for i in range(4)]
    print("  sel=%s -> bytes %s"%(n,["%02x"%b for b in bs]))
print("  input bytes b0=01 b1=02 b2=03 b3=04")
