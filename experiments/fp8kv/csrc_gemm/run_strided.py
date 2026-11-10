import ctypes, os, subprocess, torch
HERE=os.path.dirname(os.path.abspath(__file__))
SRC=os.path.join(HERE,"cpasync_test.cu"); LIB=os.path.join(HERE,"libcpasync.so")
r=subprocess.run(["/opt/maca/mxgpu_llvm/bin/mxcc","-shared","-fPIC","-O3","--offload-arch=xcore1000","-I/opt/maca/include",SRC,"-o",LIB],capture_output=True,text=True)
ok=r.returncode==0 and "error:" not in r.stderr
print("BUILD","OK" if ok else "FAIL")
if not ok: print(r.stderr[-2000:]); raise SystemExit(1)
lib=ctypes.CDLL(LIB)
lib.launch_strided.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_void_p]
dev="cuda"; st=torch.cuda.current_stream().cuda_stream
NROWS=64
src=(torch.arange(NROWS*64,device=dev,dtype=torch.int64)%251).to(torch.uint8).contiguous()
print("dst stride -> mismatches (0 of %d = works)"%(NROWS*64))
for cfg,dst in ((0,64),(1,80),(3,96),(2,128),(4,160),(5,192)):
    out=torch.zeros(NROWS*64,device=dev,dtype=torch.uint8)
    lib.launch_strided(src.data_ptr(),out.data_ptr(),NROWS,cfg,st); torch.cuda.synchronize()
    bad=int((out!=src).sum().item())
    badrows=int(((out.reshape(NROWS,64)!=src.reshape(NROWS,64)).sum(1)>0).sum().item())
    print("  dst=%3d  mismatches=%4d  badrows=%2d/64  %s"%(dst,bad,badrows,"OK" if bad==0 else ""))
