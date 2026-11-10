import ctypes, os, subprocess, torch
HERE=os.path.dirname(os.path.abspath(__file__))
LIB=os.path.join(HERE,"libcpasync.so")
r=subprocess.run(["/opt/maca/mxgpu_llvm/bin/mxcc","-shared","-fPIC","-O3","--offload-arch=xcore1000","-I/opt/maca/include",os.path.join(HERE,"cpasync_test.cu"),"-o",LIB],capture_output=True,text=True)
ok=r.returncode==0 and "error:" not in r.stderr
print("BUILD","OK" if ok else "FAIL")
if not ok: print(r.stderr[-1800:]); raise SystemExit(1)
lib=ctypes.CDLL(LIB)
lib.launch_strided.argtypes=[ctypes.c_void_p,ctypes.c_void_p]+[ctypes.c_int]*2+[ctypes.c_void_p]
dev="cuda"; st=torch.cuda.current_stream().cuda_stream
NR=64
src=(torch.arange(NR*64,device=dev,dtype=torch.int64)%251).to(torch.uint8).contiguous()
for v,wn in {0:"rowmajor (no div/mod)"}.items():
    for d,dst in ((100,64),(101,80),(103,128),(102,160)):
        out=torch.zeros(NR*64,device=dev,dtype=torch.uint8)
        rc=lib.launch_strided(src.data_ptr(),out.data_ptr(),NR,d,st)
        if rc!=0: print("  cfg unsupported",v*10+d); continue
        torch.cuda.synchronize()
        bad=int((out!=src).sum().item())
        print("  %-16s dst=%3d  mismatches=%4d/%d %s"%(wn,dst,bad,NR*64,"OK" if bad==0 else ""))
