import ctypes, os, subprocess, torch
HERE=os.path.dirname(os.path.abspath(__file__))
SRC=os.path.join(HERE,"cpasync_test.cu"); LIB=os.path.join(HERE,"libcpasync.so")
r=subprocess.run(["/opt/maca/mxgpu_llvm/bin/mxcc","-shared","-fPIC","-O3","--offload-arch=xcore1000","-I/opt/maca/include",SRC,"-o",LIB],capture_output=True,text=True)
ok=r.returncode==0 and "error:" not in r.stderr
print("BUILD","OK" if ok else "FAIL")
if not ok: print(r.stderr[-2500:]); raise SystemExit(1)
lib=ctypes.CDLL(LIB)
lib.launch_cpasync_test.argtypes=[ctypes.c_void_p,ctypes.c_void_p]+[ctypes.c_int]*2+[ctypes.c_void_p]
dev="cuda"; st=torch.cuda.current_stream().cuda_stream
NCH=1024  # 16 KB
src=torch.arange(NCH*16,device=dev,dtype=torch.int64).to(torch.uint8).contiguous()%251
NAMES={0:"cp.async + syncthreads",1:"cp.async + arrive+barrier",2:"cp.async + arrive+sync+barrier",3:"cp.async + sync+arrive+barrier",4:"plain memcpy (control)"}
for v in (4,0,1,2,3):
    out=torch.zeros(NCH*16,device=dev,dtype=torch.uint8)
    lib.launch_cpasync_test(src.data_ptr(),out.data_ptr(),NCH,v,st)
    torch.cuda.synchronize()
    nz=int((out==0).sum().item())
    bad=int((out!=src).sum().item())
    print("V%d %-32s mismatches=%4d zeros=%4d"%(v,NAMES[v],bad,nz))
    if bad and bad<200: print("   first bad idx:", torch.nonzero(out!=src).flatten()[:8].tolist())
