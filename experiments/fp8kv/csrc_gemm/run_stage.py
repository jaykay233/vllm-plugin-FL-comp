import ctypes, os, subprocess, torch
HERE=os.path.dirname(os.path.abspath(__file__))
SRC=os.path.join(HERE,"bw_stage.cu"); LIB=os.path.join(HERE,"libstage.so")
r=subprocess.run(["/opt/maca/mxgpu_llvm/bin/mxcc","-shared","-fPIC","-O3","--offload-arch=xcore1000","-I/opt/maca/include",SRC,"-o",LIB],capture_output=True,text=True)
ok = r.returncode==0 and "error:" not in r.stderr
print("BUILD","OK" if ok else "FAIL")
if not ok: print(r.stderr[-2500:]); raise SystemExit(1)
lib=ctypes.CDLL(LIB)
lib.launch_stage.argtypes=[ctypes.c_void_p]*2+[ctypes.c_void_p]+[ctypes.c_int]*3+[ctypes.c_void_p]
dev="cuda"; st=torch.cuda.current_stream().cuda_stream
CFG={0:("fp8  plain 16B",1),1:("fp8  cp.async16B",1),2:("bf16 plain 16B",2),3:("bf16 cp.async16B",2),
     4:("fp8  async KC64",1),5:("fp8  async WPB8",1),6:("fp8  async NT4",1),7:("fp8  async WPB2",1)}
def bench(fn,iters=30,warmup=5):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    a,b=torch.cuda.Event(True),torch.cuda.Event(True); a.record()
    for _ in range(iters): fn()
    b.record(); torch.cuda.synchronize(); return a.elapsed_time(b)/iters
for N,K in ((12288,2048),(130560,2048)):
    a8=torch.randint(0,100,(64,K),device=dev,dtype=torch.uint8).contiguous()
    b8=torch.randint(0,100,(N,K),device=dev,dtype=torch.uint8).contiguous()
    out=torch.zeros(2048,device=dev,dtype=torch.float32)
    print("\nN=%d K=%d  fp8 W=%.1f MB  bf16 W=%.1f MB"%(N,K,N*K/1e6,N*K*2/1e6))
    for cfg,(name,esc) in CFG.items():
        t=bench(lambda c=cfg: lib.launch_stage(a8.data_ptr(),b8.data_ptr(),out.data_ptr(),N,K,c,st))
        # bytes actually read: A re-read per tile + B
        per={0:4*2*16,1:4*2*16,2:4*2*16,3:4*2*16,4:4*2*16,5:8*16,6:4*4*16,7:2*2*16}[cfg]
        tiles=(N+per-1)//per
        arows=per
        byt=(64*K*esc + arows*K*esc)*tiles
        print("  %-16s %8.5f ms   %7.0f GB/s"%(name,t,byt/(t*1e6)))
    del a8,b8,out; torch.cuda.empty_cache()
