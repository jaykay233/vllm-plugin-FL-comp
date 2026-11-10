import ctypes, os, subprocess, torch
HERE=os.path.dirname(os.path.abspath(__file__))
LIB=os.path.join(HERE,"libdump.so")
r=subprocess.run(["/opt/maca/mxgpu_llvm/bin/mxcc","-shared","-fPIC","-O3","--offload-arch=xcore1000","-I/opt/maca/include",os.path.join(HERE,"dump_tile.cu"),"-o",LIB],capture_output=True,text=True)
ok=r.returncode==0 and "error:" not in r.stderr
print("BUILD","OK" if ok else "FAIL")
if not ok: print(r.stderr[-2000:]); raise SystemExit(1)
lib=ctypes.CDLL(LIB)
lib.launch_dump.argtypes=[ctypes.c_void_p]*4+[ctypes.c_int]*2+[ctypes.c_void_p]
dev="cuda"; st=torch.cuda.current_stream().cuda_stream
K=64; N=2048; RSB=64*2+32; RSF=64+16; BROWS=128
A=torch.randn(64,K,device=dev,dtype=torch.bfloat16)
B=(torch.randperm(N*K,device=dev,dtype=torch.int64)%200+1).to(torch.uint8).reshape(N,K).contiguous()
dA=torch.zeros(64*RSB,device=dev,dtype=torch.uint8)
dB=torch.zeros(BROWS*RSF,device=dev,dtype=torch.uint8)
lib.launch_dump(A.data_ptr(),B.data_ptr(),dA.data_ptr(),dB.data_ptr(),N,K,st)
torch.cuda.synchronize()
dA=dA.reshape(64,RSB); dB=dB.reshape(BROWS,RSF)
ab=A.view(torch.uint8)  # [64, K*2]
badA=(dA[:,:K*2]!=ab).sum().item()
badB=(dB[:,:K]!=B[:BROWS]).sum().item()
print("A tile: mismatched bytes = %d / %d"%(badA,64*K*2))
print("B tile: mismatched bytes = %d / %d"%(badB,BROWS*K))
if badA:
    print("  first bad A row:", torch.nonzero(dA[:,:K*2]!=ab)[0].tolist(), 
          " dA row0[:16] =",dA[0,:16].tolist()," exp =",ab[0,:16].tolist())
if badB:
    rr=torch.nonzero(dB[:,:K]!=B[:BROWS])[0]
    print("  first bad B:",rr.tolist()," dB =",dB[rr[0],:16].tolist()," exp =",B[rr[0],:16].tolist())
