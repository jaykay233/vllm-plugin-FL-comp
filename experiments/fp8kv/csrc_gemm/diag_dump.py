import ctypes, os, subprocess, torch
HERE=os.path.dirname(os.path.abspath(__file__))
LIB=os.path.join(HERE,"libdump.so")
dev="cuda"; st=torch.cuda.current_stream().cuda_stream
lib=ctypes.CDLL(LIB)
lib.launch_dump.argtypes=[ctypes.c_void_p]*4+[ctypes.c_int]*2+[ctypes.c_void_p]
K=64; N=2048; RSB=64*2+32; RSF=64+16; BROWS=128
Abit=(torch.arange(64*K,device=dev,dtype=torch.int64)%2048).to(torch.int16)
A=Abit.view(torch.bfloat16).reshape(64,K).contiguous()
B=(torch.arange(N*K,device=dev,dtype=torch.int64)%251).to(torch.uint8).reshape(N,K).contiguous()
dA=torch.zeros(64*RSB,device=dev,dtype=torch.uint8); dB=torch.zeros(BROWS*RSF,device=dev,dtype=torch.uint8)
lib.launch_dump(A.data_ptr(),B.data_ptr(),dA.data_ptr(),dB.data_ptr(),N,K,st); torch.cuda.synchronize()
dA=dA.reshape(64,RSB); dB=dB.reshape(BROWS,RSF)
ab=A.view(torch.uint8)
print("A: per-row mismatches (rows 0..7):",[(dA[r,:K*2]!=ab[r]).sum().item() for r in range(8)])
print("A: bad rows count:",int(((dA[:,:K*2]!=ab).sum(1)>0).sum().item()),"of 64")
# where does dA row 1 actually come from?
row1=dA[1,:K*2]
for cand in range(64):
    if torch.equal(row1,ab[cand]): print("  dA[1] == A row",cand); break
else:
    print("  dA[1] matches no A row; dA[1][:24]=",row1[:24].tolist()," exp A[1][:24]=",ab[1,:24].tolist())
# column pattern of the first bad row
bad=torch.nonzero(dA[1,:K*2]!=ab[1]).flatten()
print("  dA[1] bad byte idx (first 20):",bad[:20].tolist())
print("B: per-row mismatches (rows 0..7):",[(dB[r,:K]!=B[r]).sum().item() for r in range(8)])
print("B: bad rows count:",int(((dB[:,:K]!=B[:BROWS]).sum(1)>0).sum().item()),"of",BROWS)
row1=dB[1,:K]
for cand in range(BROWS):
    if torch.equal(row1,B[cand]): print("  dB[1] == B row",cand); break
else: print("  dB[1] matches no B row; dB[1][:24]=",row1[:24].tolist()," exp B[1][:24]=",B[1,:24].tolist())
