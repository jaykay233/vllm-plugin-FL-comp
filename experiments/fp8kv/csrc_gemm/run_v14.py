import ctypes, os, sys, torch, torch.nn.functional as F
HERE=os.path.dirname(os.path.abspath(__file__))
lib=ctypes.CDLL(os.path.join(HERE,"libgemm_v14.so"))
lib.launch_gemm_v14.argtypes=[ctypes.c_void_p]*3+[ctypes.c_int]*4+[ctypes.c_void_p]
lib.launch_gemm_v14.restype=ctypes.c_int
FP8=torch.float8_e4m3fn
dev="cuda"; torch.manual_seed(0); st=torch.cuda.current_stream().cuda_stream; M=64
TC={0:("WPB4 NT4 KC32 bufsz",4,4,32,2),1:("WPB4 NT2 KC32",4,2,32,2),
    2:("WPB2 NT4 KC32",2,4,32,2),3:("WPB4 NT2 KC64",4,2,64,2),
    4:("WPB2 NT2 KC64",2,2,64,2),5:("WPB2 NT2 KC32",2,2,32,2),
    6:("WPB2 NT4 KC16",2,4,16,2),7:("WPB2 NT1 KC64",2,1,64,2)}

def smem(w,n,kc,stg):
    pad = 32 if kc>=32 else 16
    rs=kc*2+pad; br=w*n*16
    return stg*(64*rs+br*rs)+0
def quant(t,chunk=8192):
    s=(t.abs().amax().clamp(min=1e-12)/448.0).float()
    o=torch.empty(t.shape,dtype=FP8,device=t.device)
    for r0 in range(0,t.shape[0],chunk): o[r0:r0+chunk]=(t[r0:r0+chunk].float()/s).to(FP8)
    return o,s
def bench(fn,iters=30,warmup=5):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    a,b=torch.cuda.Event(True),torch.cuda.Event(True); a.record()
    for _ in range(iters): fn()
    b.record(); torch.cuda.synchronize(); return a.elapsed_time(b)/iters
mode=sys.argv[1]
if mode=="check":
    N,K=2048,2048
    xb=torch.randn(M,K,device=dev,dtype=torch.bfloat16)
    wb=torch.randn(N,K,device=dev,dtype=torch.bfloat16)*0.02
    wq,ws=quant(wb); c=torch.zeros(M,N,device=dev,dtype=torch.float32)
    ref=(xb.float()@wq.float().t())*ws; rm=ref.abs().max().item()
    for tc in sorted(TC):
        nm=TC[tc][0]; cfg=tc*100+1
        if smem(*TC[tc][1:])>65000: print("CHK %-22s SMEM %d skip"%(nm,smem(*TC[tc][1:]))); continue
        c.zero_()
        rc=lib.launch_gemm_v14(xb.data_ptr(),wq.data_ptr(),c.data_ptr(),M,N,K,cfg,st)
        if rc!=0: print("CHK %-22s rc=%d"%(nm,rc)); continue
        torch.cuda.synchronize()
        rel=((c*ws)-ref).abs().max().item()/rm
        print("CHK %-22s smem=%5d rel=%.2e %s"%(nm,smem(*TC[tc][1:]),rel,"OK" if rel<1e-3 else "BAD"))
else:
    N,K=int(sys.argv[2]),int(sys.argv[3])
    xb=torch.randn(M,K,device=dev,dtype=torch.bfloat16)
    wb=torch.randn(N,K,device=dev,dtype=torch.bfloat16)*0.02
    wq,_=quant(wb); c=torch.zeros(M,N,device=dev,dtype=torch.float32)
    tbf=bench(lambda: F.linear(xb,wb))
    print("REF N=%d K=%d bf16 %.5f ms %.0f GB/s"%(N,K,tbf,N*K*2/(tbf*1e6)))
    res=[]
    for tc in sorted(TC):
        nm=TC[tc][0]
        if smem(*TC[tc][1:])>65000: continue
        for spl in (1,4):
            cfg=tc*100+spl
            if lib.launch_gemm_v14(xb.data_ptr(),wq.data_ptr(),c.data_ptr(),M,N,K,cfg,st)!=0: continue
            def go(cf=cfg):
                c.zero_(); lib.launch_gemm_v14(xb.data_ptr(),wq.data_ptr(),c.data_ptr(),M,N,K,cf,st)
            t=bench(go)
            print("  %-22s s%d %8.5f ms %6.0f GB/s %.2fx"%(nm,spl,t,N*K/(t*1e6),tbf/t))
            res.append((t,"%s s%d"%(nm,spl)))
    res.sort(); print("BEST %-28s %.5f ms %.2fx"%(res[0][1],res[0][0],tbf/res[0][0]))
