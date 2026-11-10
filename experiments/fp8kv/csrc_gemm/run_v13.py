import ctypes, os, sys, torch, torch.nn.functional as F
HERE=os.path.dirname(os.path.abspath(__file__))
lib=ctypes.CDLL(os.path.join(HERE,"libgemm_v13.so"))
lib.launch_gemm_v13.argtypes=[ctypes.c_void_p]*3+[ctypes.c_int]*4+[ctypes.c_void_p]
lib.launch_gemm_v13.restype=ctypes.c_int
FP8=torch.float8_e4m3fn
dev="cuda"; torch.manual_seed(0); st=torch.cuda.current_stream().cuda_stream; M=64
TC={20:("WPB4 NT4 KC64 PADA16",4,4,64,2,"p16"),21:("WPB4 NT4 KC64 PADA32",4,4,64,2,"p32"),
    22:("WPB4 NT4 KC64 PADA48",4,4,64,2,"p48"),
    23:("WPB2 NT2 KC64 PADA16",2,2,64,2,"p16"),24:("WPB2 NT2 KC64 PADA32",2,2,64,2,"p32"),
    25:("WPB2 NT2 KC64 PADA48",2,2,64,2,"p48"),
    26:("WPB4 NT4 KC32 PADA16",4,4,32,2,"p16"),27:("WPB4 NT4 KC32 PADA48",4,4,32,2,"p48")}

def smem(w,n,kc,stg,tag=""):
    pada=16 if tag in ("p16","") else (32 if tag=="p32" else 48)
    rsb=kc*2+pada; rsf=kc+16; br=w*n*16
    return stg*(64*rsb+br*rsf)+0
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
        if smem(*TC[tc][1:5],TC[tc][5])>65000: print("CHK %-22s SMEM %d skip"%(nm,smem(*TC[tc][1:5],TC[tc][5]))); continue
        c.zero_()
        rc=lib.launch_gemm_v13(xb.data_ptr(),wq.data_ptr(),c.data_ptr(),M,N,K,cfg,st)
        if rc!=0: print("CHK %-22s rc=%d"%(nm,rc)); continue
        torch.cuda.synchronize()
        rel=((c*ws)-ref).abs().max().item()/rm
        print("CHK %-22s smem=%5d rel=%.2e %s"%(nm,smem(*TC[tc][1:5],TC[tc][5]),rel,"OK" if rel<1e-3 else "BAD"))
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
        if smem(*TC[tc][1:5],TC[tc][5])>65000: continue
        for spl in (1,4):
            cfg=tc*100+spl
            if lib.launch_gemm_v13(xb.data_ptr(),wq.data_ptr(),c.data_ptr(),M,N,K,cfg,st)!=0: continue
            def go(cf=cfg):
                c.zero_(); lib.launch_gemm_v13(xb.data_ptr(),wq.data_ptr(),c.data_ptr(),M,N,K,cf,st)
            t=bench(go)
            print("  %-22s s%d %8.5f ms %6.0f GB/s %.2fx"%(nm,spl,t,N*K/(t*1e6),tbf/t))
            res.append((t,"%s s%d"%(nm,spl)))
    res.sort(); print("BEST %-28s %.5f ms %.2fx"%(res[0][1],res[0][0],tbf/res[0][0]))
