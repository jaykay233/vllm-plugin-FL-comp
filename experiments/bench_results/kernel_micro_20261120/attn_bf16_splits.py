import torch, math
from vllm_fl.dispatch.backends.vendor.metax.impl.attention.utils.fa_utils import flash_attn_with_kvcache
dev="cuda"; torch.manual_seed(0)
HQ,HKV,D,BS = 16,2,128,16
scale=1.0/math.sqrt(D)
def run(B,S,splits,iters=30,warmup=8):
    nb=(S+BS-1)//BS
    # DISTINCT blocks per request: layout [B, nb, BS, HKV, D]
    kc=torch.randn(B*nb,BS,HKV,D,device=dev,dtype=torch.bfloat16)
    vc=torch.randn(B*nb,BS,HKV,D,device=dev,dtype=torch.bfloat16)
    bt=(torch.arange(B,device=dev,dtype=torch.int32).unsqueeze(1)*nb
        + torch.arange(nb,device=dev,dtype=torch.int32).unsqueeze(0)).contiguous()
    sl=torch.full((B,),S,device=dev,dtype=torch.int32)
    q=torch.randn(B,1,HQ,D,device=dev,dtype=torch.bfloat16)
    def f():
        return flash_attn_with_kvcache(q=q,k_cache=kc,v_cache=vc,block_table=bt,
            cache_seqlens=sl,softmax_scale=scale,causal=True,num_splits=splits)
    for _ in range(warmup): f()
    torch.cuda.synchronize()
    s=torch.cuda.Event(True); e=torch.cuda.Event(True); s.record()
    for _ in range(iters): f()
    e.record(); torch.cuda.synchronize()
    ms=s.elapsed_time(e)/iters
    gb=2*B*S*HKV*D*2/1e9
    return ms, gb/ms*1000
for B,S in [(64,4096),(64,4608)]:
    print("=== batch=%d seq=%d  KV=%d MB  (42 layers per step) ==="%(B,S,2*B*S*HKV*D*2/1e6))
    for sp in [0,1,2,4,8,16,32]:
        try:
            ms,gbs=run(B,S,sp)
            print("  splits=%2d: %7.1f us  %7.1f GB/s  x42=%6.2f ms/step"%(sp,ms*1000,gbs,ms*42))
        except Exception as ex:
            print("  splits=%2d: FAIL %s"%(sp,str(ex)[:100]))
