#!/usr/bin/env python3
"""M=64 projection GEMM device_time: native vs FlagGems mm."""
from __future__ import annotations
import json, os, importlib
from collections import defaultdict
from pathlib import Path
OUT = Path(__file__).resolve().parent
SHAPES = [("qkv",2560,2048),("o_proj",2048,2048),("gate_up",12288,2048),("down",2048,6144),("lm_head",130560,2048)]
M = 64

def device_us(fn, iters=50):
    import torch
    from torch.profiler import ProfilerActivity, profile
    for _ in range(8): fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(iters): fn()
        torch.cuda.synchronize()
    by=defaultdict(float)
    for e in p.key_averages():
        t=e.self_device_time_total or 0
        if t>0: by[e.key]+=t/iters
    tot=sum(by.values())
    top=sorted(by.items(), key=lambda kv:-kv[1])[:4]
    return tot, top

def main():
    import torch
    rows=[]
    print("=== native F.linear M=64 ===", flush=True)
    for name,N,K in SHAPES:
        x=torch.randn(M,K,device='cuda',dtype=torch.bfloat16)
        w=torch.randn(N,K,device='cuda',dtype=torch.bfloat16)
        us, top = device_us(lambda: torch.nn.functional.linear(x,w))
        print(f"  {name:8s} {us:8.1f}us  top={top[0][0][:50]}", flush=True)
        rows.append({"arm":"native","name":name,"us":us,"top":[(k,v) for k,v in top]})

    print("=== FlagGems mm M=64 ===", flush=True)
    # import flag_gems mm without enabling globally if possible
    import flag_gems
    # use ops.mm
    try:
        from flag_gems.ops import mm as gems_mm
    except Exception:
        gems_mm = None
        import flag_gems.ops as ops
        gems_mm = getattr(ops, 'mm', None)
    if gems_mm is None:
        # fallback enable
        flag_gems.enable()
        gems_mm = torch.mm
    for name,N,K in SHAPES:
        a=torch.randn(M,K,device='cuda',dtype=torch.bfloat16)
        b=torch.randn(K,N,device='cuda',dtype=torch.bfloat16)  # mm wants (K,N) for A@B -> (M,N); linear uses W.T
        # warm autotune
        for _ in range(3):
            gems_mm(a,b)
        torch.cuda.synchronize()
        us, top = device_us(lambda: gems_mm(a,b))
        print(f"  {name:8s} {us:8.1f}us  top={top[0][0][:50]}", flush=True)
        rows.append({"arm":"gems","name":name,"us":us,"top":[(k,v) for k,v in top]})
    (OUT/"gemm_m64.json").write_text(json.dumps(rows, indent=2, default=str))

if __name__ == "__main__":
    main()
