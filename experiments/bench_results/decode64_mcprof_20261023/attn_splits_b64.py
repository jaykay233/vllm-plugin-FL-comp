#!/usr/bin/env python3
"""mcProfiler decode attn num_splits @ batch=64 (device_time)."""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path

OUT = Path(__file__).resolve().parent
NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, N_LAYERS = 16, 2, 128, 42
BATCH = 64
BLOCK = 16
SEQLENS = [512, 2048, 4096, 5120]
SPLITS = [0, 1, 2, 4, 8, 16]

def build(seqlen, batch, block, dev, dtype):
    import torch
    max_blocks = (seqlen + block - 1) // block
    nblocks = max_blocks * batch
    q = torch.randn(batch, 1, NUM_HEADS, HEAD_DIM, dtype=dtype, device=dev) * 0.1
    kc = torch.randn(nblocks, block, NUM_KV_HEADS, HEAD_DIM, dtype=dtype, device=dev) * 0.1
    vc = torch.randn_like(kc)
    bt = torch.arange(nblocks, dtype=torch.int32, device=dev).view(batch, max_blocks)
    cs = torch.full((batch,), seqlen, dtype=torch.int32, device=dev)
    return q, kc, vc, bt, cs

def gpu_breakdown(f, warmup=20, iters=40):
    import torch
    from torch.profiler import ProfilerActivity, profile
    for _ in range(warmup):
        f()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(iters):
            f()
        torch.cuda.synchronize()
    by = defaultdict(float)
    for e in p.key_averages():
        t = e.self_device_time_total or 0.0
        if t > 0:
            by[e.key] += t / iters
    tot = sum(by.values())
    top = sorted(({"k": k, "us": v} for k, v in by.items()), key=lambda r: -r["us"])[:6]
    return tot, top

def main():
    import torch
    import vllm  # noqa: F401
    from flash_attn import flash_attn_with_kvcache as fa

    dev = torch.device("cuda")
    dtype = torch.bfloat16
    blk = None
    for cand in (16, 32, 64, 128):
        try:
            q, kc, vc, bt, cs = build(512, 1, cand, dev, dtype)
            fa(q=q, k_cache=kc, v_cache=vc, block_table=bt, cache_seqlens=cs,
               softmax_scale=HEAD_DIM ** -0.5, causal=True, num_splits=1)
            torch.cuda.synchronize()
            blk = cand
            break
        except Exception as e:
            print(f"block {cand}: {type(e).__name__}: {e}", flush=True)
    if blk is None:
        raise SystemExit("no block size")
    print(f"using block_size={blk}", flush=True)

    rows = []
    print(f"{'seq':>5} {'splits':>6} {'us/call':>10} {'us/step42':>10}  top", flush=True)
    for seq in SEQLENS:
        q, kc, vc, bt, cs = build(seq, BATCH, blk, dev, dtype)
        for ns in SPLITS:
            def f(ns=ns, q=q, kc=kc, vc=vc, bt=bt, cs=cs):
                return fa(q=q, k_cache=kc, v_cache=vc, block_table=bt, cache_seqlens=cs,
                          softmax_scale=HEAD_DIM ** -0.5, causal=True, num_splits=ns)
            tot, top = gpu_breakdown(f)
            step = tot * N_LAYERS
            rows.append({"seqlen": seq, "num_splits": ns, "us_call": tot, "us_step42": step, "top": top})
            tops = ", ".join(f"{t['k'][:36]}={t['us']:.1f}" for t in top[:3])
            print(f"{seq:5d} {ns:6d} {tot:10.1f} {step:10.1f}  {tops}", flush=True)
        best = min((r for r in rows if r["seqlen"] == seq), key=lambda r: r["us_call"])
        base = next(r for r in rows if r["seqlen"] == seq and r["num_splits"] == 0)
        print(f"  >> seq={seq} best={best['num_splits']} {best['us_call']:.1f}us "
              f"vs0={base['us_call']:.1f}us "
              f"({100*(base['us_call']-best['us_call'])/max(base['us_call'],1e-9):+.1f}%)", flush=True)
    (OUT / "attn_splits_b64.json").write_text(json.dumps(rows, indent=2))
    print("WROTE", OUT / "attn_splits_b64.json", flush=True)

if __name__ == "__main__":
    main()
