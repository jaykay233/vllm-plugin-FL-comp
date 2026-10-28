#!/usr/bin/env python3
"""Does fp8 KV cache work on MetaX flash attention, and how much does it save?

Decode attention at 4k context is bandwidth-bound (~80% of peak): it streams
11.3 GB of KV per step at batch 64, seq 4096.  Halving KV bytes (bf16 -> fp8)
is the only lever that attacks that directly, since num_splits is already at its
optimum and the kernel is at the BW roofline.

This script runs the exact decode attention call used by the model, in bf16 and
fp8, and reports us/call and the implied GB/s.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

OUT = Path(__file__).resolve().parent
NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, N_LAYERS = 16, 2, 128, 42
BATCH = 64
BLOCK = 16
SEQLENS = [2048, 4096]


def gpu_us(f, warmup=15, iters=30):
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
        t = e.self_device_time_total or 0
        if t > 0:
            by[e.key] += t / iters
    return sum(by.values()), sorted(by.items(), key=lambda kv: -kv[1])[:3]


def kv_bytes(seqlen, batch, elem):
    return N_LAYERS * 2 * NUM_KV_HEADS * HEAD_DIM * seqlen * batch * elem


def main() -> int:
    import torch
    from flash_attn import flash_attn_with_kvcache as fa

    dev = torch.device("cuda")
    rows = {}
    for seqlen in SEQLENS:
        for label, dt, elem in (("bf16", torch.bfloat16, 2), ("fp8", torch.float8_e4m3fn, 1)):
            try:
                max_blocks = (seqlen + BLOCK - 1) // BLOCK
                nblocks = max_blocks * BATCH
                q = torch.randn(BATCH, 1, NUM_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=dev) * 0.1
                kc = (torch.randn(nblocks, BLOCK, NUM_KV_HEADS, HEAD_DIM, device=dev) * 0.1).to(dt)
                vc = (torch.randn(nblocks, BLOCK, NUM_KV_HEADS, HEAD_DIM, device=dev) * 0.1).to(dt)
                bt = torch.arange(nblocks, dtype=torch.int32, device=dev).view(BATCH, max_blocks)
                cs = torch.full((BATCH,), seqlen, dtype=torch.int32, device=dev)

                def run():
                    return fa(
                        q=q,
                        k_cache=kc,
                        v_cache=vc,
                        block_table=bt,
                        cache_seqlens=cs,
                        softmax_scale=HEAD_DIM**-0.5,
                        causal=True,
                        num_splits=0,
                    )

                us, top = gpu_us(run)
                gb = kv_bytes(seqlen, BATCH, elem) / 1e9
                per_step = us * N_LAYERS / 1000
                rows[f"{label}_seq{seqlen}"] = {
                    "us_per_call": round(us, 1),
                    "ms_per_step_42_layers": round(per_step, 3),
                    "kv_GB": round(gb, 2),
                    "GBs": round(gb / (us / 1e6), 1),
                    "top": [(k[:60], round(v, 1)) for k, v in top],
                }
                print(
                    f"{label:4s} seq={seqlen:5d}  {us:7.1f} us/call  "
                    f"{per_step:6.2f} ms/step  KV {gb:5.2f} GB  -> {gb/(us/1e6):6.1f} GB/s",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001
                rows[f"{label}_seq{seqlen}"] = {"error": f"{type(exc).__name__}: {exc}"}
                print(f"{label:4s} seq={seqlen:5d}  FAILED: {type(exc).__name__}: {str(exc)[:180]}", flush=True)

    (OUT / "attn_fp8_kv.json").write_text(json.dumps(rows, indent=2))
    print("wrote", OUT / "attn_fp8_kv.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
