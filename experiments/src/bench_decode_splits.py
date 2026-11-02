#!/usr/bin/env python3
"""Sweep decode-attention num_splits at the shapes the 4k case actually runs.

batch=64, 16 q-heads / 2 kv-heads, head_dim 128, paged KV, block 16.
`num_splits=0` = let the MACA heuristic decide (the plugin's default at
batch >= 16). Everything else is a pinned split count.

Run: python experiments/src/bench_decode_splits.py
"""
from __future__ import annotations

import torch
from flash_attn import flash_attn_with_kvcache

torch.manual_seed(0)
DEV = "cuda"
H_Q, H_KV, D, BLOCK = 16, 2, 128, 16
BATCH, KV = 64, 4608


def bench(kv_len: int, splits: list[int], iters: int = 10):
    q = torch.randn(BATCH, 1, H_Q, D, dtype=torch.bfloat16, device=DEV) * 0.1
    nb_seq = (kv_len + BLOCK - 1) // BLOCK
    kc = torch.randn(BATCH * nb_seq, BLOCK, H_KV, D,
                     dtype=torch.bfloat16, device=DEV) * 0.1
    vc = torch.randn(BATCH * nb_seq, BLOCK, H_KV, D,
                     dtype=torch.bfloat16, device=DEV) * 0.1
    bt = torch.arange(BATCH * nb_seq, dtype=torch.int32,
                      device=DEV).view(BATCH, nb_seq)
    seqlens = torch.full((BATCH,), kv_len, dtype=torch.int32, device=DEV)
    scale = D ** -0.5
    out = []

    def call(ns):
        return flash_attn_with_kvcache(
            q=q, k_cache=kc, v_cache=vc, block_table=bt, cache_seqlens=seqlens,
            softmax_scale=scale, causal=True, num_splits=ns)

    ref = call(0)
    for ns in splits:
        r = call(ns)
        d = (ref.float() - r.float()).abs().max().item()
        for _ in range(3):
            call(ns)
        torch.cuda.synchronize()
        s = torch.cuda.Event(True); e = torch.cuda.Event(True)
        s.record()
        for _ in range(iters):
            call(ns)
        e.record(); torch.cuda.synchronize()
        ms = s.elapsed_time(e) / iters
        # KV traffic: one pass over every sequence's cache
        gb = BATCH * kv_len * H_KV * D * 2 * 2 / 1e9
        out.append((ns, ms, gb / (ms / 1e3), d))
    print(f"\nkv_len={kv_len} batch={BATCH}  (per-call, 42 layers/step)")
    best = min(o[1] for o in out)
    for ns, ms, bw, d in out:
        tag = "  <-- best" if ms == best else ""
        print(f"  num_splits={ns:3d}  {ms:7.3f} ms  {bw:6.0f} GB/s  "
              f"max|Δ| vs 0={d:.2e}{tag}")
    return out


if __name__ == "__main__":
    print("device", torch.cuda.get_device_name(0))
    for kv in (4608, 16384):
        bench(kv, [0, 1, 2, 4, 8, 16, 32])
