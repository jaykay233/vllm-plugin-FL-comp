#!/usr/bin/env python3
"""Dense vs paged KV prefill attention on MetaX C500.

The plugin's prefill path hands flash_attn_varlen_func the *paged* KV cache it
just wrote, so every prefill reads K/V back through a block table. Stock vLLM
passes the dense K/V it already has when there is no cached prefix. This
measures what the paging costs.

Run: python experiments/src/bench_prefill_attn_paged.py
"""
from __future__ import annotations

import torch
from flash_attn import flash_attn_varlen_func

torch.manual_seed(0)
DEV = "cuda"
H_Q, H_KV, D = 16, 2, 128
BLOCK = 16


def make(batch: int, seq: int):
    tot = batch * seq
    q = torch.randn(tot, H_Q, D, dtype=torch.bfloat16, device=DEV) * 0.1
    k = torch.randn(tot, H_KV, D, dtype=torch.bfloat16, device=DEV) * 0.1
    v = torch.randn(tot, H_KV, D, dtype=torch.bfloat16, device=DEV) * 0.1
    cu = torch.arange(0, tot + 1, seq, dtype=torch.int32, device=DEV)
    nblk = (tot + BLOCK - 1) // BLOCK
    kc = torch.zeros(nblk, BLOCK, H_KV, D, dtype=torch.bfloat16, device=DEV)
    vc = torch.zeros(nblk, BLOCK, H_KV, D, dtype=torch.bfloat16, device=DEV)
    kc.view(-1, H_KV, D).copy_(k)
    vc.view(-1, H_KV, D).copy_(v)
    # page table: sequence i owns blocks [i*seq/BLOCK, (i+1)*seq/BLOCK)
    per = seq // BLOCK
    bt = torch.arange(nblk, dtype=torch.int32, device=DEV).view(batch, per)
    return q, k, v, cu, kc, vc, bt


def run(tag, batch, seq, iters=5):
    q, k, v, cu, kc, vc, bt = make(batch, seq)
    scale = D ** -0.5
    flops = 2 * 2 * batch * (seq * (seq + 1) / 2) * H_Q * D
    out = {}

    def timeit(fn):
        for _ in range(2):
            fn()
        torch.cuda.synchronize()
        s = torch.cuda.Event(True); e = torch.cuda.Event(True)
        s.record()
        for _ in range(iters):
            fn()
        e.record(); torch.cuda.synchronize()
        return s.elapsed_time(e) / iters

    dense = timeit(lambda: flash_attn_varlen_func(
        q, k, v, cu, cu, seq, seq, causal=True, softmax_scale=scale))
    paged = timeit(lambda: flash_attn_varlen_func(
        q, kc, vc, cu, cu, seq, seq, causal=True, softmax_scale=scale,
        block_table=bt))
    ref = flash_attn_varlen_func(q, k, v, cu, cu, seq, seq, causal=True, softmax_scale=scale)
    got = flash_attn_varlen_func(q, kc, vc, cu, cu, seq, seq, causal=True, softmax_scale=scale,
                                 block_table=bt)
    diff = (ref.float() - got.float()).abs().max().item()
    print(f"{tag:26s} batch={batch:3d} seq={seq:6d} "
          f"dense={dense:7.2f}ms ({flops/dense/1e9:6.1f} TFLOP/s)  "
          f"paged={paged:7.2f}ms ({flops/paged/1e9:6.1f} TFLOP/s)  "
          f"paged/dense={paged/dense:5.3f}  max|Δ|={diff:.2e}")
    out[tag] = (dense, paged)
    return out


if __name__ == "__main__":
    print(f"device={torch.cuda.get_device_name(0)}  heads={H_Q}/{H_KV} d={D} block={BLOCK}")
    run("4k chunk (2x4096)", 2, 4096)
    run("4k single (1x4096)", 1, 4096)
    run("8k chunk (1x8192)", 1, 8192)
    run("16k chunk (1x8192)", 1, 8192)
    run("16k full (1x16384)", 1, 16384)
    run("4k x8", 8, 4096)
