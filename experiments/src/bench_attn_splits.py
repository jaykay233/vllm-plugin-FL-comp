#!/usr/bin/env python3
"""Decode attention: what does split-KV actually cost on MetaX?

The decode kernel profile showed, per decode step (batch=1, seqlen<=77):

    flash_fwd_splitkv_combine_kernel   24.0 us/call   42 calls  = 1.007 ms/step
    flash_fwd_splitkv_kernel            8.3 us/call   42 calls  = 0.348 ms/step

The combine kernel costs 2.9x the actual attention math. For seqlen<=77 there is
essentially nothing to combine, so that 24us is pure launch/ramp overhead. If it
can be removed, it is worth ~13% of TPOT.

This measures the real decode attention call as a function of `num_splits`, at the
exact shapes MiniCPM5-2B uses (16 q heads / 2 kv heads / head_dim 128 / bf16).

Run:  conda activate mx && python /root/src/bench_attn_splits.py
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from vllm_fl.dispatch.backends.vendor.metax.impl.attention.utils.fa_utils import (
    flash_attn_varlen_func,
    get_flash_attn_version,
)

OUT = Path("/root/bench_results/attn_splits")
OUT.mkdir(parents=True, exist_ok=True)

# MiniCPM5-2B
NUM_HEADS = 16
NUM_KV_HEADS = 2
HEAD_DIM = 128
BLOCK_SIZE = 16

DEV = "cuda"
DT = torch.bfloat16


def make_inputs(seqlen: int, batch: int = 1):
    """Build the exact tensors vLLM's FA backend passes for a decode step."""
    qtokens = batch  # 1 query token per sequence
    q = torch.randn(qtokens, NUM_HEADS, HEAD_DIM, dtype=DT, device=DEV) * 0.1

    max_blocks = (seqlen + BLOCK_SIZE - 1) // BLOCK_SIZE
    nblocks = max_blocks * batch
    kv = torch.randn(2, nblocks, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM,
                     dtype=DT, device=DEV) * 0.1
    key_cache, value_cache = kv.unbind(0)

    block_table = torch.arange(nblocks, dtype=torch.int32, device=DEV).view(
        batch, max_blocks
    )
    cu_seqlens_q = torch.arange(0, batch + 1, dtype=torch.int32, device=DEV)
    seqused_k = torch.full((batch,), seqlen, dtype=torch.int32, device=DEV)
    out = torch.empty(qtokens, NUM_HEADS, HEAD_DIM, dtype=DT, device=DEV)
    return q, key_cache, value_cache, out, cu_seqlens_q, seqused_k, block_table


def call(q, key_cache, value_cache, out, cu_q, seqused_k, block_table,
         seqlen: int, num_splits: int):
    return flash_attn_varlen_func(
        q=q,
        k=key_cache,
        v=value_cache,
        out=out,
        cu_seqlens_q=cu_q,
        max_seqlen_q=1,
        seqused_k=seqused_k,
        max_seqlen_k=seqlen,
        softmax_scale=HEAD_DIM ** -0.5,
        causal=True,
        block_table=block_table,
        num_splits=num_splits,
    )


def timeit(fn, warmup: int = 50, iters: int = 200):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1000.0  # us


def kernels_of(fn):
    fn()
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        fn()
        torch.cuda.synchronize()
    return [e.name for e in prof.events()
            if e.device_type == torch.autograd.DeviceType.CUDA and e.name]


def main() -> None:
    torch.manual_seed(0)
    ver = get_flash_attn_version()
    print("=" * 100)
    print("DECODE ATTENTION vs num_splits  (MiniCPM5-2B shapes, batch=1, bf16)")
    print("=" * 100)
    print(f"  FA version: {ver}   heads={NUM_HEADS} kv_heads={NUM_KV_HEADS} "
          f"head_dim={HEAD_DIM} block={BLOCK_SIZE}")
    print()

    seqlens = [77, 128, 640, 2048]
    splits = [0, 1, 2, 4, 8, 16, 32]

    results: dict = {"fa_version": str(ver), "table": {}}

    for seqlen in seqlens:
        q, kc, vc, out, cu_q, seqused_k, bt = make_inputs(seqlen)
        print("-" * 100)
        print(f"seqlen = {seqlen}   (max_seqlen_k={seqlen}, "
              f"{(seqlen + BLOCK_SIZE - 1) // BLOCK_SIZE} kv blocks)")
        print(f"  {'num_splits':>10}  {'us/call':>9}  {'vs best':>8}   kernels launched")
        row = {}
        best = None
        for ns in splits:
            try:
                us = timeit(lambda ns=ns: call(q, kc, vc, out, cu_q, seqused_k,
                                               bt, seqlen, ns))
                kn = kernels_of(lambda ns=ns: call(q, kc, vc, out, cu_q,
                                                   seqused_k, bt, seqlen, ns))
            except Exception as exc:  # noqa: BLE001
                print(f"  {ns:>10}  {'ERR':>9}  {type(exc).__name__}: "
                      f"{str(exc)[:50]}")
                row[str(ns)] = {"error": f"{type(exc).__name__}: {exc}"}
                continue
            best = us if best is None else min(best, us)
            row[str(ns)] = {"us": us, "kernels": kn}
            print(f"  {ns:>10}  {us:>9.2f}  {'':>8}   "
                  f"{', '.join(k[:48] for k in kn)}")
        # second pass: relative
        print()
        for ns in splits:
            r = row.get(str(ns))
            if r and "us" in r:
                print(f"    num_splits={ns:<3} {r['us']:>8.2f} us  "
                      f"({r['us'] / best:>5.2f}x best)")
        results["table"][str(seqlen)] = row
        print()

    (OUT / "attn_splits.json").write_text(json.dumps(results, indent=2))
    print(f"  wrote {OUT / 'attn_splits.json'}")
    print("ATTN_SPLITS_DONE", flush=True)


if __name__ == "__main__":
    main()
