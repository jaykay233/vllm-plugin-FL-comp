#!/usr/bin/env python3
"""Unit test for the dense-KV prefill split arithmetic.

Builds a step where the first `k` prefill requests continue a cached prefix and
the rest start at token 0, then compares

  reference: one paged varlen call over every request
  candidate: paged call for [0:k) + dense call for [k:)   (plugin's split)

Outputs must match elementwise. Run: python experiments/src/test_prefill_split.py
"""
from __future__ import annotations

import torch
from flash_attn import flash_attn_varlen_func

torch.manual_seed(0)
DEV = "cuda"
H_Q, H_KV, D, BLOCK = 16, 2, 128, 16


def one(num_prefix_req: int, specs):
    """specs: list of (prefix_len, query_len) per request, in schedule order."""
    q_lens = torch.tensor([s[1] for s in specs], dtype=torch.int32, device=DEV)
    seq_lens = torch.tensor([s[0] + s[1] for s in specs], dtype=torch.int32, device=DEV)
    tot_q = int(q_lens.sum())
    n_req = len(specs)

    q = torch.randn(tot_q, H_Q, D, dtype=torch.bfloat16, device=DEV) * 0.1
    # dense k/v hold exactly the *new* tokens of this step
    kd = torch.randn(tot_q, H_KV, D, dtype=torch.bfloat16, device=DEV) * 0.1
    vd = torch.randn(tot_q, H_KV, D, dtype=torch.bfloat16, device=DEV) * 0.1

    # paged cache: lay each request's full K/V out back to back. The cached
    # prefix region is left as zeros (its values never matter for the compare,
    # only that both arms read the same thing), the new tokens are written in.
    nblk_total = sum((pre + ql + BLOCK - 1) // BLOCK for pre, ql in specs)
    kc = torch.zeros(nblk_total, BLOCK, H_KV, D, dtype=torch.bfloat16, device=DEV)
    vc = torch.zeros(nblk_total, BLOCK, H_KV, D, dtype=torch.bfloat16, device=DEV)
    flat_k, flat_v = kc.view(-1, H_KV, D), vc.view(-1, H_KV, D)

    rows, blk_ids, blk_off, tok = [], [], 0, 0
    for pre, ql in specs:
        rows.append((blk_off * BLOCK + pre, tok, ql))
        flat_k[blk_off * BLOCK + pre:blk_off * BLOCK + pre + ql] = kd[tok:tok + ql]
        flat_v[blk_off * BLOCK + pre:blk_off * BLOCK + pre + ql] = vd[tok:tok + ql]
        nb = (pre + ql + BLOCK - 1) // BLOCK
        blk_ids.append(list(range(blk_off, blk_off + nb)))
        blk_off += nb
        tok += ql

    max_nb = max(len(x) for x in blk_ids)
    bt = torch.zeros(n_req, max_nb, dtype=torch.int32, device=DEV)
    for i, x in enumerate(blk_ids):
        bt[i, : len(x)] = torch.tensor(x, dtype=torch.int32, device=DEV)

    cu_q = torch.zeros(n_req + 1, dtype=torch.int32, device=DEV)
    cu_q[1:] = torch.cumsum(q_lens, 0)
    cu_k = torch.zeros(n_req + 1, dtype=torch.int32, device=DEV)
    cu_k[1:] = torch.cumsum(seq_lens, 0)
    scale = D ** -0.5

    ref = flash_attn_varlen_func(q, kc, vc, cu_q, cu_k, int(q_lens.max()),
                                 int(seq_lens.max()), causal=True,
                                 softmax_scale=scale, block_table=bt)

    kk = num_prefix_req
    out = torch.empty_like(ref)
    qsl = cu_q
    paged_tokens = int(qsl[kk] - qsl[0])
    if kk > 0:
        out[:paged_tokens] = flash_attn_varlen_func(
            q=q[:paged_tokens], k=kc, v=vc,
            block_table=bt[:kk],
            cu_seqlens_q=qsl[:kk + 1] - qsl[0],
            cu_seqlens_k=cu_k[:kk + 1] - cu_k[0],
            max_seqlen_q=int(q_lens[:kk].max()),
            max_seqlen_k=int(seq_lens[:kk].max()),
            causal=True, softmax_scale=scale)
    if kk < n_req:
        cu_k_d = torch.cat(
            [seq_lens.new_zeros(1), seq_lens[kk:]]).cumsum(0, dtype=torch.int32)
        out[paged_tokens:] = flash_attn_varlen_func(
            q=q[paged_tokens:], k=kd[paged_tokens:], v=vd[paged_tokens:],
            cu_seqlens_q=qsl[kk:] - qsl[kk], cu_seqlens_k=cu_k_d,
            max_seqlen_q=int(q_lens[kk:].max()),
            max_seqlen_k=int(seq_lens[kk:].max()),
            causal=True, softmax_scale=scale)

    d = (ref.float() - out.float()).abs()
    print(f"k={kk} specs={specs}  max|Δ|={d.max().item():.3e}  "
          f"nonzero={int((d > 0).sum())}/{d.numel()}")
    return d.max().item()


if __name__ == "__main__":
    print("device", torch.cuda.get_device_name(0))
    worst = 0.0
    worst = max(worst, one(2, [(4096, 64), (8192, 1024), (0, 4096), (0, 2048)]))
    worst = max(worst, one(1, [(4032, 64), (0, 4096), (0, 3968)]))
    worst = max(worst, one(0, [(0, 4096), (0, 4032)]))
    worst = max(worst, one(3, [(16384, 8192), (0, 8192), (0, 4096), (0, 1024)]))
    print("\nWORST max|Δ| =", worst, "-> PASS" if worst == 0.0 else "-> FAIL")
