"""Confirm the small-batch flashinfer decode error is real, not a harness artifact.

split_decode_numeric found ixformer paged_attention_flashinfer wrong by ~4e-2 for
num_seqs < 32 at ctx >= 4096, while Triton stays at ~1e-4 against the same fp32
reference.  This varies seed, block-table layout, and page count to rule out the
scattered block table or a single unlucky draw, and checks whether the error
tracks pages-per-sequence (which would point at a split-K combine).
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("VLLM_PLUGINS", "fl")
sys.path.insert(0, "/root/bench_results")

import torch  # noqa: E402

N_QH, N_KVH, HD, BLOCK = 16, 2, 128, 16
DT, DEV = torch.bfloat16, "cuda"


def reference(q, k_hnd, v_hnd, bt, kv_len, scale):
    n = q.shape[0]
    nb = (kv_len + BLOCK - 1) // BLOCK
    rep = N_QH // N_KVH
    outs = []
    for i in range(n):
        pages = bt[i, :nb].long()
        k = k_hnd[pages].permute(1, 0, 2, 3).reshape(N_KVH, nb * BLOCK, HD)[:, :kv_len]
        v = v_hnd[pages].permute(1, 0, 2, 3).reshape(N_KVH, nb * BLOCK, HD)[:, :kv_len]
        k = k.float().repeat_interleave(rep, dim=0)
        v = v.float().repeat_interleave(rep, dim=0)
        s = torch.matmul(q[i].float().unsqueeze(1), k.transpose(1, 2)) * scale
        outs.append(torch.matmul(torch.softmax(s, -1), v).squeeze(1))
    return torch.stack(outs)


def run_case(n, ctx, seed, sequential_bt):
    from flag_gems.runtime.backend._iluvatar.fused.paged_attention_decode import (
        paged_attention_decode,
    )

    torch.manual_seed(seed)
    nb = ctx // BLOCK
    num_blocks = n * nb
    scale = 1.0 / (HD ** 0.5)
    k_hnd = torch.randn(num_blocks, N_KVH, BLOCK, HD, dtype=DT, device=DEV) * 0.5
    v_hnd = torch.randn(num_blocks, N_KVH, BLOCK, HD, dtype=DT, device=DEV) * 0.5
    kc, vc = k_hnd.permute(0, 2, 1, 3), v_hnd.permute(0, 2, 1, 3)
    q = torch.randn(n, N_QH, HD, dtype=DT, device=DEV) * 0.5
    if sequential_bt:
        bt = torch.arange(num_blocks, dtype=torch.int32, device=DEV).reshape(n, nb)
    else:
        bt = torch.randperm(num_blocks, device=DEV).to(torch.int32).reshape(n, nb)
    sk = torch.full((n,), ctx, dtype=torch.int32, device=DEV)

    ref = reference(q, k_hnd, v_hnd, bt, ctx, scale)
    out = torch.empty_like(q)
    paged_attention_decode(
        out=out, query=q, key_cache=kc, value_cache=vc, block_table=bt,
        seqused_k=sk, softmax_scale=scale, max_seqlen_k=ctx,
    )
    torch.cuda.synchronize()
    return (out.float() - ref).abs().max().item()


def main() -> int:
    import vllm_fl.dispatch.backends.vendor.iluvatar  # noqa: F401

    print("A. seed / block-table sensitivity (n=8, ctx=4096)")
    for seed in (0, 1, 2, 3):
        for seq in (False, True):
            e = run_case(8, 4096, seed, seq)
            tag = "sequential" if seq else "scattered "
            print(f"   seed={seed} bt={tag} err={e:.2e}")

    print()
    print("B. does the error track pages-per-seq? (n fixed at 8)")
    for ctx in (512, 1024, 2048, 3072, 4096, 6144, 8192):
        e = run_case(8, ctx, 0, False)
        print(f"   ctx={ctx:5d} pages/seq={ctx // BLOCK:4d} err={e:.2e}")

    print()
    print("C. boundary in n at ctx=4096 (total pages = n * 256)")
    for n in (28, 29, 30, 31, 32, 33, 34):
        e = run_case(n, 4096, 0, False)
        print(f"   n={n:3d} err={e:.2e}")

    print()
    print("D. is the boundary on n, or on total pages? (vary ctx at n=8,16)")
    for n in (8, 16):
        for ctx in (4096, 8192, 16384):
            e = run_case(n, ctx, 0, False)
            print(f"   n={n:3d} ctx={ctx:6d} total_pages={n * ctx // BLOCK:6d} err={e:.2e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
