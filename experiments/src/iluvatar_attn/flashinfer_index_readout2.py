"""Separate 'reads the wrong sequence' from 'sums only part of the context'.

The first readout was ambiguous: sequence 0 encodes as the value 0, so "every
row read sequence 0" and "every row read almost nothing" look alike.  Here the
sequence id starts at 1, and a third channel holds 1.0 everywhere, so its output
is exactly the fraction of the softmax mass the kernel actually accumulated.
Dividing the id channel by that fraction recovers the sequence genuinely read.
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("VLLM_PLUGINS", "fl")
sys.path.insert(0, "/root/bench_results")

import torch  # noqa: E402

N_QH, N_KVH, HD, BLOCK = 16, 2, 128, 16
DT, DEV = torch.bfloat16, "cuda"

CH_SEQ, CH_HEAD, CH_ONES, CH_POS = 0, 1, 2, 3


def call_flashinfer(q, k_hnd, v_hnd, bt, ctx):
    from ixformer.inference.functions.paged_attention import (
        paged_attention_flashinfer,
    )

    n, nb = bt.shape
    indptr = torch.arange(n + 1, dtype=torch.int32, device=DEV) * nb
    indices = bt.reshape(-1).contiguous().to(torch.int32)
    last_page_len = torch.full((n,), BLOCK, dtype=torch.int32, device=DEV)
    out = torch.empty_like(q)
    paged_attention_flashinfer(
        out, q, (k_hnd, v_hnd), indptr, indices, last_page_len,
        1.0 / (HD ** 0.5), max_seq_len=ctx, kv_cache_format="HND",
    )
    return out


def probe(n, ctx):
    nb = ctx // BLOCK
    num_blocks = n * nb
    k_hnd = torch.zeros(num_blocks, N_KVH, BLOCK, HD, dtype=DT, device=DEV)
    v_hnd = torch.zeros(num_blocks, N_KVH, BLOCK, HD, dtype=DT, device=DEV)

    block_idx = torch.arange(num_blocks, device=DEV)
    seq_id = (block_idx // nb + 1).to(torch.float32)      # 1..n, never 0
    pos_in_seq = (block_idx % nb).to(torch.float32) / nb  # 0..1 across context

    for kvh in range(N_KVH):
        v_hnd[:, kvh, :, CH_SEQ] = seq_id.view(-1, 1).to(DT)
        v_hnd[:, kvh, :, CH_HEAD] = float(kvh + 1)
        v_hnd[:, kvh, :, CH_ONES] = 1.0
        v_hnd[:, kvh, :, CH_POS] = pos_in_seq.view(-1, 1).to(DT)

    q = torch.zeros(n, N_QH, HD, dtype=DT, device=DEV)
    bt = torch.arange(num_blocks, dtype=torch.int32, device=DEV).reshape(n, nb)
    out = call_flashinfer(q, k_hnd, v_hnd, bt, ctx).float()
    torch.cuda.synchronize()
    return out


def main() -> int:
    import vllm_fl.dispatch.backends.vendor.iluvatar  # noqa: F401

    lines = []

    def log(s=""):
        print(s, flush=True)
        lines.append(s)

    ctx = 4096
    for n in (8, 32):
        out = probe(n, ctx)
        ones = out[:, :, CH_ONES]
        log(f"=== n={n} ctx={ctx} ===")
        log("  mass: fraction of the softmax weight actually accumulated")
        log(f"    expected 1.0, got min={ones.min().item():.4f} "
            f"max={ones.max().item():.4f}  (1/{1.0 / max(ones.max().item(), 1e-9):.1f})")

        # Recover what was read, undoing the missing normalisation.
        safe = ones.clamp_min(1e-6)
        seq_read = out[:, :, CH_SEQ] / safe
        head_read = out[:, :, CH_HEAD] / safe
        pos_read = out[:, :, CH_POS] / safe

        log("  sequence id read per row (row i should read i+1), query head 0:")
        log(f"    {[round(x, 2) for x in seq_read[:, 0].tolist()[:12]]}")
        log("  kv-head read per query head (should be 1 for h<8, 2 for h>=8):")
        log(f"    {[round(x, 2) for x in head_read[0].tolist()]}")
        log("  mean position within the context that was summed")
        log("    (0.5 if the whole context was covered):")
        log(f"    {[round(x, 3) for x in pos_read[:, 0].tolist()[:8]]}")
        log()

    with open("/root/bench_results/flashinfer_index_readout2.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
