"""Read out exactly which (sequence, kv-head) the small-batch path actually reads.

The error structure showed every query head landing on the last KV head and
different sequences producing near-identical output, which points at indexing
rather than arithmetic.  Encoding the sequence index and KV-head index directly
into V turns the kernel's output into a literal readout of what it read: with
q = 0 the attention is uniform, so each output channel returns the mean of that
channel over the context, i.e. the identity of the source it used.
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("VLLM_PLUGINS", "fl")
sys.path.insert(0, "/root/bench_results")

import torch  # noqa: E402

N_QH, N_KVH, HD, BLOCK = 16, 2, 128, 16
DT, DEV = torch.bfloat16, "cuda"


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


def readout(n, ctx):
    nb = ctx // BLOCK
    num_blocks = n * nb
    k_hnd = torch.zeros(num_blocks, N_KVH, BLOCK, HD, dtype=DT, device=DEV)
    v_hnd = torch.zeros(num_blocks, N_KVH, BLOCK, HD, dtype=DT, device=DEV)

    # Channel 0 carries the sequence index, channel 1 the kv-head index.
    seq_of_block = torch.arange(num_blocks, device=DEV) // nb
    for kvh in range(N_KVH):
        v_hnd[:, kvh, :, 0] = seq_of_block.to(DT).view(-1, 1)
        v_hnd[:, kvh, :, 1] = float(kvh)

    q = torch.zeros(n, N_QH, HD, dtype=DT, device=DEV)
    bt = torch.arange(num_blocks, dtype=torch.int32, device=DEV).reshape(n, nb)
    out = call_flashinfer(q, k_hnd, v_hnd, bt, ctx).float()
    torch.cuda.synchronize()
    return out[:, :, 0], out[:, :, 1]  # [n, N_QH] each


def main() -> int:
    import vllm_fl.dispatch.backends.vendor.iluvatar  # noqa: F401

    lines = []

    def log(s=""):
        print(s, flush=True)
        lines.append(s)

    ctx = 4096
    for n in (8, 32):
        seq_read, head_read = readout(n, ctx)
        log(f"=== n={n} ctx={ctx} ===")
        log("  sequence index each row actually read (row i should read i):")
        for i in range(min(n, 8)):
            vals = [round(x, 2) for x in seq_read[i].tolist()]
            log(f"    row {i}: {vals}")
        log("  kv-head each query head read (h<8 -> 0, h>=8 -> 1):")
        log(f"    row 0: {[round(x, 2) for x in head_read[0].tolist()]}")

        want_seq = torch.arange(n, device=DEV, dtype=torch.float32).unsqueeze(1)
        want_head = torch.tensor(
            [0.0] * (N_QH // 2) + [1.0] * (N_QH // 2), device=DEV
        ).unsqueeze(0)
        log(f"  seq-index  max deviation: "
            f"{(seq_read - want_seq).abs().max().item():.3f}")
        log(f"  kv-head    max deviation: "
            f"{(head_read - want_head).abs().max().item():.3f}")
        log()

    with open("/root/bench_results/flashinfer_index_readout.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
