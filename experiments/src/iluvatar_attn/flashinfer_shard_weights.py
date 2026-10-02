"""Pin down the split-K combine bug by solving for the shard weights.

With q = 0 the attention distribution is exactly uniform, so the correct output
is the plain mean of V over the whole context.  If the kernel splits the KV into
S shards and combines them with the wrong weights, the output instead equals a
*weighted* mean of the per-shard means.  Making V piecewise-constant over
candidate shard boundaries turns the output into a direct readout of those
weights.
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("VLLM_PLUGINS", "fl")
sys.path.insert(0, "/root/bench_results")

import torch  # noqa: E402

N_QH, N_KVH, HD, BLOCK = 16, 2, 128, 16
DT, DEV = torch.bfloat16, "cuda"


def call_flashinfer(q, k_hnd, v_hnd, bt, ctx, scale):
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
        float(scale), max_seq_len=ctx, kv_cache_format="HND",
    )
    return out


def probe_weights(n, ctx, n_segments):
    """V is constant within each segment, so the output reads out the weights."""
    nb = ctx // BLOCK
    num_blocks = n * nb
    k_hnd = torch.zeros(num_blocks, N_KVH, BLOCK, HD, dtype=DT, device=DEV)
    v_hnd = torch.zeros(num_blocks, N_KVH, BLOCK, HD, dtype=DT, device=DEV)

    # Segment s gets V = one-hot in channel s, for sequence 0.
    seg_pages = nb // n_segments
    for s in range(n_segments):
        lo, hi = s * seg_pages, (s + 1) * seg_pages
        v_hnd[lo:hi, :, :, s] = 1.0

    q = torch.zeros(n, N_QH, HD, dtype=DT, device=DEV)  # uniform attention
    bt = torch.arange(num_blocks, dtype=torch.int32, device=DEV).reshape(n, nb)
    out = call_flashinfer(q, k_hnd, v_hnd, bt, ctx, 1.0 / (HD ** 0.5))
    torch.cuda.synchronize()
    # Channel s of the output = total weight the kernel gave segment s.
    return out[0, 0, :n_segments].float().cpu()


def main() -> int:
    import vllm_fl.dispatch.backends.vendor.iluvatar  # noqa: F401

    lines = []

    def log(s=""):
        print(s, flush=True)
        lines.append(s)

    log("Uniform attention (q=0): each segment should receive weight 1/S.")
    log("Any deviation is the combine mis-weighting that segment.")
    log()

    for n in (8, 32):
        for S in (2, 4, 8, 16):
            w = probe_weights(n, 4096, S)
            ideal = 1.0 / S
            dev = (w - ideal).abs().max().item()
            tag = "OK " if dev < 0.02 * ideal else "BAD"
            log(f"{tag} n={n:3d} S={S:3d} ideal={ideal:.4f} "
                f"weights={[round(x, 4) for x in w.tolist()]}")
            log(f"        sum={w.sum().item():.4f} maxdev={dev:.4f}")
        log()

    log("If the weights are uniform but their sum != 1, the denominator is")
    log("wrong.  If some segments are over/under-weighted, the per-shard")
    log("sumexp used in the combine is wrong.")

    with open("/root/bench_results/flashinfer_shard_weights.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
