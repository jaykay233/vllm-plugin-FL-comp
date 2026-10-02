"""Which elements does the small-batch flashinfer path get wrong?

Magnitude alone has not identified the bug: the output has the right overall
scale, is not explained by reduced-precision accumulation, and the shard weights
are exact.  So look at the shape of the error instead -- whether it concentrates
in particular sequences, query heads, or head-dim channels.  A structured error
localises the bug far better than a scalar norm.
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


def main() -> int:
    import vllm_fl.dispatch.backends.vendor.iluvatar  # noqa: F401

    lines = []

    def log(s=""):
        print(s, flush=True)
        lines.append(s)

    ctx = 4096
    nb = ctx // BLOCK
    rep = N_QH // N_KVH

    for n in (8, 32):
        torch.manual_seed(0)
        num_blocks = n * nb
        k_hnd = torch.randn(num_blocks, N_KVH, BLOCK, HD, dtype=DT, device=DEV) * 0.5
        v_hnd = torch.randn(num_blocks, N_KVH, BLOCK, HD, dtype=DT, device=DEV) * 0.5
        q = torch.zeros(n, N_QH, HD, dtype=DT, device=DEV)  # uniform attention
        bt = torch.arange(num_blocks, dtype=torch.int32, device=DEV).reshape(n, nb)

        out = call_flashinfer(q, k_hnd, v_hnd, bt, ctx).float()
        torch.cuda.synchronize()

        v_seq = v_hnd.reshape(n, nb, N_KVH, BLOCK, HD).permute(0, 2, 1, 3, 4)
        v_seq = v_seq.reshape(n, N_KVH, ctx, HD)
        ref = v_seq.float().mean(dim=2).repeat_interleave(rep, dim=1)

        err = (out - ref).abs()
        log(f"=== n={n} ctx={ctx} ===")
        log(f"  overall max={err.max().item():.2e} "
            f"rms={err.pow(2).mean().sqrt().item():.2e} "
            f"(true rms={ref.pow(2).mean().sqrt().item():.2e})")

        per_seq = err.amax(dim=(1, 2))
        per_head = err.amax(dim=(0, 2))
        log(f"  per-sequence max: {[round(x, 4) for x in per_seq.tolist()[:12]]}")
        log(f"  per-qhead    max: {[round(x, 4) for x in per_head.tolist()]}")

        # Does each query head match its own KV head, or a neighbour's?
        # With GQA, heads 0..7 share kv head 0 and 8..15 share kv head 1.
        kv_means = v_seq.float().mean(dim=2)  # [n, N_KVH, HD]
        log("  closest kv-head for each query head (expect 0 for h<8, 1 for h>=8):")
        best = []
        for h in range(N_QH):
            d = [(out[:, h] - kv_means[:, kvh]).abs().max().item()
                 for kvh in range(N_KVH)]
            best.append(d.index(min(d)))
        log(f"    {best}")

        # Is the error the same across sequences (a shared workspace reused)?
        if n >= 2:
            d01 = (out[0] - out[1]).abs().max().item()
            r01 = (ref[0] - ref[1]).abs().max().item()
            log(f"  seq0 vs seq1: kernel diff={d01:.2e} reference diff={r01:.2e}")
        log()

    with open("/root/bench_results/flashinfer_error_structure.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
