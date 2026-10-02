"""Does the engine's wrapper hit the same wrong-sequence read?

The minimal reproduction says that below 32 sequences every row reads sequence
0's KV.  That cannot be the whole story: two Level 3 runs at batch 8 scored
29/30, which a failure that severe would have destroyed.  So re-run the readout
through the real call path -- paged_attention_decode, a block table as wide as
the engine allocates, and only part of it populated -- to see whether the engine
is actually affected or whether the minimal case differs in some detail.
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("VLLM_PLUGINS", "fl")
sys.path.insert(0, "/root/bench_results")

import torch  # noqa: E402

N_QH, N_KVH, HD, BLOCK = 16, 2, 128, 16
DT, DEV = torch.bfloat16, "cuda"
CH_SEQ, CH_HEAD, CH_ONES = 0, 1, 2


def run(n, ctx, max_blocks_cols, use_wrapper):
    from flag_gems.runtime.backend._iluvatar.fused.paged_attention_decode import (
        paged_attention_decode,
    )
    from ixformer.inference.functions.paged_attention import (
        paged_attention_flashinfer,
    )

    nb = ctx // BLOCK
    # Allocate a pool larger than what this batch uses, as the engine does.
    pool = max(n * nb * 2, n * max_blocks_cols)
    k_hnd = torch.zeros(pool, N_KVH, BLOCK, HD, dtype=DT, device=DEV)
    v_hnd = torch.zeros(pool, N_KVH, BLOCK, HD, dtype=DT, device=DEV)

    # Scatter each sequence's pages through the pool, as a real allocator would.
    g = torch.Generator(device=DEV).manual_seed(0)
    perm = torch.randperm(pool, device=DEV, generator=g)[: n * nb]
    used = perm.reshape(n, nb).to(torch.int32)

    for s in range(n):
        pages = used[s].long()
        for kvh in range(N_KVH):
            v_hnd[pages, kvh, :, CH_SEQ] = float(s + 1)
            v_hnd[pages, kvh, :, CH_HEAD] = float(kvh + 1)
            v_hnd[pages, kvh, :, CH_ONES] = 1.0

    q = torch.zeros(n, N_QH, HD, dtype=DT, device=DEV)
    seqused = torch.full((n,), ctx, dtype=torch.int32, device=DEV)
    out = torch.empty_like(q)

    if use_wrapper:
        # Engine path: wide block table, wrapper builds the page metadata.
        bt = torch.zeros(n, max_blocks_cols, dtype=torch.int32, device=DEV)
        bt[:, :nb] = used
        kc = k_hnd.permute(0, 2, 1, 3)
        vc = v_hnd.permute(0, 2, 1, 3)
        paged_attention_decode(
            out=out, query=q, key_cache=kc, value_cache=vc, block_table=bt,
            seqused_k=seqused, softmax_scale=1.0 / (HD ** 0.5), max_seqlen_k=ctx,
        )
    else:
        indptr = torch.arange(n + 1, dtype=torch.int32, device=DEV) * nb
        indices = used.reshape(-1).contiguous()
        last = torch.full((n,), BLOCK, dtype=torch.int32, device=DEV)
        paged_attention_flashinfer(
            out, q, (k_hnd, v_hnd), indptr, indices, last,
            1.0 / (HD ** 0.5), max_seq_len=ctx, kv_cache_format="HND",
        )
    torch.cuda.synchronize()

    mass = out[:, :, CH_ONES].clamp_min(1e-6)
    return (out[:, :, CH_SEQ] / mass)[:, 0], (out[:, :, CH_HEAD] / mass)[0], mass


def main() -> int:
    import vllm_fl.dispatch.backends.vendor.iluvatar  # noqa: F401

    lines = []

    def log(s=""):
        print(s, flush=True)
        lines.append(s)

    log("Each row should read its own sequence id (row i -> i+1).")
    log("'direct' = hand-built metadata, 'wrapper' = engine's paged_attention_decode")
    log("with a block table 8192 columns wide, as max_model_len=131072 allocates.")
    log()
    log(f"{'path':>8} {'n':>4} {'ctx':>6}  seq ids read")
    for use_wrapper in (False, True):
        tag = "wrapper" if use_wrapper else "direct"
        for n, ctx in [(8, 4096), (8, 1024), (32, 4096), (16, 2048)]:
            seq_read, head_read, mass = run(n, ctx, 8192, use_wrapper)
            ids = [round(x, 2) for x in seq_read.tolist()[:8]]
            ok = all(abs(v - (i + 1)) < 0.05 for i, v in enumerate(seq_read.tolist()))
            log(f"{tag:>8} {n:4d} {ctx:6d}  {ids} {'OK' if ok else 'WRONG'}")
        log()

    with open("/root/bench_results/flashinfer_engine_path_check.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
