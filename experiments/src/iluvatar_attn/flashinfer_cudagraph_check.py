"""Does CUDA graph capture shield the engine from the wrong-sequence read?

The minimal reproduction and the engine's own wrapper both read sequence 0's KV
below 32 sequences, yet two Level 3 runs at batch 8 scored 29/30.  Proposed
explanation: under FULL_DECODE_ONLY vLLM captures decode with seq_lens filled
to 1, so the kernel picks its launch configuration for a short context and that
choice is frozen into the graph; replaying at the real context length then runs
the non-split, correct path.

This emulates that exactly -- capture with seqused_k = 1, replay with
seqused_k = 4096 -- and compares against the same call made eagerly.
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("VLLM_PLUGINS", "fl")
sys.path.insert(0, "/root/bench_results")

import torch  # noqa: E402

N_QH, N_KVH, HD, BLOCK = 16, 2, 128, 16
DT, DEV = torch.bfloat16, "cuda"
CH_SEQ, CH_ONES = 0, 2


def build_pool(n, ctx, cols):
    nb = ctx // BLOCK
    pool = max(n * nb * 2, n * 64)
    k_hnd = torch.zeros(pool, N_KVH, BLOCK, HD, dtype=DT, device=DEV)
    v_hnd = torch.zeros(pool, N_KVH, BLOCK, HD, dtype=DT, device=DEV)
    g = torch.Generator(device=DEV).manual_seed(0)
    used = torch.randperm(pool, device=DEV, generator=g)[: n * nb]
    used = used.reshape(n, nb).to(torch.int32)
    for s in range(n):
        pages = used[s].long()
        v_hnd[pages, :, :, CH_SEQ] = float(s + 1)
        v_hnd[pages, :, :, CH_ONES] = 1.0
    bt = torch.zeros(n, cols, dtype=torch.int32, device=DEV)
    bt[:, :nb] = used
    return k_hnd.permute(0, 2, 1, 3), v_hnd.permute(0, 2, 1, 3), bt


def seq_ids(out):
    mass = out[:, :, CH_ONES].clamp_min(1e-6)
    return (out[:, :, CH_SEQ] / mass)[:, 0]


def main() -> int:
    import vllm_fl.dispatch.backends.vendor.iluvatar  # noqa: F401
    from flag_gems.runtime.backend._iluvatar.fused.paged_attention_decode import (
        paged_attention_decode,
    )

    lines = []

    def log(s=""):
        print(s, flush=True)
        lines.append(s)

    n, ctx, cols = 8, 4096, 512
    kc, vc, bt = build_pool(n, ctx, cols)
    scale = 1.0 / (HD ** 0.5)

    # Persistent buffers, as the engine keeps for graph replay.
    q = torch.zeros(n, N_QH, HD, dtype=DT, device=DEV)
    out = torch.empty_like(q)
    seqused = torch.full((n,), ctx, dtype=torch.int32, device=DEV)

    log("Row i should read sequence id i+1.")
    log()

    # 1. Eager at the real context length.
    paged_attention_decode(
        out=out, query=q, key_cache=kc, value_cache=vc, block_table=bt,
        seqused_k=seqused, softmax_scale=scale, max_seqlen_k=ctx,
    )
    torch.cuda.synchronize()
    log(f"eager, ctx={ctx}:            {[round(x, 2) for x in seq_ids(out).tolist()]}")

    # 2. Capture with seq_lens = 1 (what build_for_cudagraph_capture does),
    #    then replay after restoring the real lengths.
    seqused.fill_(1)
    out.zero_()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            paged_attention_decode(
                out=out, query=q, key_cache=kc, value_cache=vc, block_table=bt,
                seqused_k=seqused, softmax_scale=scale, max_seqlen_k=1,
            )
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        paged_attention_decode(
            out=out, query=q, key_cache=kc, value_cache=vc, block_table=bt,
            seqused_k=seqused, softmax_scale=scale, max_seqlen_k=1,
        )

    seqused.fill_(ctx)  # real lengths at replay time
    out.zero_()
    graph.replay()
    torch.cuda.synchronize()
    log(f"graph captured@1, replay@{ctx}: "
        f"{[round(x, 2) for x in seq_ids(out).tolist()]}")

    # 3. Control: capture at the real length too.
    seqused.fill_(ctx)
    out.zero_()
    g2 = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g2):
        paged_attention_decode(
            out=out, query=q, key_cache=kc, value_cache=vc, block_table=bt,
            seqused_k=seqused, softmax_scale=scale, max_seqlen_k=ctx,
        )
    out.zero_()
    g2.replay()
    torch.cuda.synchronize()
    log(f"graph captured@{ctx}, replay@{ctx}: "
        f"{[round(x, 2) for x in seq_ids(out).tolist()]}")

    with open("/root/bench_results/flashinfer_cudagraph_check.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
