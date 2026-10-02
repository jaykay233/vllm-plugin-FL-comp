"""Confirm once-per-step metadata removes the FlagGems host tax on the split path.

Under VLLM_PLUGINS=fl, every aten pointwise in _build_flashinfer_pages /
_cu_seqlens_from_seqused becomes a Triton launch (~300-1000 us).  Rebuilding
those per layer made the split path host-bound (§17.10).  prebuild_split_meta
builds them once; the 42 layers should then only pay the two real attention
kernels.
"""

from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("VLLM_PLUGINS", "fl")
os.environ["VLLM_FL_ILUVATAR_IXFORMER_ATTN"] = "1"
os.environ["VLLM_FL_ILUVATAR_IXFORMER_SPLIT"] = "1"

import torch  # noqa: E402

N_QH, N_KVH, HD, BLOCK = 16, 2, 128, 16
DT, DEV = torch.bfloat16, "cuda"
N_LAYERS = 42


def issue_ms(fn, warm=5, reps=20):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    t1 = time.perf_counter()
    torch.cuda.synchronize()
    return (t1 - t0) * 1e3 / reps


def gpu_ms(fn, warm=5, reps=20):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(reps):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / reps


def build(q_lens, ctx):
    nb = ctx // BLOCK
    n = len(q_lens)
    num_blocks = n * nb
    k_hnd = torch.randn(num_blocks, N_KVH, BLOCK, HD, dtype=DT, device=DEV) * 0.5
    v_hnd = torch.randn(num_blocks, N_KVH, BLOCK, HD, dtype=DT, device=DEV) * 0.5
    kc, vc = k_hnd.permute(0, 2, 1, 3), v_hnd.permute(0, 2, 1, 3)
    q = torch.randn(int(sum(q_lens)), N_QH, HD, dtype=DT, device=DEV) * 0.5
    bt = torch.randperm(num_blocks, device=DEV).to(torch.int32).reshape(n, nb)
    lens = torch.tensor(q_lens, dtype=torch.int32, device=DEV)
    cu = torch.zeros(n + 1, dtype=torch.int32, device=DEV)
    cu[1:] = torch.cumsum(lens, 0)
    sk = torch.full((n,), ctx, dtype=torch.int32, device=DEV)
    return q, kc, vc, bt, cu, sk


def main() -> int:
    import vllm_fl.dispatch.backends.vendor.iluvatar  # noqa: F401
    from flag_gems.runtime.backend._iluvatar.fused import (
        clear_batch_split,
        prebuild_split_meta,
        set_batch_split,
    )
    from flag_gems.runtime.backend._iluvatar.fused.paged_attention_decode import (
        paged_attention_decode,
    )
    from flag_gems.runtime.backend._iluvatar.fused.flash_attn_prefill import (
        flash_attn_prefill,
    )

    scale = 1.0 / (HD ** 0.5)
    CTX = 4096
    n_dec, chunks = 63, [2048]
    q_lens = [1] * n_dec + chunks
    q, kc, vc, bt, cu, sk = build(q_lens, CTX)
    max_q = max(q_lens)
    out = torch.empty_like(q)

    def run_rebuild():
        # Old path: publish counts only; every call rebuilds pages + cu_seqlens.
        set_batch_split(n_dec, n_dec, int(q.shape[0]))
        paged_attention_decode(
            out=out[:n_dec], query=q[:n_dec], key_cache=kc, value_cache=vc,
            block_table=bt[:n_dec], seqused_k=sk[:n_dec],
            softmax_scale=scale, max_seqlen_k=CTX,
        )
        flash_attn_prefill(
            out=out[n_dec:], query=q[n_dec:], key_cache=kc, value_cache=vc,
            cu_seqlens_q=cu[n_dec:] - cu[n_dec], max_seqlen_q=max_q,
            seqused_k=sk[n_dec:], max_seqlen_k=CTX,
            block_table=bt[n_dec:], softmax_scale=scale,
        )

    def run_prebuilt():
        # New path: metadata built once, then 42 layers reuse it.
        # Time one layer after a single prebuild (matches engine: 1 build + 42 use).
        paged_attention_decode(
            out=out[:n_dec], query=q[:n_dec], key_cache=kc, value_cache=vc,
            block_table=bt[:n_dec], seqused_k=sk[:n_dec],
            softmax_scale=scale, max_seqlen_k=CTX,
            pages=__import__(
                "flag_gems.runtime.backend._iluvatar.fused.ixformer_unified_attention",
                fromlist=["_cached_decode_pages"],
            )._cached_decode_pages(n_dec),
        )
        from flag_gems.runtime.backend._iluvatar.fused.ixformer_unified_attention import (
            _BATCH_SPLIT,
        )
        flash_attn_prefill(
            out=out[n_dec:], query=q[n_dec:], key_cache=kc, value_cache=vc,
            cu_seqlens_q=_BATCH_SPLIT["prefill_cu_q"], max_seqlen_q=max_q,
            seqused_k=sk[n_dec:], max_seqlen_k=CTX,
            block_table=bt[n_dec:], softmax_scale=scale,
            cu_seqlens_k=_BATCH_SPLIT["prefill_cu_k"],
        )

    lines = []
    def log(s=""):
        print(s, flush=True)
        lines.append(s)

    # Prime the stash once, outside the timed region for the prebuilt path.
    prebuild_split_meta(
        n_dec, n_dec, int(q.shape[0]), bt, sk, cu, BLOCK
    )
    # Also time the prebuild itself.
    def do_prebuild():
        prebuild_split_meta(
            n_dec, n_dec, int(q.shape[0]), bt, sk, cu, BLOCK
        )

    i_rebuild = issue_ms(run_rebuild)
    g_rebuild = gpu_ms(run_rebuild)
    i_prebuild = issue_ms(do_prebuild)
    i_use = issue_ms(run_prebuilt)
    g_use = gpu_ms(run_prebuilt)

    log("63 decode + 1×q2048 @ ctx=4096, under FlagGems aten dispatch")
    log(f"{'path':>18} {'issue ms':>10} {'gpu ms':>9} {'42x issue':>11} {'42x gpu':>9}")
    log(f"{'rebuild/layer':>18} {i_rebuild:10.3f} {g_rebuild:9.3f} "
        f"{i_rebuild * N_LAYERS:11.1f} {g_rebuild * N_LAYERS:9.1f}")
    log(f"{'prebuild once':>18} {i_prebuild:10.3f} {'—':>9} "
        f"{i_prebuild:11.1f} {'—':>9}")
    log(f"{'reuse/layer':>18} {i_use:10.3f} {g_use:9.3f} "
        f"{i_use * N_LAYERS:11.1f} {g_use * N_LAYERS:9.1f}")
    step_old = i_rebuild * N_LAYERS
    step_new = i_prebuild + i_use * N_LAYERS
    log()
    log(f"step host estimate: old={step_old:.1f} ms  "
        f"new={step_new:.1f} ms  "
        f"saved={step_old - step_new:.1f} ms ({100 * (1 - step_new / step_old):.0f}%)")

    clear_batch_split()
    out_path = "/root/bench_results/split_meta_once.txt"
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    log(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
