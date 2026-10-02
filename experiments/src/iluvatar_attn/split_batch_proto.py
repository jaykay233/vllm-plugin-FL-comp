"""Prototype: split a reordered mixed batch into decode rows + prefill rows.

vLLM already offers ``reorder_batch_to_split_decodes_and_prefills``, which moves
every ``q_len == 1`` row to the front of the batch.  With that ordering a mixed
batch becomes two contiguous slices, so the gate can send the decode prefix to
ixformer flashinfer and the prefill suffix to ixformer FA instead of handing the
whole batch to Triton.  This measures whether that is actually faster, and
whether it is numerically identical to the Triton path it would replace.

Timings include the per-call slicing and page-metadata construction, since those
would run on the real hot path too.
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("VLLM_PLUGINS", "fl")
sys.path.insert(0, "/root/bench_results")

import torch  # noqa: E402

N_QH, N_KVH, HD, BLOCK = 16, 2, 128, 16
DT, DEV = torch.bfloat16, "cuda"
OUT_PATH = "/root/bench_results/split_batch_proto.txt"


def timeit(fn, warm=5, reps=20):
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


def build_varlen(q_lens, kv_len):
    """Reordered batch: callers pass q_lens with every 1 first."""
    nb_per_seq = (kv_len + BLOCK - 1) // BLOCK
    n = len(q_lens)
    num_blocks = n * nb_per_seq
    # The plugin allocates KV so that the HND view is physically contiguous
    # (stride_order (1,0,3,2,4)); the BNDH view the kernels take is then
    # strided.  Build it that way round or the decode kernel rejects the cache.
    k_hnd = torch.randn(num_blocks, N_KVH, BLOCK, HD, dtype=DT, device=DEV) * 0.5
    v_hnd = torch.randn(num_blocks, N_KVH, BLOCK, HD, dtype=DT, device=DEV) * 0.5
    k_cache = k_hnd.permute(0, 2, 1, 3)
    v_cache = v_hnd.permute(0, 2, 1, 3)
    q = torch.randn(int(sum(q_lens)), N_QH, HD, dtype=DT, device=DEV) * 0.5
    perm = torch.randperm(num_blocks, device=DEV).to(torch.int32)
    bt = perm.reshape(n, nb_per_seq).contiguous()
    lens = torch.tensor(q_lens, dtype=torch.int32, device=DEV)
    cu_q = torch.zeros(n + 1, dtype=torch.int32, device=DEV)
    cu_q[1:] = torch.cumsum(lens, 0)
    seqused_k = torch.full((n,), kv_len, dtype=torch.int32, device=DEV)
    return q, k_cache, v_cache, bt, cu_q, seqused_k


def main() -> int:
    import vllm_fl.dispatch.backends.vendor.iluvatar  # noqa: F401
    import vllm.v1.attention.ops.triton_unified_attention as _uam
    from flag_gems.runtime.backend._iluvatar.fused.paged_attention_decode import (
        paged_attention_decode,
    )
    from flag_gems.runtime.backend._iluvatar.fused.flash_attn_prefill import (
        flash_attn_prefill,
    )

    triton_ua = getattr(_uam, "_orig_unified_attention", None) or _uam.unified_attention
    scale = 1.0 / (HD ** 0.5)
    lines: list[str] = []

    def log(s: str = "") -> None:
        print(s, flush=True)
        lines.append(s)

    CTX = 4096
    # (name, num decode rows, list of prefill chunk lengths)
    cases = [
        ("63 decode + 1 chunk q=2048", 63, [2048]),
        ("63 decode + 1 chunk q=512", 63, [512]),
        ("32 decode + 4 chunks q=512", 32, [512] * 4),
        ("48 decode + 2 chunks q=1024", 48, [1024] * 2),
        ("16 decode + 1 chunk q=2048", 16, [2048]),
    ]

    log("== reordered mixed batch: split(flashinfer + FA) vs all-Triton vs all-FA ==")
    log(f"   ctx(kv)={CTX}, heads q/kv={N_QH}/{N_KVH}, head_dim={HD}, page={BLOCK}")
    log("   decode rows are first, as reorder_batch_to_split_decodes_and_prefills")
    log("   would leave them; timings include slicing + page metadata build.")
    log("")

    for name, n_dec, chunks in cases:
        torch.manual_seed(0)
        q_lens = [1] * n_dec + list(chunks)
        n = len(q_lens)
        q, kc, vc, bt, cu_q, sk = build_varlen(q_lens, CTX)
        max_q = int(max(q_lens))
        total_q = int(sum(q_lens))

        out_t = torch.empty_like(q)
        out_s = torch.empty_like(q)
        out_f = torch.empty_like(q)

        # Host-side split point: vLLM's split_decodes_and_prefills returns these
        # from query_start_loc_cpu, so no device->host sync is implied here.
        n_dec_tok = n_dec

        def run_triton():
            triton_ua(
                q=q, k=kc, v=vc, out=out_t, cu_seqlens_q=cu_q, max_seqlen_q=max_q,
                seqused_k=sk, max_seqlen_k=CTX, softmax_scale=scale, causal=True,
                window_size=(-1, -1), block_table=bt, softcap=0.0,
                q_descale=None, k_descale=None, v_descale=None,
            )

        def run_split():
            paged_attention_decode(
                out=out_s[:n_dec_tok],
                query=q[:n_dec_tok],
                key_cache=kc,
                value_cache=vc,
                block_table=bt[:n_dec],
                seqused_k=sk[:n_dec],
                softmax_scale=scale,
                max_seqlen_k=CTX,
            )
            cu_pre = cu_q[n_dec:] - cu_q[n_dec]
            flash_attn_prefill(
                out=out_s[n_dec_tok:],
                query=q[n_dec_tok:],
                key_cache=kc,
                value_cache=vc,
                cu_seqlens_q=cu_pre,
                max_seqlen_q=max_q,
                seqused_k=sk[n_dec:],
                max_seqlen_k=CTX,
                block_table=bt[n_dec:],
                softmax_scale=scale,
            )

        def run_fa():
            flash_attn_prefill(
                out=out_f, query=q, key_cache=kc, value_cache=vc,
                cu_seqlens_q=cu_q, max_seqlen_q=max_q, seqused_k=sk,
                max_seqlen_k=CTX, block_table=bt, softmax_scale=scale,
            )

        try:
            run_triton()
            run_split()
            run_fa()
            torch.cuda.synchronize()
        except Exception as e:  # noqa: BLE001
            log(f"{name:30} FAILED {type(e).__name__}: {e}")
            continue

        d_dec = (out_s[:n_dec_tok].float() - out_t[:n_dec_tok].float()).abs().max()
        d_pre = (out_s[n_dec_tok:].float() - out_t[n_dec_tok:].float()).abs().max()

        ms_t = timeit(run_triton)
        ms_s = timeit(run_split)
        ms_f = timeit(run_fa)

        log(
            f"{name:30} tok={total_q:5d} seqs={n:3d}  "
            f"Triton={ms_t:7.3f}  split={ms_s:7.3f}  allFA={ms_f:7.3f}  "
            f"split/Triton={ms_t / ms_s:5.2f}x  "
            f"maxdiff dec={d_dec:.1e} pre={d_pre:.1e}"
        )

    # Where does the split's time go?
    log("")
    log("== split breakdown on 63 decode + 1 chunk q=2048 ==")
    torch.manual_seed(0)
    q_lens = [1] * 63 + [2048]
    q, kc, vc, bt, cu_q, sk = build_varlen(q_lens, CTX)
    out = torch.empty_like(q)

    def only_decode():
        paged_attention_decode(
            out=out[:63], query=q[:63], key_cache=kc, value_cache=vc,
            block_table=bt[:63], seqused_k=sk[:63], softmax_scale=scale,
            max_seqlen_k=CTX,
        )

    def only_prefill():
        cu_pre = cu_q[63:] - cu_q[63]
        flash_attn_prefill(
            out=out[63:], query=q[63:], key_cache=kc, value_cache=vc,
            cu_seqlens_q=cu_pre, max_seqlen_q=2048, seqused_k=sk[63:],
            max_seqlen_k=CTX, block_table=bt[63:], softmax_scale=scale,
        )

    log(f"  flashinfer on 63 decode rows : {timeit(only_decode):7.3f} ms")
    log(f"  FA on 1 chunk q=2048         : {timeit(only_prefill):7.3f} ms")

    # The split adds two Python-level kernel launches per layer; make sure the
    # slicing itself is not secretly syncing.
    log("")
    log("== sync check on the split path ==")
    item_calls = {"n": 0}
    orig_item = torch.Tensor.item

    def counting_item(self):
        item_calls["n"] += 1
        return orig_item(self)

    only_decode()
    only_prefill()
    torch.cuda.synchronize()
    torch.Tensor.item = counting_item
    try:
        torch.cuda.set_sync_debug_mode("warn")
    except Exception:  # noqa: BLE001
        pass
    for _ in range(5):
        only_decode()
        only_prefill()
    try:
        torch.cuda.set_sync_debug_mode("default")
    except Exception:  # noqa: BLE001
        pass
    torch.Tensor.item = orig_item
    log(f"  .item() calls over 5 split invocations = {item_calls['n']}")
    log("  (a 'called a synchronizing' warning above would come from CUDA)")

    with open(OUT_PATH, "w") as f:
        f.write("\n".join(lines) + "\n")
    log("")
    log(f"Wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
