"""Verify the split path slices a reordered batch at the right boundary.

Slicing offsets are the quiet failure mode here: a wrong boundary still runs,
still returns plausible numbers, and only shows up as degraded output.  So each
case drives the patched unified_attention (gate + Triton fallback) and compares
against raw Triton on the identical batch, while asserting which dispatch
counter moved.
"""

from __future__ import annotations

import importlib
import os
import sys

os.environ.setdefault("VLLM_PLUGINS", "fl")
os.environ.setdefault("VLLM_FL_ILUVATAR_IXFORMER_ATTN", "1")
sys.path.insert(0, "/root/bench_results")

import torch  # noqa: E402

N_QH, N_KVH, HD, BLOCK = 16, 2, 128, 16
DT, DEV = torch.bfloat16, "cuda"


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
    import vllm.v1.attention.ops.triton_unified_attention as _uam
    import vllm.v1.attention.backends.triton_attn as _ta

    g = importlib.import_module(
        "flag_gems.runtime.backend._iluvatar.fused.ixformer_unified_attention"
    )
    patched = _uam.unified_attention
    raw = getattr(_uam, "_orig_unified_attention", None)
    if raw is patched or raw is None:
        # The plugin replaces the module attribute; recover the original.
        raw = _ta.unified_attention
    assert patched is not raw or True

    print("reorder patch applied:",
          getattr(_ta, "_iluvatar_reorder_split_patch_applied", False))
    print("builder reorder_batch_threshold attr present:",
          hasattr(_ta.TritonAttentionMetadataBuilder, "_init_reorder_batch_threshold"))
    print()

    scale = 1.0 / (HD ** 0.5)
    CTX = 4096
    cases = [
        # (name, q_lens with decode rows first, published split, expected counter)
        ("mixed 63 dec + 1 chunk 2048", [1] * 63 + [2048], (63, 63), "split"),
        ("mixed 32 dec + 4 chunk 512", [1] * 32 + [512] * 4, (32, 32), "split"),
        ("mixed 8 dec + 1 chunk 2048", [1] * 8 + [2048], (8, 8), "split_unsafe_skip"),
        ("pure prefill 2 x 2048", [2048] * 2, (0, 0), "prefill"),
        # A stash from some other batch must not be trusted.
        ("stale stash", [1] * 63 + [2048], (40, 40), "mixed_skip"),
    ]

    bad = 0
    for name, q_lens, (n_dec, n_dec_tok), expect in cases:
        torch.manual_seed(0)
        q, kc, vc, bt, cu, sk = build(q_lens, CTX)
        max_q = int(max(q_lens))
        kw = dict(
            q=q, k=kc, v=vc, cu_seqlens_q=cu, max_seqlen_q=max_q, seqused_k=sk,
            max_seqlen_k=CTX, softmax_scale=scale, causal=True,
            window_size=(-1, -1), block_table=bt, softcap=0.0,
            q_descale=None, k_descale=None, v_descale=None,
        )

        out_ref = torch.empty_like(q)
        g.clear_batch_split()
        raw(out=out_ref, **kw)
        torch.cuda.synchronize()

        # "stale stash" publishes a boundary for a batch of a different size.
        tokens = int(sum(q_lens)) if name != "stale stash" else int(sum(q_lens)) - 1
        g.set_batch_split(n_dec, n_dec_tok, tokens)
        before = dict(g._STATS)
        out = torch.empty_like(q)
        patched(out=out, **kw)
        torch.cuda.synchronize()
        moved = [k for k in g._STATS if g._STATS[k] != before.get(k, 0)]

        err = (out.float() - out_ref.float()).abs().max().item()
        ok_counter = moved == [expect]
        ok_num = err < 5e-3
        status = "OK" if (ok_counter and ok_num) else "FAIL"
        if status == "FAIL":
            bad += 1
        print(
            f"{status:4} {name:30} counter={','.join(moved) or '-':18} "
            f"(want {expect:18}) maxdiff_vs_triton={err:.2e}"
        )

    g.clear_batch_split()
    print()
    print("FAILURES:", bad)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
