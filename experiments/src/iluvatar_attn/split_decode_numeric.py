"""Who is wrong on the 16-decode case: Triton or flashinfer?

split_batch_proto reported maxdiff 4.3e-02 on the decode rows when the batch had
17 sequences, versus 2.4e-04 at 64 sequences.  bf16 rounding on values of this
magnitude is ~1e-3, so one of the two kernels is wrong, not merely imprecise.
Both are scored against an fp32 gather-and-matmul reference.
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
    """fp32 full attention per decode row, gathered from the paged cache."""
    n = q.shape[0]
    outs = []
    nb = (kv_len + BLOCK - 1) // BLOCK
    rep = N_QH // N_KVH
    for i in range(n):
        pages = bt[i, :nb].long()
        k = k_hnd[pages].permute(1, 0, 2, 3).reshape(N_KVH, nb * BLOCK, HD)
        v = v_hnd[pages].permute(1, 0, 2, 3).reshape(N_KVH, nb * BLOCK, HD)
        k = k[:, :kv_len].float()
        v = v[:, :kv_len].float()
        k = k.repeat_interleave(rep, dim=0)
        v = v.repeat_interleave(rep, dim=0)
        qi = q[i].float().unsqueeze(1)
        s = torch.matmul(qi, k.transpose(1, 2)) * scale
        p = torch.softmax(s, dim=-1)
        outs.append(torch.matmul(p, v).squeeze(1))
    return torch.stack(outs)


def main() -> int:
    import vllm_fl.dispatch.backends.vendor.iluvatar  # noqa: F401
    import vllm.v1.attention.ops.triton_unified_attention as _uam
    from flag_gems.runtime.backend._iluvatar.fused.paged_attention_decode import (
        paged_attention_decode,
    )

    triton_ua = getattr(_uam, "_orig_unified_attention", None) or _uam.unified_attention
    scale = 1.0 / (HD ** 0.5)
    print(f"heads q/kv={N_QH}/{N_KVH} head_dim={HD} page={BLOCK}")
    print(f"{'n_decode':>9} {'ctx':>6} {'flashinfer_err':>15} {'triton_err':>12}  verdict")

    for n, CTX in [(n, 4096) for n in range(1, 34)] + [
        (n, ctx) for ctx in (256, 1024, 8192) for n in (4, 16, 24, 32, 48)
    ]:
        nb = CTX // BLOCK
        torch.manual_seed(0)
        num_blocks = n * nb
        k_hnd = torch.randn(num_blocks, N_KVH, BLOCK, HD, dtype=DT, device=DEV) * 0.5
        v_hnd = torch.randn(num_blocks, N_KVH, BLOCK, HD, dtype=DT, device=DEV) * 0.5
        kc = k_hnd.permute(0, 2, 1, 3)
        vc = v_hnd.permute(0, 2, 1, 3)
        q = torch.randn(n, N_QH, HD, dtype=DT, device=DEV) * 0.5
        bt = torch.randperm(num_blocks, device=DEV).to(torch.int32).reshape(n, nb)
        sk = torch.full((n,), CTX, dtype=torch.int32, device=DEV)
        cu_q = torch.arange(n + 1, dtype=torch.int32, device=DEV)

        ref = reference(q, k_hnd, v_hnd, bt, CTX, scale)

        out_ix = torch.empty_like(q)
        paged_attention_decode(
            out=out_ix, query=q, key_cache=kc, value_cache=vc, block_table=bt,
            seqused_k=sk, softmax_scale=scale, max_seqlen_k=CTX,
        )
        out_t = torch.empty_like(q)
        triton_ua(
            q=q, k=kc, v=vc, out=out_t, cu_seqlens_q=cu_q, max_seqlen_q=1,
            seqused_k=sk, max_seqlen_k=CTX, softmax_scale=scale, causal=True,
            window_size=(-1, -1), block_table=bt, softcap=0.0,
            q_descale=None, k_descale=None, v_descale=None,
        )
        torch.cuda.synchronize()

        e_ix = (out_ix.float() - ref).abs().max().item()
        e_t = (out_t.float() - ref).abs().max().item()
        bad = "flashinfer" if e_ix > 10 * e_t else ("triton" if e_t > 10 * e_ix else "both ok")
        print(f"{n:9d} {CTX:6d} {e_ix:15.2e} {e_t:12.2e}  {bad}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
