"""Map where ixformer paged_attention_flashinfer is numerically trustworthy.

Both kernels are scored against the same fp32 reference; what matters is not the
absolute error but whether flashinfer is far worse than Triton on the identical
input.  A cell is BAD when flashinfer's error exceeds 10x Triton's, which is well
outside anything bf16 accumulation order can explain.
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
    n = q.shape[0]
    nb = (kv_len + BLOCK - 1) // BLOCK
    rep = N_QH // N_KVH
    outs = []
    for i in range(n):
        pages = bt[i, :nb].long()
        k = k_hnd[pages].permute(1, 0, 2, 3).reshape(N_KVH, nb * BLOCK, HD)[:, :kv_len]
        v = v_hnd[pages].permute(1, 0, 2, 3).reshape(N_KVH, nb * BLOCK, HD)[:, :kv_len]
        k = k.float().repeat_interleave(rep, dim=0)
        v = v.float().repeat_interleave(rep, dim=0)
        s = torch.matmul(q[i].float().unsqueeze(1), k.transpose(1, 2)) * scale
        outs.append(torch.matmul(torch.softmax(s, -1), v).squeeze(1))
    return torch.stack(outs)


def main() -> int:
    import vllm_fl.dispatch.backends.vendor.iluvatar  # noqa: F401
    import vllm.v1.attention.ops.triton_unified_attention as _uam
    from flag_gems.runtime.backend._iluvatar.fused.paged_attention_decode import (
        paged_attention_decode,
    )

    triton_ua = getattr(_uam, "_orig_unified_attention", None) or _uam.unified_attention
    scale = 1.0 / (HD ** 0.5)

    ns = [1, 2, 4, 8, 16, 24, 31, 32, 40, 48, 64, 96, 128]
    ctxs = [256, 512, 1024, 2048, 4096, 8192]

    print("flashinfer decode vs Triton, same fp32 reference.")
    print("'.' = flashinfer within 10x of Triton, 'X' = worse (ratio shown below)")
    print()
    header = "  n\\ctx " + "".join(f"{c:>8}" for c in ctxs)
    print(header)

    ratios: dict[tuple[int, int], float] = {}
    for n in ns:
        row = f"{n:>6} "
        for ctx in ctxs:
            torch.manual_seed(0)
            nb = ctx // BLOCK
            num_blocks = n * nb
            k_hnd = torch.randn(num_blocks, N_KVH, BLOCK, HD, dtype=DT, device=DEV) * 0.5
            v_hnd = torch.randn(num_blocks, N_KVH, BLOCK, HD, dtype=DT, device=DEV) * 0.5
            kc, vc = k_hnd.permute(0, 2, 1, 3), v_hnd.permute(0, 2, 1, 3)
            q = torch.randn(n, N_QH, HD, dtype=DT, device=DEV) * 0.5
            bt = torch.randperm(num_blocks, device=DEV).to(torch.int32).reshape(n, nb)
            sk = torch.full((n,), ctx, dtype=torch.int32, device=DEV)
            cu_q = torch.arange(n + 1, dtype=torch.int32, device=DEV)

            ref = reference(q, k_hnd, v_hnd, bt, ctx, scale)
            o_ix = torch.empty_like(q)
            paged_attention_decode(
                out=o_ix, query=q, key_cache=kc, value_cache=vc, block_table=bt,
                seqused_k=sk, softmax_scale=scale, max_seqlen_k=ctx,
            )
            o_t = torch.empty_like(q)
            triton_ua(
                q=q, k=kc, v=vc, out=o_t, cu_seqlens_q=cu_q, max_seqlen_q=1,
                seqused_k=sk, max_seqlen_k=ctx, softmax_scale=scale, causal=True,
                window_size=(-1, -1), block_table=bt, softcap=0.0,
                q_descale=None, k_descale=None, v_descale=None,
            )
            torch.cuda.synchronize()
            e_ix = (o_ix.float() - ref).abs().max().item()
            e_t = (o_t.float() - ref).abs().max().item()
            r = e_ix / max(e_t, 1e-12)
            ratios[(n, ctx)] = r
            row += f"{('X' if r > 10 else '.'):>8}"
            del k_hnd, v_hnd, kc, vc, q, bt, ref, o_ix, o_t
            torch.cuda.empty_cache()
        print(row, flush=True)

    print()
    print("ratio (flashinfer_err / triton_err):")
    print(header)
    for n in ns:
        print(f"{n:>6} " + "".join(f"{ratios[(n, c)]:>8.0f}" for c in ctxs))

    bad = sorted(k for k, v in ratios.items() if v > 10)
    print()
    print(f"bad cells: {len(bad)} / {len(ratios)}")
    if bad:
        bad_n = sorted({n for n, _ in bad})
        bad_ctx = sorted({c for _, c in bad})
        print(f"  n values hit  : {bad_n}")
        print(f"  ctx values hit: {bad_ctx}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
