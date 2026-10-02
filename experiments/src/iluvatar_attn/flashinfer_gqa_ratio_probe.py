"""Which GQA group sizes make flashinfer decode race?

flashinfer_variant_probe.txt: 2 query heads over 2 KV heads (MHA ratio) never
went wrong in 160 calls, while the engine's 16:2 ratio did in every case.
Sweep the group size by taking query-head subsets, and time each call, to see
whether splitting the 16 heads into several safe calls is affordable.
"""

from __future__ import annotations

import os

os.environ.setdefault("VLLM_PLUGINS", "fl")

import torch  # noqa: E402

from flashinfer_pad32_e2e_check import BLOCK, make_case, timeit  # noqa: E402

HD, N_QH, N_KVH = 128, 16, 2
OUT_PATH = os.environ.get("OUT", "/root/bench_results/flashinfer_gqa_ratio_probe.txt")
REPS = int(os.environ.get("REPS", "60"))
TOL = 0.1


def reference(q, kc, vc, bt, lens, scale):
    """GQA reference for any query-head count divisible by N_KVH."""
    out = torch.empty(q.shape, dtype=torch.float32, device=q.device)
    rep = q.shape[1] // N_KVH
    for i in range(q.shape[0]):
        L = int(lens[i])
        nb = (L + BLOCK - 1) // BLOCK
        pages = bt[i, :nb].long()
        k = kc[pages].reshape(-1, N_KVH, HD)[:L].float().repeat_interleave(rep, dim=1)
        v = vc[pages].reshape(-1, N_KVH, HD)[:L].float().repeat_interleave(rep, dim=1)
        p = torch.softmax(torch.einsum("hd,lhd->hl", q[i].float(), k) * scale, dim=-1)
        out[i] = torch.einsum("hl,lhd->hd", p, v)
    return out


def main() -> int:
    import vllm_fl.dispatch.backends.vendor.iluvatar  # noqa: F401
    from flag_gems.runtime.backend._iluvatar.fused.paged_attention_decode import (
        _build_flashinfer_pages,
    )
    from ixformer.inference.functions.paged_attention import paged_attention_flashinfer

    lines = []

    def log(s=""):
        print(s, flush=True)
        lines.append(s)

    scale = 1.0 / (HD ** 0.5)
    log(f"reps={REPS}; bad = max abs err > {TOL}; t = ms per call")
    for n, ctx in [(32, 2048), (64, 2048), (64, 4096), (8, 17408)]:
        q, kc, vc, bt, lens = make_case(n, ctx, seed=n * 100003 + ctx)
        indptr, indices, last = _build_flashinfer_pages(bt, lens, BLOCK)
        k_hnd, v_hnd = kc.permute(0, 2, 1, 3), vc.permute(0, 2, 1, 3)
        max_len = int(lens.max())
        for group in (1, 2, 4, 8):
            # `group` query heads per KV head, taken from each KV head's block.
            idx = torch.cat([torch.arange(h * 8, h * 8 + group) for h in range(N_KVH)])
            qs = q[:, idx.to(q.device)].contiguous()
            ref = reference(qs, kc, vc, bt, lens, scale)
            out = torch.empty_like(qs)

            def call():
                paged_attention_flashinfer(out, qs, (k_hnd, v_hnd), indptr, indices, last,
                                           scale, max_seq_len=max_len, kv_cache_format="HND")

            bad, worst = 0, 0.0
            for _ in range(REPS):
                call()
                torch.cuda.synchronize()
                e = (out.float() - ref).abs().max().item()
                worst = max(worst, e)
                bad += e > TOL
            t = timeit(call)
            log(f"n={n:>2} ctx={ctx:>5} group={group}: bad {bad:>2}/{REPS} worst={worst:.3f} "
                f"t={t:.3f}  (x{8 // group} calls -> {t * 8 // group if False else t * (8 / group):.3f})")
        log()
    with open(OUT_PATH, "w") as f:
        f.write("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
