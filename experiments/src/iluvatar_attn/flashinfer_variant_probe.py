"""Is there a flashinfer calling convention that does not race?

flashinfer_race_probe.txt: the sporadic errors are timing-dependent (more when
another kernel runs right before), never NaN even with NaN-poisoned free
memory, and independent of ragged lengths or the max_seq_len hint -- an
intra-kernel race rather than uninitialized reads.  Try other entry shapes:
  * HND (engine default) -- baseline
  * one call per KV head (8 query heads : 1 KV head each)
  * MHA ratio control: 2 query heads : 2 KV heads
  * NHD format on the same storage
"""

from __future__ import annotations

import os

os.environ.setdefault("VLLM_PLUGINS", "fl")

import torch  # noqa: E402

from flashinfer_pad32_e2e_check import BLOCK, DT, make_case, reference  # noqa: E402

HD, N_QH, N_KVH = 128, 16, 2
OUT_PATH = os.environ.get("OUT", "/root/bench_results/flashinfer_variant_probe.txt")
REPS = int(os.environ.get("REPS", "40"))
TOL = 0.1


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
    log(f"reps={REPS}; bad = max abs err > {TOL}")

    for n, ctx in [(32, 2048), (64, 2048), (32, 4096), (64, 4096)]:
        q, kc, vc, bt, lens = make_case(n, ctx, seed=n * 100003 + ctx)
        ref = reference(q, kc, vc, bt, lens, scale)
        indptr, indices, last = _build_flashinfer_pages(bt, lens, BLOCK)
        k_hnd = kc.permute(0, 2, 1, 3)  # [pages, kvh, page, d], contiguous storage
        v_hnd = vc.permute(0, 2, 1, 3)
        max_len = int(lens.max())

        def run(fn, label):
            bad, worst = 0, 0.0
            for _ in range(REPS):
                o = fn()
                torch.cuda.synchronize()
                e = (o.float() - ref).abs().max().item()
                e = float("inf") if e != e else e
                worst = max(worst, e)
                bad += e > TOL
            log(f"{label:<34} n={n:>2} ctx={ctx:>5}: bad {bad:>2}/{REPS} worst={worst:.3f}")

        out = torch.empty_like(q)

        def hnd():
            paged_attention_flashinfer(out, q, (k_hnd, v_hnd), indptr, indices, last,
                                       scale, max_seq_len=max_len, kv_cache_format="HND")
            return out

        def per_kv_head():
            rep = N_QH // N_KVH
            for h in range(N_KVH):
                qh = q[:, h * rep:(h + 1) * rep].contiguous()
                oh = torch.empty_like(qh)
                paged_attention_flashinfer(
                    oh, qh, (k_hnd[:, h:h + 1], v_hnd[:, h:h + 1]), indptr, indices,
                    last, scale, max_seq_len=max_len, kv_cache_format="HND")
                out[:, h * rep:(h + 1) * rep] = oh
            return out

        def nhd():
            paged_attention_flashinfer(out, q, (kc, vc), indptr, indices, last,
                                       scale, max_seq_len=max_len, kv_cache_format="NHD")
            return out

        run(hnd, "HND (engine)")
        try:
            run(per_kv_head, "per KV head (8:1 GQA, 2 calls)")
        except Exception as e:  # noqa: BLE001
            log(f"per KV head: {type(e).__name__}: {e}")
        try:
            run(nhd, "NHD on HND storage")
        except Exception as e:  # noqa: BLE001
            log(f"NHD: {type(e).__name__}: {e}")

        # MHA control: 2 query heads, one per KV head.
        q2 = q[:, ::N_QH // N_KVH].contiguous()
        out2 = torch.empty_like(q2)
        rep_ref = reference_mha(q2, kc, vc, bt, lens, scale)

        bad, worst = 0, 0.0
        for _ in range(REPS):
            paged_attention_flashinfer(out2, q2, (k_hnd, v_hnd), indptr, indices, last,
                                       scale, max_seq_len=max_len, kv_cache_format="HND")
            torch.cuda.synchronize()
            e = (out2.float() - rep_ref).abs().max().item()
            worst = max(worst, e)
            bad += e > TOL
        log(f"{'MHA control (2:2)':<34} n={n:>2} ctx={ctx:>5}: bad {bad:>2}/{REPS} worst={worst:.3f}")
        log()
    with open(OUT_PATH, "w") as f:
        f.write("\n".join(lines) + "\n")
    return 0


def reference_mha(q, kc, vc, bt, lens, scale):
    out = torch.empty(q.shape, dtype=torch.float32, device=q.device)
    for i in range(q.shape[0]):
        L = int(lens[i])
        nb = (L + BLOCK - 1) // BLOCK
        pages = bt[i, :nb].long()
        k = kc[pages].reshape(-1, N_KVH, HD)[:L].float()
        v = vc[pages].reshape(-1, N_KVH, HD)[:L].float()
        p = torch.softmax(torch.einsum("hd,lhd->hl", q[i].float(), k) * scale, dim=-1)
        out[i] = torch.einsum("hl,lhd->hd", p, v)
    return out


if __name__ == "__main__":
    raise SystemExit(main())
