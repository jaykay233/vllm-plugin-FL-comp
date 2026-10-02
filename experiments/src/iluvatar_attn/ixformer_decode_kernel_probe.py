"""Is there an ixformer decode kernel that is both correct and fast for 16:2 GQA?

paged_attention_flashinfer races whenever a KV head serves more than one query
head (flashinfer_gqa_ratio_probe.txt) and is simply wrong below 32 sequences
(FINDINGS §17.9).  ixformer ships three other paged decode entry points that
take block tables directly:

    paged_attention            (HND/NHD/STD, algo switch)
    paged_attention_v7         (HND-shaped cache)
    vllm_paged_attention       (HND-shaped cache, int64 block tables)

Correctness over repeated identical calls, plus time against the Triton decode
the engine falls back to today.
"""

from __future__ import annotations

import os

os.environ.setdefault("VLLM_PLUGINS", "fl")

import torch  # noqa: E402

from flashinfer_pad32_e2e_check import BLOCK, make_case, reference, timeit  # noqa: E402

HD, N_QH, N_KVH = 128, 16, 2
OUT_PATH = os.environ.get("OUT", "/root/bench_results/ixformer_decode_kernel_probe.txt")
REPS = int(os.environ.get("REPS", "40"))
TOL = 0.1


def main() -> int:
    import vllm_fl.dispatch.backends.vendor.iluvatar  # noqa: F401
    import ixformer.inference.functions as F
    from vllm.v1.attention.ops.triton_unified_attention import unified_attention

    lines = []

    def log(s=""):
        print(s, flush=True)
        lines.append(s)

    scale = 1.0 / (HD ** 0.5)
    cases = [(int(a), int(b)) for a, b in
             (c.split(":") for c in os.environ.get(
                 "CASES", "8:17408,32:2048,32:4096,64:2048,64:4096,64:17408").split(","))]
    log(f"reps={REPS}; bad = max abs err > {TOL}; t = ms per layer call")
    for n, ctx in cases:
        q, kc, vc, bt, lens = make_case(n, ctx, seed=n * 100003 + ctx)
        ref = reference(q, kc, vc, bt, lens, scale)
        out = torch.empty_like(q)
        # kc/vc are BNDH views of HND storage; the kernels below want HND.
        k_hnd, v_hnd = kc.permute(0, 2, 1, 3), vc.permute(0, 2, 1, 3)
        max_len = int(lens.max())
        bt64 = bt.to(torch.int64)

        variants = {
            "paged_attention(algo=-1)": lambda: F.paged_attention(
                out, q, k_hnd, v_hnd, N_KVH, scale, bt, lens, BLOCK, max_len,
                kv_cache_format="HND"),
            "paged_attention(algo=0)": lambda: F.paged_attention(
                out, q, k_hnd, v_hnd, N_KVH, scale, bt, lens, BLOCK, max_len,
                kv_cache_format="HND", algo=0),
            "paged_attention(algo=1)": lambda: F.paged_attention(
                out, q, k_hnd, v_hnd, N_KVH, scale, bt, lens, BLOCK, max_len,
                kv_cache_format="HND", algo=1),
            "paged_attention_v7": lambda: F.paged_attention_v7(
                out, q, k_hnd, v_hnd, N_KVH, scale, bt, lens, BLOCK, max_len),
            "vllm_paged_attention": lambda: F.vllm_paged_attention(
                out, q, k_hnd, v_hnd, N_KVH, scale, bt64, lens, BLOCK, max_len),
        }

        cu_q = torch.arange(n + 1, dtype=torch.int32, device=q.device)
        descale = torch.ones(n, N_KVH, dtype=torch.float32, device=q.device)
        variants["triton_2d (current fallback)"] = lambda: unified_attention(
            q=q, k=kc, v=vc, out=out, cu_seqlens_q=cu_q, max_seqlen_q=1,
            seqused_k=lens, max_seqlen_k=max_len, softmax_scale=scale, causal=True,
            window_size=(-1, -1), block_table=bt, softcap=0, q_descale=None,
            k_descale=descale, v_descale=descale)

        for name, fn in variants.items():
            try:
                bad, worst = 0, 0.0
                for _ in range(REPS):
                    out.fill_(float("nan"))
                    fn()
                    torch.cuda.synchronize()
                    e = (out.float() - ref).abs().max().item()
                    e = float("inf") if e != e else e
                    worst = max(worst, e)
                    bad += e > TOL
                t = timeit(fn)
                log(f"n={n:>2} ctx={ctx:>5} {name:<28}: bad {bad:>2}/{REPS} "
                    f"worst={worst:8.3f} t={t:7.3f}")
            except Exception as e:  # noqa: BLE001
                msg = str(e).split("\n")[0][:110]
                log(f"n={n:>2} ctx={ctx:>5} {name:<28}: {type(e).__name__}: {msg}")
        log()
    with open(OUT_PATH, "w") as f:
        f.write("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
