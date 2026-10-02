"""Can small-batch long-context decode use flashinfer with a clamped max_seq_len?

Below 32 sequences, ixformer's flashinfer picks a split-KV launch when
``max_seq_len >= 2048`` and that launch reads sequence 0's KV for every row
(FINDINGS §17.9).  The non-split launch was shown correct when a graph captured
at max_seq_len=1 is replayed at ctx=4096.  This checks the same idea eagerly
and in graphs, at contexts up to 32k with ragged lengths and peaked attention,
and times it against the Triton decode the engine falls back to today.

Paths compared against an fp32 reference:
  fi_split   flashinfer, max_seq_len = real max  (the known-bad launch)
  fi_clamp   flashinfer, max_seq_len = 2047      (eager)
  fi_graph   flashinfer, captured at max_seq_len = 2047 with seq_lens=1,
             replayed at the real lengths
  tri_2d     vLLM Triton unified_attention, 2D (what NUM_PAR_SOFTMAX_SEGMENTS=1
             amounts to)
  tri_3d16   vLLM Triton unified_attention, 3D with 16 segments
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("VLLM_PLUGINS", "fl")

import torch  # noqa: E402

N_QH, N_KVH, HD, BLOCK = 16, 2, 128, 16
DT, DEV = torch.bfloat16, "cuda"
CLAMP = int(os.environ.get("CLAMP", "2047"))
OUT_PATH = os.environ.get(
    "OUT", "/root/bench_results/flashinfer_clamp_check.txt"
)


def make_case(n, ctx, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    lens = ctx - torch.randint(0, max(1, ctx // 8), (n,), device=DEV, generator=g)
    lens[0] = ctx
    lens = lens.to(torch.int32)
    nb_max = (ctx + BLOCK - 1) // BLOCK
    pool = n * nb_max + 64
    # HND storage, BNDH view -- the engine's KV layout.
    k_hnd = torch.randn(pool, N_KVH, BLOCK, HD, dtype=DT, device=DEV, generator=g)
    v_hnd = torch.randn(pool, N_KVH, BLOCK, HD, dtype=DT, device=DEV, generator=g)
    perm = torch.randperm(pool, device=DEV, generator=g)[: n * nb_max]
    bt = perm.reshape(n, nb_max).to(torch.int32).contiguous()
    # Peaked softmax so a wrong-sequence read shows as O(1) error.
    q = (torch.randn(n, N_QH, HD, dtype=torch.float32, device=DEV, generator=g) * 4).to(DT)
    return q, k_hnd.permute(0, 2, 1, 3), v_hnd.permute(0, 2, 1, 3), bt, lens


def reference(q, kc, vc, bt, lens, scale):
    out = torch.empty(q.shape, dtype=torch.float32, device=DEV)
    rep = N_QH // N_KVH
    for i in range(q.shape[0]):
        L = int(lens[i])
        nb = (L + BLOCK - 1) // BLOCK
        pages = bt[i, :nb].long()
        k = kc[pages].reshape(-1, N_KVH, HD)[:L].float()  # [L, kvh, d]
        v = vc[pages].reshape(-1, N_KVH, HD)[:L].float()
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
        s = torch.einsum("hd,lhd->hl", q[i].float(), k) * scale
        p = torch.softmax(s, dim=-1)
        out[i] = torch.einsum("hl,lhd->hd", p, v)
    return out


def timeit(fn, iters=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    st, en = torch.cuda.Event(True), torch.cuda.Event(True)
    st.record()
    for _ in range(iters):
        fn()
    en.record()
    torch.cuda.synchronize()
    return st.elapsed_time(en) / iters


def main() -> int:
    import vllm_fl.dispatch.backends.vendor.iluvatar  # noqa: F401
    from flag_gems.runtime.backend._iluvatar.fused.paged_attention_decode import (
        _build_flashinfer_pages,
        paged_attention_decode,
    )
    from vllm.v1.attention.ops.triton_unified_attention import unified_attention

    lines = []

    def log(s=""):
        print(s, flush=True)
        lines.append(s)

    scale = 1.0 / (HD ** 0.5)
    ns = [int(x) for x in os.environ.get("NS", "1,2,4,6,8,16,24,31").split(",")]
    ctxs = [int(x) for x in os.environ.get("CTXS", "2048,4096,8192,17408,32768").split(",")]
    log("max abs err vs fp32 reference (bf16 noise ~1e-2); time = ms per layer call")
    log(f"{'n':>3} {'ctx':>6} | {'fi_split':>8} {'fi_clamp':>8} {'fi_graph':>8} "
        f"{'tri_2d':>8} {'tri_3d16':>8} | {'t_clamp':>7} {'t_tri2d':>7} {'t_tri3d':>7} {'x2d':>5}")
    worst = 0.0
    for ctx in ctxs:
        for n in ns:
            q, kc, vc, bt, lens = make_case(n, ctx, seed=n * 100003 + ctx)
            ref = reference(q, kc, vc, bt, lens, scale)
            out = torch.empty_like(q)
            pages = _build_flashinfer_pages(bt, lens, BLOCK)

            def fi(max_len):
                paged_attention_decode(
                    out=out, query=q, key_cache=kc, value_cache=vc,
                    block_table=bt, seqused_k=lens, softmax_scale=scale,
                    max_seqlen_k=max_len, pages=pages,
                )

            def err():
                torch.cuda.synchronize()
                return (out.float() - ref).abs().max().item()

            out.zero_(); fi(int(lens.max())); e_split = err()
            out.zero_(); fi(CLAMP); e_clamp = err()
            t_clamp = timeit(lambda: fi(CLAMP))

            # Graph: capture with seq_lens=1 like vLLM, replay at real lengths.
            lens_g = lens.clone()
            real = lens.clone()
            lens_g.fill_(1)
            pages_g = _build_flashinfer_pages(bt, lens_g, BLOCK)

            def fi_g():
                paged_attention_decode(
                    out=out, query=q, key_cache=kc, value_cache=vc,
                    block_table=bt, seqused_k=lens_g, softmax_scale=scale,
                    max_seqlen_k=CLAMP, pages=pages_g,
                )

            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                fi_g()
            torch.cuda.current_stream().wait_stream(s)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                fi_g()
            lens_g.copy_(real)
            new = _build_flashinfer_pages(bt, lens_g, BLOCK)
            for dst, src in zip(pages_g, new):
                dst.copy_(src)
            out.zero_(); graph.replay(); e_graph = err()

            cu_q = torch.arange(n + 1, dtype=torch.int32, device=DEV)
            descale = torch.ones(n, N_KVH, dtype=torch.float32, device=DEV)

            def tri(segs):
                kw = {}
                if segs > 1:
                    kw = dict(
                        seq_threshold_3D=n,
                        num_par_softmax_segments=segs,
                        softmax_segm_output=torch.empty(
                            n, N_QH, segs, HD, dtype=torch.float32, device=DEV),
                        softmax_segm_max=torch.empty(
                            n, N_QH, segs, dtype=torch.float32, device=DEV),
                        softmax_segm_expsum=torch.empty(
                            n, N_QH, segs, dtype=torch.float32, device=DEV),
                    )
                unified_attention(
                    q=q, k=kc, v=vc, out=out, cu_seqlens_q=cu_q, max_seqlen_q=1,
                    seqused_k=lens, max_seqlen_k=int(lens.max()),
                    softmax_scale=scale, causal=True, window_size=(-1, -1),
                    block_table=bt, softcap=0, q_descale=None,
                    k_descale=descale, v_descale=descale, **kw,
                )

            out.zero_(); tri(1); e_t2 = err()
            t_t2 = timeit(lambda: tri(1))
            out.zero_(); tri(16); e_t3 = err()
            t_t3 = timeit(lambda: tri(16))

            worst = max(worst, e_clamp, e_graph)
            log(f"{n:>3} {ctx:>6} | {e_split:8.3f} {e_clamp:8.3f} {e_graph:8.3f} "
                f"{e_t2:8.3f} {e_t3:8.3f} | {t_clamp:7.3f} {t_t2:7.3f} {t_t3:7.3f} "
                f"{t_t2 / t_clamp:5.1f}")
            del graph
    log()
    log(f"worst fi_clamp/fi_graph err = {worst:.4f}")
    with open(OUT_PATH, "w") as f:
        f.write("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
