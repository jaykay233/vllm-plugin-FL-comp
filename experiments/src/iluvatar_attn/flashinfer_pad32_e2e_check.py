"""Small-batch decode on flashinfer by padding the batch to 32 rows.

Below 32 sequences flashinfer's split-KV launch reads sequence 0's KV for every
row (FINDINGS §17.9), so the engine sends those batches to Triton, which with
NUM_PAR_SOFTMAX_SEGMENTS=1 walks the whole context in one program per head
(~220 ms/step at 17k ctx, §17.15).  Clamping max_seq_len to stay on the
non-split launch is not an option: it returns garbage once the real context
exceeds the clamp (flashinfer_clamp_check.txt).

Rows of a varlen batch are independent, so pad to 32 with 1-token dummy rows
and discard their outputs.  Checked eagerly (max_seq_len = real max) and the
way vLLM captures FULL decode graphs (seq_lens filled to 1, max_seq_len =
max_model_len), against an fp32 reference, with ragged lengths and peaked
attention.  Timed against the Triton decode the engine uses today.
"""

from __future__ import annotations

import os

os.environ.setdefault("VLLM_PLUGINS", "fl")

import torch  # noqa: E402

N_QH, N_KVH, HD, BLOCK = 16, 2, 128, 16
DT, DEV = torch.bfloat16, "cuda"
PAD_TO = 32
CAPTURE_MAX = 131072
OUT_PATH = os.environ.get("OUT", "/root/bench_results/flashinfer_pad32_e2e_check.txt")


def make_case(n, ctx, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    lens = ctx - torch.randint(0, max(1, ctx // 8), (n,), device=DEV, generator=g)
    lens[0] = ctx
    lens = lens.to(torch.int32)
    nb_max = (ctx + BLOCK - 1) // BLOCK
    pool = n * nb_max + 64
    k_hnd = torch.randn(pool, N_KVH, BLOCK, HD, dtype=DT, device=DEV, generator=g)
    v_hnd = torch.randn(pool, N_KVH, BLOCK, HD, dtype=DT, device=DEV, generator=g)
    perm = torch.randperm(pool, device=DEV, generator=g)[: n * nb_max]
    bt = perm.reshape(n, nb_max).to(torch.int32).contiguous()
    q = (torch.randn(n, N_QH, HD, dtype=torch.float32, device=DEV, generator=g) * 4).to(DT)
    return q, k_hnd.permute(0, 2, 1, 3), v_hnd.permute(0, 2, 1, 3), bt, lens


def reference(q, kc, vc, bt, lens, scale):
    out = torch.empty(q.shape, dtype=torch.float32, device=DEV)
    rep = N_QH // N_KVH
    for i in range(q.shape[0]):
        L = int(lens[i])
        nb = (L + BLOCK - 1) // BLOCK
        pages = bt[i, :nb].long()
        k = kc[pages].reshape(-1, N_KVH, HD)[:L].float().repeat_interleave(rep, dim=1)
        v = vc[pages].reshape(-1, N_KVH, HD)[:L].float().repeat_interleave(rep, dim=1)
        p = torch.softmax(torch.einsum("hd,lhd->hl", q[i].float(), k) * scale, dim=-1)
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


class Padded:
    """Static 32-row buffers, as a graph-safe wrapper would keep them."""

    def __init__(self, n, bt):
        self.n = n
        cols = bt.shape[1]
        self.q = torch.zeros(PAD_TO, N_QH, HD, dtype=DT, device=DEV)
        self.out = torch.empty_like(self.q)
        self.bt = torch.zeros(PAD_TO, cols, dtype=torch.int32, device=DEV)
        self.lens = torch.ones(PAD_TO, dtype=torch.int32, device=DEV)

    def load(self, q, bt, lens):
        n = self.n
        self.q[:n].copy_(q)
        self.bt[:n].copy_(bt)
        # Dummy rows read one token of a page the real batch already owns.
        self.bt[n:, 0] = bt[0, 0]
        self.lens[:n].copy_(lens)
        self.lens[n:] = 1


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
    log("max abs err vs fp32 reference (bf16 noise ~1e-2..1e-1 with peaked q)")
    log("time = ms per layer call; pad times include the copies in/out of the 32-row buffers")
    log(f"{'n':>3} {'ctx':>6} | {'fi_raw':>7} {'pad_eag':>7} {'pad_gr':>7} {'tri_2d':>7} | "
        f"{'t_pad':>6} {'t_padgr':>7} {'t_tri':>6} {'speedup':>7}")
    worst = 0.0
    for ctx in ctxs:
        for n in ns:
            q, kc, vc, bt, lens = make_case(n, ctx, seed=n * 100003 + ctx)
            ref = reference(q, kc, vc, bt, lens, scale)
            out = torch.empty_like(q)

            def err(o):
                torch.cuda.synchronize()
                return (o[:n].float() - ref).abs().max().item()

            # Unpadded flashinfer at the real max: the known-bad launch.
            paged_attention_decode(
                out=out, query=q, key_cache=kc, value_cache=vc, block_table=bt,
                seqused_k=lens, softmax_scale=scale, max_seqlen_k=int(lens.max()),
            )
            e_raw = err(out)

            pad = Padded(n, bt)

            def run_pad(max_len, lens_buf=None):
                paged_attention_decode(
                    out=pad.out, query=pad.q, key_cache=kc, value_cache=vc,
                    block_table=pad.bt, seqused_k=pad.lens if lens_buf is None else lens_buf,
                    softmax_scale=scale, max_seqlen_k=max_len,
                )

            def eager_step():
                pad.load(q, bt, lens)
                run_pad(int(lens.max()))
                out.copy_(pad.out[:n])

            out.zero_()
            eager_step()
            e_pad = err(out)
            t_pad = timeit(eager_step)

            # Graph, captured the way vLLM does it.
            pad.load(q, bt, lens)
            real_lens = pad.lens.clone()
            pad.lens[:n] = 1
            pages = _build_flashinfer_pages(pad.bt, pad.lens, BLOCK)

            def graph_body():
                paged_attention_decode(
                    out=pad.out, query=pad.q, key_cache=kc, value_cache=vc,
                    block_table=pad.bt, seqused_k=pad.lens, softmax_scale=scale,
                    max_seqlen_k=CAPTURE_MAX, pages=pages,
                )
                out.copy_(pad.out[:n])

            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                graph_body()
            torch.cuda.current_stream().wait_stream(s)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                graph_body()
            pad.lens.copy_(real_lens)
            for dst, src in zip(pages, _build_flashinfer_pages(pad.bt, pad.lens, BLOCK)):
                dst.copy_(src)
            out.zero_()
            graph.replay()
            e_gr = err(out)
            t_gr = timeit(graph.replay)

            cu_q = torch.arange(n + 1, dtype=torch.int32, device=DEV)
            descale = torch.ones(n, N_KVH, dtype=torch.float32, device=DEV)

            def tri():
                unified_attention(
                    q=q, k=kc, v=vc, out=out, cu_seqlens_q=cu_q, max_seqlen_q=1,
                    seqused_k=lens, max_seqlen_k=int(lens.max()),
                    softmax_scale=scale, causal=True, window_size=(-1, -1),
                    block_table=bt, softcap=0, q_descale=None,
                    k_descale=descale, v_descale=descale,
                )

            out.zero_()
            tri()
            e_tri = err(out)
            t_tri = timeit(tri)

            worst = max(worst, e_pad, e_gr)
            log(f"{n:>3} {ctx:>6} | {e_raw:7.3f} {e_pad:7.3f} {e_gr:7.3f} {e_tri:7.3f} | "
                f"{t_pad:6.3f} {t_gr:7.3f} {t_tri:6.3f} {t_tri / t_gr:6.1f}x")
            del graph
    log()
    log(f"worst padded err (eager + graph) = {worst:.4f}")
    with open(OUT_PATH, "w") as f:
        f.write("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
