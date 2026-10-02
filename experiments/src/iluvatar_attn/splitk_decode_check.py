"""Correctness and speed of the self-written split-K paged decode kernel.

Compared against an fp32 reference and against the two kernels the engine has
today: ixformer flashinfer (unusable below 32 sequences) and vLLM's Triton
decode (correct everywhere, but one program per sequence and KV head).

Covers eager calls and the way vLLM captures FULL decode graphs -- seq_lens
filled to 1 at capture, real lengths at replay -- since the split count is
frozen into the graph while the lengths are not.
"""

from __future__ import annotations

import os

os.environ.setdefault("VLLM_PLUGINS", "fl")

import torch  # noqa: E402

from flashinfer_pad32_e2e_check import BLOCK, make_case, reference, timeit  # noqa: E402

HD, N_QH, N_KVH = 128, 16, 2
OUT_PATH = os.environ.get("OUT", "/root/bench_results/splitk_decode_check.txt")


def main() -> int:
    import vllm_fl.dispatch.backends.vendor.iluvatar  # noqa: F401
    from flag_gems.runtime.backend._iluvatar.fused.paged_decode_splitk import (
        choose_num_splits,
        paged_decode_splitk,
    )
    from flag_gems.runtime.backend._iluvatar.fused.paged_attention_decode import (
        paged_attention_decode,
    )
    from vllm.v1.attention.ops.triton_unified_attention import unified_attention

    lines = []

    def log(s=""):
        print(s, flush=True)
        lines.append(s)

    scale = 1.0 / (HD ** 0.5)
    cases = [(int(a), int(b)) for a, b in
             (c.split(":") for c in os.environ.get(
                 "CASES",
                 "1:17408,4:17408,8:17408,8:4096,16:17408,24:17408,31:17408,"
                 "32:4096,64:4096,64:17408,64:2048").split(","))]
    log("err = max abs vs fp32 reference (bf16 noise ~1e-2); t = ms per layer call")
    log(f"{'n':>3} {'ctx':>6} {'spl':>4} | {'splitk':>8} {'triton':>8} {'flashinf':>8} | "
        f"{'t_splitk':>8} {'t_graph':>8} {'t_triton':>8} {'t_fi':>7} | {'vs_triton':>9}")
    worst = 0.0
    for n, ctx in cases:
        q, kc, vc, bt, lens = make_case(n, ctx, seed=n * 100003 + ctx)
        ref = reference(q, kc, vc, bt, lens, scale)
        out = torch.empty_like(q)
        splits = choose_num_splits(n, N_KVH)

        def err():
            torch.cuda.synchronize()
            return (out.float() - ref).abs().max().item()

        out.fill_(float("nan"))
        paged_decode_splitk(out, q, kc, vc, bt, lens, scale)
        e_sk = err()
        t_sk = timeit(lambda: paged_decode_splitk(out, q, kc, vc, bt, lens, scale))

        # Graph: split count frozen at capture with seq_lens = 1.
        lens_buf = torch.ones_like(lens)
        out_g = torch.empty_like(q)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            paged_decode_splitk(out_g, q, kc, vc, bt, lens_buf, scale)
        torch.cuda.current_stream().wait_stream(s)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            paged_decode_splitk(out_g, q, kc, vc, bt, lens_buf, scale)
        lens_buf.copy_(lens)
        out_g.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        e_graph = (out_g.float() - ref).abs().max().item()
        t_graph = timeit(graph.replay)

        cu_q = torch.arange(n + 1, dtype=torch.int32, device=q.device)
        descale = torch.ones(n, N_KVH, dtype=torch.float32, device=q.device)

        def tri():
            unified_attention(
                q=q, k=kc, v=vc, out=out, cu_seqlens_q=cu_q, max_seqlen_q=1,
                seqused_k=lens, max_seqlen_k=int(lens.max()), softmax_scale=scale,
                causal=True, window_size=(-1, -1), block_table=bt, softcap=0,
                q_descale=None, k_descale=descale, v_descale=descale)

        out.fill_(float("nan"))
        tri()
        e_tri = err()
        t_tri = timeit(tri)

        def fi():
            paged_attention_decode(
                out=out, query=q, key_cache=kc, value_cache=vc, block_table=bt,
                seqused_k=lens, softmax_scale=scale, max_seqlen_k=int(lens.max()))

        out.fill_(float("nan"))
        fi()
        e_fi = err()
        t_fi = timeit(fi)

        worst = max(worst, e_sk, e_graph)
        log(f"{n:>3} {ctx:>6} {splits:>4} | {e_sk:8.3f} {e_tri:8.3f} {e_fi:8.3f} | "
            f"{t_sk:8.3f} {t_graph:8.3f} {t_tri:8.3f} {t_fi:7.3f} | {t_tri / t_graph:8.2f}x")
        del graph
    log()
    log(f"worst split-K err (eager + graph) = {worst:.4f}")
    with open(OUT_PATH, "w") as f:
        f.write("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
