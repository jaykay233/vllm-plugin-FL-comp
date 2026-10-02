"""Does the engine's own decode path (batch >= 32 flashinfer graphs) go wrong?

flashinfer_repeat_check.txt shows eager flashinfer calls at n=32/64 returning
O(1) errors on random rows in a fraction of identical calls.  The engine runs
batch >= 32 decode as FULL graphs captured with seq_lens=1 and max_seq_len =
max_model_len (131072), replayed at the real lengths.  Reproduce exactly that
and replay many times.
"""

from __future__ import annotations

import os

os.environ.setdefault("VLLM_PLUGINS", "fl")

import torch  # noqa: E402

from flashinfer_pad32_e2e_check import BLOCK, make_case, reference  # noqa: E402

HD = 128
OUT_PATH = os.environ.get("OUT", "/root/bench_results/flashinfer_graph_repeat_check.txt")
REPS = int(os.environ.get("REPS", "50"))
CAPTURE_MAX = int(os.environ.get("CAPTURE_MAX", "131072"))
TOL = 0.1


def main() -> int:
    import vllm_fl.dispatch.backends.vendor.iluvatar  # noqa: F401
    from flag_gems.runtime.backend._iluvatar.fused.paged_attention_decode import (
        _build_flashinfer_pages,
        paged_attention_decode,
    )

    lines = []

    def log(s=""):
        print(s, flush=True)
        lines.append(s)

    scale = 1.0 / (HD ** 0.5)
    ns = [int(x) for x in os.environ.get("NS", "32,48,64").split(",")]
    ctxs = [int(x) for x in os.environ.get("CTXS", "2048,4096,17408").split(",")]
    log(f"graph capture: seq_lens=1, max_seq_len={CAPTURE_MAX}; reps={REPS}; bad = err > {TOL}")
    for ctx in ctxs:
        for n in ns:
            q, kc, vc, bt, lens = make_case(n, ctx, seed=n * 100003 + ctx)
            ref = reference(q, kc, vc, bt, lens, scale)
            out = torch.empty_like(q)
            lens_buf = torch.ones_like(lens)
            pages = _build_flashinfer_pages(bt, lens_buf, BLOCK)

            def body():
                paged_attention_decode(
                    out=out, query=q, key_cache=kc, value_cache=vc, block_table=bt,
                    seqused_k=lens_buf, softmax_scale=scale,
                    max_seqlen_k=CAPTURE_MAX, pages=pages,
                )

            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                body()
            torch.cuda.current_stream().wait_stream(s)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                body()
            lens_buf.copy_(lens)
            for dst, src in zip(pages, _build_flashinfer_pages(bt, lens_buf, BLOCK)):
                dst.copy_(src)
            bad, errs, rows = 0, [], {}
            for _ in range(REPS):
                out.zero_()
                graph.replay()
                torch.cuda.synchronize()
                row_err = (out.float() - ref).abs().amax(dim=(1, 2))
                errs.append(row_err.max().item())
                if errs[-1] > TOL:
                    bad += 1
                    for r in torch.nonzero(row_err > TOL).flatten().tolist():
                        rows[r] = rows.get(r, 0) + 1
            errs.sort()
            log(f"n={n:>2} ctx={ctx:>5}: bad {bad}/{REPS}; err median={errs[len(errs) // 2]:.3f} "
                f"max={errs[-1]:.3f}; bad rows={dict(sorted(rows.items()))}")
            del graph
    with open(OUT_PATH, "w") as f:
        f.write("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
