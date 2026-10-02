"""Pick the split count for the shapes the split-K kernel will actually serve.

It only runs where flashinfer is refused -- fewer than 32 sequences with a
context of at least 2048 (FINDINGS §17.9) -- so sweep that regime and compare
against the Triton decode it replaces.
"""

from __future__ import annotations

import os

os.environ.setdefault("VLLM_PLUGINS", "fl")

import torch  # noqa: E402
import triton  # noqa: E402

from flashinfer_pad32_e2e_check import make_case, reference, timeit  # noqa: E402
from splitk_config_sweep import _run  # noqa: E402

HD, N_QH, N_KVH = 128, 16, 2
OUT_PATH = os.environ.get("OUT", "/root/bench_results/splitk_heuristic_sweep.txt")


def main() -> int:
    import vllm_fl.dispatch.backends.vendor.iluvatar  # noqa: F401
    from flag_gems.runtime.backend._iluvatar.fused import paged_decode_splitk as M
    from vllm.v1.attention.ops.triton_unified_attention import unified_attention

    lines = []

    def log(s=""):
        print(s, flush=True)
        lines.append(s)

    scale = 1.0 / (HD ** 0.5)
    ns = [int(x) for x in os.environ.get("NS", "1,2,4,8,16,24,31").split(",")]
    ctxs = [int(x) for x in os.environ.get("CTXS", "2048,4096,8192,17408,32768").split(",")]
    split_opts = [int(x) for x in os.environ.get("SPLITS", "1,2,4,8,16,32,64").split(",")]
    log("ms per layer call; BLOCK_N=32 warps=4 stages=1; heur = choose_num_splits()")
    header = f"{'n':>3} {'ctx':>6} {'triton':>7} | " + " ".join(f"{s:>6}" for s in split_opts)
    log(header + f" | {'best':>6} {'heur':>5} {'gain':>6}")
    for ctx in ctxs:
        for n in ns:
            q, kc, vc, bt, lens = make_case(n, ctx, seed=n * 100003 + ctx)
            ref = reference(q, kc, vc, bt, lens, scale)
            out = torch.empty_like(q)
            cu_q = torch.arange(n + 1, dtype=torch.int32, device=q.device)
            descale = torch.ones(n, N_KVH, dtype=torch.float32, device=q.device)

            def tri():
                unified_attention(
                    q=q, k=kc, v=vc, out=out, cu_seqlens_q=cu_q, max_seqlen_q=1,
                    seqused_k=lens, max_seqlen_k=int(lens.max()), softmax_scale=scale,
                    causal=True, window_size=(-1, -1), block_table=bt, softcap=0,
                    q_descale=None, k_descale=descale, v_descale=descale)

            t_tri = timeit(tri)
            row, best, best_sp = [], None, None
            for sp in split_opts:
                try:
                    def call():
                        _run(M, out, q, kc, vc, bt, lens, scale, sp, 32, 4, 1)

                    out.fill_(float("nan"))
                    call()
                    torch.cuda.synchronize()
                    e = (out.float() - ref).abs().max().item()
                    if e != e or e > 0.1:
                        row.append("  WRONG")
                        continue
                    t = timeit(call)
                    row.append(f"{t:6.3f}")
                    if best is None or t < best:
                        best, best_sp = t, sp
                except Exception:  # noqa: BLE001
                    row.append("   oom")
            heur = M.choose_num_splits(n, N_KVH)
            log(f"{n:>3} {ctx:>6} {t_tri:7.3f} | " + " ".join(row) +
                f" | {best_sp:>6} {heur:>5} {t_tri / best:5.1f}x")
        log()
    with open(OUT_PATH, "w") as f:
        f.write("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
