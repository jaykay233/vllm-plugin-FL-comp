"""Tune the split-K decode kernel's tile shape, warps, stages and split count.

Reference points at n=8 / ctx=17408: Triton 4.35 ms per layer call, ixformer
flashinfer 0.81 ms (wrong below 32 sequences, but it sets the bar).
"""

from __future__ import annotations

import itertools
import os

os.environ.setdefault("VLLM_PLUGINS", "fl")

import torch  # noqa: E402
import triton  # noqa: E402

from flashinfer_pad32_e2e_check import make_case, reference, timeit  # noqa: E402

HD, N_QH, N_KVH = 128, 16, 2
OUT_PATH = os.environ.get("OUT", "/root/bench_results/splitk_config_sweep.txt")


def main() -> int:
    import vllm_fl.dispatch.backends.vendor.iluvatar  # noqa: F401
    from flag_gems.runtime.backend._iluvatar.fused import paged_decode_splitk as M

    lines = []

    def log(s=""):
        print(s, flush=True)
        lines.append(s)

    scale = 1.0 / (HD ** 0.5)
    cases = [(int(a), int(b)) for a, b in
             (c.split(":") for c in os.environ.get("CASES", "8:17408,64:4096").split(","))]
    block_ns = [int(x) for x in os.environ.get("BLOCK_NS", "32,64,128,256").split(",")]
    warps = [int(x) for x in os.environ.get("WARPS", "4,8").split(",")]
    stages = [int(x) for x in os.environ.get("STAGES", "1,2,3").split(",")]
    split_opts = [int(x) for x in os.environ.get("SPLITS", "8,16,32,64").split(",")]

    for n, ctx in cases:
        q, kc, vc, bt, lens = make_case(n, ctx, seed=n * 100003 + ctx)
        ref = reference(q, kc, vc, bt, lens, scale)
        out = torch.empty_like(q)
        log(f"=== n={n} ctx={ctx}")
        results = []
        for bn, nw, ns, sp in itertools.product(block_ns, warps, stages, split_opts):
            M._BLOCK_N = bn
            try:
                def call():
                    _run(M, out, q, kc, vc, bt, lens, scale, sp, bn, nw, ns)

                out.fill_(float("nan"))
                call()
                torch.cuda.synchronize()
                e = (out.float() - ref).abs().max().item()
                if e != e or e > 0.1:
                    log(f"  BLOCK_N={bn:<4} warps={nw} stages={ns} splits={sp:<3}: WRONG err={e}")
                    continue
                t = timeit(call)
                results.append((t, bn, nw, ns, sp, e))
            except Exception as exc:  # noqa: BLE001
                log(f"  BLOCK_N={bn:<4} warps={nw} stages={ns} splits={sp:<3}: "
                    f"{type(exc).__name__}: {str(exc).splitlines()[0][:80]}")
        results.sort()
        for t, bn, nw, ns, sp, e in results[:12]:
            log(f"  BLOCK_N={bn:<4} warps={nw} stages={ns} splits={sp:<3}: {t:7.3f} ms  err={e:.3f}")
        log()
    with open(OUT_PATH, "w") as f:
        f.write("\n".join(lines) + "\n")
    return 0


def _run(M, out, q, kc, vc, bt, lens, scale, splits, bn, nw, ns):
    num_seqs, num_q_heads, head_dim = q.shape
    page_size, num_kv_heads = kc.shape[1], kc.shape[2]
    group = num_q_heads // num_kv_heads
    acc, lse_m, lse_l = M._workspace(num_seqs, num_q_heads, splits, head_dim, q.device)
    M._splitk_partial_kernel[(num_seqs, num_kv_heads, splits)](
        q, kc, vc, bt, lens, acc, lse_m, lse_l, scale,
        q.stride(0), q.stride(1),
        kc.stride(0), kc.stride(1), kc.stride(2),
        vc.stride(0), vc.stride(1), vc.stride(2),
        bt.stride(0),
        acc.stride(0), acc.stride(1), acc.stride(2),
        lse_m.stride(0), lse_m.stride(1),
        splits, bt.shape[1],
        HEAD_DIM=head_dim, PAGE_SIZE=page_size, GROUP=group,
        BLOCK_M=max(16, triton.next_power_of_2(group)), BLOCK_N=bn,
        num_warps=nw, num_stages=ns,
    )
    M._splitk_reduce_kernel[(num_seqs, num_q_heads)](
        acc, lse_m, lse_l, out,
        acc.stride(0), acc.stride(1), acc.stride(2),
        lse_m.stride(0), lse_m.stride(1),
        out.stride(0), out.stride(1),
        splits, HEAD_DIM=head_dim, BLOCK_P=triton.next_power_of_2(splits),
        num_warps=4,
    )


if __name__ == "__main__":
    raise SystemExit(main())
