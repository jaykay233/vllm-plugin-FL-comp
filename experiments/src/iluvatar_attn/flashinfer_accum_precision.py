"""Is the split-K path losing precision in accumulation rather than mis-weighting?

The shard-weight probe found no mis-weighting, but it used piecewise-constant V,
which accumulates exactly.  With random V and uniform attention the output is
just mean(V), so the kernel's answer can be compared against the same mean
accumulated at different precisions.  If the small-batch path matches a
low-precision accumulation far better than an fp32 one, the split-K workspace
(not the combine logic) is the problem.
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("VLLM_PLUGINS", "fl")
sys.path.insert(0, "/root/bench_results")

import torch  # noqa: E402

N_QH, N_KVH, HD, BLOCK = 16, 2, 128, 16
DT, DEV = torch.bfloat16, "cuda"


def call_flashinfer(q, k_hnd, v_hnd, bt, ctx):
    from ixformer.inference.functions.paged_attention import (
        paged_attention_flashinfer,
    )

    n, nb = bt.shape
    indptr = torch.arange(n + 1, dtype=torch.int32, device=DEV) * nb
    indices = bt.reshape(-1).contiguous().to(torch.int32)
    last_page_len = torch.full((n,), BLOCK, dtype=torch.int32, device=DEV)
    out = torch.empty_like(q)
    paged_attention_flashinfer(
        out, q, (k_hnd, v_hnd), indptr, indices, last_page_len,
        1.0 / (HD ** 0.5), max_seq_len=ctx, kv_cache_format="HND",
    )
    return out


def main() -> int:
    import vllm_fl.dispatch.backends.vendor.iluvatar  # noqa: F401

    lines = []

    def log(s=""):
        print(s, flush=True)
        lines.append(s)

    ctx = 4096
    nb = ctx // BLOCK
    log("q = 0 so attention is exactly uniform; correct output is mean(V).")
    log("Comparing the kernel against that mean at several accumulation widths,")
    log("and against a shard-wise fp32 mean (what a correct split-K would give).")
    log()
    log(f"{'n':>4} {'vs fp32':>10} {'vs bf16 seq':>12} {'vs fp16 seq':>12} "
        f"{'out_rms':>10} {'true_rms':>10}")

    for n in (8, 16, 31, 32, 64):
        torch.manual_seed(0)
        num_blocks = n * nb
        k_hnd = torch.randn(num_blocks, N_KVH, BLOCK, HD, dtype=DT, device=DEV) * 0.5
        v_hnd = torch.randn(num_blocks, N_KVH, BLOCK, HD, dtype=DT, device=DEV) * 0.5
        q = torch.zeros(n, N_QH, HD, dtype=DT, device=DEV)
        bt = torch.arange(num_blocks, dtype=torch.int32, device=DEV).reshape(n, nb)

        out = call_flashinfer(q, k_hnd, v_hnd, bt, ctx).float()
        torch.cuda.synchronize()

        # V laid out per sequence: [n, kv_heads, ctx, head_dim]
        v_seq = v_hnd.reshape(n, nb, N_KVH, BLOCK, HD).permute(0, 2, 1, 3, 4)
        v_seq = v_seq.reshape(n, N_KVH, ctx, HD)
        rep = N_QH // N_KVH

        ref32 = v_seq.float().mean(dim=2).repeat_interleave(rep, dim=1)

        # Sequential accumulation at reduced width, same order a naive kernel uses.
        def seq_mean(dtype):
            acc = torch.zeros(n, N_KVH, HD, dtype=dtype, device=DEV)
            for i in range(0, ctx, 256):
                acc = acc + v_seq[:, :, i : i + 256].to(dtype).sum(dim=2).to(dtype)
            return (acc.float() / ctx).repeat_interleave(rep, dim=1)

        refbf = seq_mean(torch.bfloat16)
        reffp16 = seq_mean(torch.float16)

        e32 = (out - ref32).abs().max().item()
        ebf = (out - refbf).abs().max().item()
        e16 = (out - reffp16).abs().max().item()
        log(f"{n:4d} {e32:10.2e} {ebf:12.2e} {e16:12.2e} "
            f"{out.pow(2).mean().sqrt().item():10.2e} "
            f"{ref32.pow(2).mean().sqrt().item():10.2e}")

    log()
    log("out_rms vs true_rms shows whether the kernel's answer is even the right")
    log("size; 'vs fp32' is the honest error.")

    with open("/root/bench_results/flashinfer_accum_precision.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
