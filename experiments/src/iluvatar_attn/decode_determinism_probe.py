"""Are the sporadic decode errors nondeterminism, or a wrong fp32 reference?

ixformer_decode_kernel_probe.txt shows the Triton fallback disagreeing with the
fp32 reference on some repeats too, which the earlier runs did not.  Compare
each call's output against the *first* call's output on identical inputs: no
reference involved, so any difference is the kernel (or the machine) being
nondeterministic.  Also report the reference gap for context.
"""

from __future__ import annotations

import os

os.environ.setdefault("VLLM_PLUGINS", "fl")

import torch  # noqa: E402

from flashinfer_pad32_e2e_check import BLOCK, make_case, reference  # noqa: E402

HD, N_QH, N_KVH = 128, 16, 2
OUT_PATH = os.environ.get("OUT", "/root/bench_results/decode_determinism_probe.txt")
REPS = int(os.environ.get("REPS", "40"))


def main() -> int:
    import vllm_fl.dispatch.backends.vendor.iluvatar  # noqa: F401
    from flag_gems.runtime.backend._iluvatar.fused.paged_attention_decode import (
        _build_flashinfer_pages,
    )
    from ixformer.inference.functions.paged_attention import paged_attention_flashinfer
    from vllm.v1.attention.ops.triton_unified_attention import unified_attention

    lines = []

    def log(s=""):
        print(s, flush=True)
        lines.append(s)

    scale = 1.0 / (HD ** 0.5)
    log(f"reps={REPS}; self = max |out_i - out_0| over repeats; ref = max |out - fp32 ref|")
    log(f"{'case':<22} {'kernel':<12} {'self_max':>9} {'n_differ':>9} {'ref_max':>9}")
    for n, ctx in [(8, 17408), (32, 2048), (32, 4096), (64, 2048), (64, 4096)]:
        q, kc, vc, bt, lens = make_case(n, ctx, seed=n * 100003 + ctx)
        ref = reference(q, kc, vc, bt, lens, scale)
        out = torch.empty_like(q)
        k_hnd, v_hnd = kc.permute(0, 2, 1, 3), vc.permute(0, 2, 1, 3)
        indptr, indices, last = _build_flashinfer_pages(bt, lens, BLOCK)
        max_len = int(lens.max())
        cu_q = torch.arange(n + 1, dtype=torch.int32, device=q.device)
        descale = torch.ones(n, N_KVH, dtype=torch.float32, device=q.device)

        kernels = {
            "flashinfer": lambda: paged_attention_flashinfer(
                out, q, (k_hnd, v_hnd), indptr, indices, last, scale,
                max_seq_len=max_len, kv_cache_format="HND"),
            "triton_2d": lambda: unified_attention(
                q=q, k=kc, v=vc, out=out, cu_seqlens_q=cu_q, max_seqlen_q=1,
                seqused_k=lens, max_seqlen_k=max_len, softmax_scale=scale,
                causal=True, window_size=(-1, -1), block_table=bt, softcap=0,
                q_descale=None, k_descale=descale, v_descale=descale),
        }
        prefill_modes = {
            "nan": lambda: out.fill_(float("nan")),
            "zero": lambda: out.zero_(),
            "none": lambda: None,
        }
        for name, fn in kernels.items():
          for pname, pre in prefill_modes.items():
            first, self_max, differ, ref_max = None, 0.0, 0, 0.0
            for _ in range(REPS):
                pre()
                fn()
                torch.cuda.synchronize()
                cur = out.float().clone()
                ref_max = max(ref_max, (cur - ref).abs().max().item())
                if first is None:
                    first = cur
                else:
                    d = (cur - first).abs().max().item()
                    self_max = max(self_max, d)
                    differ += d > 1e-6
            log(f"n={n:<3} ctx={ctx:<7}  {name:<12} pre={pname:<5} {self_max:9.3f} "
                f"{differ:>6}/{REPS - 1} {ref_max:9.3f}")
        log()
    with open(OUT_PATH, "w") as f:
        f.write("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
