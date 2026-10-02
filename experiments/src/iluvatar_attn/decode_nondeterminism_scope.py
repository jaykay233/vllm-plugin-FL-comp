"""How much of the decode nondeterminism is real, and does it matter?

decode_determinism_probe.txt: at >= 32 sequences both flashinfer and the Triton
fallback return different outputs for identical inputs, which a deterministic
kernel cannot do -- so either the kernels race, or the probe amplifies tiny
differences.  The probe uses q scaled by 4, which makes softmax nearly one-hot:
a near-tie between the top two keys turns a 1-ulp difference into an O(1) output
change.

Separate the two.  For each q scale, report
  self_max   max |out_i - out_0| over repeats
  self_mean  mean |out_i - out_0|
  rows       how many of the n rows ever differ
  ref_max    max |out - fp32 reference|
A racing kernel moves many rows a little; near-tie amplification moves one or
two rows a lot while the mean stays at noise.
"""

from __future__ import annotations

import os

os.environ.setdefault("VLLM_PLUGINS", "fl")

import torch  # noqa: E402

from flashinfer_pad32_e2e_check import BLOCK, DT, DEV, reference  # noqa: E402

HD, N_QH, N_KVH = 128, 16, 2
OUT_PATH = os.environ.get("OUT", "/root/bench_results/decode_nondeterminism_scope.txt")
REPS = int(os.environ.get("REPS", "30"))


def make_case(n, ctx, seed, qscale):
    g = torch.Generator(device=DEV).manual_seed(seed)
    lens = ctx - torch.randint(0, max(1, ctx // 8), (n,), device=DEV, generator=g)
    lens[0] = ctx
    lens = lens.to(torch.int32)
    nb_max = (ctx + BLOCK - 1) // BLOCK
    pool = n * nb_max + 64
    k = torch.randn(pool, N_KVH, BLOCK, HD, dtype=DT, device=DEV, generator=g)
    v = torch.randn(pool, N_KVH, BLOCK, HD, dtype=DT, device=DEV, generator=g)
    perm = torch.randperm(pool, device=DEV, generator=g)[: n * nb_max]
    bt = perm.reshape(n, nb_max).to(torch.int32).contiguous()
    q = (torch.randn(n, N_QH, HD, dtype=torch.float32, device=DEV, generator=g) * qscale).to(DT)
    return q, k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3), bt, lens


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
    log(f"reps={REPS}")
    log(f"{'case':<26} {'kernel':<11} {'qscale':>6} {'self_max':>9} {'self_mean':>10} "
        f"{'rows':>7} {'ref_max':>9}")
    for n, ctx in [(32, 2048), (64, 4096), (8, 17408)]:
        for qscale in (1.0, 4.0):
            q, kc, vc, bt, lens = make_case(n, ctx, n * 100003 + ctx, qscale)
            ref = reference(q, kc, vc, bt, lens, scale)
            out = torch.empty_like(q)
            k_hnd, v_hnd = kc.permute(0, 2, 1, 3), vc.permute(0, 2, 1, 3)
            indptr, indices, last = _build_flashinfer_pages(bt, lens, BLOCK)
            max_len = int(lens.max())
            cu_q = torch.arange(n + 1, dtype=torch.int32, device=DEV)
            descale = torch.ones(n, N_KVH, dtype=torch.float32, device=DEV)
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
            for name, fn in kernels.items():
                first, self_max, self_sum, cnt, ref_max = None, 0.0, 0.0, 0, 0.0
                rows = torch.zeros(n, dtype=torch.bool, device=DEV)
                for _ in range(REPS):
                    fn()
                    torch.cuda.synchronize()
                    cur = out.float().clone()
                    ref_max = max(ref_max, (cur - ref).abs().max().item())
                    if first is None:
                        first = cur
                        continue
                    d = (cur - first).abs()
                    self_max = max(self_max, d.max().item())
                    self_sum += d.mean().item()
                    cnt += 1
                    rows |= d.amax(dim=(1, 2)) > 1e-3
                log(f"n={n:<3} ctx={ctx:<6}        {name:<11} {qscale:>6.1f} {self_max:9.3f} "
                    f"{self_sum / max(cnt, 1):10.5f} {int(rows.sum()):>3}/{n:<3} {ref_max:9.3f}")
        log()
    with open(OUT_PATH, "w") as f:
        f.write("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
