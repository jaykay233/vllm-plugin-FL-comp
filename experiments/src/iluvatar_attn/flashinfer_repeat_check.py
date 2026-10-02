"""Are flashinfer's sporadic large errors deterministic, and do they need padding?

flashinfer_pad32_e2e_check.txt shows a few cells with max error 0.5-2.4 while
Triton stays at bf16 noise.  Re-run identical inputs many times, for padded
small batches and for plain batches of >= 32 (what the engine already runs),
and report how often each call exceeds the noise floor and which rows go wrong.
"""

from __future__ import annotations

import os

os.environ.setdefault("VLLM_PLUGINS", "fl")

import torch  # noqa: E402

from flashinfer_pad32_e2e_check import BLOCK, DEV, Padded, make_case, reference  # noqa: E402

HD = 128
OUT_PATH = os.environ.get("OUT", "/root/bench_results/flashinfer_repeat_check.txt")
REPS = int(os.environ.get("REPS", "50"))
TOL = 0.1


def main() -> int:
    import vllm_fl.dispatch.backends.vendor.iluvatar  # noqa: F401
    from flag_gems.runtime.backend._iluvatar.fused.paged_attention_decode import (
        paged_attention_decode,
    )

    lines = []

    def log(s=""):
        print(s, flush=True)
        lines.append(s)

    scale = 1.0 / (HD ** 0.5)
    cases = [
        # (n, ctx, padded)
        (4, 2048, True), (2, 4096, True), (24, 2048, True), (31, 8192, True),
        (8, 17408, True),
        (32, 2048, False), (32, 4096, False), (48, 2048, False), (64, 4096, False),
        (64, 17408, False),
    ]
    log(f"reps={REPS}, bad = max abs err > {TOL}")
    for n, ctx, padded in cases:
        q, kc, vc, bt, lens = make_case(n, ctx, seed=n * 100003 + ctx)
        ref = reference(q, kc, vc, bt, lens, scale)
        out = torch.empty_like(q)
        pad = Padded(n, bt) if padded else None
        bad, bad_rows, errs = 0, {}, []
        first = None
        for _ in range(REPS):
            if padded:
                pad.load(q, bt, lens)
                paged_attention_decode(
                    out=pad.out, query=pad.q, key_cache=kc, value_cache=vc,
                    block_table=pad.bt, seqused_k=pad.lens, softmax_scale=scale,
                    max_seqlen_k=int(lens.max()),
                )
                o = pad.out[:n].float()
            else:
                paged_attention_decode(
                    out=out, query=q, key_cache=kc, value_cache=vc, block_table=bt,
                    seqused_k=lens, softmax_scale=scale, max_seqlen_k=int(lens.max()),
                )
                o = out.float()
            torch.cuda.synchronize()
            if first is None:
                first = o.clone()
            row_err = (o - ref).abs().amax(dim=(1, 2))
            errs.append(row_err.max().item())
            if row_err.max().item() > TOL:
                bad += 1
                for r in torch.nonzero(row_err > TOL).flatten().tolist():
                    bad_rows[r] = bad_rows.get(r, 0) + 1
        same = "deterministic" if all(abs(e - errs[0]) < 1e-6 for e in errs) else "varies"
        errs.sort()
        log(f"n={n:>2} ctx={ctx:>5} {'pad' if padded else 'raw'}: bad {bad}/{REPS} ({same}); "
            f"err median={errs[len(errs) // 2]:.3f} max={errs[-1]:.3f}; bad rows={dict(sorted(bad_rows.items()))}")
    with open(OUT_PATH, "w") as f:
        f.write("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
