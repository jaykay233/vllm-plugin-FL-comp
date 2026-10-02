"""What makes flashinfer decode go wrong sporadically?

Hypotheses probed, each on identical inputs repeated REPS times:
  * uninitialized workspace: poison the caching allocator's free blocks with
    NaN (or zeros) right before every call;
  * ragged lengths / empty splits: all rows at the same length vs ragged;
  * split count: max_seq_len hint = real max, 2x, 131072.
"""

from __future__ import annotations

import os

os.environ.setdefault("VLLM_PLUGINS", "fl")

import torch  # noqa: E402

from flashinfer_pad32_e2e_check import BLOCK, make_case, reference  # noqa: E402

HD = 128
OUT_PATH = os.environ.get("OUT", "/root/bench_results/flashinfer_race_probe.txt")
REPS = int(os.environ.get("REPS", "40"))
TOL = 0.1


def poison(value):
    x = torch.empty(256 * 1024 * 1024, dtype=torch.float32, device="cuda")
    x.fill_(value)
    del x


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
    log(f"reps={REPS}; bad = max abs err > {TOL}")

    def trial(n, ctx, ragged=True, hint="real", pre=None, prebuilt=False, label=""):
        q, kc, vc, bt, lens = make_case(n, ctx, seed=n * 100003 + ctx)
        if not ragged:
            lens.fill_(ctx)
        ref = reference(q, kc, vc, bt, lens, scale)
        out = torch.empty_like(q)
        max_len = {"real": int(lens.max()), "2x": 2 * int(lens.max()), "131072": 131072}[hint]
        pages = _build_flashinfer_pages(bt, lens, BLOCK) if prebuilt else None
        bad, worst = 0, 0.0
        for _ in range(REPS):
            if pre is not None:
                poison(pre)
            out.fill_(float("nan"))
            paged_attention_decode(
                out=out, query=q, key_cache=kc, value_cache=vc, block_table=bt,
                seqused_k=lens, softmax_scale=scale, max_seqlen_k=max_len, pages=pages,
            )
            torch.cuda.synchronize()
            e = (out.float() - ref).abs().max().item()
            e = float("inf") if e != e else e
            worst = max(worst, e)
            bad += e > TOL
        log(f"{label:<44} n={n:>2} ctx={ctx:>5}: bad {bad:>2}/{REPS} worst={worst:.3f}")

    for n, ctx in [(32, 2048), (64, 2048), (32, 4096)]:
        trial(n, ctx, label="baseline (ragged, hint=real)")
        trial(n, ctx, prebuilt=True, label="pages prebuilt once")
        trial(n, ctx, pre=float("nan"), label="free memory poisoned with NaN")
        trial(n, ctx, pre=0.0, label="free memory zeroed")
        trial(n, ctx, ragged=False, label="uniform lengths")
        trial(n, ctx, hint="2x", label="hint = 2x max")
        trial(n, ctx, hint="131072", label="hint = 131072")
        log()
    with open(OUT_PATH, "w") as f:
        f.write("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
