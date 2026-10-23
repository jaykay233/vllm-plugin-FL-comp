#!/usr/bin/env python3
"""Decode-shape micro: norm and rope variants, counting kernels.

Baseline is what the 4k trace actually shows inside the cudagraph:

* norm: 1 aten add + 1 FlagGems ``rms_norm_kernel`` per call (the ``split``
  mode of the flagos ``fused_add_rms_norm`` provider), 2 calls per layer.
* rope: the eager ``forward_static`` path, which inductor fuses into
  ``triton_poi_fused_cat_1`` / ``_2`` (one per q and k), 2 kernels per layer.

Candidates:
* norm: ``flag_gems.fused_add_rms_norm`` (in-place, 1 kernel).
* rope: ``flag_gems.apply_rotary_pos_emb(..., inplace=True)`` (1 kernel, q+k).

The eager rope is measured through ``torch.compile`` so it is compared in the
same fused form production uses, not as 5 unfused aten kernels.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

OUT = Path(__file__).resolve().parent
M, HIDDEN, NLAYERS = 64, 2048, 42
NH, NKV, HD = 16, 2, 128


def dev_us(fn, warmup=15, iters=50):
    import torch
    from torch.profiler import ProfilerActivity, profile

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    by = defaultdict(float)
    cnt = defaultdict(int)
    for e in p.key_averages():
        t = e.self_device_time_total or 0
        if t > 0:
            by[e.key] += t / iters
            cnt[e.key] += max(1, round(e.count / iters))
    n = sum(cnt.values())
    top = sorted(by.items(), key=lambda kv: -kv[1])[:3]
    return sum(by.values()), n, top


def main() -> int:
    import torch
    import flag_gems
    from flag_gems.fused.rotary_embedding import apply_rotary_pos_emb as gems_rope

    rows = {}

    def report(tag, fn):
        us, n, top = dev_us(fn)
        rows[tag] = {"us": round(us, 2), "kernels": n, "top": [(a[:50], round(b, 2)) for a, b in top]}
        print(f"{tag:38s} {us:8.2f} us  {n:2d} kern  top={[a[:34] for a, _ in top]}", flush=True)
        return us

    x = torch.randn(M, HIDDEN, device="cuda", dtype=torch.bfloat16)
    r = torch.randn(M, HIDDEN, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(HIDDEN, device="cuda", dtype=torch.bfloat16)
    eps = 1e-6

    print("=== add + rms_norm (2 calls per layer) ===", flush=True)
    split = report(
        "A split  add + gems rms_norm",
        lambda: flag_gems.rms_norm(x + r, [HIDDEN], w, eps),
    )
    fused = report(
        "B inplace gems fused_add_rms_norm",
        lambda: flag_gems.fused_add_rms_norm(x, r, [HIDDEN], w, eps),
    )

    print("\n=== rope (1 call per layer, q+k) ===", flush=True)
    pos = torch.arange(M, device="cuda", dtype=torch.int64) + 1000
    cache = torch.randn(131072, HD, device="cuda", dtype=torch.bfloat16)
    q0 = torch.randn(M, NH, HD, device="cuda", dtype=torch.bfloat16)
    k0 = torch.randn(M, NKV, HD, device="cuda", dtype=torch.bfloat16)

    def eager_rope(q, k):
        from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb

        cos_sin = cache.index_select(0, pos)
        cos, sin = cos_sin.chunk(2, dim=-1)
        qs = ApplyRotaryEmb.forward_static(q, cos, sin, True)
        ks = ApplyRotaryEmb.forward_static(k, cos, sin, True)
        return qs, ks

    compiled = torch.compile(eager_rope, fullgraph=True, dynamic=False)
    compiled(q0, k0)
    eager_us = report("C eager rope under inductor", lambda: compiled(q0, k0))
    gems_us = report(
        "D gems rope inplace (q+k, 1 kern)",
        lambda: gems_rope(
            q0, k0, cache[:, : HD // 2], cache[:, HD // 2 :], pos, False, True
        ),
    )

    per = {
        "norm_current_ms": round(split * 2 * NLAYERS / 1000, 3),
        "norm_fused_ms": round(fused * 2 * NLAYERS / 1000, 3),
        "rope_current_ms": round(eager_us * NLAYERS / 1000, 3),
        "rope_fused_ms": round(gems_us * NLAYERS / 1000, 3),
    }
    per["saving_ms"] = round(
        per["norm_current_ms"] + per["rope_current_ms"] - per["norm_fused_ms"] - per["rope_fused_ms"], 3
    )
    per["saving_pct_of_itl_18_45ms"] = round(100 * per["saving_ms"] / 18.45, 2)
    rows["per_step"] = per
    print("\n=== per decode step (42 layers) ===", flush=True)
    for kk, v in per.items():
        print(f"  {kk:34s} {v}", flush=True)

    # correctness of the fused rope vs the eager path
    qa, ka = q0.clone(), k0.clone()
    qe, ke = eager_rope(qa, ka)
    qg, kg = q0.clone(), k0.clone()
    gems_rope(qg, kg, cache[:, : HD // 2], cache[:, HD // 2 :], pos, False, True)
    rows["rope_max_abs_diff"] = {
        "q": float((qe - qg).abs().max()),
        "k": float((ke - kg).abs().max()),
        "q_ref_absmax": float(qe.abs().max()),
    }
    print("\nrope max|diff| vs eager:", rows["rope_max_abs_diff"], flush=True)

    (OUT / "norm_rope_micro.json").write_text(json.dumps(rows, indent=2))
    print("wrote", OUT / "norm_rope_micro.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
