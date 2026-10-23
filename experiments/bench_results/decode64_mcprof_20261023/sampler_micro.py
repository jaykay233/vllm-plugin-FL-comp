#!/usr/bin/env python3
"""Decode-step sampler cost: how much of a 25 ms TPOT goes to top-p?

Model generation_config is temperature=1.0 / top_p=0.95 and `vllm bench serve`
does not override it, so every decode step runs the full sampling chain on an
(M, 130560) logits tensor:

    bf16 logits -> logits.to(float32) -> apply_top_k_top_p -> random_sample

stock vLLM picks `apply_top_k_top_p_triton` for M>=8.  Measures each stage's
device time at the real decode shape (M=64 is what conc=64 runs at).
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

OUT = Path(__file__).resolve().parent
VOCAB = 130560


def device_us(fn, iters=30):
    import torch
    from torch.profiler import ProfilerActivity, profile

    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    by = defaultdict(float)
    for e in p.key_averages():
        t = e.self_device_time_total or 0
        if t > 0:
            by[e.key] += t / iters
    return sum(by.values()), sorted(by.items(), key=lambda kv: -kv[1])[:3]


def main() -> int:
    import torch

    # Outside `vllm serve` current_platform is Unspecified, so the Triton top-p
    # kernel cannot ask for the SM count. Point it at the real device; this is
    # exactly what MetaXPlatform.num_compute_units returns in a live server.
    import vllm.v1.sample.ops.topk_topp_triton as _ttt
    from vllm.utils import platform_utils

    _ncu = lambda device_id=0: torch.cuda.get_device_properties(  # noqa: E731
        device_id or torch.cuda.current_device()
    ).multi_processor_count
    platform_utils.num_compute_units = _ncu
    _ttt.num_compute_units = _ncu

    from vllm.v1.sample.ops.topk_topp_sampler import (
        apply_top_k_top_p,
        apply_top_k_top_p_pytorch,
    )

    rows = {}
    for M in (8, 32, 64):
        bf16 = torch.randn(M, VOCAB, device="cuda", dtype=torch.bfloat16)
        f32 = bf16.float()
        p = torch.full((M,), 0.95, device="cuda", dtype=torch.float32)

        cast_us, _ = device_us(lambda: bf16.to(torch.float32))
        topp_us, topp_top = device_us(lambda: apply_top_k_top_p(f32, None, p))
        topp_pt_us, _ = device_us(lambda: apply_top_k_top_p_pytorch(f32, None, p))
        sm_us, _ = device_us(lambda: f32.softmax(dim=-1))
        sort_us, _ = device_us(lambda: f32.sort(dim=-1, descending=False))

        def full():
            lg = bf16.to(torch.float32)
            lg = apply_top_k_top_p(lg, None, p)
            lg = lg.softmax(dim=-1)
            q = torch.empty_like(lg).exponential_()
            return lg.div(q).argmax(dim=-1)

        full_us, full_top = device_us(full)

        rows[f"M{M}"] = {
            "cast_f32_us": round(cast_us, 1),
            "topp_triton_us": round(topp_us, 1),
            "topp_pytorch_us": round(topp_pt_us, 1),
            "softmax_us": round(sm_us, 1),
            "sort_only_us": round(sort_us, 1),
            "full_chain_us": round(full_us, 1),
            "topp_top": [(k, round(v, 1)) for k, v in topp_top],
            "full_top": [(k, round(v, 1)) for k, v in full_top],
        }
        print(
            f"M={M:3d}  cast={cast_us:6.1f}  topp_triton={topp_us:7.1f}  "
            f"topp_pytorch={topp_pt_us:7.1f}  softmax={sm_us:6.1f}  "
            f"sort={sort_us:7.1f}  |  full_chain={full_us:7.1f}us",
            flush=True,
        )
        print(f"        topp top={[(k[:44], round(v, 1)) for k, v in topp_top]}", flush=True)
        print(f"        full top={[(k[:44], round(v, 1)) for k, v in full_top]}", flush=True)

    (OUT / "sampler_micro.json").write_text(json.dumps(rows, indent=2))
    print("wrote", OUT / "sampler_micro.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
