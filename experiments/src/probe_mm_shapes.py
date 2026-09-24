#!/usr/bin/env python3
"""Log the (M, N, K) of every FlagGems MetaX `mm` call during decode.

The zero_ overhead is a symptom of the split-k path in
`flag_gems/runtime/backend/_metax/ops/mm.py`.  The same path is also the single
most expensive kernel in the decode budget (mm_kernel_splitk ~1692 us/step,
i.e. ~20 us per M=1 call), so the shapes matter: FlagGems specialises `N == 1`
as GEMV, but decode produces `M == 1` with a large `N`, which lands in the
general split-k path instead.

Run:  python /root/src/probe_mm_shapes.py
"""

from __future__ import annotations

import importlib
import os
from collections import Counter

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import torch
from torch.profiler import ProfilerActivity, profile
from vllm import LLM, SamplingParams

CALLS: Counter = Counter()
PATHS: Counter = Counter()


def patch() -> None:
    m = importlib.import_module("flag_gems.runtime.backend._metax.ops.mm")

    orig_mm = m.mm
    orig_mm_out = m.mm_out

    def _record(a, b, tag):
        M, K = a.shape
        _, N = b.shape
        CALLS[(tag, M, N, K, str(a.dtype))] += 1

    def mm(a, b):
        _record(a, b, "mm")
        return orig_mm(a, b)

    def mm_out(a, b, *, out):
        _record(a, b, "mm_out")
        return orig_mm_out(a, b, out=out)

    m.mm = mm
    m.mm_out = mm_out
    print("[patch] metax mm/mm_out wrapped", flush=True)

    # also record which scenario each call selects
    from flag_gems.runtime.backend._metax.ops import mm as mmmod  # noqa: F401
    for nm in ("splitk_mm_scenario", "nt_mm_scenario", "nn_mm_scenario",
               "general_mm"):
        pass


def main() -> None:
    patch()
    llm = LLM(
        model="/root/models/MiniCPM5-2B",
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=2048,
        gpu_memory_utilization=0.30,
        compilation_config={"mode": "VLLM_COMPILE", "cudagraph_mode": "NONE"},
    )
    sp = SamplingParams(max_tokens=16, temperature=0.0, ignore_eos=True)
    prompt = "The theory of general relativity describes gravity as"
    llm.generate([prompt], sp, use_tqdm=False)

    base = dict(CALLS)
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        out = llm.generate([prompt], sp, use_tqdm=False)
        torch.cuda.synchronize()
    n = len(out[0].outputs[0].token_ids)
    steps = max(n - 1, 1)

    print(f"\nsteps={steps}", flush=True)
    print(f"  {'tag':8s} {'M':>4s} {'N':>7s} {'K':>7s} {'dtype':>8s} "
          f"{'calls':>7s} {'per_step':>9s}", flush=True)
    for (tag, M, N, K, dt), c in CALLS.most_common(20):
        print(f"  {tag:8s} {M:4d} {N:7d} {K:7d} {dt:>8s} {c:7d} {c / steps:9.2f}",
              flush=True)

    CM = torch.autograd.DeviceType.CUDA
    kern = Counter()
    for ev in prof.events():
        if ev.device_type == CM and ev.name and "mm_kernel" in ev.name:
            kern[ev.name] += ev.count
    print("\n=== mm kernel counts per step ===", flush=True)
    for k, v in kern.most_common():
        print(f"  {k[:44]:46s} {v / steps:8.2f}/step", flush=True)
    print("\nMM_SHAPES_DONE", flush=True)


if __name__ == "__main__":
    main()
