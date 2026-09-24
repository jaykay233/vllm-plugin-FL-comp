#!/usr/bin/env python3
"""Shapes of every aten::mm / addmm reached during decode.

FlagGems' MetaX mm takes the split-k path (and therefore calls c.zero_()) for
certain shapes, while it specialises only `N == 1` as GEMV.  Decode gives
`M == 1` with large `N`, so the shapes tell us whether an M==1 GEMV mapping
would avoid split-k altogether.

Run:  python /root/src/probe_mm_aten_shapes.py
"""

from __future__ import annotations

import os
from collections import Counter

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import torch
from torch.profiler import ProfilerActivity, profile
from vllm import LLM, SamplingParams


def main() -> None:
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

    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        out = llm.generate([prompt], sp, use_tqdm=False)
        torch.cuda.synchronize()
    n = len(out[0].outputs[0].token_ids)
    steps = max(n - 1, 1)

    CPU = torch.autograd.DeviceType.CPU
    c = Counter()
    for ev in prof.events():
        if ev.device_type != CPU:
            continue
        if ev.name not in ("aten::mm", "aten::addmm", "aten::linear", "aten::bmm"):
            continue
        c[f"{ev.name} {ev.input_shapes}"] += ev.count

    print(f"\nsteps={steps}", flush=True)
    for k, v in c.most_common(20):
        print(f"  {v:6d}  {v / steps:7.2f}/step   {k[:120]}", flush=True)
    print("\nMM_ATEN_SHAPES_DONE", flush=True)


if __name__ == "__main__":
    main()
