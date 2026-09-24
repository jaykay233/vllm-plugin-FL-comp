#!/usr/bin/env python3
"""Attribute every `zero_persistent_kernel` launch to its CPU parent.

Facts so far:
  * GPU runs ~85 `zero_persistent_kernel` launches per decode step (~223 us/step)
  * Python-level `fill_` / `zero_` on the attention output happen ~0 times/step
  * the compiled kernels live under /tmp/torchinductor_root/triton/... with
    dtypes i32 / bf16 / i64 / f32 / i8, i.e. many distinct call sites

So the launches are produced by compiled code.  Running with CUDA graphs
disabled makes every launch eager, which lets the profiler record the CPU
parent op (`aten.zero_`, `aten::fill_`, ...) and its input shapes.

Run:  python /root/src/probe_zero_eager.py
"""

from __future__ import annotations

import os
from collections import Counter

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import torch
from torch.profiler import ProfilerActivity, profile
from vllm import LLM, SamplingParams

STEPS = 32
PROMPT = "The theory of general relativity describes gravity as"


def main() -> None:
    llm = LLM(
        model="/root/models/MiniCPM5-2B",
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=2048,
        gpu_memory_utilization=0.30,
        compilation_config={"mode": "VLLM_COMPILE", "cudagraph_mode": "NONE"},
    )
    sp = SamplingParams(max_tokens=STEPS, temperature=0.0, ignore_eos=True)
    llm.generate([PROMPT], sp, use_tqdm=False)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        out = llm.generate([PROMPT], sp, use_tqdm=False)
        torch.cuda.synchronize()
    n = len(out[0].outputs[0].token_ids)
    steps = max(n - 1, 1)

    CM = torch.autograd.DeviceType.CUDA
    CPU = torch.autograd.DeviceType.CPU
    name_of: dict[int, tuple[str, str]] = {}
    for ev in prof.events():
        if ev.device_type == CPU:
            name_of.setdefault(ev.id, (ev.name, ev.input_shapes or ""))

    z_kern = Counter()
    z_kern_us = Counter()
    parents = Counter()
    for ev in prof.events():
        if ev.device_type != CM or not ev.name:
            continue
        if "zero" not in ev.name.lower():
            continue
        z_kern[ev.name] += ev.count
        z_kern_us[ev.name] += ev.device_time
        p = getattr(ev, "cpu_parent", None)
        if p is None:
            parents["<no cpu parent>"] += ev.count
        else:
            nm, shp = name_of.get(p.id, (p.name, ""))
            parents[f"{nm}  {shp}"[:110]] += ev.count

    print(f"\nsteps={steps}", flush=True)
    print("=== zero kernels per step ===", flush=True)
    for k, v in z_kern.most_common():
        print(f"  {k[:40]:42s} {v / steps:7.2f}/step  {z_kern_us[k] / steps:8.1f} us/step",
              flush=True)
    print("=== zero kernel CPU parents (per step) ===", flush=True)
    for k, v in parents.most_common(20):
        print(f"  {v / steps:7.2f}/step  {k}", flush=True)

    # Also list the biggest CPU `zero`/`fill` ops with shapes, for cross-check.
    c_cpu = Counter()
    for ev in prof.events():
        if ev.device_type == CPU:
            ln = ev.name.lower()
            if "zero" in ln or "fill" in ln:
                c_cpu[f"{ev.name} {ev.input_shapes}"] += ev.count
    print("=== CPU zero/fill ops (total over run) ===", flush=True)
    for k, v in c_cpu.most_common(15):
        print(f"  {v:7d}  {k[:130]}", flush=True)
    print("\nZERO_EAGER_DONE", flush=True)


if __name__ == "__main__":
    main()
