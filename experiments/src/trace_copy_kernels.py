#!/usr/bin/env python3
"""Who issues the 168 `_copy_kernel_kernel_rank_2` calls per decode step?

Kernel attribution for the MATH-500-shaped decode step shows two copy kernels
that together cost ~576 us/step (~7.5% of TPOT):

    _copy_kernel_kernel_rank_2   441 us/step   168 launches/step  (4 per layer)
    _copy_kernel_kernel_rank_3   135 us/step    42 launches/step  (1 per layer)

168 = 42 layers x 4, so these are model-internal, not graph plumbing. To find out
exactly which op, profile with CPU stack traces. Stacks are only meaningful when
kernels are launched eagerly, so cudagraph is disabled here -- the copy structure
of the forward pass is unchanged, only the launch path differs.

Run:  conda activate mx && python /root/src/trace_copy_kernels.py
"""

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

MODEL = "/root/models/MiniCPM5-2B"
PROMPT = "The theory of general relativity describes gravity as"
OUT = Path("/root/bench_results/copy_kernels")
OUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    import torch
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=MODEL,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=2048,
        gpu_memory_utilization=0.30,
        # eager: no cudagraph so kernels are launched from Python and carry stacks
        compilation_config={"cudagraph_mode": "NONE"},
    )

    sp = SamplingParams(max_tokens=32, temperature=0.0, ignore_eos=True)
    llm.generate([PROMPT], sp, use_tqdm=False)  # warmup

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CUDA,
            torch.profiler.ProfilerActivity.CPU,
        ],
        with_stack=True,
    ) as prof:
        out = llm.generate([PROMPT], sp, use_tqdm=False)
        torch.cuda.synchronize()

    print("=" * 108)
    print("COPY-KERNEL ATTRIBUTION  (eager, with CPU stacks)")
    print("=" * 108)

    # --- aggregate per kernel -------------------------------------------
    counts: Counter[str] = Counter()
    dur: Counter[str] = Counter()
    stacks: dict[str, Counter] = {}
    for ev in prof.events():
        if ev.device_type != torch.autograd.DeviceType.CUDA or not ev.name:
            continue
        if "_copy_kernel" not in ev.name:
            continue
        counts[ev.name] += ev.count
        dur[ev.name] += ev.device_time
        st = getattr(ev, "stack", None) or []
        # resolve the most interesting frames: our plugin + vllm model code
        frames = []
        for f in st:
            fn = getattr(f, "filename", "") or ""
            if ("vllm" in fn and "site-packages" in fn) or "vllm_fl" in fn:
                frames.append(f"{Path(fn).name}:{getattr(f, 'lineno', 0)}"
                              f" {getattr(f, 'name', '')}")
        key = " <- ".join(frames[-4:]) if frames else "<no stack>"
        stacks.setdefault(ev.name, Counter())[key] += ev.count

    for name, c in counts.most_common():
        print(f"\n  {name}")
        print(f"    launches={c}  total={dur[name] / 1000:.2f} ms  "
              f"per_launch={dur[name] / c:.2f} us")
        print(f"    top call sites:")
        for site, n in stacks.get(name, Counter()).most_common(6):
            print(f"      x{n:<7} {site}")

    # --- also list everything, to see what else is hot -------------------
    print()
    print("=" * 108)
    print("ALL KERNELS (eager, with per-launch cost)")
    print("=" * 108)
    allc: Counter[str] = Counter()
    alld: Counter[str] = Counter()
    for ev in prof.events():
        if ev.device_type != torch.autograd.DeviceType.CUDA or not ev.name:
            continue
        allc[ev.name] += ev.count
        alld[ev.name] += ev.device_time
    for name, d in alld.most_common(20):
        print(f"  {d / 1000:9.2f} ms  x{allc[name]:<7} "
              f"({d / allc[name]:6.2f} us/launch)  {name[:64]}")

    print("\nTRACE_COPY_KERNELS_DONE", flush=True)


if __name__ == "__main__":
    main()
