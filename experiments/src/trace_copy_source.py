#!/usr/bin/env python3
"""Name the op that issues `_copy_kernel_kernel_rank_{2,3}`.

Kernel attribution (eager, with_stack) gave exact counts but the MACA profiler
returned `<no stack>` for every CUDA kernel, so the source op stayed unknown:

    _copy_kernel_kernel_rank_2   168/step  (4 per layer)  2.92 us/launch
    _copy_kernel_kernel_rank_3    42/step  (1 per layer)  3.90 us/launch

CPU-side FunctionEvents *do* carry python stacks on this backend, so profile the
CPU + CUDA activity together, find the aten ops whose per-step call count matches
168 / 42, and print their stacks with the torch/_inductor/vllm frames highlighted.

Run:  conda activate mx && python /root/src/trace_copy_source.py
"""

from __future__ import annotations

import os
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

MODEL = "/root/models/MiniCPM5-2B"
PROMPT = "The theory of general relativity describes gravity as"
OUT = Path("/root/bench_results/copy_kernels")
OUT.mkdir(parents=True, exist_ok=True)
STEPS = 16

# op names that can lower to a device-side copy/pad
COPY_LIKE = (
    "copy_", "clone", "contiguous", "_to_copy", "to_", "resize",
    "cat", "slice", "split", "view", "reshape", "expand", "narrow",
    "zero_", "fill_", "pad", "new_empty", "detach", "transpose", "permute",
)


def interesting(fn: str) -> bool:
    """Frames worth showing: inductor codegen, the plugin, vLLM model code."""
    if not fn:
        return False
    if "_inductor" in fn or "inductor" in fn:
        return True
    if "vllm_fl" in fn:
        return True
    if "/vllm/" in fn and "site-packages" in fn:
        return True
    if "/models/" in fn or "modeling_" in fn:
        return True
    return False


def main() -> None:
    import torch
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=MODEL,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=2048,
        gpu_memory_utilization=0.30,
        # eager launches: otherwise the copies happen at cudagraph capture time,
        # outside the profiled window, and the CPU side shows only ~17 ops/step
        # while the GPU side replays 168 kernels/step.
        compilation_config={"cudagraph_mode": "NONE"},
    )
    sp = SamplingParams(max_tokens=STEPS, temperature=0.0, ignore_eos=True)
    llm.generate([PROMPT], sp, use_tqdm=False)  # warmup

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        with_stack=True,
        record_shapes=True,
    ) as prof:
        llm.generate([PROMPT], sp, use_tqdm=False)
        torch.cuda.synchronize()

    CUDA = torch.autograd.DeviceType.CUDA
    CPU = torch.autograd.DeviceType.CPU

    # ---------------------------------------------------------------- CUDA
    # MACA does not record stacks, but each kernel event carries `cpu_parent`:
    # the CPU op that launched it.  That is the attribution we need.
    cu = Counter()
    cu_dur = Counter()
    parents: dict[str, Counter] = {}
    for ev in prof.events():
        if ev.device_type != CUDA or not ev.name or "_copy_kernel" not in ev.name:
            continue
        cu[ev.name] += ev.count
        cu_dur[ev.name] += ev.device_time
        chain = []
        p = getattr(ev, "cpu_parent", None)
        while p is not None and len(chain) < 6:
            chain.append(p.name)
            p = getattr(p, "cpu_parent", None)
        parents.setdefault(ev.name, Counter())[" <- ".join(chain) or "<none>"] += 1

    print("=" * 110)
    print(f"CUDA copy kernels for {STEPS} output tokens")
    print("=" * 110)
    for n, c in cu.most_common():
        print(f"  {n:34s} total={c:<6} per_step={c / STEPS:6.1f} "
              f"per_launch={cu_dur[n] / c:5.2f}us  "
              f"per_step_us={cu_dur[n] / STEPS:7.1f}")
        for chain, k in parents.get(n, Counter()).most_common(6):
            print(f"      launched by x{k:<5} {chain}")

    # ----------------------------------------------------------------- CPU
    cpus = Counter()
    cpu_stacks = defaultdict(Counter)
    for ev in prof.events():
        if ev.device_type != CPU or not ev.name:
            continue
        nm = ev.name
        cpus[nm] += 1
        if any(k in nm for k in COPY_LIKE):
            stack = getattr(ev, "stack", None) or []
            frames = []
            for f in stack:
                fn = getattr(f, "filename", "") or ""
                if interesting(fn):
                    frames.append(f"{Path(fn).name}:{getattr(f, 'lineno', 0)}"
                                  f":{getattr(f, 'name', '')}")
            if frames:
                cpu_stacks[nm][" <- ".join(frames[-5:])] += 1

    print()
    print("=" * 110)
    print("TOP CPU OPS  (aten/inductor, with per-step counts)")
    print("=" * 110)
    top = [(n, c) for n, c in cpus.most_common() if "::" in n][:45]
    for n, c in top:
        flag = "  <<< COPY-LIKE" if any(k in n for k in COPY_LIKE) else ""
        print(f"  {c:7d}  {c / STEPS:8.2f}/step   {n[:72]}{flag}")

    print()
    print("=" * 110)
    print("STACKS FOR COPY-LIKE CPU OPS  (inductor/vllm frames highlighted)")
    print("=" * 110)
    # order by per-step count so the 168/42 suspects come first
    for n, c in sorted(cpus.items(), key=lambda kv: -kv[1]):
        if not any(k in n for k in COPY_LIKE):
            continue
        st = cpu_stacks.get(n)
        if not st:
            continue
        print(f"\n  {n}   x{c} ({c / STEPS:.1f}/step)")
        for site, k in st.most_common(5):
            print(f"    x{k:<6} {site}")

    print()
    print("=" * 110)
    print("SHAPES OF THE COPY SOURCES  (what is being cloned, and how big)")
    print("=" * 110)
    shp: dict[str, Counter] = defaultdict(Counter)
    for ev in prof.events():
        if ev.device_type != CPU or not ev.name:
            continue
        if ev.name not in ("aten::clone", "aten::mm", "aten::copy_",
                           "vllm::unified_attention_with_output"):
            continue
        shapes = getattr(ev, "input_shapes", None) or []
        key = " ; ".join(str(s) for s in shapes)
        shp[ev.name][key] += 1
    for nm in ("aten::clone", "aten::mm", "aten::copy_",
               "vllm::unified_attention_with_output"):
        ent = shp.get(nm)
        if not ent:
            continue
        print(f"\n  {nm}")
        for key, k in ent.most_common(8):
            print(f"    x{k:<6} ({k / STEPS:.1f}/step)  {key[:100]}")

    print("\nTRACE_COPY_SOURCE_DONE", flush=True)


if __name__ == "__main__":
    main()
