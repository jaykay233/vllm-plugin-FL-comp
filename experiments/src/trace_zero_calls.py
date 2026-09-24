#!/usr/bin/env python3
"""Who zeroes a tensor 2x per layer at decode?

The compile-mode kernel budget contains

    zero_persistent_kernel   223 us/step   x 84.7/step   (2 per layer)

and the kernel's ttir names its own source:

    #loc = loc(".../flag_gems/runtime/backend/_metax/ops/zero.py":50:0)

so it is not an inductor glue kernel at all -- it is FlagGems' MetaX zero
override, which intercepts every `tensor.zero_()` in the process.  This patches
`_launch_zero` to record each call's shape/dtype and python stack, which names
the caller.

Run:  conda activate mx && python /root/src/trace_zero_calls.py
"""

from __future__ import annotations

import os
import traceback
from collections import Counter
from pathlib import Path

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

MODEL = "/root/models/MiniCPM5-2B"
PROMPT = "The theory of general relativity describes gravity as"
OUT = Path("/root/bench_results/glue_kernels")
OUT.mkdir(parents=True, exist_ok=True)

shapes: Counter = Counter()
stacks: Counter = Counter()
n_calls = [0]


def install_probe() -> None:
    # NOTE: `from ...ops import zero` yields the *function* `zero` (exported by
    # the package __init__), not the submodule -- importlib is required here.
    import importlib

    zmod = importlib.import_module("flag_gems.runtime.backend._metax.ops.zero")

    orig = zmod._launch_zero

    def patched(tensor):
        n_calls[0] += 1
        shapes[(tuple(tensor.shape), str(tensor.dtype), str(tensor.device))] += 1
        frames = []
        for f in traceback.extract_stack()[:-1]:
            fn = f.filename or ""
            if ("vllm" in fn or "vllm_fl" in fn or "_inductor" in fn
                    or "flag_gems" in fn or "modeling_" in fn):
                frames.append(f"{Path(fn).name}:{f.lineno}:{f.name}")
        stacks[" <- ".join(frames[-8:]) or "<none>"] += 1
        return orig(tensor)

    zmod._launch_zero = patched
    print("probe installed on flag_gems..._metax.ops.zero._launch_zero", flush=True)


def main() -> None:
    install_probe()

    import torch
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=MODEL,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=2048,
        gpu_memory_utilization=0.30,
        # compile generates the glue kernels; no cudagraph so the calls are
        # issued from python on every step and carry usable stacks
        compilation_config={"mode": "VLLM_COMPILE", "cudagraph_mode": "NONE"},
    )

    sp = SamplingParams(max_tokens=32, temperature=0.0, ignore_eos=True)
    llm.generate([PROMPT], sp, use_tqdm=False)  # warmup

    shapes.clear()
    stacks.clear()
    n_calls[0] = 0
    llm.generate([PROMPT], sp, use_tqdm=False)
    torch.cuda.synchronize()
    steps = 32

    print("=" * 108)
    print(f"zero_ CALLS: {n_calls[0]} total, {n_calls[0] / steps:.1f} per step")
    print("=" * 108)
    for (shp, dt, dev), k in shapes.most_common(10):
        numel = 1
        for d in shp:
            numel *= d
        print(f"  x{k:<6} ({k / steps:6.1f}/step)  shape={shp} dtype={dt} "
              f"numel={numel} bytes={numel * (4 if '32' in dt else 2)}")
    print()
    print("  -- call sites --")
    for site, k in stacks.most_common(8):
        print(f"    x{k:<6} {site}")

    # ---- cross-check via CPU events (in case the patch misses a path) ----
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU],
                 record_shapes=True) as prof:
        llm.generate([PROMPT], sp, use_tqdm=False)
        torch.cuda.synchronize()

    CPU = torch.autograd.DeviceType.CPU
    CUDA = torch.autograd.DeviceType.CUDA
    zc = Counter()
    zshapes = Counter()
    zcu = Counter()
    zcu_us = Counter()
    for ev in prof.events():
        if ev.device_type == CPU and ev.name in ("aten::zero_", "aten::zeros"):
            zc[ev.name] += 1
            zshapes[(ev.name, " ; ".join(str(s) for s in (getattr(ev, "input_shapes", None) or [])))] += 1
        if ev.device_type == CUDA and ev.name and "zero" in ev.name.lower():
            zcu[ev.name] += ev.count
            zcu_us[ev.name] += ev.device_time

    print()
    print("=" * 108)
    print("CROSS-CHECK: aten::zero_ events and zero kernels")
    print("=" * 108)
    for name, k in zc.most_common():
        print(f"  {name:14s} {k:6d} total  {k / steps:6.2f}/step")
    for (name, shp), k in zshapes.most_common(8):
        print(f"    x{k:<6} ({k / steps:6.1f}/step)  {name}  shapes={shp}")
    print()
    for name, k in zcu.most_common():
        print(f"  {name[:50]:52s} {k:6d} total  {k / steps:6.1f}/step  "
              f"{zcu_us[name] / steps:8.1f} us/step")

    print("\nTRACE_ZERO_DONE", flush=True)


if __name__ == "__main__":
    main()
