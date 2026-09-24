#!/usr/bin/env python3
"""Catch the stack of every `aten.zero_` call on a [1, 2048]-shaped tensor.

Established so far:
  * eager (cudagraph_mode NONE): `aten::zero_ [[1, 2048]]` is called exactly
    84 times per decode step -> 2 per layer
  * the GPU cost is `zero_persistent_kernel` ~86.7 launches / 281 us per step
  * with CUDA graphs on, those launches are captured into the graph and replayed
    every step, yet no Python-level call happens at decode time

FlagGems installs its own kernel for `aten.zero_`, so `torch.Tensor.zero_`
patching is unreliable.  A TorchDispatchMode sits at the dispatcher and sees
every call, so it yields the real Python caller.

Run:  python /root/src/probe_zero_stack.py
"""

from __future__ import annotations

import os
import traceback
from collections import Counter

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import torch
from torch.utils._python_dispatch import TorchDispatchMode

HITS: Counter = Counter()
FIRST: dict = {}
TARGETS = {(1, 2048)}


class ZeroSpy(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        try:
            name = func._schema.name if hasattr(func, "_schema") else str(func)
        except Exception:
            name = str(func)
        if name.startswith("aten.zero_") or name == "aten.zero_.default":
            for a in args:
                if isinstance(a, torch.Tensor) and tuple(a.shape) in TARGETS:
                    st = traceback.format_stack()[:-2]
                    keep = [
                        l for l in st
                        if "/vllm/" in l or "/root/vllm-plugin-FL-comp/" in l
                        or "/flag_gems/" in l
                    ]
                    key = "".join(keep)[:900] or "<no vllm/flag_gems frames>"
                    HITS[(name, tuple(a.shape), str(a.dtype))] += 1
                    FIRST.setdefault((name, tuple(a.shape), str(a.dtype)), keep)
                    break
        return func(*args, **kwargs)


def main() -> None:
    from vllm import LLM, SamplingParams

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
    llm.generate([prompt], sp, use_tqdm=False)  # warmup / capture

    before = sum(HITS.values())
    with ZeroSpy():
        llm.generate([prompt], sp, use_tqdm=False)
        torch.cuda.synchronize()
    after = sum(HITS.values())
    print(f"\nzero_ hits during warmup={before}  during 16-step decode={after - before}",
          flush=True)

    print("=== distinct call sites ===", flush=True)
    for (name, shape, dtype), c in HITS.most_common():
        print(f"\n--- {name} {shape} {dtype}  x{c} ---", flush=True)
        print("".join(FIRST[(name, shape, dtype)][-26:]), flush=True)
    print("\nZERO_STACK_DONE", flush=True)


if __name__ == "__main__":
    main()
