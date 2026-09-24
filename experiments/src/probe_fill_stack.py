#!/usr/bin/env python3
"""Who calls fill_(0) on the hidden state (84x per forward = 2 per layer)?

`zero_persistent_kernel` is 84x/step with 221 us/step, and the CPU op behind it
is `aten::fill_` (896 calls total ~= 84 per model forward).  The dynamo graph
has no zero/fill node, so the call is introduced after dynamo.

The IR provider is a prime suspect: `fused_add_rms_norm` is traced as the
`.maybe_inplace` overload, and the functionalization/lowering path may allocate
and zero a fresh activation buffer.  Patch Tensor.fill_/zero_ at the Python
level and report the distinct stacks for 2048-wide tensors.

Run:  python /root/src/probe_fill_stack.py
"""

from __future__ import annotations

import os
import traceback
from collections import Counter

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import torch

HITS: Counter = Counter()
FIRST: dict = {}
TARGET_NUMEL = 2048


def _patch(name: str) -> None:
    orig = getattr(torch.Tensor, name)

    def wrapped(self, *args, **kwargs):
        try:
            numel = self.numel()
        except Exception:
            numel = -1
        if numel == TARGET_NUMEL:
            st = traceback.format_stack()[:-1]
            key = "".join(
                l.strip() for l in st if "site-packages/vllm" in l or "/root/" in l
            )
            key = key[:600] or "<no vllm/root frames>"
            HITS[(name, str(tuple(self.shape)), key)] += 1
            FIRST.setdefault(key, st)
        return orig(self, *args, **kwargs)

    setattr(torch.Tensor, name, wrapped)


def main() -> None:
    _patch("fill_")
    _patch("zero_")

    from vllm import LLM, SamplingParams

    print(f"### fill_/zero_ stack probe (numel=={TARGET_NUMEL}) ###", flush=True)
    llm = LLM(
        model="/root/models/MiniCPM5-2B",
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=2048,
        gpu_memory_utilization=0.30,
    )
    sp = SamplingParams(max_tokens=64, temperature=0.0, ignore_eos=True)
    llm.generate(["The theory of general relativity describes gravity as"], sp,
                 use_tqdm=False)

    print(f"total distinct keys: {len(HITS)}", flush=True)
    for (name, shape, key), c in HITS.most_common(6):
        print(f"\n=== {name} {shape}  x{c} ===", flush=True)
        print("\n".join(FIRST[key][-14:]), flush=True)
    print("\nFILL_STACK_DONE", flush=True)


if __name__ == "__main__":
    main()
