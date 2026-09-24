#!/usr/bin/env python3
"""Dump the dynamo graph with source locations so the `zero_`/`zeros_like`
node that fires 2x per layer can be named.

The profiler showed:

    aten::zero_   84.0/step   shapes=[1, 2048]
    zero_persistent_kernel   271.0 us/step   (3.2 us/launch -- pure overhead)

[1, 2048] is exactly the hidden state, so something allocates+zeroes an
activation twice per layer.  Reasoning from the vLLM source did not find it, so
print the traced graph: `TORCH_LOGS=graph_code` annotates every FX node with
`# File: ...:NN in <func>`, which names the caller outright.

Run:
    TORCH_LOGS=graph_code python /root/src/trace_zero_graph.py 2>&1 | tee out.log
    grep -n -B4 "zero" out.log
"""

from __future__ import annotations

import os

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

MODEL = "/root/models/MiniCPM5-2B"
PROMPT = "The theory of general relativity describes gravity as"


def main() -> None:
    import torch
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=MODEL,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=2048,
        gpu_memory_utilization=0.30,
        compilation_config={"mode": "VLLM_COMPILE", "cudagraph_mode": "NONE"},
    )
    sp = SamplingParams(max_tokens=8, temperature=0.0, ignore_eos=True)
    llm.generate([PROMPT], sp, use_tqdm=False)

    # Also print the graph right before inductor's pre-fusion lowering, which
    # keeps the source locations and shows how zero_ was folded.
    try:
        from torch._inductor import config as icfg
        print(f"[probe] inductor trace enabled={icfg.trace.enabled}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[probe] inductor config unavailable: {e}", flush=True)

    print("\nTRACE_ZERO_GRAPH_DONE", flush=True)


if __name__ == "__main__":
    main()
