#!/usr/bin/env python3
"""Task 3 (decisive): does the OOT -> FlagGems path stay active under
torch.compile + CUDAGraph, the mode `vllm serve` actually runs in?

Offline runs with enforce_eager=True log:
    Op 'rms_norm'         using 'default.flagos'
    Op 'rotary_embedding' using 'default.flagos'
    Op 'silu_and_mul'     using 'default.flagos'
but the `vllm serve` logs only ever show attention_backend. This script lets us
switch between eager and compiled mode with everything else identical.

Run:  EAGER=1 python /root/src/oot_mode_check.py    (eager)
      EAGER=0 python /root/src/oot_mode_check.py    (compile + cudagraph)
"""

from __future__ import annotations

import logging
import os

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

EAGER = os.environ.get("EAGER", "1") == "1"
MODEL = "/root/models/MiniCPM5-2B"


def main() -> None:
    import torch  # noqa: F401
    from vllm import LLM, SamplingParams

    print(f"\n########## OOT MODE CHECK: enforce_eager={EAGER} ##########", flush=True)

    llm = LLM(
        model=MODEL,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=2048,
        gpu_memory_utilization=0.85,
        enforce_eager=EAGER,
    )
    out = llm.generate(
        ["The capital of France is"],
        SamplingParams(max_tokens=8, temperature=0.0, ignore_eos=True),
    )
    print(f"--- generated: {out[0].outputs[0].text[:60]!r}", flush=True)
    print(f"OOT_MODE_CHECK_DONE eager={EAGER}", flush=True)


if __name__ == "__main__":
    main()
