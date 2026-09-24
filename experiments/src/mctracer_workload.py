#!/usr/bin/env python3
"""Controlled workload for mcTracer: offline MiniCPM5-2B inference.

Kept deliberately small so the mcTracer trace stays parseable.

MODE=eager   -> enforce_eager=True  (plugin -> FlagGems path is active)
MODE=compile -> default vLLM compile + cudagraph (custom_ops=['none'], FlagGems bypassed)

Run:  MODE=eager python /root/src/mctracer_workload.py
      mcTracer --mctx --odname /root/bench_results/mctracer/eager \
          python /root/src/mctracer_workload.py
"""

from __future__ import annotations

import os
import time

MODE = os.environ.get("MODE", "eager")
MODEL = "/root/models/MiniCPM5-2B"
STEPS = int(os.environ.get("STEPS", "16"))

os.environ.setdefault("VLLM_FL_PREFER", "flagos")
os.environ.setdefault("USE_FLAGGEMS", "1")


def main() -> None:
    import torch
    from vllm import LLM, SamplingParams

    kwargs = dict(
        model=MODEL,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=2048,
        gpu_memory_utilization=0.85,
    )
    if MODE == "eager":
        kwargs["enforce_eager"] = True

    print(f"########## mcTracer workload MODE={MODE} STEPS={STEPS} ##########", flush=True)
    print(f"device: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"whitelist={os.environ.get('VLLM_FL_FLAGOS_WHITELIST', '<none: all GEMS>')}",
          flush=True)

    t0 = time.perf_counter()
    llm = LLM(**kwargs)
    print(f"init={time.perf_counter() - t0:.1f}s", flush=True)

    sp = SamplingParams(max_tokens=STEPS, temperature=0.0, ignore_eos=True)
    prompt = "Explain the theory of relativity in detail. " * 40  # ~500 tokens

    # warmup keeps compile/cudagraph capture out of the traced region
    for _ in range(3):
        llm.generate([prompt], SamplingParams(max_tokens=4, temperature=0.0, ignore_eos=True))
    torch.cuda.synchronize()
    print("warmup done -- entering traced region", flush=True)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = llm.generate([prompt], sp)
    torch.cuda.synchronize()
    gen_s = time.perf_counter() - t0

    n_out = len(out[0].outputs[0].token_ids)
    print(f"gen={gen_s:.3f}s tokens={n_out} => {n_out / gen_s:.1f} tok/s", flush=True)
    print("MCTRACER_WORKLOAD_DONE", flush=True)


if __name__ == "__main__":
    main()
