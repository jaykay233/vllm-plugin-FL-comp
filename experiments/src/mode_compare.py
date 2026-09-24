#!/usr/bin/env python3
"""Can torch.compile coexist with FlagGems?

vLLM gates its CustomOp layer on `CompilationConfig.custom_ops`:
    default_on() == (custom_ops != ['none'])
    enabled()    == (default_on() or "+name" in custom_ops) and "-name" not in custom_ops
With inductor, custom_ops defaults to ['none'] -> every custom op resolves to
`forward_native` (inductor-compiled) and the plugin's `forward_oot` -> CachedOp
-> FlagGems chain never runs. That is why `vllm serve` logs only one dispatch
resolution.

This script compares three modes on an identical workload:
    eager            : enforce_eager=True,  FlagGems should be active
    compile          : default (custom_ops=['none']), FlagGems bypassed
    compile_forced   : compile + custom_ops=['+rms_norm','+silu_and_mul','+rotary_embedding']

Run:  MODE=eager|compile|compile_forced python /root/src/mode_compare.py
"""

from __future__ import annotations

import json
import os
import statistics
import time
from pathlib import Path

MODE = os.environ.get("MODE", "eager")
MODEL = "/root/models/MiniCPM5-2B"
OUT = Path("/root/bench_results/mode_compare")
OUT.mkdir(parents=True, exist_ok=True)

FORCE_OPS = os.environ.get(
    "FORCE_OPS", "+rms_norm,+silu_and_mul,+rotary_embedding"
).split(",")


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
    elif MODE == "compile_forced":
        kwargs["compilation_config"] = {"custom_ops": FORCE_OPS}

    print(f"\n########## MODE={MODE} ##########", flush=True)
    print(f"vllm kwargs: { {k: v for k, v in kwargs.items() if k != 'model'} }", flush=True)

    t0 = time.perf_counter()
    llm = LLM(**kwargs)
    init_s = time.perf_counter() - t0

    sp = SamplingParams(max_tokens=128, temperature=0.0, ignore_eos=True)
    prompt = "Explain the theory of relativity in detail. " * 40  # ~500 tokens

    # warmup
    llm.generate([prompt], SamplingParams(max_tokens=8, temperature=0.0, ignore_eos=True))

    # timed: decode throughput
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = llm.generate([prompt], sp)
    torch.cuda.synchronize()
    gen_s = time.perf_counter() - t0

    n_out = len(out[0].outputs[0].token_ids)
    print(f"init={init_s:.1f}s  gen={gen_s:.2f}s  tokens={n_out}  "
          f"=> {n_out / gen_s:.1f} tok/s (includes prefill)", flush=True)

    # repeat 3x, report median
    times = []
    for _ in range(3):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        llm.generate([prompt], sp)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    med = statistics.median(times)
    print(f"median gen={med:.3f}s => {n_out / med:.1f} tok/s", flush=True)

    rec = {"mode": MODE, "init_s": init_s, "n_out": n_out,
           "median_gen_s": med, "tok_per_s": n_out / med,
           "runs": times}
    (OUT / f"{MODE}.json").write_text(json.dumps(rec, indent=2))
    print(f"MODE_COMPARE_DONE {MODE} tok_per_s={n_out / med:.1f}", flush=True)


if __name__ == "__main__":
    main()
