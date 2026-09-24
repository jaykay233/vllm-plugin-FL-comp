#!/usr/bin/env python3
"""How much could a megakernel possibly buy? Measure the per-step kernel budget.

A megakernel (one persistent kernel looping over all 42 layers) can only remove
two things: host launch overhead and GPU-side inter-kernel gaps.

Decode already runs under CUDA Graph (FULL) capture, so host launch overhead is
already gone. That leaves the inter-kernel gaps. This measures the actual
budget:

  * kernels launched per decode step
  * total GPU kernel time per decode step
  * TPOT (wall time per decode step)

  gap = TPOT - total_kernel_time   <- the absolute ceiling for a megakernel

If `gap` is a small share of TPOT, a megakernel cannot pay for itself.

Run:  conda activate mx && python /root/src/decode_kernel_budget.py
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

MODEL = "/root/models/MiniCPM5-2B"
OUT = Path("/root/bench_results/megakernel")
OUT.mkdir(parents=True, exist_ok=True)

PROMPT = "The theory of general relativity describes gravity as"
DECODE_TOKENS = 64


def main() -> None:
    import torch
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=MODEL,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=2048,
        gpu_memory_utilization=0.85,
        # defaults: torch.compile ON + CUDA graph ON (the fast path)
    )

    def gen(n_tokens: int) -> tuple[float, int]:
        sp = SamplingParams(max_tokens=n_tokens, temperature=0.0, ignore_eos=True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = llm.generate([PROMPT], sp, use_tqdm=False)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        n_out = len(out[0].outputs[0].token_ids)
        return dt, n_out

    # warmup (triggers compile + cudagraph capture)
    print("  warming up (compile + cudagraph capture)...", flush=True)
    gen(DECODE_TOKENS)
    print("  warmup done", flush=True)

    # --- TPOT: (time for N+1 tokens - time for 1 token) / N ----------------
    t1, n1 = gen(1)
    t2, n2 = gen(DECODE_TOKENS)
    steps = max(1, n2 - n1)
    tpot_s = (t2 - t1) / steps
    print(f"  TTFT-ish (1 token)      : {t1 * 1e3:.2f} ms")
    print(f"  {n2} tokens total        : {t2 * 1e3:.2f} ms")
    print(f"  decode steps (n2-n1)    : {steps}")
    print(f"  TPOT (wall per step)    : {tpot_s * 1e3:.3f} ms")

    # --- profile one decode run -------------------------------------------
    sp = SamplingParams(max_tokens=DECODE_TOKENS, temperature=0.0, ignore_eos=True)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
    ) as prof:
        out = llm.generate([PROMPT], sp, use_tqdm=False)
        torch.cuda.synchronize()

    counts: Counter[str] = Counter()
    dur: Counter[str] = Counter()
    for e in prof.events():
        if e.device_type == torch.autograd.DeviceType.CUDA and e.name:
            counts[e.name] += e.count
            dur[e.name] += e.device_time

    total_kernels = sum(counts.values())
    total_gpu_us = sum(dur.values())
    n_out = len(out[0].outputs[0].token_ids)
    decode_steps = max(1, n_out - 1)

    print()
    print("=" * 92)
    print("PER DECODE STEP BUDGET")
    print("=" * 92)
    print(f"  profiled run            : {n_out} tokens "
          f"({decode_steps} decode steps + prefill)")
    print(f"  total kernel launches   : {total_kernels}")
    print(f"  total GPU kernel time   : {total_gpu_us / 1000:.3f} ms")
    print(f"  distinct kernels        : {len(counts)}")
    print()
    print(f"  kernels per token+prefill: {total_kernels / n_out:.0f}")
    print(f"  GPU us per token         : {total_gpu_us / n_out:.1f} us")
    print(f"  TPOT wall per step       : {tpot_s * 1e6:.1f} us")
    gap_us = tpot_s * 1e6 - total_gpu_us / n_out
    print(f"  gap (wall - GPU) per step: {gap_us:.1f} us  "
          f"= {gap_us / (tpot_s * 1e6) * 100:.1f}% of the step")

    print()
    print("  top kernels by GPU time:")
    for name, d in dur.most_common(12):
        print(f"    {d / 1000:8.3f} ms  {counts[name]:6d} launches  {name[:60]}")

    print()
    print("=" * 92)
    print("MEGAKERNEL CEILING")
    print("=" * 92)
    print(f"  A megakernel removes inter-kernel gaps and host launch overhead.")
    print(f"  Host launch is already removed by the CUDA graph.")
    print(f"  So its ABSOLUTE ceiling is the gap: {gap_us:.1f} us/step "
          f"= {gap_us / (tpot_s * 1e6) * 100:.1f}%")
    print()
    print(f"  It does NOT reduce the {total_gpu_us / n_out / 1000:.3f} ms of actual GPU "
          f"kernel time,")
    print(f"  because at M=1 the weights must still be read in full.")

    result = {
        "tpot_ms": tpot_s * 1e3,
        "gpu_us_per_token": total_gpu_us / n_out,
        "gap_us_per_step": gap_us,
        "gap_pct": gap_us / (tpot_s * 1e6) * 100,
        "kernels_per_token": total_kernels / n_out,
        "total_kernels": total_kernels,
        "decode_steps": decode_steps,
        "top_kernels": [
            {"name": n, "ms": d / 1000, "launches": counts[n]}
            for n, d in dur.most_common(20)
        ],
    }
    (OUT / "budget.json").write_text(json.dumps(result, indent=2))
    print(f"\n  Wrote {OUT / 'budget.json'}")
    print("MEGAKERNEL_BUDGET_DONE", flush=True)


if __name__ == "__main__":
    main()
