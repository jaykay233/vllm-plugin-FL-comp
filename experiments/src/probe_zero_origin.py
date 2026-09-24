#!/usr/bin/env python3
"""Is the 84x/step `zero_` ours, or vLLM's?

Compile-mode budget:

    zero_persistent_kernel   271 us/step   x 84.0/step   (3.2 us/launch)
    aten::zero_              84.0/step     shape [1, 2048] bf16

84/step is exactly the number of fused_add_rms_norm calls per step (2 per
layer), and the dynamo graph contains no zero node at all, so it is introduced
between dynamo and inductor.  This isolates the owner by toggling our provider:

    ir_split   IR kernels on, split impl     (current default)
    ir_fused   IR kernels on, fused impl     (clones activations)
    ir_off     IR kernels off (VLLM_FL_IR_KERNELS=0) -> norm goes native

If zero_ disappears when the IR path is off, it belongs to our IR provider;
if it stays, it is vLLM/inductor.

Run:  CFG=ir_split python /root/src/probe_zero_origin.py
"""

from __future__ import annotations

import json
import os
import statistics
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

CFG = os.environ.get("CFG", "ir_split")
MODEL = "/root/models/MiniCPM5-2B"
PROMPT = "The theory of general relativity describes gravity as"
OUT = Path("/root/bench_results/zero_origin")
OUT.mkdir(parents=True, exist_ok=True)

if CFG == "ir_split":
    os.environ["VLLM_FL_IR_KERNELS"] = "1"
    os.environ["VLLM_FL_IR_FUSED_ADD"] = "split"
elif CFG == "ir_fused":
    os.environ["VLLM_FL_IR_KERNELS"] = "1"
    os.environ["VLLM_FL_IR_FUSED_ADD"] = "fused"
elif CFG == "ir_off":
    os.environ["VLLM_FL_IR_KERNELS"] = "0"
else:
    raise SystemExit(f"unknown CFG={CFG}")

STEPS = 128


def main() -> None:
    import torch
    from torch.profiler import ProfilerActivity, profile
    from vllm import LLM, SamplingParams

    print(f"########## CFG={CFG}  IR={os.environ.get('VLLM_FL_IR_KERNELS')} "
          f"FUSED_ADD={os.environ.get('VLLM_FL_IR_FUSED_ADD')} ##########", flush=True)

    llm = LLM(
        model=MODEL,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=2048,
        gpu_memory_utilization=0.30,
        compilation_config={"mode": "VLLM_COMPILE"},
    )
    sp = SamplingParams(max_tokens=STEPS, temperature=0.0, ignore_eos=True)
    llm.generate([PROMPT], sp, use_tqdm=False)  # warmup

    times = []
    for _ in range(3):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = llm.generate([PROMPT], sp, use_tqdm=False)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    n = len(out[0].outputs[0].token_ids)
    ids = list(out[0].outputs[0].token_ids)
    med = statistics.median(times)
    print(f"  tok/s={n / med:7.1f}  us/token={med * 1e6 / n:8.0f}", flush=True)

    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU],
                 record_shapes=False) as prof:
        llm.generate([PROMPT], sp, use_tqdm=False)
        torch.cuda.synchronize()

    CPU = torch.autograd.DeviceType.CPU
    CUDA = torch.autograd.DeviceType.CUDA
    steps = n - 1

    z_cpu = Counter()
    z_cuda = Counter()
    z_cuda_us = Counter()
    rms_cuda = Counter()
    rms_cuda_us = Counter()
    for ev in prof.events():
        if ev.device_type == CPU and ev.name in ("aten::zero_", "aten::zeros",
                                                 "aten::zeros_like", "aten::fill_"):
            z_cpu[ev.name] += 1
        if ev.device_type == CUDA and ev.name:
            low = ev.name.lower()
            if "zero" in low:
                z_cuda[ev.name] += ev.count
                z_cuda_us[ev.name] += ev.device_time
            if "rms_norm" in low:
                rms_cuda[ev.name] += ev.count
                rms_cuda_us[ev.name] += ev.device_time

    print(f"  --- CPU zero-ish ops (per decode step, {steps} steps) ---", flush=True)
    for name, k in z_cpu.most_common():
        print(f"    {name:18s} {k:6d}  {k / steps:7.2f}/step", flush=True)
    print("  --- CUDA zero kernels ---", flush=True)
    for name, k in z_cuda.most_common():
        print(f"    {name[:46]:48s} {k:6d}  {k / steps:7.2f}/step  "
              f"{z_cuda_us[name] / steps:8.1f} us/step", flush=True)
    print("  --- CUDA rms_norm kernels ---", flush=True)
    for name, k in rms_cuda.most_common():
        print(f"    {name[:46]:48s} {k:6d}  {k / steps:7.2f}/step  "
              f"{rms_cuda_us[name] / steps:8.1f} us/step", flush=True)

    rec = {
        "cfg": CFG,
        "tok_per_s": n / med,
        "us_per_token": med * 1e6 / n,
        "zero_cpu_total": dict(z_cpu),
        "zero_cuda_us_per_step": sum(z_cuda_us.values()) / steps,
        "zero_cuda_launches_per_step": sum(z_cuda.values()) / steps,
        "rms_cuda_us_per_step": sum(rms_cuda_us.values()) / steps,
        "token_ids_head": ids[:8],
    }
    (OUT / f"{CFG}.json").write_text(json.dumps(rec, indent=2))
    print(f"\nPROBE_ZERO_ORIGIN_DONE {CFG} zero_us_per_step="
          f"{sum(z_cuda_us.values()) / steps:.1f} tok_per_s={n / med:.1f}", flush=True)


if __name__ == "__main__":
    main()
