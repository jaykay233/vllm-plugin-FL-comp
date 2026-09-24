#!/usr/bin/env python3
"""End-to-end A/B of the M==1 GEMV routing in FlagGems' MetaX `mm`.

Arm is selected by the env var FLAG_GEMS_METAX_MM_GEMV (1 = routed, 0 = stock).

Reports:
  * TPOT (min of 3 samples) and output throughput
  * the decode kernel budget, so the stock arm's 84/step `zero_persistent_kernel`
    (the split-k accumulator `c.zero_()`) can be seen disappearing
  * a hash of the generated token ids, as a cheap correctness cross-check

Run:  FLAG_GEMS_METAX_MM_GEMV=1 python /root/src/ab_mm_gemv_e2e.py
"""

from __future__ import annotations

import hashlib
import os
import statistics
import time
from collections import Counter

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import torch
from torch.profiler import ProfilerActivity, profile
from vllm import LLM, SamplingParams

ARM = os.environ.get("FLAG_GEMS_METAX_MM_GEMV", "1")
STEPS = 128
RUNS = 3
PROMPT = "The theory of general relativity describes gravity as"


def main() -> None:
    llm = LLM(
        model="/root/models/MiniCPM5-2B",
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=2048,
        gpu_memory_utilization=0.30,
        compilation_config={"mode": "VLLM_COMPILE"},
    )
    sp = SamplingParams(max_tokens=STEPS, temperature=0.0, ignore_eos=True)

    llm.generate([PROMPT], sp, use_tqdm=False)

    tpots, tputs, tokids = [], [], None
    for _ in range(RUNS):
        t0 = time.perf_counter()
        out = llm.generate([PROMPT], sp, use_tqdm=False)
        dt = time.perf_counter() - t0
        ids = list(out[0].outputs[0].token_ids)
        tokids = ids
        # first token is prefill-dominated; average over the rest
        tpots.append((dt / len(ids)) * 1e6)
        tputs.append(len(ids) / dt)
    tpot = min(tpots)

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        out = llm.generate([PROMPT], sp, use_tqdm=False)
        torch.cuda.synchronize()
    n = len(out[0].outputs[0].token_ids)
    steps = max(n - 1, 1)

    CM = torch.autograd.DeviceType.CUDA
    kern = Counter()
    for ev in prof.events():
        if ev.device_type == CM and ev.name and ev.device_time > 0:
            kern[ev.name] += ev.device_time

    h = hashlib.sha1(str(tokids).encode()).hexdigest()[:12]

    print(f"\n===== ARM FLAG_GEMS_METAX_MM_GEMV={ARM} =====", flush=True)
    print(f"TPOT          = {tpot:.1f} us/token   ({tpot / 1000:.3f} ms)", flush=True)
    print(f"TPOT samples  = {[round(x, 1) for x in tpots]}", flush=True)
    print(f"throughput    = {max(tputs):.2f} tok/s", flush=True)
    print(f"token-hash    = {h}", flush=True)
    print("decode kernel budget (us/step):", flush=True)
    for k, v in kern.most_common(14):
        print(f"   {k[:50]:52s} {v / steps:9.1f} us   x{v / steps:.2f}",
              flush=True)
    print("\nAB_MM_GEMV_DONE", flush=True)


if __name__ == "__main__":
    main()
