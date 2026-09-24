#!/usr/bin/env python3
"""Full per-kernel budget for the three IR-provider configs.

The earlier probe showed FlagGems' norm kernels are 354-374 us/step vs native
inductor's 412 us/step, yet end-to-end tok/s was WORSE with FlagGems enabled
(147.8 / 144.1 vs 151.2).  The norm kernel time cannot explain that, so dump
the whole kernel budget to find where the extra time goes.

Configs: ir_split / ir_fused / ir_off  (see probe_zero_origin.py)
Run:  CFG=ir_split python /root/src/probe_cfg_budget.py
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
OUT = Path("/root/bench_results/cfg_budget")
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
PROMPT = "The theory of general relativity describes gravity as"


def main() -> None:
    import torch
    from torch.profiler import ProfilerActivity, profile
    from vllm import LLM, SamplingParams

    print(f"### CFG={CFG} ###", flush=True)
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

    times = []
    for _ in range(3):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = llm.generate([PROMPT], sp, use_tqdm=False)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    n = len(out[0].outputs[0].token_ids)
    med = statistics.median(times)
    tpot = med * 1e6 / n
    print(f"  tok/s={n / med:7.1f}  us/token={tpot:8.0f}", flush=True)

    with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof:
        llm.generate([PROMPT], sp, use_tqdm=False)
        torch.cuda.synchronize()

    CM = torch.autograd.DeviceType.CUDA
    steps = n - 1
    per: Counter = Counter()
    cnt: Counter = Counter()
    total = 0.0
    for ev in prof.events():
        if ev.device_type != CM or not ev.name:
            continue
        if ev.device_time <= 0:
            continue
        per[ev.name] += ev.device_time
        cnt[ev.name] += ev.count
        total += ev.device_time

    print(f"  total GPU {total / steps:9.1f} us/step  (tpot {tpot:.0f})", flush=True)
    print(f"  {'kernel':52s} {'us/step':>9s} {'n/step':>8s} {'share':>7s}", flush=True)
    for name, t in per.most_common(18):
        print(f"  {name[:52]:52s} {t / steps:9.1f} {cnt[name] / steps:8.2f} "
              f"{100 * t / total:6.1f}%", flush=True)

    rec = {
        "cfg": CFG,
        "tok_per_s": n / med,
        "us_per_token": tpot,
        "gpu_us_per_step": total / steps,
        "kernels": [
            {"name": k, "us_per_step": v / steps, "n_per_step": cnt[k] / steps,
             "share": v / total}
            for k, v in per.most_common(60)
        ],
    }
    (OUT / f"{CFG}.json").write_text(json.dumps(rec, indent=2))
    print(f"\nCFG_BUDGET_DONE {CFG}", flush=True)


if __name__ == "__main__":
    main()
