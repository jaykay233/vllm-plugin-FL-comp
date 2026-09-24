#!/usr/bin/env python3
"""Export a chrome trace and read the Python stack of every `aten::zero_` op.

A short run (4 decode steps, CPU-only, with_stack=True) keeps the trace small
enough to parse, and the chrome trace records the Python frames for each CPU
op, which the torch profiler API does not expose.

Run:  python /root/src/probe_zero_trace.py
"""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import torch
from torch.profiler import ProfilerActivity, profile
from vllm import LLM, SamplingParams

OUT = Path("/root/bench_results/zero_stack")
OUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    llm = LLM(
        model="/root/models/MiniCPM5-2B",
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=2048,
        gpu_memory_utilization=0.30,
        compilation_config={"mode": "VLLM_COMPILE", "cudagraph_mode": "NONE"},
    )
    sp = SamplingParams(max_tokens=4, temperature=0.0, ignore_eos=True)
    prompt = "The theory of general relativity describes gravity as"
    llm.generate([prompt], sp, use_tqdm=False)

    trace = OUT / "trace.json"
    with profile(activities=[ProfilerActivity.CPU], with_stack=True) as prof:
        llm.generate([prompt], sp, use_tqdm=False)
        torch.cuda.synchronize()
    prof.export_chrome_trace(str(trace))
    print(f"[trace] {trace} ({trace.stat().st_size / 1e6:.1f} MB)", flush=True)

    data = json.loads(trace.read_text())
    evs = data["events"] if isinstance(data, dict) else data

    hit = Counter()
    shown = set()
    frames_of: dict = {}

    for ev in evs:
        if ev.get("ph") != "X":
            continue
        name = ev.get("name", "")
        if not (name.startswith("aten::zero") or name.startswith("aten::fill")):
            continue
        args = ev.get("args", {})
        stack = args.get("stack") or []
        keep = [
            f"{f.get('filename','?')}:{f.get('line','?')} in {f.get('name','?')}"
            for f in stack
            if any(k in (f.get("filename") or "")
                   for k in ("/vllm/", "/root/vllm-plugin-FL-comp/", "/flag_gems/"))
        ]
        key = (name, tuple(keep))
        hit[key] += 1
        frames_of.setdefault(key, keep)

    print(f"distinct zero/fill call sites: {len(hit)}", flush=True)
    for (name, _), c in hit.most_common(8):
        print(f"\n===== {name}   x{c} =====", flush=True)
        for line in frames_of[(name, _)][-18:]:
            print(f"   {line}", flush=True)
    print("\nZERO_TRACE_DONE", flush=True)


if __name__ == "__main__":
    main()
