#!/usr/bin/env python3
"""Split the trace into decode steps and separate in-graph from out-of-graph work.

The capture window starts 20 s into the official 4k run, which is the end of
wave 1's decode -- so it holds ~19 decode steps followed by wave 2's prefill
chunks.  Summing everything and dividing by step count (what step_budget.py does)
gives 262 ms/step, which is prefill, not decode.  This restricts to the decode
region first.

Decode steps are delimited by `_top_p_mass_v2_kernel`, which runs exactly once
per step (section 10).  Step i is [ts_i, ts_{i+1}).

For the sampler question the split that matters is inside vs outside the CUDA
graph.  vLLM cannot capture the sampler -- it consumes values the step produces
-- so lm_head, softmax, the top-p kernels and the RNG run as individual launches
on the critical path.  Per step we therefore count:
  * device events (kernels + memcpys)           -- everything that happens
  * host launch calls (mcLaunchKernel, mcModuleLaunchKernel)
  * mcGraphLaunch                                -- replays of the captured step
The difference between device events and individual launches is what rides
inside a graph; the kernels that appear once per step while everything else
appears 42 times are the un-captured tail.
"""
from __future__ import annotations

import gzip
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def load(path):
    p = Path(path)
    op = gzip.open if p.suffix == ".gz" else open
    with op(p) as f:
        tr = json.load(f)
    return tr.get("traceEvents", tr.get("events", []))


def main() -> int:
    path = sys.argv[1]
    ev = load(path)

    # --- decode step boundaries ---
    steps = sorted(
        (e["ts"], e.get("dur", 0))
        for e in ev
        if e.get("cat") == "kernel" and e.get("name") == "_top_p_mass_v2_kernel"
    )
    if len(steps) < 3:
        print("not enough sampler kernels to delimit steps:", len(steps))
        return 1
    # drop the first and last, they may be clipped by the capture window
    windows = [(steps[i][0], steps[i + 1][0]) for i in range(1, len(steps) - 1)]
    n = len(windows)
    print(f"decode steps delimited: {n}")

    kern_us = defaultdict(float)
    kern_cnt = defaultdict(int)
    rt_cnt = Counter()
    dev_total = 0
    cp_us = 0.0
    cp_cnt = 0

    for e in ev:
        ts = e.get("ts")
        if ts is None:
            continue
        idx = None
        for i, (a, b) in enumerate(windows):
            if a <= ts < b:
                idx = i
                break
        if idx is None:
            continue
        cat = e.get("cat")
        if cat == "kernel":
            dev_total += 1
            kern_us[e["name"]] += (e.get("dur") or 0) / 1000.0
            kern_cnt[e["name"]] += 1
        elif cat in ("gpu_memcpy", "gpu_memset"):
            dev_total += 1
            cp_us += (e.get("dur") or 0) / 1000.0
            cp_cnt += 1
        elif cat == "cuda_runtime":
            rt_cnt[e.get("name", "?")] += 1

    total_kern = sum(kern_us.values())
    print(f"\ndecode-only: {total_kern/n:.2f} ms/step kernel ({total_kern/n*1000:.0f} us)")
    print(f"             {cp_us/n:.3f} ms/step memcpy ({cp_cnt/n:.1f} copies/step)")
    print(f"             {dev_total/n:.1f} device events/step")

    launches = rt_cnt["mcLaunchKernel"] + rt_cnt["mcModuleLaunchKernel"]
    graphs = rt_cnt["mcGraphLaunch"]
    print(f"\nhost: {launches/n:.1f} individual kernel launches/step, "
          f"{graphs/n:.1f} graph replays/step")
    print(f"      => {dev_total/n - launches/n:.0f} device events ride inside graphs")
    print(f"\n  {'per step':>9}  {'us/call':>8}  {'ms/step':>8}  kernel")
    print("  " + "-" * 96)
    for name, us in sorted(kern_us.items(), key=lambda kv: -kv[1])[:18]:
        c = kern_cnt[name] / n
        print(f"  {c:9.1f}  {us/kern_cnt[name]*1000:8.1f}  {us/n:8.3f}  {name[:62]}")

    # --- the un-captured tail: kernels that run roughly once per step ---
    print("\n=== candidates for the out-of-graph tail (<=2 calls/step) ===")
    tail_us = 0.0
    tail_rows = []
    for name, us in kern_us.items():
        c = kern_cnt[name] / n
        if c <= 2.0:
            tail_us += us / n
            tail_rows.append((us / n, c, us / kern_cnt[name] * 1000, name))
    for ms, c, per_call, name in sorted(tail_rows, reverse=True):
        print(f"  {c:5.1f}/step  {per_call:8.1f} us  {ms:7.3f} ms/step  {name[:58]}")
    print(f"\n  tail total: {tail_us:.3f} ms/step "
          f"({100*tail_us/(total_kern/n):.1f}% of decode kernel time)")

    (Path(path).parent.parent / "decode_step_split.json").write_text(json.dumps({
        "steps": n,
        "kernel_ms_per_step": total_kern / n,
        "memcpy_ms_per_step": cp_us / n,
        "device_events_per_step": dev_total / n,
        "individual_launches_per_step": launches / n,
        "graph_replays_per_step": graphs / n,
        "tail_ms_per_step": tail_us,
        "tail": [{"kern": k, "calls_per_step": c, "us_per_call": pc, "ms_per_step": ms}
                 for ms, c, pc, k in sorted(tail_rows, reverse=True)],
    }, indent=2))
    print("\nwrote decode_step_split.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
