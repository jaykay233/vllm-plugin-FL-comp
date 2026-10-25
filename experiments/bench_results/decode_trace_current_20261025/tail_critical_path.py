#!/usr/bin/env python3
"""Where the decode step actually goes: graph replay vs the eager tail.

split_decode.py answers "how many kernels and how long" per step.  This asks the
follow-up question that decides whether the tail is worth attacking: what is the
step's WALL time, how much of it is GPU busy, and how much is the device idle
waiting for the host to launch the out-of-graph kernels one at a time.

Decode steps are delimited by _top_p_mass_v2_kernel, which runs exactly once per
step as part of the sampler tail.

Reported per step:
  wall     -- end-to-end step duration (also cross-checked against /metrics)
  gpu      -- union of all kernel/memcpy intervals inside the step
  gap      -- wall - gpu: host launch latency and sync stalls
  ingraph  -- kernels covered by a graph replay event
  tail     -- kernels outside any graph, i.e. the sampler chain
"""
from __future__ import annotations

import gzip
import json
import sys
from collections import defaultdict

path = sys.argv[1]
with gzip.open(path) as f:
    tr = json.load(f)
ev = tr.get("traceEvents") or tr.get("events")

kern = [e for e in ev if e.get("cat") in ("kernel", "Kernel", "gpu_memcpy", "gpu_memset")
        and e.get("ph") == "X" and e.get("ts") is not None and e.get("dur") is not None]
memcpy = [e for e in ev if e.get("cat") in ("gpu_memcpy", "Memcpy", "gpu_memset")]
graphs = [e for e in ev if e.get("cat") in ("cuda_graph", "CudaGraph") or
          (e.get("ph") == "X" and "graph" in str(e.get("name", "")).lower())]


def name(e):
    return str(e.get("name", ""))


step_ts = sorted(e["ts"] for e in kern if name(e) == "_top_p_mass_v2_kernel")
print(f"steps delimited: {len(step_ts) - 1}")

# graph replay extents, for deciding what rode inside a graph
gext = [(g["ts"], g["ts"] + g["dur"]) for g in graphs]


def in_graph(e):
    if gext:
        for a, b in gext:
            if a <= e["ts"] and e["ts"] + e["dur"] <= b:
                return True
        return False
    # Fall back to the profiler's own annotation when present.
    return bool(e.get("args", {}).get("graph"))


def union_len(iv):
    iv = sorted(iv)
    tot, ca, cb = 0, None, None
    for a, b in iv:
        if ca is None:
            ca, cb = a, b
        elif a <= cb:
            cb = max(cb, b)
        else:
            tot += cb - ca
            ca, cb = a, b
    if ca is not None:
        tot += cb - ca
    return tot


tot_wall = tot_gpu = tot_tail = tot_mc = 0.0
n = 0
tail_k = defaultdict(lambda: [0, 0.0])
mc_by_kind = defaultdict(lambda: [0, 0.0, 0])
for i in range(len(step_ts) - 1):
    a, b = step_ts[i], step_ts[i + 1]
    w = [e for e in kern if a <= e["ts"] < b]
    if not w:
        continue
    n += 1
    tot_wall += b - a
    tot_gpu += union_len([(e["ts"], e["ts"] + e["dur"]) for e in w])
    for e in w:
        if not in_graph(e):
            tot_tail += e["dur"]
            k = tail_k[name(e)[:70]]
            k[0] += 1
            k[1] += e["dur"]
for e in memcpy:
    if step_ts and step_ts[0] <= e["ts"] < step_ts[-1]:
        tot_mc += e["dur"]
        k = name(e)[:40]
        d = mc_by_kind[k]
        d[0] += 1
        d[1] += e["dur"]
        d[2] = max(d[2], int(e.get("args", {}).get("bytes", 0) or 0))

if not n:
    print("no decode steps")
    sys.exit(1)

print(f"\nper step, mean over {n} steps")
print(f"  wall            {tot_wall / n / 1000:7.2f} ms")
print(f"  gpu busy        {tot_gpu / n / 1000:7.2f} ms  ({tot_gpu / tot_wall * 100:.1f}% of wall)")
print(f"  idle / gaps     {(tot_wall - tot_gpu) / n / 1000:7.2f} ms")
print(f"  graph replays   {len(gext) / n:7.2f}")
print(f"  out-of-graph    {tot_tail / n / 1000:7.2f} ms  "
      f"({tot_tail / tot_gpu * 100:.1f}% of gpu busy)")
print(f"  memcpy total    {tot_mc / n / 1000:7.2f} ms")

print("\nmemcpy breakdown (per step)")
for k, (cnt, dur, mx) in sorted(mc_by_kind.items(), key=lambda kv: -kv[1][1]):
    print(f"  {cnt / n:6.1f}x  {dur / n:7.1f} us  max {mx:>10} B  {k}")

print("\nout-of-graph tail, grouped (per step)")
groups = {
    "lm_head (mm_kernel_nt)": lambda s: "mm_kernel_nt" in s,
    "top-p chain": lambda s: "_top_p_" in s,
    "softmax": lambda s: "SoftMaxForward" in s,
    "thrust / cub scan": lambda s: "thrust::" in s or "cub::" in s,
    "RNG sampling": lambda s: "distribution_elementwise" in s,
}
for label, pred in groups.items():
    c = sum(v[0] for k, v in tail_k.items() if pred(k))
    d = sum(v[1] for k, v in tail_k.items() if pred(k))
    if c:
        print(f"  {label:<24} {c / n:5.1f}x {d / n:8.1f} us  {d / n / 1000:5.2f} ms")

print("\nfull out-of-graph list")
for k, (cnt, dur) in sorted(tail_k.items(), key=lambda kv: -kv[1][1])[:25]:
    print(f"  {cnt / n:6.1f}x {dur / n:8.1f} us  {dur / n / 1000:5.2f} ms  {k}")
