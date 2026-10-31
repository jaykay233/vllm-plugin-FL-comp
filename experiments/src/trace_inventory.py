#!/usr/bin/env python3
"""Full per-step kernel inventory from a torch.profiler chrome trace.

Aggregates by kernel name and reports ms/step and calls/step for a decode trace
(steps inferred from the number of times the most frequent kernel appears).

Run: python experiments/src/trace_inventory.py <trace.json> [steps]
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict

path = sys.argv[1]
forced = int(sys.argv[2]) if len(sys.argv) > 2 else 0

ev = json.load(open(path))["traceEvents"]
kern = [e for e in ev if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_user_annotation", "Kernel", "gpu_memcpy", "Memcpy")]
if not kern:
    kern = [e for e in ev if e.get("ph") == "X" and e.get("cat") == "gpu_op"]

agg: dict[str, list] = defaultdict(lambda: [0, 0.0])
for e in kern:
    n = e["name"]
    agg[n][0] += 1
    agg[n][1] += e.get("dur", 0.0)

total = sum(v[1] for v in agg.values())
calls = sum(v[0] for v in agg.values())
steps = forced or max(v[0] for v in agg.values())
print(f"trace={path}")
print(f"gpu events={calls}  total={total/1e3:.1f} ms  steps~{steps}  "
      f"per-step={total/1e3/steps:.2f} ms\n")
print(f"{'ms/step':>9s} {'%':>6s} {'calls/step':>11s}  kernel")
for n, (c, d) in sorted(agg.items(), key=lambda kv: -kv[1][1])[:45]:
    print(f"{d/1e3/steps:9.3f} {100*d/total:5.1f}% {c/steps:11.2f}  {n[:84]}")
