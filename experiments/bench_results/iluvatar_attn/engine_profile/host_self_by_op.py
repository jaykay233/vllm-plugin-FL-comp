"""Host self-time per cpu_op, averaged over steps that launch prefill (FA) kernels.

Usage: host_self_by_op.py <trace.json>
"""

import collections
import json
import sys

tr = json.load(open(sys.argv[1]))
ev = [e for e in tr["traceEvents"] if e.get("ph") == "X"]
steps = sorted((e for e in ev if e["name"].startswith("execute_new_")), key=lambda e: e["ts"])
ops = [e for e in ev if e.get("cat") == "cpu_op"]
kernels = [e for e in ev if e.get("cat") == "kernel"]
runtime = [e for e in ev if e.get("cat") in ("cuda_runtime", "cuda_driver")]
launch_ts = {e["args"]["correlation"]: e["ts"] for e in runtime if "correlation" in e.get("args", {})}

def in_step(ts, s):
    return s["ts"] <= ts < s["ts"] + s["dur"]

prefill_steps = []
for s in steps:
    for k in kernels:
        t = launch_ts.get(k.get("args", {}).get("correlation"))
        if t is not None and in_step(t, s) and "flash_attn" in k["name"].lower():
            prefill_steps.append(s)
            break

by_tid = collections.defaultdict(list)
for e in ops:
    by_tid[e["tid"]].append(e)
self_us = {}
for tid, lst in by_tid.items():
    lst.sort(key=lambda e: (e["ts"], -e["dur"]))
    stack = []
    for e in lst:
        while stack and stack[-1]["ts"] + stack[-1]["dur"] <= e["ts"]:
            stack.pop()
        self_us[id(e)] = e["dur"]
        if stack:
            self_us[id(stack[-1])] -= e["dur"]
        stack.append(e)

agg = collections.defaultdict(lambda: [0, 0.0])
for s in prefill_steps:
    for e in ops:
        if in_step(e["ts"], s):
            a = agg[e["name"]]
            a[0] += 1
            a[1] += self_us[id(e)]

n = len(prefill_steps)
gpu = collections.defaultdict(lambda: [0, 0.0])
for s in prefill_steps:
    for k in kernels:
        t = launch_ts.get(k.get("args", {}).get("correlation"))
        if t is not None and in_step(t, s):
            g = gpu[k["name"]]
            g[0] += 1
            g[1] += k["dur"]

print(f"steps with prefill: {n} / {len(steps)}; avg wall {sum(s['dur'] for s in prefill_steps)/n/1e3:.1f} ms")
print(f"{'op':60s} {'calls/step':>10s} {'self ms/step':>12s} {'self us/call':>12s}")
for name, (c, us) in sorted(agg.items(), key=lambda kv: -kv[1][1])[:20]:
    print(f"{name[:60]:60s} {c/n:10.1f} {us/n/1e3:12.1f} {us/c:12.1f}")
print()
for name, (c, us) in sorted(gpu.items(), key=lambda kv: -kv[1][1]):
    if "add_func" in name:
        print(f"GPU {name}: {c/n:.1f}/step avg {us/c/1e3:.3f} ms total {us/n/1e3:.1f} ms/step")
