#!/usr/bin/env python3
"""Diff two decode traces on the one thing the fix was supposed to change.

FINDINGS 25 removed 42 device-to-device copies of 262144 B per decode step by
writing the attention output back through an elementwise kernel instead of
`output[:n] = attn_out`. The end-to-end gain reconciled with the trace (4096
steps x 1.30 ms = 5.30 s vs 5.35 s measured), but that is an inference. This
checks the device timeline directly: same shape, same server flags, before and
after, and reports per-step step time and the copy that should be gone.

Usage: compare_before_after.py before.pt.trace.json.gz after.pt.trace.json.gz
"""
from __future__ import annotations

import gzip
import json
import sys
from collections import defaultdict

COPY_BYTES = 262144      # 64 tok x 2048 hidden x 2 B -- the attention writeback
# the elementwise kernel the fix substitutes in. Annotated with the dtype
# because the sampler also launches CUDAFunctorOnSelf_add<int> for token ids.
ADD_NAME = "CUDAFunctorOnSelf_add<c10::BFloat16>"


def load(path):
    with gzip.open(path) as f:
        tr = json.load(f)
    return tr.get("traceEvents") or tr.get("events")


def analyse(ev, label):
    kern = [e for e in ev if e.get("cat") == "kernel" and e.get("ph") == "X"
            and e.get("ts") is not None]
    mcs = [e for e in ev if e.get("cat") == "gpu_memcpy" and e.get("ph") == "X"]
    steps = sorted(e["ts"] for e in kern if str(e.get("name")) == "_top_p_mass_v2_kernel")
    n = len(steps) - 1
    if n <= 0:
        print(f"{label}: no decode steps found")
        return None

    wall = (steps[-1] - steps[0]) / n / 1000
    # step time from /metrics-independent signal: count kernels per step too
    per_step = defaultdict(int)
    dur = defaultdict(float)
    d2d = defaultdict(lambda: [0, 0.0])
    for i in range(n):
        a, b = steps[i], steps[i + 1]
        for e in kern:
            if a <= e["ts"] < b:
                per_step[str(e.get("name"))] += 1
                dur[str(e.get("name"))] += e["dur"]
        for e in mcs:
            if a <= e["ts"] < b:
                by = int(e.get("args", {}).get("bytes", 0) or 0)
                d2d[by][0] += 1
                d2d[by][1] += e["dur"]
    kernel_ms = sum(dur.values()) / n / 1000
    copy_cnt = d2d.get(COPY_BYTES, [0, 0.0])[0] / n
    copy_ms = d2d.get(COPY_BYTES, [0, 0.0])[1] / n / 1000
    # both the writeback kernel and the GEMM epilogue show up with a namespace
    # prefix, so match on substring
    add_cnt = sum(c for k, c in per_step.items() if ADD_NAME in k) / n
    add_ms = sum(d for k, d in dur.items() if ADD_NAME in k) / n / 1000
    print(f"\n{label}  ({n} steps)")
    print(f"  wall / step          {wall:7.2f} ms")
    print(f"  kernel time / step   {kernel_ms:7.2f} ms")
    print(f"  {COPY_BYTES} B copies    {copy_cnt:7.2f}x  {copy_ms:6.3f} ms"
          f"   <-- the writeback that was replaced")
    print(f"  {ADD_NAME:<18} {add_cnt:7.2f}x  {add_ms:6.3f} ms")
    return {"wall": wall, "kernel": kernel_ms, "copy_n": copy_cnt,
            "copy_ms": copy_ms, "add_n": add_cnt, "add_ms": add_ms}


def main() -> int:
    before = analyse(load(sys.argv[1]), "BEFORE (copy_ writeback)")
    after = analyse(load(sys.argv[2]), "AFTER  (add(0,out=) writeback)")
    if not before or not after:
        return 1
    print("\ndelta")
    print(f"  step time        {after['wall'] - before['wall']:+7.2f} ms"
          f"  ({(after['wall'] / before['wall'] - 1) * 100:+.2f}%)")
    print(f"  kernel time      {after['kernel'] - before['kernel']:+7.2f} ms")
    print(f"  copies/step      {after['copy_n'] - before['copy_n']:+7.2f}x"
          f"  {after['copy_ms'] - before['copy_ms']:+7.3f} ms")
    print(f"  add/step         {after['add_n'] - before['add_n']:+.2f}x")
    removed = before["copy_ms"] - after["copy_ms"]
    added = after["add_ms"] - before["add_ms"]
    gained = before["wall"] - after["wall"]
    print(f"\n  copy time removed {removed:6.3f} ms/step")
    print(f"  writeback kernel  {added:+6.3f} ms/step  (the add replacing it)")
    print(f"  net expected      {removed - added:6.3f} ms/step")
    print(f"  actual step gain  {gained:6.3f} ms/step")
    if removed > 0:
        print(f"  closure           {gained / (removed - added) * 100:5.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
