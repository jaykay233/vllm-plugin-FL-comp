#!/usr/bin/env python3
"""Sanity-check raw mcTracer event units (ts / dur) on a few known kernels."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone

TRACE = sys.argv[1]

WANT = [
    "linear_kernel",
    "flash_fwd_splitkv_kernel",
    "fused_add_rms_norm_kernel",
    "silu_and_mul_kernel",
    "apply_rotary_pos_emb_inplace_kernel",
    "reshape_and_cache_flash_kernel",
    "mcGetDevice",
]

with open(TRACE) as f:
    ev = json.load(f).get("traceEvents", [])

print(f"{len(ev)} events\n")
print(f"{'name':46s} {'ph':3s} {'ts':>22s} {'dur':>12s}")
shown = set()
for e in ev:
    n = e.get("name", "")
    if any(w in n for w in WANT) and e.get("ph") == "X" and n not in shown:
        ts = e.get("ts")
        dur = e.get("dur")
        shown.add(n)
        ts_s = ts / 1e9 if isinstance(ts, int) else None
        print(f"{n[:46]:46s} {str(e.get('ph')):3s} {ts if ts is not None else '-':>22} "
              f"{dur if dur is not None else '-':>12}")

# interpret ts as ns -> wall clock
xs = [e["ts"] for e in ev if isinstance(e.get("ts"), int)]
print(f"\nts range: {min(xs)} .. {max(xs)}")
print(f"as ns since epoch: {datetime.fromtimestamp(min(xs) / 1e9, timezone.utc)} .. "
      f"{datetime.fromtimestamp(max(xs) / 1e9, timezone.utc)}")
print(f"as us since epoch: {datetime.fromtimestamp(min(xs) / 1e6, timezone.utc)} .. "
      f"{datetime.fromtimestamp(max(xs) / 1e6, timezone.utc)}")
print(f"span(ns-interp) = {(max(xs) - min(xs)) / 1e9:.2f} s")

# dur stats per wanted kernel
print("\n-- dur stats (raw units) --")
for w in WANT:
    ds = [e["dur"] for e in ev if w in e.get("name", "") and isinstance(e.get("dur"), (int, float))]
    if ds:
        print(f"  {w:46s} n={len(ds):6d} min={min(ds):10.0f} med={sorted(ds)[len(ds)//2]:10.0f} max={max(ds):10.0f}")

print("\nUNIT_CHECK_DONE")
