#!/usr/bin/env python3
"""Parse the exported chrome trace for the Python stacks of aten::zero_.

Run:  python /root/src/parse_zero_trace.py [trace.json]
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

TRACE = Path(sys.argv[1] if len(sys.argv) > 1
             else "/root/bench_results/zero_stack/trace.json")

KEEP = ("/vllm/", "/root/vllm-plugin-FL-comp/", "/flag_gems/", "/torch/_inductor/")


def main() -> None:
    print(f"loading {TRACE} ...", flush=True)
    data = json.loads(TRACE.read_text())
    if isinstance(data, dict):
        print(f"trace with_stack={data.get('with_stack')}", flush=True)
        evs = data.get("traceEvents") or data.get("events") or []
    else:
        evs = data
    print(f"events: {len(evs)}", flush=True)

    hit = Counter()
    stacks: dict = {}
    for ev in evs:
        if ev.get("ph") != "X":
            continue
        name = ev.get("name", "")
        if "zero" not in name.lower() and "fill" not in name.lower():
            continue
        stack = (ev.get("args") or {}).get("stack") or []
        frames = [
            f"{(f.get('filename') or '?')}:{f.get('line')} in {f.get('name')}"
            for f in stack
        ]
        keep = tuple(f for f in frames if any(k in f for k in KEEP))
        key = (name, keep)
        hit[key] += 1
        stacks.setdefault(key, frames)

    print(f"\ndistinct zero/fill op+stack combos: {len(hit)}", flush=True)
    for (name, keep), c in hit.most_common(12):
        print(f"\n===== {name}  x{c} =====", flush=True)
        for line in stacks[(name, keep)][-22:]:
            print(f"   {line}", flush=True)
    print("\nPARSE_DONE", flush=True)


if __name__ == "__main__":
    main()
