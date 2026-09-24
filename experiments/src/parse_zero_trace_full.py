#!/usr/bin/env python3
"""Print full Python stacks for aten::zero_ / _launch_zero events in the trace."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

TRACE = Path("/root/bench_results/zero_stack/trace.json")


def main() -> None:
    data = json.loads(TRACE.read_text())
    evs = data.get("traceEvents") or []
    hit = Counter()
    stacks: dict = {}
    for ev in evs:
        if ev.get("ph") != "X":
            continue
        name = ev.get("name", "")
        if not (name.startswith("<built-in method zero_") or "zero.py(90)" in name):
            continue
        frames = [
            f"{(f.get('filename') or '?')}:{f.get('line')} {f.get('name')}"
            for f in ((ev.get("args") or {}).get("stack") or [])
        ]
        key = (name, tuple(frames))
        hit[key] += 1
        stacks.setdefault(key, frames)

    for (name, _), c in hit.most_common(6):
        print(f"\n########## {name}  x{c} ##########", flush=True)
        for line in stacks[(name, _)]:
            print(f"  {line}", flush=True)
    print("\nPARSE_FULL_DONE", flush=True)


if __name__ == "__main__":
    main()
