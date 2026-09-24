#!/usr/bin/env python3
"""Rebuild the Python call chain for the per-step `Tensor.zero_` calls.

The exported trace has no `stack` arrays, but every `python_function` event
carries `Python id` / `Python parent id`, so the call tree can be walked.

Run:  python /root/src/parse_zero_parents.py
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

TRACE = Path("/root/bench_results/zero_stack/trace.json")


def main() -> None:
    data = json.loads(TRACE.read_text())
    evs = data.get("traceEvents") or []
    by_id: dict[int, dict] = {}
    for ev in evs:
        if ev.get("cat") == "python_function":
            pid = (ev.get("args") or {}).get("Python id")
            if pid is not None:
                by_id[pid] = ev

    chains = Counter()
    first: dict = {}
    for ev in evs:
        if ev.get("cat") != "python_function":
            continue
        nm = ev.get("name", "")
        if not (nm.startswith("<built-in method zero_")
                or (nm.startswith("aten::zero_") and "zero.py" not in nm)):
            continue
        chain = []
        cur = ev
        for _ in range(80):
            args = cur.get("args") or {}
            chain.append(cur.get("name", "?"))
            par = args.get("Python parent id")
            if par is None or par not in by_id:
                break
            cur = by_id[par]
        chain.reverse()
        key = tuple(chain)
        chains[key] += 1
        first.setdefault(key, chain)

    print(f"distinct zero_ call chains: {len(chains)}", flush=True)
    for key, c in chains.most_common(5):
        print(f"\n########## chain x{c} ##########", flush=True)
        for i, line in enumerate(first[key]):
            print(f"  [{i:2d}] {line[:150]}", flush=True)
    print("\nPARENTS_DONE", flush=True)


if __name__ == "__main__":
    main()
