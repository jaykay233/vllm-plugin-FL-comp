#!/usr/bin/env python3
"""D2H / H2D transfer budget for the MetaX decode path.

The per-token kernel budget showed "Memcpy HtoD (Pinned -> Device)" at ~7
launches per token, and raw mcTracer events show individual H2D copies with
``args.bytes == 4``.  A copy that small is pure launch overhead, so this
script aggregates every memcpy in a trace by direction, byte size and
duration to expose how much of the transfer cost is data vs. fixed overhead.

Usage:
    python memcpy_budget.py [trace.json ...]
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

DEFAULT_TRACES = [
    "/root/bench_results/mctracer/marked/compile/trace_compile/tracer_out-166589.json",
    "/root/bench_results/mctracer/marked/eager/trace_eager/tracer_out-164200.json",
]


def load(path: str) -> list[dict]:
    d = json.load(open(path))
    return d if isinstance(d, list) else d.get("traceEvents", d.get("events", []))


def analyze(path: str) -> None:
    evs = load(path)
    print("=" * 96)
    print(f"trace: {path}")
    print(f"  total events: {len(evs)}")

    by_name: Counter[str] = Counter()
    dur_by_name: Counter[str] = Counter()
    bytes_hist: dict[str, Counter[int]] = defaultdict(Counter)
    nbytes_by_name: Counter[str] = Counter()
    dur_sum_by_name: Counter[str] = Counter()

    for e in evs:
        name = str(e.get("name", ""))
        if "emcpy" not in name or e.get("ph") != "X":
            continue
        if not name.startswith("memcpy"):
            # host-side API events (mcMemcpyAsync); the GPU side is "memcpy ..."
            by_name["api:" + name] += 1
            continue
        args = e.get("args") or {}
        nb = int(args.get("bytes", -1))
        dur = int(e.get("dur", 0))
        by_name[name] += 1
        dur_by_name[name] += dur
        dur_sum_by_name[name] += dur
        if nb >= 0:
            bytes_hist[name][nb] += 1
            nbytes_by_name[name] += nb

    print("\n  -- by direction (GPU-side copy events) --")
    print(f"  {'name':34s} {'count':>7} {'count/tok':>9} {'total_us':>10} "
          f"{'avg_us':>8} {'MB total':>9}")
    # 16 decode steps in this workload (1344 kernels / 84 per step)
    steps = 16
    for name, cnt in by_name.most_common():
        if name.startswith("api:"):
            continue
        tot_us = dur_by_name[name] / 1e3
        mb = nbytes_by_name[name] / 1e6
        print(f"  {name:34s} {cnt:7d} {cnt / steps:9.1f} {tot_us:10.2f} "
              f"{tot_us / cnt:8.3f} {mb:9.4f}")

    print("\n  -- host-side memcpy API calls --")
    for name, cnt in by_name.most_common():
        if not name.startswith("api:"):
            continue
        print(f"  {name[4:]:34s} {cnt:7d} {cnt / steps:9.1f}")

    for name, hist in bytes_hist.items():
        print(f"\n  -- byte-size histogram: {name} --")
        print(f"  {'bytes':>10} {'count':>7} {'count/tok':>9} {'total_us':>10}")
        small_cnt = 0
        for nb, cnt in hist.most_common(14):
            print(f"  {nb:10d} {cnt:7d} {cnt / steps:9.1f}")
        for nb, cnt in hist.items():
            if nb <= 16:
                small_cnt += cnt
        tot = sum(hist.values())
        print(f"  distinct sizes: {len(hist)}   "
              f"<=16B copies: {small_cnt}/{tot} = {small_cnt / tot * 100:.1f}%")


def main() -> None:
    paths = sys.argv[1:] or DEFAULT_TRACES
    for p in paths:
        if Path(p).exists():
            analyze(p)
        else:
            print(f"missing: {p}")
    print("\nMEMCPY_BUDGET_DONE", flush=True)


if __name__ == "__main__":
    main()
