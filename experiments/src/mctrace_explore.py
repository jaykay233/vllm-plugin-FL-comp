#!/usr/bin/env python3
"""Explore the raw mcTracer (Perfetto) trace before aggregating.

Answers: what categories/phases exist, what the top kernels are, and how the
process timeline splits (init vs warmup vs generation) so later aggregation can
be restricted to the region of interest.

Run:  python /root/src/mctrace_explore.py <trace.json>
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict

TRACE = sys.argv[1] if len(sys.argv) > 1 else (
    "/root/bench_results/mctracer/file_mode/trace_eager/tracer_out-161979.json"
)


def main() -> None:
    print(f"loading {TRACE} ...", flush=True)
    with open(TRACE) as f:
        d = json.load(f)
    ev = d.get("traceEvents", [])
    print(f"loaded {len(ev)} events\n", flush=True)

    cats = Counter(e.get("cat") for e in ev)
    phases = Counter(e.get("ph") for e in ev)
    print("-- phases --")
    for k, v in phases.most_common():
        print(f"  ph={k!r:8s} {v}")
    print("\n-- categories (top 30) --")
    for k, v in cats.most_common(30):
        print(f"  {str(k):40s} {v}")

    # thread / pid landscape
    pids = Counter(e.get("pid") for e in ev)
    tids = Counter((e.get("pid"), e.get("tid")) for e in ev)
    print(f"\n-- num pids: {len(pids)}, num (pid,tid): {len(tids)} --")
    for k, v in pids.most_common(10):
        print(f"  pid={k} events={v}")
    print("  top (pid,tid):")
    for (p, t), v in tids.most_common(15):
        print(f"    pid={p} tid={t} n={v}")

    # name-space landscape for duration events
    print("\n-- top 40 names by event count (all phases) --")
    names = Counter(e.get("name") for e in ev)
    for k, v in names.most_common(40):
        print(f"  {v:9d}  {str(k)[:90]}")

    # duration events: aggregate self time by name
    dur = [e for e in ev if e.get("ph") == "X" and isinstance(e.get("dur"), (int, float))]
    print(f"\n-- duration(X) events: {len(dur)} --")
    agg = defaultdict(lambda: [0.0, 0])
    for e in dur:
        key = e.get("name")
        agg[key][0] += float(e["dur"])
        agg[key][1] += 1
    print("-- top 40 by total duration (us) --")
    print(f"  {'total_ms':>12s} {'count':>9s} {'avg_us':>10s}  name")
    for k, (tot, cnt) in sorted(agg.items(), key=lambda kv: -kv[1][0])[:40]:
        print(f"  {tot / 1e3:12.2f} {cnt:9d} {tot / cnt:10.2f}  {str(k)[:80]}")

    # time extent
    ts = [e.get("ts") for e in ev if e.get("ts") is not None]
    if ts:
        lo, hi = min(ts), max(ts)
        print(f"\ntimeline: ts {lo} -> {hi}  span={(hi - lo) / 1e6:.2f}s")

    print("\nMCTRACE_EXPLORE_DONE", flush=True)


if __name__ == "__main__":
    main()
