#!/usr/bin/env python3
"""Per-decode-step kernel budget from a vLLM rank0 torch trace.

Unlike analyze_trace.py (which prints raw trace totals), this normalises
everything by the number of captured decode steps, so each row answers
"what does this kernel cost in one 24.45 ms step, and how many times is it
launched?"  That is the form needed to judge whether a kernel is worth
attacking.

Also dumps the immediate predecessor/successor of a named kernel, which is how
the 8.9 us `DevDstElementwiseAdd` gets attributed to a source line: 42 calls
per step means one per layer, but *which* per-layer tensor is only visible from
its neighbours in stream order.
"""
from __future__ import annotations

import gzip
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load(path: Path) -> dict:
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as f:
            return json.load(f)
    return json.loads(path.read_text())


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    neighbours_of = None
    for i, a in enumerate(sys.argv[1:]):
        if a == "--neighbours" and i + 2 <= len(sys.argv[1:]):
            neighbours_of = sys.argv[2 + i]

    path = Path(args[0]) if args else max(
        HERE.joinpath("trace").glob("rank0*.gz"), key=lambda p: p.stat().st_mtime
    )
    print(f"trace: {path.name}\n", flush=True)

    tr = load(path)
    ev = tr.get("traceEvents", tr.get("events", []))

    # --- per-kernel GPU time and launch counts ---
    # `kernel` alone undercounts: on this stack a per-layer 256 KB D2D copy
    # (gpu_memcpy) cost 1.43 ms/step, 10% of GPU time, and is invisible if only
    # the `kernel` category is summed.
    dur_sum: dict[str, float] = defaultdict(float)
    calls: Counter[str] = Counter()
    for e in ev:
        if e.get("ph") != "X" or e.get("cat") not in ("kernel", "Kernel", "gpu_memcpy"):
            continue
        name = e.get("name", "")
        if e.get("cat") == "gpu_memcpy":
            nbytes = (e.get("args") or {}).get("bytes")
            if nbytes:
                name = f"{name} [{nbytes/1024:.0f} KiB]"
        dur_sum[name] += e.get("dur") or 0
        calls[name] += 1

    total = sum(dur_sum.values())
    if not total:
        print("no kernel events -- trace captured no device activity")
        return 1

    # --- how many decode steps does this trace cover? ---
    # Every decode step launches exactly one attention kernel per layer, so the
    # most-launched flash kernel count divided by ASM layers gives the steps.
    # Instead of guessing the layer count, infer steps from the graph replay
    # count when available, else from the sampler (1 per step).
    sampler = [n for n in calls if "_top_p_mass" in n or "top_p_sample" in n]
    steps = calls[sampler[0]] if sampler else None
    if steps:
        print(f"decode steps in trace: {steps} (from {sampler[0]})")
    else:
        candidate = max(calls.values())
        steps = candidate
        print(f"decode steps in trace: {steps} (heuristic: most-launched kernel)")

    print(f"GPU kernel total: {total/1000:.1f} ms over {steps} steps "
          f"-> {total/steps/1000:.2f} ms/step\n")

    print(f"{'ms/step':>9s} {'%step':>6s} {'calls/step':>11s}  kernel")
    for name, us in sorted(dur_sum.items(), key=lambda kv: -kv[1])[:24]:
        per_step = us / steps / 1000
        per_call = us / calls[name]
        print(
            f"{per_step:9.3f} {100*per_step/(total/steps/1000):5.1f}% "
            f"{calls[name]/steps:11.1f}  {name[:70]}"
            f"   [{per_call:.1f} us/call]"
        )

    if neighbours_of:
        print(f"\n=== stream order around kernels matching '{neighbours_of}' ===")
        ks = [
            e for e in ev
            if e.get("ph") == "X" and e.get("cat") in ("kernel", "Kernel")
            and neighbours_of in e.get("name", "")
        ]
        ks.sort(key=lambda e: e["ts"])
        if len(ks) > 60:
            ks = ks[:60]
        for i, e in enumerate(ks):
            prev = ks[i - 1] if i else None
            nxt = ks[i + 1] if i + 1 < len(ks) else None
            def short(x):
                return (x.get("name", "")[:52], round(x.get("dur", 0), 1)) if x else None
            print(f"  {e['dur']:7.1f} us   prev={short(prev)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
