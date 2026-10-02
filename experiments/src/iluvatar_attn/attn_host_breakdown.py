"""Host self time of every op nested inside vllm::unified_attention_with_output.

Usage: attn_host_breakdown.py <trace.json>

Self time = an op's duration minus its direct children, so the list shows
where the attention call's host time actually goes, per attention call.
"""

from __future__ import annotations

import collections
import json
import sys

TARGET = "vllm::unified_attention_with_output"
CATS = ("cpu_op", "user_annotation", "cuda_runtime", "cuda_driver")


def main(path: str) -> None:
    with open(path) as f:
        ev = json.load(f)["traceEvents"]
    ev = [e for e in ev if e.get("ph") == "X" and e.get("cat") in CATS]
    ev.sort(key=lambda e: (e.get("tid"), e["ts"], -e["dur"]))
    calls = [e for e in ev if e["name"] == TARGET]
    if not calls:
        print("no attention calls in trace")
        return
    n = len(calls)
    print(f"{TARGET}: {n} calls, avg inclusive {sum(e['dur'] for e in calls) / n / 1e3:.2f} ms")

    by_tid = collections.defaultdict(list)
    for e in ev:
        by_tid[e.get("tid")].append(e)
    for lst in by_tid.values():
        stack = []
        for e in lst:
            while stack and stack[-1]["ts"] + stack[-1]["dur"] <= e["ts"]:
                stack.pop()
            if any(s["name"] == TARGET for s in stack):
                stack[-1]["_child"] = stack[-1].get("_child", 0) + e["dur"]
                e["_in"] = True
            stack.append(e)

    incl, selft, cnt = collections.Counter(), collections.Counter(), collections.Counter()
    for e in ev:
        if e.get("_in"):
            incl[e["name"]] += e["dur"]
            selft[e["name"]] += e["dur"] - e.get("_child", 0)
            cnt[e["name"]] += 1
    print("per attention call (us):")
    for name, us in selft.most_common(25):
        print(f"  self {us / n:8.1f}  incl {incl[name] / n:8.1f}  "
              f"calls {cnt[name] / n:5.1f}  {name[:80]}")


if __name__ == "__main__":
    main(sys.argv[1])
