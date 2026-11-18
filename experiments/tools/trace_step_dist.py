"""Step-level wall-time distribution from the existing 16k trace.

linear_kernel (lm_head) runs exactly once at the end of every engine step, so
the interval between consecutive occurrences is one engine step.  Steps that
carry prefill work show up as much longer intervals, so the distribution splits
cleanly into decode steps and prefill/mixed steps.
"""
from __future__ import annotations
import sqlite3
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/trace16k/t16k.ixrep"
NS = 1e6

c = sqlite3.connect(path)
rows = list(c.execute("""
    SELECT k.ts, k.dur, f.name
    FROM cuda_kernel_data k JOIN function_name_map f ON f.id = k.name_id
    ORDER BY k.ts
"""))

idx = [i for i, (_, _, n) in enumerate(rows) if n.startswith("linear_kernel")]
print(f"engine steps = {len(idx) - 1}")

steps = []
for a, b in zip(idx, idx[1:]):
    dur = (rows[b][0] - rows[a][0]) / NS
    nkern = b - a
    # sum of visible (out-of-graph) kernel time in the step
    vis = sum(rows[i][1] for i in range(a, b)) / NS
    steps.append((dur, nkern, vis))

steps_sorted = sorted(steps)
n = len(steps_sorted)
print(f"\nstep duration percentiles (ms):")
for p in (1, 5, 10, 25, 50, 75, 90, 95, 99):
    print(f"  p{p:<3d} {steps_sorted[int(n * p / 100)] [0]:8.2f}")

med = steps_sorted[n // 2][0]
print(f"\nmedian step = {med:.2f} ms")

# split: a step is "decode-like" if its visible kernel count is ~43
dec = [s for s in steps if 40 <= s[1] <= 50]
oth = [s for s in steps if not (40 <= s[1] <= 50)]
print(f"decode-like steps (40..50 kernels): {len(dec)}  "
      f"median {sorted(x[0] for x in dec)[len(dec)//2]:.2f} ms")
print(f"other steps:                       {len(oth)}")
if oth:
    d = sorted(x[0] for x in oth)
    print(f"  their durations: min {d[0]:.1f}  median {d[len(d)//2]:.1f}  "
          f"max {d[-1]:.1f} ms   total {sum(d):.1f} ms")
    for lbl, grp in (("decode-like", dec), ("other", oth)):
        tot = sum(x[0] for x in grp)
        print(f"  wall share {lbl:<12} {tot:8.1f} ms "
              f"({100 * tot / sum(x[0] for x in steps):4.1f}%)")

print("\nthe 8 slowest steps (likely prefill chunks):")
for dur, nkern, vis in sorted(steps, reverse=True)[:8]:
    print(f"  {dur:8.2f} ms  {nkern:5d} kernels  visible {vis:7.2f} ms")
print("DONE")
