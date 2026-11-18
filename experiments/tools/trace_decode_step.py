"""Full breakdown of one decode step's out-of-graph tail.

Picks the step whose duration is closest to the median, then prints every
kernel in it with its duration and the gap before it.  This separates
"kernel is slow" from "launch is slow", which the aggregate numbers cannot.
"""
from __future__ import annotations
import sqlite3
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/trace16k/t16k.ixrep"
NS = 1e6

c = sqlite3.connect(path)
rows = list(c.execute("""
    SELECT k.ts, k.dur, k.streamId, f.name
    FROM cuda_kernel_data k JOIN function_name_map f ON f.id = k.name_id
    ORDER BY k.ts
"""))
idx = [i for i, r in enumerate(rows) if r[3].startswith("linear_kernel")]

steps = []
for a, b in zip(idx, idx[1:]):
    steps.append(((rows[b][0] - rows[a][0]) / NS, a, b))
steps.sort()
med = [s for s in steps if 40 <= s[2] - s[1] <= 50][len([s for s in steps if 40 <= s[2] - s[1] <= 50]) // 2]
dur, a, b = med
print(f"median decode step: {dur:.2f} ms, {b - a} kernels\n")

print(f"{'#':>3} {'dur us':>8} {'gap us':>7}  name")
tot_k = tot_g = 0.0
for i in range(a, b):
    d = rows[i][1] / 1e3
    g = (rows[i][0] - (rows[i - 1][0] + rows[i - 1][1])) / 1e3 if i > a else 0.0
    tot_k += d
    tot_g += g
    print(f"{i - a:3d} {d:8.1f} {g:7.1f}  {rows[i][3][:64]}")
tail_end = rows[b][0]
print(f"\n{'':>11} {tot_k:8.1f} {tot_g:7.1f}  (sum of kernel dur / sum of gaps)")

# visible tail = last kernel's end minus first kernel's start
print(f"\nvisible tail from first to last kernel = {(tail_end - rows[a][0]) / NS:.2f} ms")
print(f"sum kernel duration                    = {tot_k / 1e3:.2f} ms")
print(f"sum gaps                               = {tot_g / 1e3:.2f} ms")
print(f"=> launch/queue overhead               = {(tot_g) / 1e3:.2f} ms "
      f"({100 * tot_g / 1e3 / dur:.1f}% of the step)")

# gap distribution
gaps = []
for i in range(a + 1, b):
    gaps.append((rows[i][0] - (rows[i - 1][0] + rows[i - 1][1])) / 1e3)
gaps.sort()
print(f"\ngap percentiles (us): p10 {gaps[len(gaps)//10]:.1f}  "
      f"p50 {gaps[len(gaps)//2]:.1f}  p90 {gaps[9*len(gaps)//10]:.1f}  "
      f"max {gaps[-1]:.1f}")
print("DONE")
