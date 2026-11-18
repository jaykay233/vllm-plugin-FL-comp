"""Are CUDA-graph-internal kernels visible to ixsys?

linear_kernel (lm_head) runs once per decode step, outside the graph.  Count the
kernels between consecutive occurrences: a fully-visible 42-layer decode step
should show on the order of 42 x (6..10) kernels; a graph-opaque step shows a
handful.
"""
from __future__ import annotations
import sqlite3
import sys
from collections import Counter

path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/trace16k/t16k.ixrep"
c = sqlite3.connect(path)
rows = list(c.execute("""
    SELECT k.ts, k.dur, f.name
    FROM cuda_kernel_data k JOIN function_name_map f ON f.id = k.name_id
    ORDER BY k.ts
"""))

idx = [i for i, (_, _, n) in enumerate(rows) if n.startswith("linear_kernel")]
print(f"linear_kernel occurrences = {len(idx)}")

per_step = []
for a, b in zip(idx, idx[1:]):
    per_step.append(b - a)
cnt = Counter(per_step)
print("\nkernels between consecutive linear_kernel calls (top 12):")
for k, v in cnt.most_common(12):
    print(f"  {k:6d} kernels  x{v}")

# how many distinct kernel names appear INSIDE a step
if per_step:
    mode = cnt.most_common(1)[0][0]
    a = idx[0]
    b = idx[0] + mode
    names = [rows[i][2] for i in range(a, b)]
    print(f"\ninside a {mode}-kernel step, distinct names = {len(set(names))}")
    for n in sorted(set(names)):
        print(f"    {n[:70]}")
    # total duration of that step
    print(f"\nstep duration = {(rows[b][0] - rows[a][0]) / 1e6:.2f} ms, "
          f"sum of kernel dur = {sum(rows[i][1] for i in range(a, b)) / 1e6:.2f} ms")
print("DONE")
