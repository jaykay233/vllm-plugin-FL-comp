"""Attribute pure-prefill steps and compute per-component TFLOPS.

Setup: prompts are 4096 tokens and max_num_batched_tokens is 4096, so one
engine step is one sequence's full 4096-token prefill.  That makes the FLOP
counts exact:

  linear : 2 * n_params_layer * seq_len * n_layers   (n_params_layer = 47.2M)
  attn   : 42 layers * 4096 * s^2  (derived from s(s+1)/2 pairs * 4 * d * heads)
"""
from __future__ import annotations
import sqlite3
import sys
from collections import defaultdict

path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/trace16k/pf.ixrep"
NS = 1e6
PARAM_PER_LAYER = 47.2e6      # q,k,v,o (9.44M) + gate,up,down (37.75M)
N_LAYERS = 42
PEAK_FLOPS = 96.6e12
SEQ = 4096
DEC = 64

c = sqlite3.connect(path)
rows = list(c.execute("""
    SELECT k.ts, k.dur, f.name
    FROM cuda_kernel_data k JOIN function_name_map f ON f.id = k.name_id
    ORDER BY k.ts
"""))
print(f"kernels = {len(rows)}")
t0 = rows[0][0]
print(f"span = {(rows[-1][0] + rows[-1][1] - t0) / NS / 1000:.1f}s")

# find engine steps via linear_kernel (lm_head, once per step)
idx = [i for i, r in enumerate(rows) if r[2].startswith("linear_kernel")]
print(f"steps = {max(0, len(idx) - 1)}")

# classify each step by kernel count and pick prefill-only steps (they have
# ~42 mm_kernel + 42 fa + 42 rms_norm, and no paged_decode)
steps = []
for a, b in zip(idx, idx[1:]):
    names = [rows[i][2] for i in range(a, b)]
    steps.append((a, b,
                  sum(1 for n in names if n.startswith("mm_kernel")),
                  sum(1 for n in names if "flash_attn" in n),
                  sum(1 for n in names if "_paged_decode" in n),
                  (rows[b][0] - rows[a][0]) / NS))
pure = [s for s in steps if s[2] >= 40 and s[3] >= 40]
mixed = [s for s in steps if s[2] >= 40 and s[4] > 0]
dec = [s for s in steps if s[2] < 40]
print(f"pure-prefill steps {len(pure)} | mixed {len(mixed)} | decode-only {len(dec)}")
if not pure:
    print("no pure prefill steps")
    sys.exit(1)

durs = sorted(s[5] for s in pure)
print(f"pure prefill step ms: min {durs[0]:.1f}  median {durs[len(durs) // 2]:.1f}  "
      f"max {durs[-1]:.1f}")

# aggregate kernels across pure prefill steps
a, b = pure[len(pure) // 2][0], pure[len(pure) // 2][1]
wall = (rows[b][0] - rows[a][0]) / NS
agg = defaultdict(lambda: [0, 0.0])
for i in range(a, b):
    k = agg[rows[i][2]]
    k[0] += 1
    k[1] += rows[i][1] / NS
ktot = sum(v[1] for v in agg.values())
print(f"\nmedian pure-prefill step: wall {wall:.2f} ms, kernels {ktot:.2f} ms, "
      f"gaps {wall - ktot:.2f} ms")
print(f"\n{'ms':>8} {'%':>6} {'n':>4} {'avg us':>9}  kernel")
for n, (cnt, tot) in sorted(agg.items(), key=lambda x: -x[1][1])[:14]:
    print(f"{tot:8.2f} {100 * tot / ktot:6.1f} {cnt:4d} {tot * 1e3 / cnt:9.1f}  {n[:58]}")

gemm_ms = sum(v[1] for n, v in agg.items()
              if n.startswith("mm_kernel") or n.startswith("linear_kernel"))
fa_ms = sum(v[1] for n, v in agg.items() if "flash_attn" in n)
other_ms = ktot - gemm_ms - fa_ms

lin_flops = 2 * PARAM_PER_LAYER * (SEQ + DEC) * N_LAYERS
attn_flops = N_LAYERS * 4096 * SEQ * SEQ
print(f"\n=== per-component efficiency ===")
print(f"  GEMM      {gemm_ms:7.2f} ms  {lin_flops / 1e12:6.2f} TFLOP  "
      f"-> {lin_flops / (gemm_ms * 1e-3) / 1e12:5.1f} TFLOPS "
      f"({100 * lin_flops / (gemm_ms * 1e-3) / PEAK_FLOPS:.0f}% peak)")
print(f"  attention {fa_ms:7.2f} ms  {attn_flops / 1e12:6.2f} TFLOP  "
      f"-> {attn_flops / (fa_ms * 1e-3) / 1e12:5.1f} TFLOPS "
      f"({100 * attn_flops / (fa_ms * 1e-3) / PEAK_FLOPS:.0f}% peak)")
print(f"  other     {other_ms:7.2f} ms  (rmsnorm / silu / cast / kv write)")
print(f"  total FLOPs {(lin_flops + attn_flops) / 1e12:.2f} TFLOP over "
      f"{ktot:.2f} ms -> {100 * (lin_flops + attn_flops) / (ktot * 1e-3) / PEAK_FLOPS:.0f}% of peak")
print("DONE")
