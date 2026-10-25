#!/usr/bin/env python3
"""Post-fix decode budget: where the remaining ~17.7 ms/step goes, and how much
of each part is real headroom versus bandwidth already at the wall.

The attention writeback memcpy is gone (FINDINGS 25), so the 42 per-layer
Memcpy D2D entries that used to show up are excluded here. What remains is
grouped by role, and each group is converted to an effective bandwidth so it can
be compared against what this card actually sustains:

  attention  -- KV is read once per step; bytes are exact and unavoidable
  projections -- weights are read once per step; bytes are exact
  lm_head    -- weights read once per step; bytes are exact
  top-p      -- elementwise passes over the [64, vocab] logits tile

Anything sitting near the attention bandwidth is at the wall and not worth
touching; anything well below it is leaving bandwidth on the table.

Model: 42 layers, hidden 2048, 16 q heads / 2 kv heads, head_dim 128,
intermediate 6144, vocab 130560, bf16.
"""
from __future__ import annotations

import gzip
import json
import re
import sys
from collections import defaultdict

PATH = sys.argv[1] if len(sys.argv) > 1 else (
    "trace/rank0.1792907289481674743.pt.trace.json.gz")

# ── model geometry ────────────────────────────────────────────────────────
LAYERS, HID, KVH, HD, INTER, VOCAB = 42, 2048, 2, 128, 6144, 130560
BATCH = 64          # concurrent decode requests
CTX = 4096          # decode sequence length
BYTES = 2           # bf16

kv_bytes_step = BATCH * CTX * KVH * HD * 2 * BYTES * LAYERS
attn_w = sum([
    HID * HID,          # q
    HID * KVH * HD,     # k
    HID * KVH * HD,     # v
    HID * HID,          # o
]) * BYTES
mlp_w = 3 * HID * INTER * BYTES
w_step = (attn_w + mlp_w) * LAYERS
lmhead_w = HID * VOCAB * BYTES
logits_b = BATCH * VOCAB * BYTES

with gzip.open(PATH) as f:
    tr = json.load(f)
ev = tr.get("traceEvents") or tr.get("events")
kern = [e for e in ev if e.get("cat") == "kernel" and e.get("ph") == "X"
        and e.get("ts") is not None]
mcs = [e for e in ev if e.get("cat") == "gpu_memcpy" and e.get("ph") == "X"]

step_ts = sorted(e["ts"] for e in kern if str(e.get("name")) == "_top_p_mass_v2_kernel")
n = len(step_ts) - 1
print(f"decode steps: {n}\n")

# attribute every event to a step, then to a role
ROLES = [
    ("attention (K/V read)", lambda s: "flash_fwd_splitkv_kernel" in s),
    ("attention combine", lambda s: "flash_fwd_splitkv_combine" in s),
    ("projections (weights)", lambda s: "Mck_bf16gemm_tn" in s),
    ("lm_head", lambda s: "mm_kernel_nt" in s),
    ("sampling top-p", lambda s: "_top_p_" in s),
    ("softmax", lambda s: "SoftMaxForward" in s),
    ("RNG sample", lambda s: "distribution_elementwise" in s),
    ("thrust/cub scans", lambda s: "thrust::" in s or "cub::" in s),
    ("rmsnorm (fused add)", lambda s: "FusedAddRMSNorm" in s),
    ("rmsnorm (final)", lambda s: "rms_norm_kernel" in s),
    ("GEMM epilogue bias add", lambda s: "DevDstElementwiseAdd" in s),
    ("silu+gate mul", lambda s: "mul_silu" in s),
    ("qkv/pack concat", lambda s: "triton_poi_fused_cat" in s),
    ("kv cache write", lambda s: "reshape_and_cache" in s),
    ("embedding gather", lambda s: "_to_copy_embedding" in s),
    ("slot mapping", lambda s: "_compute_slot_mapping" in s),
]

tot = defaultdict(float)
for i in range(n):
    a, b = step_ts[i], step_ts[i + 1]
    for e in kern:
        if not (a <= e["ts"] < b):
            continue
        s = str(e.get("name"))
        for label, pred in ROLES:
            if pred(s):
                tot[label] += e["dur"]
                break
        else:
            tot["other"] += e["dur"]

wall = (step_ts[-1] - step_ts[0]) / n / 1000
gpu = sum(tot.values()) / n / 1000

print(f"wall {wall:.2f} ms/step   attributed gpu {gpu:.2f} ms/step\n")
print(f"{'role':<26} {'ms/step':>8} {'%':>6} {'GB/s':>8} {'vs attn':>8}")
print("-" * 62)

by = {}
for label, _ in ROLES:
    ms = tot[label] / n / 1000
    if ms <= 0:
        continue
    if label == "attention (K/V read)":
        bw = kv_bytes_step / (ms / 1000) / 1e9
        ref = bw
    elif label == "projections (weights)":
        bw = w_step / (ms / 1000) / 1e9
    elif label == "lm_head":
        bw = lmhead_w / (ms / 1000) / 1e9
    elif label.startswith(("sampling", "softmax", "thrust")):
        bw = logits_b / (ms / 1000) / 1e9
    else:
        bw = float("nan")
    by[label] = (ms, bw)
    rel = f"{bw / by['attention (K/V read)'][1] * 100:.0f}%" if label != "attention (K/V read)" and bw == bw else "—"
    print(f"{label:<26} {ms:8.2f} {ms / wall * 100:5.1f}% "
          f"{(f'{bw:8.0f}' if bw == bw else '       -')} {rel:>8}")

print(f"\nbyte budgets that set those bandwidths")
print(f"  KV read          {kv_bytes_step / 2**30:6.2f} GiB/step "
      f"({BATCH} x {CTX} tok x {KVH} kvh x {HD} d x 2 x 2 B x {LAYERS} L) -- unavoidable")
print(f"  weights          {w_step / 2**30:6.2f} GiB/step -- unavoidable")
print(f"  lm_head weights  {lmhead_w / 2**30:6.2f} GiB/step -- unavoidable")
print(f"  logits tile      {logits_b / 2**20:6.1f} MiB -- per elementwise pass")

# ── headroom if the slow groups reached attention's bandwidth ─────────────
ref_bw = by["attention (K/V read)"][1]
print(f"\nheadroom if a group reached the attention bandwidth ({ref_bw:.0f} GB/s)")
save = 0.0
for label, (ms, bw) in by.items():
    if bw != bw or label == "attention (K/V read)":
        continue
    ideal = ms * bw / ref_bw
    d = ms - ideal
    if d > 0.05:
        print(f"  {label:<26} {ms:5.2f} -> {ideal:5.2f} ms   save {d:5.2f} ms")
        save += d
print(f"  {'total':<26} {'':>14}   save {save:5.2f} ms "
      f"({save / wall * 100:.1f}% of step)")

# ── host traffic per step ─────────────────────────────────────────────────
print("\nhost<->device traffic per step")
kind = defaultdict(lambda: [0, 0, 0])
for e in mcs:
    if not (step_ts[0] <= e["ts"] < step_ts[-1]):
        continue
    nm = str(e.get("name"))
    b = int(e.get("args", {}).get("bytes", 0) or 0)
    k = (nm, b)
    kind[k][0] += 1
    kind[k][1] += e["dur"]
    kind[k][2] = max(kind[k][2], e["dur"])
for (nm, b), (c, d, mx) in sorted(kind.items(), key=lambda kv: -kv[1][1])[:8]:
    print(f"  {c / n:5.1f}x  {d / n:7.1f} us  {b:>8} B  max {mx:6.1f} us  {nm}")

print(f"\nidle / launch gaps: {wall - gpu:.2f} ms/step "
      f"({(wall - gpu) / wall * 100:.1f}% of wall)")
