"""Verify align32_geometric is registered and measure the real tuning count.

Runs the actual MetaX mm kernel over an M sweep with a fresh DB and counts how
many times the tuner genuinely tunes, for the strategy the installed mm.py
declares. Compare against a run with align32 to quantify the win.

Reports bucket count too, which is the deterministic quantity that predicts
worst-case tuning time.
"""

import importlib
import json
import os
import time
from collections import Counter

MMAX = int(os.environ.get("MM_MAX", "2048"))
N = int(os.environ.get("MM_N", "2048"))
K = int(os.environ.get("MM_K", "2048"))
STEP = int(os.environ.get("MM_STEP", "1"))
LABEL = os.environ.get("MM_LABEL", "?")

import torch  # noqa: E402

import flag_gems  # noqa: E402

LE = importlib.import_module("flag_gems.utils.libentry")
TS = importlib.import_module(
    "flag_gems.runtime.backend._metax.tuner_strategies")

print(f"策略注册检查: align32_geometric -> {TS.align32_geometric}")
print(f"  threshold = {TS.ALIGN32_GEOMETRIC_THRESHOLD}")
print(f"  抽样: " + ", ".join(
    f"{m}->{TS.align32_geometric(m)}" for m in (1, 33, 100, 128, 129, 256, 1500, 2048)))

RUNS = []
_orig_run = LE.LibTuner.run


def probe_run(self, *a, **k):
    self.bench_time = None
    r = _orig_run(self, *a, **k)
    tuned = getattr(self, "bench_time", None) is not None
    if tuned:
        RUNS.append((getattr(self, "__name__", "?"), float(self.bench_time)))
    return r


LE.LibTuner.run = probe_run

flag_gems.only_enable(include=["mm"], record=False, once=True)

mmmod = importlib.import_module("flag_gems.runtime.backend._metax.ops.mm")
strats = {}
for kn in ("mm_kernel", "mm_kernel_nn", "mm_kernel_nt", "mm_kernel_splitk"):
    inner = getattr(getattr(mmmod, kn, None), "fn", None)
    if inner is not None:
        strats[kn] = [getattr(s, "__name__", "?") for s in (inner.strategy or [])]
print("运行时 mm 族 strategy:")
for kn, s in strats.items():
    print(f"  {kn:22s} {s}")

dev = "cuda"
w = torch.randn(N, K, device=dev, dtype=torch.bfloat16)
bt = w.t()

Ms = list(range(1, MMAX + 1, STEP))
t0 = time.perf_counter()
for M in Ms:
    a = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
    torch.mm(a, bt)
    torch.cuda.synchronize()
wall = time.perf_counter() - t0

res = {
    "label": LABEL,
    "M_swept": f"1..{MMAX} step {STEP} ({len(Ms)} values)",
    "tunings": len(RUNS),
    "total_bench_s": round(sum(b for _, b in RUNS), 2),
    "wall_s": round(wall, 1),
    "by_kernel": dict(Counter(n for n, _ in RUNS)),
}
print(json.dumps(res, indent=2))
