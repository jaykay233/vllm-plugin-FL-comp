"""Clean A/B: does the strategy-normalized ConfigCache key bound hot-path tuning?

Runs the real MetaX mm kernel over a decode-shaped M ramp twice, on separate
fresh DBs, forcing the tuner strategy via the .fn handle (so the on-disk mm.py
state does not matter):

  identity : key = raw M   -> one tuning per distinct M
  align32  : key = bucketed -> tunings only at bucket boundaries

Reports tuning count and, more importantly, the TOTAL tuning wall time, since a
single tuning measured ~9.5 s on MetaX (8 configs, each Triton-compiled then
benchmarked under BenchmarkMode.REPLAY).
"""

import importlib
import json
import os
import time
from collections import Counter

MMAX = int(os.environ.get("MM_MAX", "32"))
N = int(os.environ.get("MM_N", "2048"))
K = int(os.environ.get("MM_K", "2048"))
ARM = os.environ.get("MM_STRATEGY", "identity")

import torch  # noqa: E402

import flag_gems  # noqa: E402

LE = importlib.import_module("flag_gems.utils.libentry")

RUNS = []
_orig_run = LE.LibTuner.run


def probe_run(self, *a, **k):
    self.bench_time = None
    r = _orig_run(self, *a, **k)
    tuned = getattr(self, "bench_time", None) is not None
    RUNS.append((getattr(self, "__name__", "?"), tuned,
                 float(self.bench_time) if tuned else 0.0))
    return r


LE.LibTuner.run = probe_run

flag_gems.only_enable(include=["mm"], record=False, once=True)

mmmod = importlib.import_module("flag_gems.runtime.backend._metax.ops.mm")
KERNS = {
    "mm_kernel": 5, "mm_kernel_nn": 3, "mm_kernel_nt": 3,
    "mm_kernel_splitk": 5, "mm_kernel_splitk_partial": 5,
    "gemv_kernel": 4, "gemv_kernel_k_parallel_partial": 4,
}
n_set = 0
for kn, nkey in KERNS.items():
    inner = getattr(getattr(mmmod, kn, None), "fn", None)
    if inner is None or not hasattr(inner, "strategy"):
        continue
    inner.strategy = [(LE.align32_strategy if ARM == "align32" else LE.default_strategy)] * nkey
    n_set += 1

print(f"ARM={ARM} kernels_configured={n_set} M=1..{MMAX} N={N} K={K}")
print(f"  DB={os.environ.get('FLAGGEMS_DB_URL')}")

dev = "cuda"
w = torch.randn(N, K, device=dev, dtype=torch.bfloat16)
bt = w.t()

t0 = time.perf_counter()
for M in range(1, MMAX + 1):
    a = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
    torch.mm(a, bt)
    torch.cuda.synchronize()
wall = time.perf_counter() - t0

tuned = [(n, b) for n, t, b in RUNS if t]
res = {
    "arm": ARM,
    "kernels_configured": n_set,
    "M_max": MMAX,
    "tuner_calls": len(RUNS),
    "tunings": len(tuned),
    "total_bench_s": round(sum(b for _, b in tuned), 3),
    "wall_s": round(wall, 3),
    "per_tuning_s": round(sum(b for _, b in tuned) / len(tuned), 2) if tuned else 0,
    "by_kernel": dict(Counter(n for n, _ in tuned)),
}
print(json.dumps(res, indent=2))
