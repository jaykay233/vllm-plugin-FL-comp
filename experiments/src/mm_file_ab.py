"""A/B the strategy via the on-disk mm.py (the mechanism that is known to work),
not by mutating the tuner object at runtime.

The previous attempt assigned tuner.strategy directly and both arms behaved as
identity, so that override is unreliable. This script only counts tunings; the
strategy comes from whatever mm.py is installed.
"""

import importlib
import json
import os
import time
from collections import Counter

MMAX = int(os.environ.get("MM_MAX", "32"))
N = int(os.environ.get("MM_N", "2048"))
K = int(os.environ.get("MM_K", "2048"))
ARM = os.environ.get("MM_ARM", "?")

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
strategies = {}
for kn in ("mm_kernel", "mm_kernel_nn", "mm_kernel_nt", "mm_kernel_splitk",
           "mm_kernel_splitk_partial", "gemv_kernel",
           "gemv_kernel_k_parallel_partial"):
    inner = getattr(getattr(mmmod, kn, None), "fn", None)
    if inner is None:
        continue
    strategies[kn] = [getattr(s, "__name__", str(s)) for s in (inner.strategy or [])]

print(f"ARM={ARM} M=1..{MMAX} N={N} K={K} DB={os.environ.get('FLAGGEMS_DB_URL')}")
print(f"  .fn type = {type(getattr(mmmod, 'mm_kernel_nt', None).fn).__name__}")
for kn, s in strategies.items():
    print(f"  {kn:30s} {s}")

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
print(json.dumps({
    "arm": ARM,
    "M_max": MMAX,
    "tuner_calls": len(RUNS),
    "tunings": len(tuned),
    "total_bench_s": round(sum(b for _, b in tuned), 3),
    "wall_s": round(wall, 3),
    "by_kernel": dict(Counter(n for n, _ in tuned)),
}, indent=2))
