"""Count how many times the MetaX mm tuner actually tunes when M sweeps.

Mechanism under test (flag_gems/utils/libentry.py LibTuner.run):

    config_key = self.get_key(_args)        # strategy-normalized
    if config_key not in self.cache:        # miss -> benchmark every config
        ... bench ...; self.cache[config_key] = best_config

get_key applies self.strategy to each key element:
  * identity  -> raw M -> a new ConfigCache key for EVERY M -> tune every step
  * align32   -> M bucketed to 32 -> a fixed, small set of keys -> tunes once per bucket

This script drives the real kernel through torch.mm, counts genuine tuning
events (detected via bench_time being written), and reports wall time.
Run it twice with a fresh FLAGGEMS_DB_URL to compare the arms fairly.
"""

import json
import os
import sys
import time

ALIGN = os.environ.get("MM_STRATEGY", "identity") == "align32"
MMAX = int(os.environ.get("MM_MAX", "64"))
N = int(os.environ.get("MM_N", "2048"))
K = int(os.environ.get("MM_K", "2048"))

import torch  # noqa: E402
import triton  # noqa: E402

import importlib  # noqa: E402

import flag_gems  # noqa: E402

# NB: `flag_gems.utils.libentry` the submodule is shadowed by the `libentry`
# decorator re-exported in flag_gems.utils, so import it by module path.
LE = importlib.import_module("flag_gems.utils.libentry")

STATS = {"tuning_events": 0, "bench_s": 0.0, "calls": 0, "by_kernel": {}}

_orig_run = LE.LibTuner.run


def counting_run(self, *a, **k):
    self.bench_time = None          # reset so we can detect a tune this call
    t0 = time.perf_counter()
    r = _orig_run(self, *a, **k)
    dt = time.perf_counter() - t0
    STATS["calls"] += 1
    if getattr(self, "bench_time", None) is not None:
        STATS["tuning_events"] += 1
        STATS["bench_s"] += float(self.bench_time)
        name = getattr(self, "__name__", "?")
        STATS["by_kernel"][name] = STATS["by_kernel"].get(name, 0) + 1
    STATS.setdefault("wall_s", 0.0)
    STATS["wall_s"] += dt
    return r


LE.LibTuner.run = counting_run

flag_gems.only_enable(include=["mm"], record=False, once=True)

# Configure the strategy on every MetaX mm-family tuner, mirroring what the
# patched decorators would do.
import flag_gems.runtime.backend._metax.ops.mm as mmmod  # noqa: E402

KERNELS = {
    "mm_kernel": 5,
    "mm_kernel_nn": 3,
    "mm_kernel_nt": 3,
    "mm_kernel_splitk": 5,
    "mm_kernel_splitk_partial": 5,
    "gemv_kernel": 4,
    "gemv_kernel_k_parallel_partial": 4,
}
changed = []
for kn, nkey in KERNELS.items():
    obj = getattr(mmmod, kn, None)
    if obj is None:
        continue
    tuner = getattr(obj, "fn", None)
    if tuner is None or not hasattr(tuner, "strategy"):
        continue
    if ALIGN:
        tuner.strategy = [LE.align32_strategy] * nkey
    else:
        tuner.strategy = [LE.default_strategy] * nkey
    changed.append(kn)

print(f"MM_STRATEGY={os.environ.get('MM_STRATEGY','identity')} "
      f"kernels_configured={len(changed)} align={ALIGN}")
print(f"  DB={os.environ.get('FLAGGEMS_DB_URL')}")

dev = "cuda"
# mm_nt layout: a (M,K) contiguous, b = W.t() with W (N,K) contiguous
w = torch.randn(N, K, device=dev, dtype=torch.bfloat16)
bt = w.t()

sweep_t0 = time.perf_counter()
for M in range(1, MMAX + 1):
    a = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
    torch.mm(a, bt)
    torch.cuda.synchronize()
sweep_wall = time.perf_counter() - sweep_t0

# second pass over the same M values: with a working key strategy this must add
# zero tunings; with identity it depends on whether raw M values were cached.
before = STATS["tuning_events"]
for M in range(1, MMAX + 1):
    a = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
    torch.mm(a, bt)
    torch.cuda.synchronize()
second_pass_events = STATS["tuning_events"] - before

out = {
    "arm": os.environ.get("MM_STRATEGY", "identity"),
    "M_range": [1, MMAX],
    "N": N,
    "K": K,
    "tuning_events_pass1": before,
    "tuning_events_pass2": second_pass_events,
    "tuning_events_total": STATS["tuning_events"],
    "total_bench_s": round(STATS["bench_s"], 3),
    "total_wall_s": round(sweep_wall, 3),
    "by_kernel": STATS["by_kernel"],
}
print(json.dumps(out, indent=2))
