"""Log the actual autotune keys MetaX mm builds as M varies, plus the tuner
invocation count. This settles whether raw M reaches the ConfigCache key.
"""

import importlib
import os
import sys

MMAX = int(os.environ.get("MM_MAX", "12"))
N = int(os.environ.get("MM_N", "2048"))
K = int(os.environ.get("MM_K", "2048"))

import torch  # noqa: E402

import flag_gems  # noqa: E402

LE = importlib.import_module("flag_gems.utils.libentry")

RUNS = []
_orig_run = LE.LibTuner.run


def probe_run(self, *a, **k):
    self.bench_time = None
    self.nargs = dict(zip(self.arg_names, a))
    all_args = {**self.nargs, **k}
    _args = {x: y for x, y in all_args.items() if x in self.arg_names}
    try:
        key = self.get_key(_args)
    except Exception as e:  # noqa: BLE001
        key = f"<{type(e).__name__}: {e}>"
    r = _orig_run(self, *a, **k)
    tuned = getattr(self, "bench_time", None) is not None
    RUNS.append({
        "name": getattr(self, "__name__", "?"),
        "keys": list(self.keys),
        "strategy": [getattr(s, "__name__", str(s)) for s in (self.strategy or [])],
        "cfg_key": str(key),
        "tuned": tuned,
        "bench_s": round(float(self.bench_time), 3) if tuned else 0.0,
        "nconfigs": len(self.configs),
    })
    return r


LE.LibTuner.run = probe_run

flag_gems.only_enable(include=["mm"], record=False, once=True)

mmmod = importlib.import_module("flag_gems.runtime.backend._metax.ops.mm")
print("=== mmmod 里带 tuner 的候选 ===")
for attr in dir(mmmod):
    obj = getattr(mmmod, attr)
    inner = getattr(obj, "fn", None)
    if inner is not None and hasattr(inner, "keys"):
        print(f"  {attr:34s} .fn.keys={inner.keys} strategy="
              f"{[getattr(s,'__name__','?') for s in (inner.strategy or [])]}")

dev = "cuda"
w = torch.randn(N, K, device=dev, dtype=torch.bfloat16)
bt = w.t()

print(f"\n=== 扫描 M=1..{MMAX}  N={N} K={K} ===")
for M in range(1, MMAX + 1):
    before = len(RUNS)
    a = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
    torch.mm(a, bt)
    torch.cuda.synchronize()
    for r in RUNS[before:]:
        mark = "TUNE" if r["tuned"] else "hit "
        print(f"  M={M:4d} {r['name']:22s} ncfgs={r['nconfigs']:3d} {mark} "
              f"key={r['cfg_key']} bench={r['bench_s']}s")

print("\n=== 汇总 ===")
tuned = [r for r in RUNS if r["tuned"]]
print(f"  总 tuner 调用 {len(RUNS)}, 其中真 tuning {len(tuned)}, "
      f"累计 bench {round(sum(r['bench_s'] for r in tuned), 2)}s")
from collections import Counter
print(f"  tuning 分布: {dict(Counter(r['name'] for r in tuned))}")
print(f"  不同 cfg_key 数: {len(set(r['cfg_key'] for r in RUNS))}")
print("\n  所有 cfg_key:")
for k in sorted(set(r["cfg_key"] for r in RUNS)):
    print(f"    {k}")
