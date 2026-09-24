"""Decisive check: is the align32 M-bucketing strategy actually live on the
MetaX mm tuners, or does it resolve to None (raw M key -> autotune storm)?

Hypothesis that fits all three arms:
  linear_kernel   passes strategy=["align32",...]   -> M bucketed -> no storm
  MetaX mm_*      passes no strategy                -> raw M      -> storm
  rms_norm        key is ["N"] only                 -> no M       -> no storm

If the mm tuners come back with strategy None (or all-default), the fix is to
pass the align32 strategy that DEFAULT_STRATEGIES already declares for mm/mm_nt.
"""

import importlib
import os

os.environ.setdefault("USE_FLAGGEMS", "1")
import flag_gems  # noqa: F401
from flag_gems.runtime import common

GR = "flag_gems.runtime.backend._metax.ops"


def describe(tag, obj):
    print(f"  --- {tag} ---")
    for attr in ("keys", "strategy", "_flagtune_op_name", "_flagtune_expand_op_name",
                 "_flagtune_mode", "_flagtune_default_strategy"):
        try:
            v = getattr(obj, attr)
        except Exception as e:  # noqa: BLE001
            v = f"<{type(e).__name__}>"
        print(f"      {attr:28s} = {v}")


print("=== DEFAULT_STRATEGIES 声称的值 ===")
for k in ("mm", "mm_nt", "mm_nn", "mm_splitk"):
    print(f"  {k:12s} {common.DEFAULT_STRATEGIES.get(k)}")

print("\n=== MetaX mm 各 kernel 的实际 strategy ===")
mmmod = importlib.import_module(GR + ".mm")
for name in ("mm_kernel", "mm_kernel_nt", "mm_kernel_nn", "mm_kernel_splitk"):
    fn = getattr(mmmod, name, None)
    if fn is None:
        print(f"  --- {name}: 未找到 ---")
        continue
    describe(name, fn)
    # libentry 会把原 JITFunction 藏在 wrapper 里，尝试下钻
    for inner_attr in ("fn", "jit_fn", "wrapped", "_fn", "_jit_fn"):
        inner = getattr(fn, inner_attr, None)
        if inner is not None and inner is not fn:
            describe(f"{name}.{inner_attr}", inner)

print("\n=== 对照：通用 linear_kernel 的 strategy ===")
lin = importlib.import_module("flag_gems.ops.linear")
describe("linear_kernel", lin.linear_kernel)
for inner_attr in ("fn", "jit_fn", "wrapped", "_fn", "_jit_fn"):
    inner = getattr(lin.linear_kernel, inner_attr, None)
    if inner is not None and inner is not lin.linear_kernel:
        describe(f"linear_kernel.{inner_attr}", inner)

print("\n=== 运行时 TuningMode ===")
try:
    from flag_gems.runtime import TuningMode
    print(f"  TuningMode = {list(TuningMode)}")
except Exception as e:  # noqa: BLE001
    print(f"  TuningMode 取不到: {e}")
for env in ("FLAGGEMS_TUNING_MODE", "FLAGGEMS_TUNE", "FLAGTUNE_MODE", "TUNING_MODE"):
    print(f"  env {env} = {os.environ.get(env)}")
