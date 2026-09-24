"""Verify the mm_splitk pipeline fix: speed AND numerics.

The fix changed every `pipeline: "null"` META entry in _metax/tune_configs.yaml
to "basic". The Triton MetaX backend only lowers basic / cpasync / cpasync-mixed
/ mixed, so "null" used to hit the else branch ("no avalilable pipeline for
maca") and compile the kernel with no software pipeline.

Coverage:
  o_proj / down at M=2, 8   -> splitk_mm_scenario is true, so these are the
                               paths the fix is supposed to change
  o_proj / down at M=32,2048 -> general/nt path, a control that must not regress
  qkv at M=8                -> nt path, second control

Native is measured before flag_gems is enabled; both arms then call torch.mm,
so only the backend differs. Numerics are checked as gems-vs-native max abs
diff on identical inputs.
"""

import torch

torch.manual_seed(0)
DEV = "cuda"

CASES = [
    ("o_proj", 2048, 2048, 2), ("o_proj", 2048, 2048, 8),
    ("o_proj", 2048, 2048, 32), ("o_proj", 2048, 2048, 2048),
    ("down", 2048, 6144, 2), ("down", 2048, 6144, 8),
    ("down", 2048, 6144, 32), ("down", 2048, 6144, 2048),
    ("qkv", 2560, 2048, 8), ("qkv", 2560, 2048, 2048),
]
ITERS = {2: 200, 8: 200, 32: 100, 2048: 30}


def mk(M, N, K):
    a = torch.randn(M, K, device=DEV, dtype=torch.bfloat16)
    w = torch.randn(N, K, device=DEV, dtype=torch.bfloat16)
    return a, w.t()


def timed(fn, a, b, iters):
    for _ in range(3):
        fn(a, b)
    torch.cuda.synchronize()
    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(iters):
        fn(a, b)
    t1.record()
    torch.cuda.synchronize()
    return t0.elapsed_time(t1) / iters


print("=== NATIVE (before flag_gems) ===", flush=True)
native_ms, ref = {}, {}
for name, N, K, M in CASES:
    a, b = mk(M, N, K)
    native_ms[(name, M)] = timed(lambda x, y: torch.mm(x, y), a, b, ITERS[M])
    ref[(name, M)] = torch.mm(a, b)

print("=== enable flag_gems (mm only) ===", flush=True)
import flag_gems  # noqa: E402

flag_gems.only_enable(include=["mm"], record=False, once=True)

print("=== FLAGGEMS (after fix) ===", flush=True)
gems_ms, out = {}, {}
for name, N, K, M in CASES:
    a, b = mk(M, N, K)
    gems_ms[(name, M)] = timed(lambda x, y: torch.mm(x, y), a, b, ITERS[M])
    out[(name, M)] = torch.mm(a, b)

print("\n=== RESULT ===")
print(f"  {'case':18s} {'native ms':>10s} {'gems ms':>10s} {'ratio':>7s} {'max|diff|':>11s}")
for name, N, K, M in CASES:
    r = gems_ms[(name, M)] / native_ms[(name, M)]
    d = (out[(name, M)].float() - ref[(name, M)].float()).abs().max().item()
    print(f"  {name + ' M' + str(M):18s} {native_ms[(name, M)]:10.4f} "
          f"{gems_ms[(name, M)]:10.4f} {r:7.2f} {d:11.4f}", flush=True)
