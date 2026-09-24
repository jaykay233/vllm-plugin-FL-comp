"""Why is FlagGems' MetaX `mm` slow?  Steady-state kernel quality vs autotune cost.

Design notes:
  * Native arm is measured FIRST, before flag_gems is imported, so a global
    aten patch cannot leak into the baseline. FlagGems is imported afterwards.
  * Each (shape, M) is timed twice: the very first call ("cold", which on a
    cache miss includes the Triton/libtuner autotune) and then a warmed loop
    ("warm"). cold - warm isolates the autotune cost from kernel quality.

MiniCPM5-2B on MetaX C500, bf16. Shapes come from the real model:
  hidden=2048, inter=6144, heads=16, kv_heads=2, head_dim=128
  qkv (2560,2048) | gate_up (12288,2048) | o_proj (2048,2048) | down (2048,6144)
"""

import time

import torch

torch.manual_seed(0)
DEV = "cuda"

SHAPES = [
    ("qkv", 2560, 2048),
    ("gate_up", 12288, 2048),
    ("o_proj", 2048, 2048),
    ("down", 2048, 6144),
]
M_LIST = [1, 32, 512, 2048]
COLD_M = [1009, 1777]  # odd values unlikely to be cached


def mk(M, N, K):
    a = torch.randn(M, K, device=DEV, dtype=torch.bfloat16)
    b = torch.randn(K, N, device=DEV, dtype=torch.bfloat16)
    return a, b


def timed(fn, a, b, iters):
    """Device time per call, in ms."""
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


def cold_ms(fn, a, b):
    fn(a, b)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn(a, b)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e3


# ---------------------------------------------------------------- native arm
print("=== NATIVE (torch.mm, before flag_gems import) ===", flush=True)
native = {}
for name, N, K in SHAPES:
    for M in M_LIST:
        a, b = mk(M, N, K)
        native[(name, M)] = timed(lambda x, y: torch.mm(x, y), a, b, 30)
    print(f"  {name:8s} N={N:6d} K={K:5d}  " + "  ".join(
        f"M{M}={native[(name, M)]:6.3f}" for _, M in [("", m) for m in M_LIST]
    ), flush=True)
del a, b
torch.cuda.empty_cache()

# ------------------------------------------------------------------ gems arm
print("\n=== load flag_gems (slow) ===", flush=True)
import flag_gems  # noqa: E402

print("  loaded:", flag_gems.__file__, flush=True)

print("\n=== FLAGGEMS (flag_gems.mm) ===", flush=True)
gems = {}
for name, N, K in SHAPES:
    for M in M_LIST:
        a, b = mk(M, N, K)
        gems[(name, M)] = timed(lambda x, y: flag_gems.mm(x, y), a, b, 30)
    print(f"  {name:8s} N={N:6d} K={K:5d}  " + "  ".join(
        f"M{m}={gems[(name, m)]:6.3f}" for m in M_LIST
    ), flush=True)

# ------------------------------------------------------- comparison + autotune
print("\n=== RATIO gems/native (warm; <1 means FlagGems faster) ===", flush=True)
print(f"  {'shape':8s} " + "  ".join(f"M{m:>5d}" for m in M_LIST))
for name, N, K in SHAPES:
    row = "  ".join(f"{gems[(name, m)] / native[(name, m)]:6.2f}" for m in M_LIST)
    print(f"  {name:8s} {row}", flush=True)

print("\n=== AUTOTUNE COST (first call on a fresh, odd M; shape o_proj) ===", flush=True)
N, K = 2048, 2048
for M in COLD_M:
    a, b = mk(M, N, K)
    c = cold_ms(lambda x, y: flag_gems.mm(x, y), a, b)
    w = timed(lambda x, y: flag_gems.mm(x, y), a, b, 30)
    print(f"  M={M:5d}  cold={c:9.1f} ms   warm={w:7.3f} ms   autotune~{c - w:8.1f} ms",
          flush=True)
