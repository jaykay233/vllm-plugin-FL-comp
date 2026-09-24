"""mm probe v2 - faithful layouts and the aten dispatch path the server uses.

v1 mistakes this fixes:
  * v1 used b = randn(K, N) contiguous, so b.stride() == (N, 1). The MetaX
    M==1 GEMV fast path requires the *transposed view* layout
    b.stride() == (1, K), i.e. b = W.t() with W (N, K) contiguous -- which is
    exactly what the model does (x @ W.t()). v1 therefore skipped the GEMV
    path and its M==1 numbers were meaningless.
  * v1 called flag_gems.mm() directly. The server goes through aten, so here
    both arms call torch.mm and only the backend differs.

Native arm is measured before flag_gems is enabled, so it cannot be patched.
"""

import torch

torch.manual_seed(0)
DEV = "cuda"

SHAPES = [
    ("qkv", 2560, 2048),
    ("gate_up", 12288, 2048),
    ("o_proj", 2048, 2048),
    ("down", 2048, 6144),
]
M_LIST = [1, 2, 8, 32, 512, 2048]
ITERS = {1: 200, 2: 200, 8: 100, 32: 60, 512: 30, 2048: 20}


def mk_nt(M, N, K):
    """a: (M,K) contiguous; W: (N,K) contiguous; b = W.t() -> stride (1,K)."""
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


def run(label):
    out = {}
    for name, N, K in SHAPES:
        for M in M_LIST:
            a, b = mk_nt(M, N, K)
            out[(name, M)] = timed(lambda x, y: torch.mm(x, y), a, b, ITERS[M])
        print(f"  {name:8s} " + "  ".join(
            f"M{m}={out[(name, m)]:7.4f}" for m in M_LIST), flush=True)
    return out


print("=== NATIVE (torch.mm, aten, before flag_gems) ===", flush=True)
native = run("native")

print("\n=== enable flag_gems (aten dispatch, mm only) ===", flush=True)
import flag_gems  # noqa: E402

flag_gems.only_enable(include=["mm"], record=False, once=True)
print("  enabled", flush=True)

print("\n=== FLAGGEMS (torch.mm -> FlagGems metax mm) ===", flush=True)
gems = run("gems")

print("\n=== RATIO gems/native  (<1 = FlagGems faster) ===", flush=True)
print(f"  {'shape':8s} " + "  ".join(f"M{m:>7d}" for m in M_LIST))
for name, N, K in SHAPES:
    print(f"  {name:8s} " + "  ".join(
        f"{gems[(name, m)] / native[(name, m)]:8.2f}" for m in M_LIST), flush=True)

print("\n=== NATIVE us -> effective GB/s at M=1 (weight bytes / time) ===", flush=True)
for name, N, K in SHAPES:
    wb = N * K * 2
    print(f"  {name:8s} w={wb / 1e6:7.1f}MB  native={wb / (native[(name, 1)] * 1e-3) / 1e9:7.1f} GB/s"
          f"   gems={wb / (gems[(name, 1)] * 1e-3) / 1e9:7.1f} GB/s", flush=True)
