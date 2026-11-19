"""No-L2-reuse decode floor: 42 distinct weight sets and 42 distinct KV caches,
exactly like a real step.  The earlier microbenches re-read one buffer, so L2
could serve part of it; this measures the true per-step kernel floor.
"""
import math

import torch

from flag_gems.runtime.backend._iluvatar.ops.mm import _launch_mm
from flag_gems.runtime.backend._iluvatar.fused.paged_decode_triton import (
    paged_decode_triton,
)

dev = "cuda"
dt = torch.bfloat16
M = 64
SHAPES = [("qkv", 2560, 2048), ("o", 2048, 2048),
          ("gate_up", 12288, 2048), ("down", 2048, 6144)]


def timed_graph(build, reps=5):
    g = torch.cuda.CUDAGraph()
    build()  # warm workspace/JIT outside the graph
    torch.cuda.synchronize()
    with torch.cuda.graph(g):
        build()
    torch.cuda.synchronize()
    s = torch.cuda.Event(True)
    e = torch.cuda.Event(True)
    s.record()
    for _ in range(reps):
        g.replay()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / reps  # ms per step


# ---------------- GEMM: 42 layers x 4 distinct weight matrices ----------------
w = {}
a = {}
c = {}
for name, N, K in SHAPES:
    a[name] = torch.randn(M, K, device=dev, dtype=dt)
    c[name] = torch.empty(M, N, device=dev, dtype=dt)
    w[name] = [torch.randn(K, N, device=dev, dtype=dt) for _ in range(42)]


def build_gemm():
    for L in range(42):
        for name, N, K in SHAPES:
            _launch_mm(a[name], w[name][L], c[name], M, N, K)


t_gemm = timed_graph(build_gemm)
gemm_bytes = sum(K * N * 2 for _, N, K in SHAPES) * 42
print(f"GEMM  (42 distinct weights): {t_gemm:7.2f} ms  "
      f"{gemm_bytes/t_gemm/1e6:6.1f} GB/s  ({gemm_bytes/1e9:.2f} GB)")

del w, a, c
torch.cuda.empty_cache()

# ---------------- Attention: 42 distinct KV caches --------------------------
B, HKV, D, P, CTX = 64, 2, 128, 16, 4608
npages = CTX // P
kv = []
for _ in range(42):
    k = torch.randn(B * npages, P, HKV, D, device=dev, dtype=dt)
    v = torch.randn(B * npages, P, HKV, D, device=dev, dtype=dt)
    kv.append((k, v))
q = torch.randn(B, 16, D, device=dev, dtype=dt)
out = torch.empty_like(q)
bt = torch.arange(B * npages, device=dev, dtype=torch.int32).reshape(B, npages)
sl = torch.full((B,), CTX, device=dev, dtype=torch.int32)
scale = 1.0 / math.sqrt(D)

# warm the workspace + JIT with the first cache (outside capture)
paged_decode_triton(out, q, kv[0][0], kv[0][1], bt, sl, scale)
torch.cuda.synchronize()


def build_attn():
    for L in range(42):
        paged_decode_triton(out, q, kv[L][0], kv[L][1], bt, sl, scale)


t_attn = timed_graph(build_attn)
kv_bytes = 42 * B * CTX * HKV * D * 2 * 2
print(f"ATTN  (42 distinct KV)     : {t_attn:7.2f} ms  "
      f"{kv_bytes/t_attn/1e6:6.1f} GB/s  ({kv_bytes/1e9:.2f} GB)")

print(f"\nkernel floor (GEMM+ATTN) = {t_gemm + t_attn:7.2f} ms")
print(f"7000 tok/s needs         ~ 34.9 ms/step (incl. tail/small ops)")
