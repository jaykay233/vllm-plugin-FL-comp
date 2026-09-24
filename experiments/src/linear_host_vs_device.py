"""Is the ~0.1 ms/call host overhead in `linear` too?

The model's projections are F.linear(x, W) (aten::linear), and metax has a
separate `linear.py`. The whitelist drops both mm and linear to native, so if
`linear` carries the same per-call host cost, that (not kernel quality) is what
the whitelist was really buying on the decode side.

Same measurement as mm_host_vs_device.py: native arm before flag_gems, both arms
call F.linear, device time from torch.profiler, host time from a launch-only
loop.
"""

import time

import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile

torch.manual_seed(0)
DEV = "cuda"

# (label, N, K, M) -- model projections, decode-ish M first
CASES = [
    ("qkv M1", 2560, 2048, 1), ("qkv M8", 2560, 2048, 8),
    ("qkv M2048", 2560, 2048, 2048),
    ("o_proj M1", 2048, 2048, 1), ("o_proj M8", 2048, 2048, 8),
    ("o_proj M2048", 2048, 2048, 2048),
    ("down M1", 2048, 6144, 1), ("down M8", 2048, 6144, 8),
    ("down M2048", 2048, 6144, 2048),
    ("gate_up M1", 12288, 2048, 1), ("gate_up M2048", 12288, 2048, 2048),
    ("lm_head M1", 130560, 2048, 1), ("lm_head M8", 130560, 2048, 8),
]


def mk(M, N, K):
    x = torch.randn(M, K, device=DEV, dtype=torch.bfloat16)
    w = torch.randn(N, K, device=DEV, dtype=torch.bfloat16)
    return x, w


def event_ms(fn, iters=100):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(iters):
        fn()
    t1.record()
    torch.cuda.synchronize()
    return t0.elapsed_time(t1) / iters


def device_ms(fn, iters=50):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    return sum(e.device_time_total for e in prof.key_averages() if e.device_time_total) / iters / 1e3


def host_ms(fn, iters=50):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    dt = time.perf_counter() - t0
    torch.cuda.synchronize()
    return dt / iters * 1e3


data = {}
print("=== NATIVE ===", flush=True)
nat = {}
for lab, N, K, M in CASES:
    x, w = mk(M, N, K)
    fn = lambda x=x, w=w: F.linear(x, w)  # noqa: E731
    nat[lab] = (event_ms(fn), device_ms(fn), host_ms(fn))
    data[lab] = (x, w, F.linear(x, w))

print("=== enable flag_gems (linear only) ===", flush=True)
import flag_gems  # noqa: E402

flag_gems.only_enable(include=["linear"], record=False, once=True)

print("=== RESULT ===")
print(f"  {'case':14s} {'nat_ev':>8s} {'nat_dev':>8s} {'nat_host':>9s} | "
      f"{'gms_ev':>8s} {'gms_dev':>8s} {'gms_host':>9s} | {'r_ev':>6s} {'r_dev':>6s} {'max|d|':>8s}")
for lab, N, K, M in CASES:
    x, w, ref = data[lab]
    fn = lambda x=x, w=w: F.linear(x, w)  # noqa: E731
    ev, dv, ho = event_ms(fn), device_ms(fn), host_ms(fn)
    o = F.linear(x, w)
    d = (o.float() - ref.float()).abs().max().item()
    ne, nd, nh = nat[lab]
    print(f"  {lab:14s} {ne:8.4f} {nd:8.4f} {nh:9.4f} | "
          f"{ev:8.4f} {dv:8.4f} {ho:9.4f} | {ev / ne:6.2f} {dv / nd:6.2f} {d:8.3f}", flush=True)
