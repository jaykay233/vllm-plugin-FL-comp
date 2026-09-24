"""Separate host overhead from device time for FlagGems metax mm.

Probe #2 showed flag_gems times pinned at ~0.11 ms for every shape and every M,
which is the signature of a fixed per-call host cost rather than kernel work.
This script settles it: same inputs for both arms, and device time measured with
torch.profiler instead of wall/event time.

Fixes probe #2's methodology bug as well: it regenerated inputs between the two
arms, so its gems-vs-native numeric diff compared different random tensors.
"""

import torch
from torch.profiler import ProfilerActivity, profile

torch.manual_seed(0)
DEV = "cuda"

CASES = [("qkv", 2560, 2048, 1), ("qkv", 2560, 2048, 8), ("qkv", 2560, 2048, 2048),
         ("o_proj", 2048, 2048, 2), ("o_proj", 2048, 2048, 32), ("o_proj", 2048, 2048, 2048),
         ("down", 2048, 6144, 8), ("down", 2048, 6144, 2048),
         ("gate_up", 12288, 2048, 2048)]


def mk(M, N, K):
    a = torch.randn(M, K, device=DEV, dtype=torch.bfloat16)
    w = torch.randn(N, K, device=DEV, dtype=torch.bfloat16)
    return a, w.t()


def event_ms(fn, a, b, iters=100):
    for _ in range(5):
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


def device_ms(fn, a, b, iters=50):
    """Total CUDA kernel time per call, from the profiler."""
    for _ in range(5):
        fn(a, b)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn(a, b)
        torch.cuda.synchronize()
    tot = 0.0
    for e in prof.key_averages():
        if e.device_time_total:
            tot += e.device_time_total
    return tot / iters / 1e3  # us -> ms


def host_ms(fn, a, b, iters=50):
    """Host wall time to *launch* iters calls (queued, so this is pure CPU)."""
    for _ in range(5):
        fn(a, b)
    torch.cuda.synchronize()
    import time
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(a, b)
    dt = time.perf_counter() - t0
    torch.cuda.synchronize()
    return dt / iters * 1e3


inputs = {(n, M): mk(M, N, K) for n, N, K, M in CASES}

print("=== NATIVE (before flag_gems) ===", flush=True)
nat = {}
for n, N, K, M in CASES:
    a, b = inputs[(n, M)]
    nat[(n, M)] = (event_ms(lambda x, y: torch.mm(x, y), a, b),
                   device_ms(lambda x, y: torch.mm(x, y), a, b),
                   host_ms(lambda x, y: torch.mm(x, y), a, b))
    ref = torch.mm(a, b)
    inputs[(n, M)] = (a, b, ref)

print("=== enable flag_gems ===", flush=True)
import flag_gems  # noqa: E402

flag_gems.only_enable(include=["mm"], record=False, once=True)

gems = {}
print("=== RESULT ===")
print(f"  {'case':16s} {'nat_ev':>8s} {'nat_dev':>8s} {'nat_host':>9s} | "
      f"{'gms_ev':>8s} {'gms_dev':>8s} {'gms_host':>9s} | {'r_ev':>6s} {'r_dev':>6s} {'max|d|':>8s}")
for n, N, K, M in CASES:
    a, b, ref = inputs[(n, M)]
    ev = event_ms(lambda x, y: torch.mm(x, y), a, b)
    dv = device_ms(lambda x, y: torch.mm(x, y), a, b)
    ho = host_ms(lambda x, y: torch.mm(x, y), a, b)
    o = torch.mm(a, b)
    d = (o.float() - ref.float()).abs().max().item()
    ne, nd, nh = nat[(n, M)]
    print(f"  {n + ' M' + str(M):16s} {ne:8.4f} {nd:8.4f} {nh:9.4f} | "
          f"{ev:8.4f} {dv:8.4f} {ho:9.4f} | {ev / ne:6.2f} {dv / nd:6.2f} {d:8.3f}",
          flush=True)
