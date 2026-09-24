#!/usr/bin/env python3
"""CORRECTED rms_norm comparison -- the previous one was measuring the fallback.

The earlier runs called `op.set_priority([...])` as a plain function. That API
is a @contextlib.contextmanager, so calling it without `with` did nothing, and
`IrOp.dispatch()` fell through to

    if not self._priority_impls: ... return self.impls["native"]

Both "flagos" and "native" runs therefore executed the torch composite
fallback (8 elementwise kernels, ~29us), which is why they looked identical.

`set_default()` is the persistent setter. This script uses it, and reports
both CUDA-graph replay time and the profiler kernel count, so the provider
actually in use is unambiguous.

Run:  conda activate mx && python /root/src/rms_norm_correct.py
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

OUT = Path("/root/bench_results/rms_norm_cmp")
OUT.mkdir(parents=True, exist_ok=True)

HIDDEN = 2048
EPS = 1e-6
CALLS = 84  # 42 layers x 2 norms
DECODE_STEP_MS = 10.15


def graph_ms(fn, calls=CALLS, n=50) -> float:
    import torch

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(calls):
            fn()
    torch.cuda.synchronize()

    for _ in range(10):
        g.replay()
    torch.cuda.synchronize()
    runs = []
    for _ in range(5):
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        a.record()
        for _ in range(n):
            g.replay()
        b.record()
        torch.cuda.synchronize()
        runs.append(a.elapsed_time(b) / n)
    runs.sort()
    return runs[len(runs) // 2]


def kern_per_call(fn, iters=20):
    import torch

    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    counts: Counter[str] = Counter()
    dur: Counter[str] = Counter()
    for e in prof.events():
        if e.device_type == torch.autograd.DeviceType.CUDA and e.name:
            counts[e.name] += e.count
            dur[e.name] += e.device_time
    total = sum(dur.values())
    n = sum(counts.values())
    top = counts.most_common(1)[0][0] if counts else "-"
    return n / iters, total / iters, top


def main() -> None:
    import torch
    from vllm.ir import ops as irops

    op = irops.rms_norm
    dev = "cuda"
    dt = torch.bfloat16

    print("=" * 96)
    print("CORRECTED: rms_norm provider comparison (set_default, not set_priority)")
    print("=" * 96)
    print(f"  providers: {sorted(op.impls.keys())}")
    print()

    rows = {}
    for M in (1, 512):
        torch.manual_seed(0)
        x = torch.randn(M, HIDDEN, device=dev, dtype=dt)
        w = torch.randn(HIDDEN, device=dev, dtype=dt)

        print(f"  M={M}")
        for label, priority in (("flagos", ["flagos"]), ("native", ["native"])):
            op.set_default(priority)
            used = op.dispatch(x, w, EPS).provider
            km, kus, top = kern_per_call(lambda: op(x, w, EPS))
            gm = graph_ms(lambda: op(x, w, EPS))
            rows[f"M{M}_{label}"] = {
                "provider_selected": used, "kernels_per_call": km,
                "gpu_us_per_call": kus, "graph_total_ms": gm,
                "graph_us_per_call": gm * 1e3 / CALLS, "top_kernel": top,
            }
            print(f"    {label:8s} -> provider={used!r:10s} "
                  f"kernels/call={km:5.2f}  gpu/call={kus:7.3f}us  "
                  f"graph={gm:7.4f}ms ({gm * 1e3 / CALLS:6.2f}us/call)")
            print(f"             top kernel: {top[:60]}")

    # non-ir reference: FlagGems called directly
    try:
        from flag_gems.modules.normalization import gems_rms_forward

        torch.manual_seed(0)
        x = torch.randn(1, HIDDEN, device=dev, dtype=dt)
        w = torch.randn(HIDDEN, device=dev, dtype=dt)
        km, kus, top = kern_per_call(lambda: gems_rms_forward(x, None, w, EPS))
        gm = graph_ms(lambda: gems_rms_forward(x, None, w, EPS))
        rows["M1_direct_gems"] = {
            "kernels_per_call": km, "gpu_us_per_call": kus,
            "graph_total_ms": gm, "graph_us_per_call": gm * 1e3 / CALLS,
            "top_kernel": top,
        }
        print(f"\n  M=1 direct FlagGems (no ir wrapper)")
        print(f"    kernels/call={km:5.2f}  gpu/call={kus:7.3f}us  "
              f"graph={gm:7.4f}ms ({gm * 1e3 / CALLS:6.2f}us/call)")
        print(f"    top kernel: {top[:60]}")
    except Exception as e:  # noqa: BLE001
        print(f"\n  direct gems unavailable: {e}")

    op.set_default(["flagos", "native"])

    print()
    print("=" * 96)
    print("DECODE STEP IMPACT (84 norms at M=1)")
    print("=" * 96)
    f = rows.get("M1_flagos")
    n = rows.get("M1_native")
    if f and n:
        print(f"  flagos (engaged) : {f['graph_total_ms']:7.4f} ms  "
              f"({f['graph_total_ms'] / DECODE_STEP_MS * 100:6.2f}% of step)")
        print(f"  native (fallback): {n['graph_total_ms']:7.4f} ms  "
              f"({n['graph_total_ms'] / DECODE_STEP_MS * 100:6.2f}% of step)")
        saved = n["graph_total_ms"] - f["graph_total_ms"]
        print(f"  FlagGems saves   : {saved:7.4f} ms  "
              f"({saved / DECODE_STEP_MS * 100:6.2f}% of the step)")
        print()
        if f["kernels_per_call"] < n["kernels_per_call"]:
            print(f"  >> FlagGems serves rms_norm in "
                  f"{f['kernels_per_call']:.0f} kernel(s) vs "
                  f"{n['kernels_per_call']:.0f} for the torch fallback.")
            print(f"     The FlagGems mandate is a real ASSET here, not a cost.")
        else:
            print("  >> FlagGems offers no advantage on rms_norm.")

    (OUT / "corrected.json").write_text(json.dumps(rows, indent=2))
    print("\nRMS_NORM_CORRECT_DONE", flush=True)


if __name__ == "__main__":
    main()
