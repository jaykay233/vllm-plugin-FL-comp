#!/usr/bin/env python3
"""Where does the 19.4us per rms_norm under graph replay actually go?

Graph replay shows 84 rms_norm calls costing 1.6274 ms at M=1 -- 16% of the
decode step -- yet each call only reads 8 KiB. So it is entirely overhead, and
the question is which overhead:

  (a) the irreducible per-kernel launch cost inside a CUDA graph (a floor that
      no kernel work can go below), or
  (b) the ir/dispatch/python path being re-entered per call, or
  (c) the kernel itself genuinely taking that long.

Calibrate the floor with trivial elementwise kernels in the same graph, then
compare the ir op against a direct FlagGems call to see whether the dispatch
layer is adding anything.

Run:  conda activate mx && python /root/src/calibrate_rms_norm.py
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path("/root/bench_results/rms_norm_cmp")
OUT.mkdir(parents=True, exist_ok=True)

HIDDEN = 2048
EPS = 1e-6
N = 84
DECODE_STEP_MS = 10.15


def replay_ms(graph, n=50) -> float:
    import torch

    for _ in range(10):
        graph.replay()
    torch.cuda.synchronize()
    runs = []
    for _ in range(5):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        s.record()
        for _ in range(n):
            graph.replay()
        e.record()
        torch.cuda.synchronize()
        runs.append(s.elapsed_time(e) / n)
    runs.sort()
    return runs[len(runs) // 2]


def capture(fn, calls: int):
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
    return g


def main() -> None:
    import torch
    from vllm.ir import ops as irops

    op = irops.rms_norm
    dev = "cuda"
    dt = torch.bfloat16

    M = 1
    torch.manual_seed(0)
    x = torch.randn(M, HIDDEN, device=dev, dtype=dt)
    w = torch.randn(HIDDEN, device=dev, dtype=dt)
    y = torch.empty_like(x)

    print("=" * 92)
    print(f"CALIBRATION: {N} calls captured per graph, M={M}, hidden={HIDDEN}")
    print("=" * 92)
    print(f"  {'variant':44s} {'total ms':>10s} {'us/call':>9s}")

    results = {}
    rows = []

    # (a) floor: trivial elementwise copy -- one kernel, minimal work
    g = capture(lambda: y.copy_(x), N)
    t = replay_ms(g)
    results["floor copy_ (1 kernel)"] = t
    print(f"  {'FLOOR: y.copy_(x)  [1 trivial kernel]':44s} {t:10.4f} {t * 1e3 / N:9.3f}")

    # floor with a tiny elementwise add
    g = capture(lambda: torch.add(x, 1.0, out=y), N)
    t = replay_ms(g)
    results["floor add scalar (1 kernel)"] = t
    print(f"  {'FLOOR: add(x,1)     [1 trivial kernel]':44s} {t:10.4f} {t * 1e3 / N:9.3f}")

    # (b) the ir op, flagos provider
    op.set_priority(["flagos"])
    g = capture(lambda: op(x, w, EPS), N)
    t = replay_ms(g)
    results["ir op flagos"] = t
    print(f"  {'ir.ops.rms_norm  flagos (FlagGems)':44s} {t:10.4f} {t * 1e3 / N:9.3f}")

    # (c) the ir op, native provider
    op.set_priority(["native"])
    g = capture(lambda: op(x, w, EPS), N)
    t = replay_ms(g)
    results["ir op native"] = t
    print(f"  {'ir.ops.rms_norm  native (torch)':44s} {t:10.4f} {t * 1e3 / N:9.3f}")
    op.set_priority(["flagos", "native"])

    # (d) direct FlagGems call, bypassing the ir + dispatch layers
    try:
        from flag_gems.modules.normalization import gems_rms_forward

        g = capture(lambda: gems_rms_forward(x, None, w, EPS), N)
        t = replay_ms(g)
        results["direct gems_rms_forward"] = t
        print(f"  {'DIRECT gems_rms_forward (no ir/dispatch)':44s} "
              f"{t:10.4f} {t * 1e3 / N:9.3f}")
    except Exception as e:  # noqa: BLE001
        print(f"  direct gems_rms_forward failed: {e}")

    # (e) prefill shape for contrast
    x5 = torch.randn(512, HIDDEN, device=dev, dtype=dt)
    op.set_priority(["flagos"])
    g = capture(lambda: op(x5, w, EPS), N)
    t = replay_ms(g)
    results["ir op flagos M=512"] = t
    print(f"  {'ir.ops.rms_norm  flagos  (M=512)':44s} {t:10.4f} {t * 1e3 / N:9.3f}")
    op.set_priority(["flagos", "native"])

    print()
    print("=" * 92)
    print("READING")
    print("=" * 92)
    floor = results.get("floor copy_ (1 kernel)", float("nan"))
    flagos = results.get("ir op flagos", float("nan"))
    direct = results.get("direct gems_rms_forward", float("nan"))
    print(f"  irreducible per-kernel floor in a graph : "
          f"{floor * 1e3 / N:7.3f} us")
    print(f"  ir op flagos                            : "
          f"{flagos * 1e3 / N:7.3f} us")
    if direct == direct:
        print(f"  direct gems_rms_forward (no ir/dispatch): "
              f"{direct * 1e3 / N:7.3f} us")
        print(f"  -> ir+dispatch overhead                 : "
              f"{(flagos - direct) * 1e3 / N:7.3f} us")
    print()
    print(f"  overhead above the floor, ir op flagos   : "
          f"{(flagos - floor) * 1e3 / N:7.3f} us")
    print()
    print(f"  For a {DECODE_STEP_MS} ms decode step, {N} calls at the floor would be "
          f"{floor / DECODE_STEP_MS * 100:.2f}%")
    print(f"  Actual flagos share is {flagos / DECODE_STEP_MS * 100:.2f}%")
    print(f"  -> Attackable headroom (flagos minus floor): "
          f"{(flagos - floor) / DECODE_STEP_MS * 100:.2f}% of the step")
    print()
    if floor * 1e3 / N > 12:
        print("  >> The per-kernel graph floor ITSELF is the dominant cost.")
        print("     No kernel optimisation can help; only FUSING calls would.")
    else:
        print("  >> The floor is low, so the kernels/dispatch carry real time.")
        print("     Kernel work is a genuine lever.")
    print("CALIBRATE_DONE", flush=True)

    (OUT / "calibrate.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
