#!/usr/bin/env python3
"""rms_norm under CUDA Graph replay -- the actual decode condition.

Previous attempts failed for two different reasons:

  1. wall-clock + synchronize per iteration  -> measured sync latency (~106us)
  2. CUDA events over a raw launch loop      -> measured CPU enqueue cost
     (~82us/call), because the GPU was starved and never saturated

Decode does not look like either. vLLM captures the whole decode step into a
CUDA Graph (`Capturing CUDA graphs (decode, FULL)`) and then replays it, so
Python dispatch is paid once at capture and never during steady state. What
remains is pure GPU kernel time.

This reproduces that: capture N rms_norm calls into a graph, replay it, and
time the replay with events. Same for the native provider, so the two can be
compared under graph conditions.

Run:  conda activate mx && python /root/src/bench_rms_norm_graph.py
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path("/root/bench_results/rms_norm_cmp")
OUT.mkdir(parents=True, exist_ok=True)

HIDDEN = 2048
EPS = 1e-6
LAYERS = 42
NORMS_PER_LAYER = 2
CALLS = LAYERS * NORMS_PER_LAYER  # 84
DECODE_STEP_MS = 10.15


def replay_ms(graph, n=50) -> float:
    import torch

    for _ in range(10):
        graph.replay()
    torch.cuda.synchronize()
    runs = []
    for _ in range(5):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        for _ in range(n):
            graph.replay()
        end.record()
        torch.cuda.synchronize()
        runs.append(start.elapsed_time(end) / n)
    runs.sort()
    return runs[len(runs) // 2]


def capture_calls(op, x, w, calls: int):
    """Capture `calls` successive rms_norm invocations into one CUDA graph."""
    import torch

    # warmup on a side stream (required before capture)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            op(x, w, EPS)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(calls):
            op(x, w, EPS)
    torch.cuda.synchronize()
    return graph


def main() -> None:
    import torch
    from vllm.ir import ops as irops

    op = irops.rms_norm
    dev = "cuda"
    dt = torch.bfloat16

    print("=" * 94)
    print(f"rms_norm under CUDA Graph replay ({CALLS} calls captured per graph)")
    print("=" * 94)
    print(f"  {'M':>5s} {'calls':>6s} {'flagos ms':>11s} {'native ms':>11s} "
          f"{'flagos/native':>14s} {'flagos us/call':>16s}")

    rows = []
    for M in (1, 8, 128, 512):
        torch.manual_seed(0)
        x = torch.randn(M, HIDDEN, device=dev, dtype=dt)
        w = torch.randn(HIDDEN, device=dev, dtype=dt)

        op.set_priority(["flagos"])
        g_flagos = capture_calls(op, x, w, CALLS)
        t_flagos = replay_ms(g_flagos)

        op.set_priority(["native"])
        g_native = capture_calls(op, x, w, CALLS)
        t_native = replay_ms(g_native)

        ratio = t_flagos / t_native
        per_call_us = t_flagos * 1e3 / CALLS
        rows.append({"M": M, "calls": CALLS, "flagos_ms": t_flagos,
                     "native_ms": t_native, "ratio": ratio,
                     "flagos_us_per_call": per_call_us})
        print(f"  {M:5d} {CALLS:6d} {t_flagos:11.4f} {t_native:11.4f} "
              f"{ratio:13.3f}x {per_call_us:15.3f}")

    op.set_priority(["flagos", "native"])

    print()
    print("=" * 94)
    print("WHAT IT MEANS FOR A DECODE STEP")
    print("=" * 94)
    r1 = rows[0]
    print(f"  all {CALLS} M=1 norms, flagos : {r1['flagos_ms']:7.4f} ms "
          f"({r1['flagos_ms'] / DECODE_STEP_MS * 100:6.3f}% of a "
          f"{DECODE_STEP_MS} ms step)")
    print(f"  all {CALLS} M=1 norms, native : {r1['native_ms']:7.4f} ms "
          f"({r1['native_ms'] / DECODE_STEP_MS * 100:6.3f}%)")
    print(f"  flagos/native at M=1        : {r1['ratio']:.3f}x")
    print()
    print(f"  absolute ceiling if flagos rms_norm were made free: "
          f"{r1['flagos_ms'] / DECODE_STEP_MS * 100:.3f}% of the step")
    print()
    if r1["flagos_ms"] / DECODE_STEP_MS < 0.02:
        print("  >> rms_norm is a rounding error in the decode step. Even making")
        print("     it free buys <2%. It is NOT the lever.")
    else:
        print("  >> rms_norm is a measurable share of the step.")

    (OUT / "graph.json").write_text(json.dumps(rows, indent=2))
    print("\nRMS_NORM_GRAPH_DONE", flush=True)


if __name__ == "__main__":
    main()
