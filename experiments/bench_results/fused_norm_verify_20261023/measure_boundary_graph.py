#!/usr/bin/env python3
"""Boundary cost *inside a CUDA graph*, which is how serving actually runs.

Why this file exists: measure_boundary.py measured the two providers in eager
mode and got flagos = 106.92 us/call wall against 7.40 us/call of device time.
That 99.5 us is not a kernel boundary -- it is the FL dispatch manager running
in Python on every call (fused_add_rms_norm_op -> _dispatch_rms_norm ->
call_op("rms_norm", ...)).  Serving does not pay it: vLLM captures the decode
step into a CUDA graph and replays device work only, which is exactly why the
end-to-end penalty was 1.5% and not 900%.

So the only honest way to price the extra kernel boundary is to put the calls in
a graph and time the replays.  A graph with CALLS calls inside is replayed N
times; wall/replays/CALLS is per call.  Profiling the same replays gives device
time, and the difference is the boundary cost the decode step really pays.

  vllm_c   1 kernel/call  (FusedAddRMSNormKernelOpt)
  flagos   2 kernels/call (add, then functional rms_norm)
  -> the difference in (wall - device) is what the extra boundary costs
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent
M, HIDDEN = 64, 2048
CALLS = 20      # calls captured inside one graph
REPLAYS = 50    # graph replays timed
WARMUP = 5


def main() -> int:
    import torch
    from torch.profiler import ProfilerActivity, profile

    from vllm_fl.ops import ir_kernels  # noqa: F401

    try:
        import vllm.kernels  # noqa: F401
    except Exception as e:  # noqa: BLE001
        print("warn: vllm.kernels unavailable:", e)

    import vllm.ir.ops as ir_ops

    op = ir_ops.fused_add_rms_norm
    prio = ir_kernels.default_ir_op_priority().get("fused_add_rms_norm") or []
    print("published priority:", prio)

    w = torch.randn(HIDDEN, device="cuda", dtype=torch.bfloat16)
    eps = 1e-6
    total_calls = CALLS * REPLAYS

    results = {}
    print(f"\ngraph: {M}x{HIDDEN} bf16, {CALLS} calls inside, {REPLAYS} replays "
          f"= {total_calls} calls\n")
    print(f"{'provider':8s} {'wall us':>9s} {'device us':>10s} {'boundary':>9s}  kernels")

    for prov in prio:
        impl = op.impls.get(prov)
        if impl is None:
            continue
        fn_impl = impl.impl_fn

        x = torch.randn(M, HIDDEN, device="cuda", dtype=torch.bfloat16)
        r = torch.randn(M, HIDDEN, device="cuda", dtype=torch.bfloat16)
        stream = torch.cuda.Stream()
        try:
            # warm up on a side stream, as graph capture requires
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(10):
                    fn_impl(x, r, w, eps)
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()

            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for _ in range(CALLS):
                    fn_impl(x, r, w, eps)

            for _ in range(WARMUP):
                g.replay()
            torch.cuda.synchronize()

            s, e = torch.cuda.Event(True), torch.cuda.Event(True)
            s.record()
            for _ in range(REPLAYS):
                g.replay()
            e.record()
            torch.cuda.synchronize()
            wall = s.elapsed_time(e) * 1000.0 / total_calls

            with profile(activities=[ProfilerActivity.CUDA]) as p:
                for _ in range(REPLAYS):
                    g.replay()
                torch.cuda.synchronize()
            dev = 0.0
            nk = 0
            for ev in p.key_averages():
                t = ev.self_device_time_total or 0
                if t > 0 and "Memcpy" not in ev.key and "memset" not in ev.key:
                    dev += t / total_calls
                    nk += max(1, round(ev.count / total_calls))
            if dev <= 0:
                print(f"{prov:8s} no device time")
                continue

            results[prov] = {"wall_us": wall, "device_us": dev,
                             "boundary_us": wall - dev, "kern_count": nk}
            print(f"{prov:8s} {wall:9.2f} {dev:10.2f} {wall-dev:9.2f}  {nk}")
        except Exception as exc:  # noqa: BLE001
            print(f"{prov:8s} FAILED {type(exc).__name__}: {str(exc)[:90]}")

    if "vllm_c" in results and "flagos" in results:
        a, b = results["vllm_c"], results["flagos"]
        print(f"\nflagos - vllm_c:  wall {b['wall_us']-a['wall_us']:+.2f} us/call   "
              f"device {b['device_us']-a['device_us']:+.2f} us/call")
        print(f"  boundary delta {b['boundary_us']-a['boundary_us']:+.2f} us/call")
        # serving measured +4.11 us/call for flagos-first (0.345 ms/step / 84)
        print(f"  serving measured      +4.11 us/call (0.345 ms/step over 84 calls)")

    (OUT / "boundary_cost_graph.json").write_text(json.dumps(results, indent=2))
    print("\nwrote", OUT / "boundary_cost_graph.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
