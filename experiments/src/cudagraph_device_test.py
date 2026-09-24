#!/usr/bin/env python3
"""Does MetaX C500 support CUDA graph capture/replay at the device level?

Independent of vLLM: capture a small compute graph, replay it with changed
inputs, and verify the replayed output matches eager. Also probes the pieces a
capture actually needs (stream capture, graph API, memory pools) so a failure
can be attributed.

Run:  python /root/src/cudagraph_device_test.py
"""

from __future__ import annotations

import torch


def main() -> None:
    print(f"torch {torch.__version__}")
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"cc: {torch.cuda.get_device_capability(0)}")
    print()

    print("-- API availability --")
    for name in ("CUDAGraph", "graph_pool_handle", "is_current_stream_capturing"):
        print(f"  torch.cuda.{name:32s} {'OK' if hasattr(torch.cuda, name) else 'MISSING'}")
    print(f"  torch.cuda.graphs.graph_pool_handle {'OK' if hasattr(torch.cuda.graphs, 'graph_pool_handle') else 'MISSING'}")
    print()

    dev, dt = "cuda", torch.bfloat16
    M, K, N = 8, 2048, 2048

    # ---- capture ----
    print("-- capture --")
    x = torch.randn(M, K, device=dev, dtype=dt)
    w = torch.randn(N, K, device=dev, dtype=dt)
    x_static = x.clone()

    # warmup on a side stream (required before capture)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            y = torch.nn.functional.linear(x_static, w)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(g):
            y_static = torch.nn.functional.linear(x_static, w)
        torch.cuda.synchronize()
        print("  capture: OK")
    except Exception as e:
        print(f"  capture: FAILED -> {type(e).__name__}: {e}")
        return

    # ---- replay correctness with new input ----
    print("\n-- replay --")
    x_new = torch.randn(M, K, device=dev, dtype=dt)
    x_static.copy_(x_new)
    g.replay()
    torch.cuda.synchronize()
    ref = torch.nn.functional.linear(x_new, w)
    d = (y_static.float() - ref.float()).abs().max().item()
    print(f"  replayed vs eager max abs diff: {d:.4e}  {'OK' if d < 1e-1 else 'MISMATCH'}")

    g.replay()  # replay twice to be sure it is stable
    torch.cuda.synchronize()
    print("  second replay: OK")

    # ---- timing: does replay actually cut host overhead? ----
    import time

    def timed(fn, n=200):
        for _ in range(20):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / n * 1e6

    t_eager = timed(lambda: torch.nn.functional.linear(x_static, w))
    t_graph = timed(g.replay)
    print(f"\n-- host-side latency (single small GEMM) --")
    print(f"  eager launch : {t_eager:8.2f} us/call")
    print(f"  graph replay : {t_graph:8.2f} us/call")
    if t_graph > 0:
        print(f"  speedup      : {t_eager / t_graph:8.2f}x")

    # multi-op graph: the case graphs actually help
    def chain():
        a = torch.nn.functional.linear(x_static, w)
        a = torch.nn.functional.silu(a)
        return a

    y_ref = chain()
    y_static2 = None
    s2 = torch.cuda.Stream()
    s2.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s2):
        for _ in range(3):
            chain()
    torch.cuda.current_stream().wait_stream(s2)
    torch.cuda.synchronize()

    g2 = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g2):
        y_static2 = chain()
    torch.cuda.synchronize()

    t_chain = timed(chain)
    t_graph2 = timed(g2.replay)
    d2 = (y_static2.float() - chain().float()).abs().max().item()
    print(f"\n-- host-side latency (GEMM + silu chain) --")
    print(f"  eager   : {t_chain:8.2f} us/call")
    print(f"  replay  : {t_graph2:8.2f} us/call")
    print(f"  speedup : {t_chain / t_graph2:8.2f}x")
    print(f"  chain replay correctness diff: {d2:.4e}")

    print("\nCUDAGRAPH_DEVICE_TEST_DONE", flush=True)


if __name__ == "__main__":
    main()
