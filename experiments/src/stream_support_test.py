#!/usr/bin/env python3
"""Does the MetaX / MACA backend support CUDA-style streams (work queues)?

Context: the plugin reports device_type="cuda" and dispatch_key="CUDA", and the
marked trace showed the 16 pinned D2H copies running on stream_id=2 while the
default stream was 0 -- so a second stream really was doing work.  This script
pins down how much of the CUDA stream model is actually available.

Deliberately memory-light (~100 MB) so it can run alongside another session's
vLLM engine without fighting for HBM.

Usage:
    /opt/conda/envs/mx/bin/python /root/src/stream_support_test.py
"""

from __future__ import annotations

import time

import torch

DEV = "cuda"
N = 4096  # 4096^2 bf16 = 32 MB per operand
ITERS = 24


def hr(t: str) -> None:
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}", flush=True)


def main() -> None:
    print(f"torch {torch.__version__}  device {torch.cuda.get_device_name(0)}")

    hr("1. device properties relevant to queues")
    p = torch.cuda.get_device_properties(0)
    for attr in ("multi_processor_count", "major", "minor", "total_memory"):
        if hasattr(p, attr):
            print(f"  {attr:24s} = {getattr(p, attr)}")
    # CUDA exposes asyncEngineCount (hardware copy engines); MACA may or may not.
    for attr in ("concurrent_kernels", "async_engine_count", "asyncEngineCount"):
        print(f"  {attr:24s} = {getattr(p, attr, '<not exposed>')}")

    hr("2. stream creation / current_stream / context manager")
    try:
        s = torch.cuda.Stream()
        print(f"  Stream()                       -> {s}")
        print(f"  current_stream (default)       -> {torch.cuda.current_stream()}")
        with torch.cuda.stream(s):
            print(f"  inside torch.cuda.stream(s)    -> {torch.cuda.current_stream()}")
            print(f"  is same object as s            -> {torch.cuda.current_stream() is s or torch.cuda.current_stream().cuda_stream == s.cuda_stream}")
        print(f"  after context, current_stream  -> {torch.cuda.current_stream()}")
        # stream id exposed?
        print(f"  s.cuda_stream                  -> {getattr(s, 'cuda_stream', '<none>')}")
    except Exception as e:  # noqa: BLE001
        print(f"  FAILED: {type(e).__name__}: {e}")

    hr("3. events: record / synchronize / elapsed_time")
    try:
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        a = torch.randn(512, 512, device=DEV)
        for _ in range(50):
            a = a @ a
        e1.record()
        e1.synchronize()
        print(f"  Event(enable_timing) elapsed   -> {e0.elapsed_time(e1):.3f} ms")
        plain = torch.cuda.Event()
        plain.record()
        plain.synchronize()
        print("  Event() + record + synchronize -> OK (no timing)")
        ev = torch.Event()  # generic alias, used by the plugin
        ev.record()
        ev.synchronize()
        print("  torch.Event() alias            -> OK")
    except Exception as e:  # noqa: BLE001
        print(f"  FAILED: {type(e).__name__}: {e}")

    hr("4. stream-ordered dependency: wait_stream preserves ordering")
    try:
        s1, s2 = torch.cuda.Stream(), torch.cuda.Stream()
        x = torch.zeros(1024, device=DEV)
        with torch.cuda.stream(s1):
            x.fill_(7.0)
        s2.wait_stream(s1)
        with torch.cuda.stream(s2):
            y = x + 1.0  # must observe 7.0, not 0.0
        torch.cuda.synchronize()
        ok = bool((y == 8.0).all().item())
        print(f"  s2.wait_stream(s1) then read   -> y[0]={y[0].item()}, correct={ok}")
    except Exception as e:  # noqa: BLE001
        print(f"  FAILED: {type(e).__name__}: {e}")

    hr("5. do two streams actually OVERLAP? (wall-clock, the decisive test)")
    try:
        a = torch.randn(N, N, dtype=torch.bfloat16, device=DEV)
        b = torch.randn(N, N, dtype=torch.bfloat16, device=DEV)

        def work(stream):
            with torch.cuda.stream(stream) if stream else torch.cuda.stream(torch.cuda.current_stream()):
                c = a
                for _ in range(ITERS):
                    c = c @ b

        # warmup
        work(None)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        work(None)
        work(None)
        torch.cuda.synchronize()
        seq_ms = (time.perf_counter() - t0) * 1e3

        s1, s2 = torch.cuda.Stream(), torch.cuda.Stream()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        work(s1)
        work(s2)
        torch.cuda.synchronize()
        par_ms = (time.perf_counter() - t0) * 1e3

        print(f"  sequential (1 stream, 2x work)  -> {seq_ms:8.2f} ms")
        print(f"  parallel   (2 streams)          -> {par_ms:8.2f} ms")
        print(f"  speedup from overlap            -> {seq_ms / par_ms:.2f}x "
              f"(~1.0 = no overlap, ~2.0 = perfect)")
    except Exception as e:  # noqa: BLE001
        print(f"  FAILED: {type(e).__name__}: {e}")

    hr("6. stream priorities")
    try:
        lo, hi = torch.cuda.Stream.priority_range()
        print(f"  priority_range  -> ({lo}, {hi})")
        hp = torch.cuda.Stream(priority=lo)  # CUDA: numerically lower = higher priority
        lp = torch.cuda.Stream(priority=hi)
        print(f"  high-prio stream={hp}, low-prio={lp}  -> creation OK")
    except Exception as e:  # noqa: BLE001
        print(f"  FAILED: {type(e).__name__}: {e}")

    hr("7. non_blocking H2D/D2H on a side stream (the vLLM async-output pattern)")
    try:
        src = torch.randn(4096, dtype=torch.float32, pin_memory=True)
        gpu = torch.empty_like(src, device=DEV)
        dst = torch.empty_like(src, pin_memory=True)
        copy_stream = torch.cuda.Stream()
        ev = torch.cuda.Event()
        with torch.cuda.stream(copy_stream):
            gpu.copy_(src, non_blocking=True)
            dst.copy_(gpu, non_blocking=True)
            ev.record()
        ev.synchronize()
        print(f"  pinned H2D + D2H on side stream -> OK, max err={((dst - src).abs().max().item()):.3e}")
    except Exception as e:  # noqa: BLE001
        print(f"  FAILED: {type(e).__name__}: {e}")

    hr("8. CUDA graph capture on a non-default stream")
    try:
        a = torch.randn(1024, 1024, dtype=torch.bfloat16, device=DEV)
        b = torch.randn(1024, 1024, dtype=torch.bfloat16, device=DEV)
        side = torch.cuda.Stream()
        # warmup on the capture stream
        s = torch.cuda.Stream()
        with torch.cuda.stream(s):
            for _ in range(3):
                a @ b
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            out = a @ b
        g.replay()
        torch.cuda.synchronize()
        print(f"  torch.cuda.graph(stream=side)  -> capture+replay OK, out[0,0]={out[0, 0].item():.4f}")
    except Exception as e:  # noqa: BLE001
        print(f"  FAILED: {type(e).__name__}: {e}")

    hr("9. stream pool / query helpers")
    for name in ("stream", "current_stream", "default_stream", "set_stream",
                 "synchronize", "Stream", "Event", "ExternalStream", "graph_pool_handle"):
        print(f"  torch.cuda.{name:24s} -> {hasattr(torch.cuda, name)}")

    print("\nSTREAM_SUPPORT_DONE", flush=True)


if __name__ == "__main__":
    main()
