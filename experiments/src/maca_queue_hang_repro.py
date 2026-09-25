"""Minimal reproducer: MACA host-queue deadlock in torch.cuda.Stream().

On 2026-09-25 this container's MACA runtime deadlocked while creating a host
command queue. `torch.cuda.Stream()` never returned (observed > 13 minutes),
which blocked CUDA-graph capture and therefore the whole scored eval path,
because vllm_fl/worker/model_runner.py:757 calls Stream() unconditionally when
async scheduling is on (and async scheduling defaults to on).

What this script shows
----------------------
1. The device is *not* dead: after init, real compute (matmul) works fine.
2. `torch.cuda.init()` itself is slow (~21 s) because the MACA compiler
   (`libmccompiler.so`) decompresses its embedded device-code database. This is
   the dominant cost, but it does finish.
3. `torch.cuda.Stream()` does NOT finish. Its native stack is:

       torch/cuda/streams.py:37  __new__
         -> c10::cuda::getStreamFromPool
         -> c10::cuda::initDeviceStreamState -> initSingleStream
         -> mcStreamCreateWithPriority (mc_runtime_api.cpp:610)
         -> imcStreamCreate (mc_stream.cpp:875)
         -> mcr::Stream::Create (mc_stream.cpp:348)
         -> mxr::HostQueue::HostQueue (mx_commandqueue.cpp:33)
         -> mxr::Monitor::wait -> mxr::Semaphore::timedWait -> futex   <-- hangs

   Note the contrast: calling `mcStreamCreateWithPriority` directly through
   ctypes returns in ~0.02 s. Same function, so the block is not a broken API:
   it is a lock/semaphore inside the MACA execution runtime that is never
   signalled.

How to confirm it is the environment and not the plugin
------------------------------------------------------
    python maca_queue_hang_repro.py --health      # fast health check
    python maca_queue_hang_repro.py               # full: init timings + Stream()

If --health reports STREAM_HANG, the GPU needs a host-level reset. Restarting
the container does NOT help: the leaked GPU context lives in the host driver
(`mx-smi` showed 826 MiB used with `no process found`), so a container restart
leaves it in place.
"""

import argparse
import sys
import time

T0 = time.time()


def stamp(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def health():
    """Return 0 if torch.cuda.Stream() works, 2 if it hangs."""
    import torch

    stamp(f"import torch ok")
    t = time.time()
    avail = torch.cuda.is_available()
    stamp(f"is_available={avail} (init took {time.time() - t:.1f}s)")

    t = time.time()
    stream = torch.cuda.Stream()
    stamp(f"Stream() ok in {time.time() - t:.1f}s -> {stream}")
    print("GPU_HEALTHY", flush=True)
    return 0


def full():
    import torch

    stamp("--- 1) torch.cuda.init() timing (expect ~21 s when mcc decompresses)")
    t = time.time()
    torch.cuda.init()
    stamp(f"    init done in {time.time() - t:.1f}s")

    stamp("--- 2) real compute, to prove the device still works")
    t = time.time()
    x = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
    y = x @ x
    torch.cuda.synchronize()
    stamp(f"    matmul ok in {time.time() - t:.1f}s sum={y.float().sum().item():.0f}")

    stamp("--- 3) torch.cuda.Stream() -- this is what hangs")
    t = time.time()
    stream = torch.cuda.Stream()
    stamp(f"    *** Stream() returned in {time.time() - t:.1f}s -> {stream} ***")
    print("GPU_HEALTHY", flush=True)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--health", action="store_true", help="only the Stream() check")
    args = ap.parse_args()
    sys.exit(health() if args.health else full())
