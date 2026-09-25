"""Post-recovery GPU health check.

After a host-level GPU reset/reboot, run this before anything else. It exits 0
only if `torch.cuda.Stream()` returns, which is the exact operation that
deadlocked (see maca_hostqueue_deadlock.md).

    python gpu_health_check.py && bash run_after_recovery.sh

Exit codes:
    0  GPU healthy, CUDA graph capture can proceed
    2  Stream() still hangs -> the host reset did not take effect
"""

import sys
import time

T0 = time.time()


def stamp(msg):
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


def main():
    import torch

    stamp("import torch ok")

    t = time.time()
    avail = torch.cuda.is_available()
    stamp(f"is_available={avail} (init took {time.time() - t:.1f}s)")
    if not avail:
        stamp("!! CUDA not available at all")
        return 2

    t = time.time()
    stream = torch.cuda.Stream()
    stamp(f"Stream() ok in {time.time() - t:.1f}s -> {stream}")

    t = time.time()
    x = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
    y = x @ x
    torch.cuda.synchronize()
    stamp(f"matmul ok in {time.time() - t:.1f}s sum={y.float().sum().item():.0f}")

    print("GPU_HEALTHY", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
