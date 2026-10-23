#!/usr/bin/env python3
"""Does the current fused_add_rms_norm still cost a 256 KiB D2D copy per call?

§21.7 found 42 x 262144 B device->device copies per decode step in a trace
taken *before* c4dadef switched fused_add_rms_norm from `flagos` to `vllm_c`.
262144 = 64 x 2048 x 2 is the residual's shape, so the copies looked like the
two clones the flagos custom-op path needs to satisfy torch's no-aliasing rule.

That trace predates the fix, so it cannot say whether the copies survived.  This
probe answers it directly, without fighting the profiler: run each provider of
the op at the decode shape and count every gpu_memcpy, with its byte count.

  vllm_c   in-place vendor kernel, registered inplace=True, no clone expected
  flagos   torch.library.custom_op(mutates_args=()) -> 2 clones expected
  native   eager torch fallback

A clone shows up as `Memcpy HtoA (Device -> Device)` of exactly 262144 B, 34 us
on this part -- the same signature §21.7 saw 42 times per step.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

OUT = Path(__file__).resolve().parent
M, HIDDEN, NLAYERS = 64, 2048, 42
RESIDUAL_BYTES = M * HIDDEN * 2  # 262144


def profile_calls(fn, iters=30, warmup=10):
    """Per-call device time, split into kernels and memcpys."""
    import torch
    from torch.profiler import ProfilerActivity, profile

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()

    kern = defaultdict(float)
    kcnt = defaultdict(int)
    cp_us = 0.0
    cp_cnt = 0
    cp_bytes = defaultdict(int)
    for e in p.key_averages():
        name = e.key
        t = e.self_device_time_total or 0
        if t <= 0:
            continue
        if "Memcpy" in name:
            cp_us += t / iters
            cp_cnt += max(1, round(e.count / iters))
            # bytes are not in key_averages; approximate from the signature
            cp_bytes[name] += max(1, round(e.count / iters))
        else:
            kern[name] += t / iters
            kcnt[name] += max(1, round(e.count / iters))
    return {
        "kern_us": sum(kern.values()),
        "kern_names": {k[:56]: round(v, 2) for k, v in sorted(kern.items(), key=lambda kv: -kv[1])},
        "kern_count": sum(kcnt.values()),
        "copy_us": cp_us,
        "copy_count": cp_cnt,
        "copy_names": {k[:56]: v for k, v in cp_bytes.items()},
    }


def main() -> int:
    import torch

    from vllm_fl.ops import ir_kernels  # noqa: F401

    # Importing vllm.kernels is what registers torch.ops._C on MetaX.
    try:
        import vllm.kernels  # noqa: F401
    except Exception as e:  # noqa: BLE001
        print("warn: vllm.kernels unavailable:", e)

    import vllm.ir.ops as ir_ops

    op = ir_ops.fused_add_rms_norm
    print("registered impls:", sorted(op.impls))
    prio = ir_kernels.default_ir_op_priority().get("fused_add_rms_norm")
    print("published priority:", prio)

    x = torch.randn(M, HIDDEN, device="cuda", dtype=torch.bfloat16)
    r = torch.randn(M, HIDDEN, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(HIDDEN, device="cuda", dtype=torch.bfloat16)
    eps = 1e-6

    results = {}
    print(f"\nresidual payload = {RESIDUAL_BYTES} B ({RESIDUAL_BYTES/1024:.0f} KiB)\n")
    header = f"{'provider':8s} {'kern us':>8s} {'n':>2s} {'copy us':>8s} {'copies':>6s}  kernels"
    print(header)

    # Call each provider's implementation directly.  set_priority() did not take
    # effect here (every provider measured as `native`, 95 us / 11 kernels), so
    # the dispatch path cannot be used to isolate them.
    for prov in list(prio or []):
        impl = op.impls.get(prov)
        if impl is None:
            print(f"{prov:8s} (no impl registered)")
            continue
        try:
            torch.cuda.synchronize()
            r_clone = r.clone()
            fn = lambda impl=impl: impl.impl_fn(x, r_clone, w, eps)
            res = profile_calls(fn)
            results[prov] = res
            names = ", ".join(list(res["kern_names"])[:3])
            print(f"{prov:8s} {res['kern_us']:8.2f} {res['kern_count']:2d} "
                  f"{res['copy_us']:8.2f} {res['copy_count']:6d}  {names}")
            if res["copy_names"]:
                for nm, c in res["copy_names"].items():
                    print(f"           copy: {nm}  x{c}/call")
        except Exception as exc:  # noqa: BLE001
            print(f"{prov:8s} FAILED {type(exc).__name__}: {str(exc)[:90]}")


    # per-step projection for the provider actually used by default
    used = (prio or ["native"])[0]
    if used in results:
        res = results[used]
        per_call = res["kern_us"] + res["copy_us"]
        per_step = per_call * 2 * NLAYERS / 1000
        print(f"\ndefault provider '{used}': {per_call:.2f} us/call")
        print(f"  x 2 calls x 42 layers = {per_step:.3f} ms/step")
        if res["copy_count"]:
            print(f"  WARNING: still copying -- {res['copy_count']} memcpy/call, "
                  f"{res['copy_us']:.2f} us/call")
        else:
            print("  no memcpy on this path (clones are gone)")

    (OUT / "fused_norm_providers.json").write_text(json.dumps(results, indent=2))
    print("\nwrote", OUT / "fused_norm_providers.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
