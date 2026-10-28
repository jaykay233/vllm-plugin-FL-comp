#!/usr/bin/env python3
"""Measure the *boundary* cost of an extra kernel, directly.

Sections 19 and 22 disagree in an instructive way: torch.profiler says the
flagos split form of fused_add_rms_norm is cheaper on the GPU (7.42 us over 2
kernels) than the vllm_c fused kernel (10.21 us over 1), yet serving is ~1.5%
slower with flagos.  The gap has to be the cost of the kernel *boundary*, which
profiler's self_device_time never shows.

This measures it two ways on the same tensors, no decode loop around them:

  wall        CUDA events around a chain of K calls.  Consecutive calls are
              dependent (each writes x / residual in place), so the chain
              serialises the way a decode step does.  wall/K = kernel + boundary.
  profiler    the same call profiled, self_device_time_total / K = kernel only.

boundary = wall - kernel, and since vllm_c crosses one boundary per call while
flagos crosses two, the marginal cost of a boundary is

    (wall_flagos - wall_vllmc) - (kernel_flagos - kernel_vllmc)

If wall_flagos > wall_vllmc while kernel_flagos < kernel_vllmc, the ranking
inverts: the microbench was measuring the wrong quantity, not just optimistically.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent
M, HIDDEN = 64, 2048
K = 200
WARMUP = 30


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

    results = {}
    print(f"\n{M} x {HIDDEN} bf16, chain of {K} dependent calls\n")
    print(f"{'provider':8s} {'wall us':>9s} {'kernel us':>10s} {'boundary':>9s}  kernels")

    for prov in prio:
        impl = op.impls.get(prov)
        if impl is None:
            continue
        fn_impl = impl.impl_fn

        # fresh tensors per provider; the op writes them in place
        x = torch.randn(M, HIDDEN, device="cuda", dtype=torch.bfloat16)
        r = torch.randn(M, HIDDEN, device="cuda", dtype=torch.bfloat16)

        def chain(n=K):
            for _ in range(n):
                fn_impl(x, r, w, eps)

        try:
            for _ in range(WARMUP):
                fn_impl(x, r, w, eps)
            torch.cuda.synchronize()

            # --- wall: one event pair around the whole chain ---
            s, e = torch.cuda.Event(True), torch.cuda.Event(True)
            s.record()
            chain()
            e.record()
            torch.cuda.synchronize()
            wall = s.elapsed_time(e) * 1000.0 / K

            # --- kernel only ---
            with profile(activities=[ProfilerActivity.CUDA]) as p:
                chain()
                torch.cuda.synchronize()
            kus = 0.0
            nk = 0
            for ev in p.key_averages():
                t = ev.self_device_time_total or 0
                if t > 0 and "Memcpy" not in ev.key:
                    kus += t / K
                    nk += max(1, round(ev.count / K))
            if kus <= 0:
                print(f"{prov:8s} no device time recorded")
                continue

            results[prov] = {"wall_us": wall, "kern_us": kus,
                             "boundary_us": wall - kus, "kern_count": nk}
            print(f"{prov:8s} {wall:9.2f} {kus:10.2f} {wall-kus:9.2f}  {nk}")
        except Exception as exc:  # noqa: BLE001
            print(f"{prov:8s} FAILED {type(exc).__name__}: {str(exc)[:80]}")

    if "vllm_c" in results and "flagos" in results:
        a, b = results["vllm_c"], results["flagos"]
        d_wall = b["wall_us"] - a["wall_us"]
        d_kern = b["kern_us"] - a["kern_us"]
        print(f"\nflagos - vllm_c:  wall {d_wall:+.2f} us/call   kernel {d_kern:+.2f} us/call")
        print(f"  -> extra boundary costs {d_wall - d_kern:+.2f} us")
        print(f"  -> vllm_c crosses {a['kern_count']} boundary/call, "
              f"flagos {b['kern_count']}")
        if d_wall > 0 and d_kern < 0:
            print("\n  RANKING INVERTS: the cheaper-on-GPU provider is slower on the wall clock.")
            print("  The microbench measured kernel time, not the quantity that decide the step.")
        elif d_wall < 0:
            print("\n  flagos wins on the wall clock too -- the serving result would then need")
            print("  a different explanation (graph interactions, not boundary cost).")

    (OUT / "boundary_cost.json").write_text(json.dumps(results, indent=2))
    print("\nwrote", OUT / "boundary_cost.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
