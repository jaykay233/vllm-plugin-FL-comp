#!/usr/bin/env python3
"""Probe: which backend does dispatch actually resolve for MiniCPM5-2B hot ops?

Answers task 3 -- is FlagGems really being used for the 42-layer hot ops?
Run:  conda activate mx && python /root/src/probe_dispatch.py
"""

from __future__ import annotations

import os
import sys

HOT_OPS = [
    # op_name in dispatch,   what MiniCPM5-2B actually calls every layer
    ("silu_and_mul", "MLP activation (42x)"),
    ("rms_norm", "input/post layernorm (84x)"),
    ("rotary_embedding", "RoPE (42x)"),
    ("topk_softmax", "MoE router (n/a: dense model)"),
]


def main() -> None:
    os.environ.setdefault("VLLM_FL_DISPATCH_DEBUG", "1")

    from vllm_fl.utils import use_flaggems_op, use_flaggems, get_flag_gems_whitelist_blacklist

    print("=" * 72)
    print("FlagGems switch state")
    print("=" * 72)
    print(f"USE_FLAGGEMS / use_flaggems()        : {use_flaggems()}")
    wl, bl = get_flag_gems_whitelist_blacklist()
    print(f"whitelist                            : {wl}")
    print(f"blacklist                            : {bl}")

    import vllm_fl.dispatch as d
    from vllm_fl.dispatch import get_default_manager

    mgr = get_default_manager()
    mgr.ensure_initialized()
    snap = mgr.registry.snapshot()

    print()
    print("=" * 72)
    print("Per-op resolution")
    print("=" * 72)
    for op_name, why in HOT_OPS:
        impls = snap.impls_by_op.get(op_name, [])
        print(f"\n[{op_name}]  ({why})")
        print(f"  use_flaggems_op({op_name!r}) = {use_flaggems_op(op_name)}")
        if not impls:
            print("  registered impls: <none>")
            continue
        for impl in sorted(impls, key=lambda x: (x.priority, x.impl_id), reverse=True):
            avail = "?"
            try:
                avail = impl.is_available()
            except Exception as e:  # pragma: no cover
                avail = f"err:{e}"
            print(
                f"    - {impl.impl_id:34s} kind={impl.kind.value:9s} "
                f"vendor={str(impl.vendor):8s} prio={impl.priority:4d} avail={avail}"
            )
        try:
            chosen = mgr.resolve(op_name)
            chosen_id = mgr.resolve_impl_id(op_name) if hasattr(mgr, "resolve_impl_id") else "?"
            print(f"  => RESOLVED: {chosen_id}  fn={getattr(chosen, '__name__', chosen)}")
        except Exception as e:
            print(f"  => RESOLVED: <error> {e}")

    print()
    print("=" * 72)
    print("Is the FlagGems hot-op module actually importable?")
    print("=" * 72)
    checks = [
        ("flag_gems.modules.normalization", "gems_rms_forward"),
        ("flag_gems.modules.activation", "gems_silu_and_mul"),
        ("flag_gems.fused", "gelu_and_mul"),
    ]
    for mod_name, attr in checks:
        try:
            mod = __import__(mod_name, fromlist=[attr])
            ok = hasattr(mod, attr)
            print(f"  {mod_name}.{attr:22s} -> {'OK' if ok else 'MISSING'}")
        except Exception as e:
            print(f"  {mod_name}.{attr:22s} -> IMPORT FAILED: {type(e).__name__}: {e}")

    print()
    print("=" * 72)
    print("OOT layer registration (does CustomOp get replaced?)")
    print("=" * 72)
    from vllm_fl.ops.custom_ops import OOT_OPS

    from vllm.model_executor.custom_op import CustomOp, PluggableLayer

    for op_name, (cls, reg_name) in OOT_OPS.items():
        registered = None
        try:
            registered = CustomOp._OOT_OPS.get(reg_name) or PluggableLayer._OOT_OPS.get(reg_name)
        except Exception:
            pass
        print(f"  {op_name:22s} reg_name={reg_name:22s} oot_registered={registered is not None}")

    print("\nPROBE_DONE", flush=True)


if __name__ == "__main__":
    sys.exit(main())
