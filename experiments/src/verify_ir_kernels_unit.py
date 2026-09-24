#!/usr/bin/env python3
"""Unit check of the plugin's IR-op bridge (no LLM / no engine needed).

Exercises ``vllm_fl.ops.ir_kernels`` directly:
  * are the ``flagos`` providers registered for vllm.ir rms_norm ops?
  * does the platform priority we publish get applied?
  * eager call -> dispatch manager picks which impl?
  * torch.compile(fullgraph=True, dynamic=False) (what vLLM uses) -> does the
    opaque custom op survive and does the runtime call re-enter dispatch?

Run:  conda activate mx && cd /workspace/vllm-plugin-FL && IR=1 python /root/src/verify_ir_kernels_unit.py
"""

from __future__ import annotations

import os
import traceback

import torch

DT = torch.bfloat16
DEV = "cuda"
EPS = 1e-6
H = 256
TOK = 32

CALLS = {"rms": 0, "fused": 0}


def line(msg: str = "", **kw) -> None:
    kw.setdefault("flush", True)
    print(msg, **kw)


def sep(title: str) -> None:
    line()
    line("=" * 74)
    line(title)
    line("=" * 74)


def main() -> None:
    line(f"torch {torch.__version__}  device={torch.cuda.get_device_name(0)}")
    line(f"VLLM_FL_IR_KERNELS={os.environ.get('VLLM_FL_IR_KERNELS', '(unset)')}")

    import vllm_fl.ops.ir_kernels as ik

    # count dispatches that go through the bridge (works for eager *and* for
    # the runtime execution of the opaque custom op)
    orig_dispatch = ik._dispatch_rms_norm

    def counted(x, residual, weight, epsilon):
        if residual is None:
            CALLS["rms"] += 1
        else:
            CALLS["fused"] += 1
        return orig_dispatch(x, residual, weight, epsilon)

    ik._dispatch_rms_norm = counted

    # ---------------------------------------------------------------- register
    sep("1. register flagos providers + publish platform priority")
    ok = ik.register_compile_safe_ops()
    line(f"  register_compile_safe_ops() -> {ok}")
    line(f"  default_ir_op_priority()    -> {ik.default_ir_op_priority()}")
    if not ok:
        line("  => registration failed, stop")
        return

    import vllm.ir.ops  # noqa: F401
    from vllm.ir.op import IrOp

    rms = IrOp.registry["rms_norm"]
    fused = IrOp.registry["fused_add_rms_norm"]
    line(f"  rms_norm providers={rms.supported_providers()} "
         f"priority={rms.get_priority()}")
    line(f"  fused_add_rms_norm providers={fused.supported_providers()} "
         f"priority={fused.get_priority()}")

    # what vllm/config/kernel.py does with our dict
    prio = ik.default_ir_op_priority()
    line(f"  complete priority      -> {prio}")
    for name, ir_op in (("rms_norm", rms), ("fused_add_rms_norm", fused)):
        ir_op.set_default(prio[name])
    line(f"  after set_default: rms={rms.get_priority()} fused={fused.get_priority()}")

    # ---------------------------------------------------------------- which impl
    sep("2. which FL implementation does the dispatch manager resolve?")
    from vllm_fl.dispatch import get_default_manager

    mgr = get_default_manager()
    for op_name in ("rms_norm",):
        impl = mgr._resolve_impl(op_name)
        line(f"  {op_name}: impl_id={impl.impl_id!r} backend={getattr(impl, 'backend', '?')!r} "
             f"fn={getattr(impl.fn, '__module__', '?')}.{getattr(impl.fn, '__name__', '?')}")

    # ---------------------------------------------------------------- data
    from flag_gems.modules.normalization import gems_rms_forward

    x = torch.randn(TOK, H, device=DEV, dtype=DT)
    w = torch.randn(H, device=DEV, dtype=DT)
    res = torch.randn(TOK, H, device=DEV, dtype=DT)

    ref = gems_rms_forward(x, None, w, EPS)
    ref_f, ref_res = gems_rms_forward(x.clone(), res.clone(), w, EPS)
    torch.cuda.synchronize()

    # ---------------------------------------------------------------- eager
    sep("3. eager: vllm_ir.rms_norm(...) -> flagos provider")
    CALLS["rms"] = 0
    out = rms(x, w, EPS)
    torch.cuda.synchronize()
    line(f"  bridge calls={CALLS['rms']} max|diff|={float((out - ref).abs().max()):.3e}")

    CALLS["fused"] = 0
    o1, o2 = fused(x.clone(), res.clone(), w, EPS)
    torch.cuda.synchronize()
    line(f"  fused: bridge calls={CALLS['fused']} "
         f"max|diff (out)={float((o1 - ref_f).abs().max()):.3e} "
         f"max|diff (res)={float((o2 - ref_res).abs().max()):.3e}")

    # ---------------------------------------------------------------- compiled
    sep("4. torch.compile(fullgraph=True, dynamic=False) - vLLM's setting")
    CALLS["rms"] = 0
    try:
        cf = torch.compile(lambda a, b: rms(a, b, EPS), fullgraph=True, dynamic=False)
        out_c = cf(x, w)
        torch.cuda.synchronize()
        line(f"  first call : bridge calls={CALLS['rms']} "
             f"max|diff|={float((out_c - ref).abs().max()):.3e}")
        n0 = CALLS["rms"]
        out_c2 = cf(x, w)
        torch.cuda.synchronize()
        line(f"  second call: bridge calls delta={CALLS['rms'] - n0} "
             f"max|diff|={float((out_c2 - ref).abs().max()):.3e}")
    except Exception as e:  # noqa: BLE001
        line(f"  FAILED: {type(e).__name__}: {str(e)[:600]}")
        for ln in traceback.format_exc().strip().splitlines()[-8:]:
            line("    " + ln)

    CALLS["fused"] = 0
    try:
        cf2 = torch.compile(
            lambda a, b, c: fused(a, b, c, EPS), fullgraph=True, dynamic=False
        )
        o1c, o2c = cf2(x.clone(), res.clone(), w)
        torch.cuda.synchronize()
        line(f"  fused compiled: bridge calls={CALLS['fused']} "
             f"max|diff (out)={float((o1c - ref_f).abs().max()):.3e} "
             f"max|diff (res)={float((o2c - ref_res).abs().max()):.3e}")
    except Exception as e:  # noqa: BLE001
        line(f"  fused FAILED: {type(e).__name__}: {str(e)[:600]}")
        for ln in traceback.format_exc().strip().splitlines()[-8:]:
            line("    " + ln)

    # ---------------------------------------------------------------- called ops
    sep("5. OpManager bookkeeping")
    line(f"  _called_ops (fl) = {sorted(mgr._called_ops)}")
    line()
    line("VERIFY_IR_KERNELS_UNIT_DONE", flush=True)


if __name__ == "__main__":
    main()
