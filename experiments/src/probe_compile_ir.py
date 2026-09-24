#!/usr/bin/env python3
"""Does torch.compile (fullgraph, as vLLM uses) cooperate with the dispatch chain?

Three questions:
  Q1  raw FlagGems triton call inside torch.compile(fullgraph=True) -> ok?
  Q2  same call wrapped as an opaque torch.library.custom_op -> ok?
  Q3  can a FlagGems impl be plugged into vLLM 0.24's IR-op provider list
      (vllm.ir.ops.rms_norm.register_impl + set_default) and survive compile?

Run:  conda activate mx && python /root/src/probe_compile_ir.py
"""

from __future__ import annotations

import traceback

import torch

DEV = "cuda"
DT = torch.bfloat16
EPS = 1e-6
H = 256
TOK = 32

CALLS: dict[str, int] = {"flagos": 0}


def line(msg: str = "") -> None:
    print(msg, flush=True)


def sep(title: str) -> None:
    line()
    line("=" * 74)
    line(title)
    line("=" * 74)


def main() -> None:
    line(f"torch {torch.__version__}  device={torch.cuda.get_device_name(0)}")

    from flag_gems.modules.normalization import gems_rms_forward

    x = torch.randn(TOK, H, device=DEV, dtype=DT)
    w = torch.randn(H, device=DEV, dtype=DT)

    ref = gems_rms_forward(x, None, w, EPS)
    torch.cuda.synchronize()

    # ---------------------------------------------------------------- Q1
    sep("Q1: raw FlagGems triton under torch.compile(fullgraph=True)")

    def raw_fn(xx, ww):
        return gems_rms_forward(xx, None, ww, EPS)

    q1_ok = False
    try:
        c_raw = torch.compile(raw_fn, fullgraph=True)
        out = c_raw(x, w)
        torch.cuda.synchronize()
        line(f"  compiled OK, max|diff|={float((out - ref).abs().max()):.3e}")
        q1_ok = True
    except Exception as e:  # noqa: BLE001
        line(f"  FAILED: {type(e).__name__}: {str(e)[:600]}")
        tb = traceback.format_exc().strip().splitlines()
        line("  last frames:")
        for ln in tb[-8:]:
            line("    " + ln)
    line(f"  => Q1 {'PASS' if q1_ok else 'FAIL (expected: dynamo cannot trace triton)'}")

    # ---------------------------------------------------------------- Q2
    sep("Q2: same op wrapped in an opaque torch.library.custom_op")

    @torch.library.custom_op("vllm_fl_probe::rms_norm", mutates_args=())
    def rms_norm_op(xx: torch.Tensor, ww: torch.Tensor, eps: float) -> torch.Tensor:
        return gems_rms_forward(xx, None, ww, eps)

    @rms_norm_op.register_fake
    def _rms_norm_op_fake(xx: torch.Tensor, ww: torch.Tensor, eps: float) -> torch.Tensor:
        return torch.empty_like(xx)

    def wrapped_fn(xx, ww):
        return rms_norm_op(xx, ww, EPS)

    q2_ok = False
    try:
        c_wrapped = torch.compile(wrapped_fn, fullgraph=True)
        out2 = c_wrapped(x, w)
        torch.cuda.synchronize()
        d = float((out2 - ref).abs().max())
        line(f"  compiled OK, max|diff|={d:.3e}")
        q2_ok = d < 1e-2
    except Exception as e:  # noqa: BLE001
        line(f"  FAILED: {type(e).__name__}: {str(e)[:600]}")
    line(f"  => Q2 {'PASS' if q2_ok else 'FAIL'}")

    # ---------------------------------------------------------------- Q3
    sep("Q3: vLLM IR-op provider (vllm.ir.ops.rms_norm) + priority + compile")

    try:
        from vllm import ir
        import vllm.ir.ops  # noqa: F401  (registers rms_norm / fused_add_rms_norm)

        np_ = ir.ops.rms_norm

        @np_.register_impl("flagos_probe", supported=True)
        def rms_norm_flagos(
            x: torch.Tensor,
            weight: torch.Tensor | None,
            epsilon: float,
            variance_size: int | None = None,
        ) -> torch.Tensor:
            CALLS["flagos"] += 1
            assert variance_size is None
            return rms_norm_op(x, weight, epsilon)

        np_.set_default(["flagos_probe", "native"])
        line(f"  registered providers: {np_.supported_providers()}")
        line(f"  priority            : {np_.get_priority()}")
        line(f"  torch op            : {np_.torch_op}")

        # eager
        CALLS["flagos"] = 0
        out_e = np_(x, w, EPS)
        torch.cuda.synchronize()
        line(f"  eager: provider calls={CALLS['flagos']} "
             f"max|diff|={float((out_e - ref).abs().max()):.3e}")

        # compiled (inductor) - the IR op must stay opaque to dynamo
        CALLS["flagos"] = 0
        c_ir = torch.compile(lambda xx, ww: np_(xx, ww, EPS), fullgraph=True)
        out_c = c_ir(x, w)
        torch.cuda.synchronize()
        line(f"  compiled(fullgraph): max|diff|={float((out_c - ref).abs().max()):.3e} "
             f"eager-provider-calls={CALLS['flagos']}")

        # second call - runtime path (the traced graph calls the IR op again)
        n0 = CALLS["flagos"]
        c_ir(x, w)
        torch.cuda.synchronize()
        line(f"  compiled second call: provider calls delta={CALLS['flagos'] - n0}")
        line("  => Q3 PASS" if CALLS["flagos"] > n0 else "  => Q3 INCONCLUSIVE (provider not re-entered)")
    except Exception as e:  # noqa: BLE001
        line(f"  FAILED: {type(e).__name__}: {str(e)[:800]}")
        for ln in traceback.format_exc().strip().splitlines()[-10:]:
            line("    " + ln)
        line("  => Q3 FAIL")

    line()
    line("PROBE_COMPILE_IR_DONE", flush=True)


if __name__ == "__main__":
    main()
