# Copyright 2026 FlagOS Contributors
"""Static W8A8 linears for Iluvatar, SmoothQuant scales from math_500 Level 3.

Uses ``torch._int_mm`` (BI-V150 int8 tensor core).  On by default on Iluvatar;
set ``VLLM_FL_ILUVATAR_STATIC_INT8=0`` to disable.

Compile / CUDAGraph notes (2026-10-07):
* Preload the CUDA extension so Dynamo never traces ``cpp_extension.load``.
* Use an opaque ``custom_op`` with ``mutates_args`` writing into **per-layer**
  persistent ``out`` buffers (cudagraph-safe addresses).
* ``FULL_DECODE_ONLY``: decode capture sizes are ``<=72``.  Always take the
  bf16 path for ``M <= _DECODE_WS_CAP`` (and while the stream is capturing) so
  CUDAGraph never records the INT8 kernel path.  INT8 is eager-prefill only
  (``M > 72``).  Return ``out[:m].clone()`` so inductor/CG cannot alias the
  persistent workspace into the next op.
* ``at::_int_mm`` rejects ``M <= 16``; pad activations to 32 when INT8 runs.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

_CALIB = Path(__file__).with_name("minicpm5_2b_l3_static_int8.pt")


def _parse_hf_name(name: str) -> tuple[int, str] | None:
    parts = name.split(".")
    if "layers" not in parts:
        return None
    i = parts.index("layers")
    if i + 1 >= len(parts):
        return None
    try:
        idx = int(parts[i + 1])
    except ValueError:
        return None
    tail = ".".join(parts[i + 2 :])
    mapping = {
        "self_attn.q_proj": "q",
        "self_attn.k_proj": "k",
        "self_attn.v_proj": "v",
        "self_attn.o_proj": "o",
        "mlp.gate_proj": "gate",
        "mlp.up_proj": "up",
        "mlp.down_proj": "down",
    }
    role = mapping.get(tail)
    if role is None:
        return None
    return idx, role


def _quant_weight(weight: torch.Tensor, xmax: torch.Tensor, alpha: float):
    n, k = weight.shape
    xm = xmax.to(device=weight.device, dtype=torch.float32).reshape(-1)
    if xm.numel() != k:
        return None
    w = weight.detach().float()
    wmax = w.abs().amax(dim=0).clamp_min(1e-6)
    s = (xm.clamp_min(1e-6).pow(alpha)) / wmax.pow(1.0 - alpha)
    s = s.clamp(1e-3, 1e3)
    w_s = w * s
    w_scale = (w_s.abs().amax(dim=1).clamp_min(1e-6) / 127.0)
    w8 = (w_s / w_scale.unsqueeze(1)).round().clamp(-128, 127).to(torch.int8)
    a_scale = ((xm / s).amax() / 127.0).clamp_min(1e-8)
    q_scale = (s.reciprocal() / a_scale).contiguous()
    out_scale = (a_scale * w_scale).contiguous()
    return {
        "w8": w8.t().contiguous(),
        "q_scale": q_scale,
        "out_scale": out_scale,
    }


_CUDA_MOD = None
_CUDA_FAIL = False


@torch.compiler.disable
def _cuda_mod():
    global _CUDA_MOD, _CUDA_FAIL
    if _CUDA_FAIL:
        return None
    if _CUDA_MOD is not None:
        return _CUDA_MOD
    src_cu = Path(__file__).with_name("static_w8a8_cuda.cu")
    src_cpp = Path(__file__).with_name("static_w8a8_cuda.cpp")
    try:
        from torch.utils.cpp_extension import load

        _CUDA_MOD = load(
            name="fl_static_w8a8_v1",
            sources=[str(src_cpp), str(src_cu)],
            extra_cuda_cflags=["-O3", "-x", "ivcore"],
            extra_cflags=["-O3"],
            verbose=False,
        )
    except Exception as e:
        logger.warning("static_w8a8 cuda load failed: %s", e)
        _CUDA_FAIL = True
        return None
    return _CUDA_MOD


def _preload_cuda_ext() -> bool:
    return _cuda_mod() is not None


# FULL_DECODE_ONLY capture list ends at 72; keep persistent buffers to that.
_DECODE_WS_CAP = 72


def _apply_to_linear(mod: torch.nn.Module, meta: dict) -> None:
    mod._iluvatar_static_int8 = meta
    mod._iluvatar_w8 = meta["w8"]
    mod._iluvatar_q_scale = meta["q_scale"]
    mod._iluvatar_out_scale = meta["out_scale"]
    # Fixed-shape decode workspaces registered now so cudagraph/inductor cannot
    # specialize them down to the capture batch size (e.g. 16).
    k = meta["w8"].shape[0]
    n = meta["w8"].shape[1]
    device = meta["w8"].device
    cap = _DECODE_WS_CAP
    mod._iluvatar_x_pad = torch.empty((cap, k), dtype=torch.bfloat16, device=device)
    mod._iluvatar_x8 = torch.empty((cap, k), dtype=torch.int8, device=device)
    mod._iluvatar_acc = torch.empty((cap, n), dtype=torch.int32, device=device)
    mod._iluvatar_out = torch.empty((cap, n), dtype=torch.bfloat16, device=device)
    mod._iluvatar_static_int8_ws = None


def _make_ws(m: int, k: int, n: int, device):
    return {
        "cap": m,
        "k": k,
        "n": n,
        "x_pad": torch.empty((m, k), dtype=torch.bfloat16, device=device),
        "x8": torch.empty((m, k), dtype=torch.int8, device=device),
        "acc": torch.empty((m, n), dtype=torch.int32, device=device),
        "out": torch.empty((m, n), dtype=torch.bfloat16, device=device),
    }


def _ensure_ws(layer, m: int, k: int, n: int, device):
    """Persistent per-layer WS for decode-sized M; ephemeral for large prefill."""
    if m > _DECODE_WS_CAP:
        return _make_ws(m, k, n, device)
    ws = getattr(layer, "_iluvatar_static_int8_ws", None)
    if (
        ws is None
        or ws["cap"] < m
        or ws["k"] != k
        or ws["n"] != n
        or ws["out"].device != device
    ):
        # Allocate to decode cap once so all capture sizes share addresses.
        cap = _DECODE_WS_CAP
        ws = _make_ws(cap, k, n, device)
        layer._iluvatar_static_int8_ws = ws
    return ws


def _use_bf16_path(m: int) -> bool:
    """Decode-sized M (and CG capture) must stay on bf16; INT8 is prefill-only."""
    if m <= _DECODE_WS_CAP:
        return True
    try:
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            return True
    except Exception:
        pass
    return False


if hasattr(torch.library, "custom_op"):

    @torch.library.custom_op(
        "vllm_fl::static_w8a8_linear_v5",
        mutates_args={"x_pad", "x8", "acc", "out"},
    )
    def _static_w8a8_linear_op(
        x: torch.Tensor,
        w8: torch.Tensor,
        w_bf16: torch.Tensor,
        q_scale: torch.Tensor,
        out_scale: torch.Tensor,
        bias: torch.Tensor,
        x_pad: torch.Tensor,
        x8: torch.Tensor,
        acc: torch.Tensor,
        out: torch.Tensor,
    ) -> None:
        """bf16 for decode/CG; W8A8 only for eager prefill M > decode cap."""
        m = x.shape[0]
        n = w8.shape[1]
        if out.shape[0] < m:
            raise RuntimeError(
                f"static_w8a8 workspace too small: m={m} out={tuple(out.shape)}"
            )
        if bias is None or bias.numel() == 0:
            b = None
        else:
            b = bias
        if _use_bf16_path(m):
            # Decode / CG capture: never record INT8 into the CUDAGraph.
            y = torch.nn.functional.linear(x, w_bf16, b)
            out[:m].copy_(y)
            return
        mod = _cuda_mod()
        if mod is None:
            raise RuntimeError("static_w8a8 cuda extension is required")
        if x_pad.shape[0] < m or x8.shape[0] < m:
            raise RuntimeError(
                f"static_w8a8 workspace too small: m={m} "
                f"x_pad={tuple(x_pad.shape)}"
            )
        x_pad[:m].copy_(x)
        bb = (
            torch.empty(0, dtype=torch.bfloat16, device=x.device)
            if b is None
            else b
        )
        mod.static_w8a8_linear(
            x_pad[:m],
            w8,
            q_scale,
            out_scale,
            x8[:m],
            acc[:m],
            out[:m],
            bb,
        )

    @_static_w8a8_linear_op.register_fake
    def _static_w8a8_linear_op_fake(
        x, w8, w_bf16, q_scale, out_scale, bias, x_pad, x8, acc, out
    ):
        return None

else:
    _static_w8a8_linear_op = None


def _static_int8_linear(x: torch.Tensor, meta: dict, bias: torch.Tensor | None, layer):
    orig = x.shape
    k = orig[-1]
    x2 = x.reshape(-1, k).contiguous()
    m = x2.shape[0]
    n = meta["w8"].shape[1]
    b = bias if bias is not None else getattr(layer, "bias", None)
    if b is None:
        b = torch.empty(0, dtype=torch.bfloat16, device=x.device)
    else:
        b = b.contiguous()
    w8 = getattr(layer, "_iluvatar_w8", meta["w8"])
    w_bf16 = layer.weight
    q_scale = getattr(layer, "_iluvatar_q_scale", meta["q_scale"])
    out_scale = getattr(layer, "_iluvatar_out_scale", meta["out_scale"])
    if m <= _DECODE_WS_CAP and getattr(layer, "_iluvatar_out", None) is not None:
        x_pad = layer._iluvatar_x_pad
        x8 = layer._iluvatar_x8
        acc = layer._iluvatar_acc
        out = layer._iluvatar_out
    else:
        ws = _ensure_ws(layer, max(m, 32), k, n, x.device)
        x_pad, x8, acc, out = ws["x_pad"], ws["x8"], ws["acc"], ws["out"]
    if _static_w8a8_linear_op is not None:
        _static_w8a8_linear_op(
            x2, w8, w_bf16, q_scale, out_scale, b, x_pad, x8, acc, out
        )
    else:
        if _use_bf16_path(m):
            y = torch.nn.functional.linear(
                x2, w_bf16, None if b.numel() == 0 else b
            )
            return y.reshape(*orig[:-1], n)
        x_pad[:m].copy_(x2)
        mod = _cuda_mod()
        if mod is None:
            raise RuntimeError("static_w8a8 cuda extension is required")
        mod.static_w8a8_linear(
            x_pad[:m], w8, q_scale, out_scale, x8[:m], acc[:m], out[:m], b
        )
    # Clone so CG / inductor cannot keep an alias into the persistent workspace.
    return out[:m].clone().reshape(*orig[:-1], n)


def patch_static_int8_linear_for_iluvatar() -> None:
    try:
        from vllm_fl.dispatch.backends.vendor.iluvatar.iluvatar import (
            _is_iluvatar_platform,
        )

        if not _is_iluvatar_platform():
            return
        if os.getenv("VLLM_FL_ILUVATAR_STATIC_INT8", "1") == "0":
            logger.info(
                "patch_static_int8_linear_for_iluvatar: off "
                "(VLLM_FL_ILUVATAR_STATIC_INT8=0)."
            )
            return
        if not _CALIB.is_file():
            logger.warning(
                "patch_static_int8_linear_for_iluvatar: missing calib file %s",
                _CALIB,
            )
            return

        if not _preload_cuda_ext():
            logger.warning(
                "patch_static_int8_linear_for_iluvatar: CUDA ext missing; leave off."
            )
            return

        from vllm.model_executor.layers.linear import UnquantizedLinearMethod

        if getattr(UnquantizedLinearMethod, "_iluvatar_static_int8_applied", False):
            return

        orig_apply = UnquantizedLinearMethod.apply

        def apply(self, layer, x, bias=None):
            meta = getattr(layer, "_iluvatar_static_int8", None)
            if meta is None:
                return orig_apply(self, layer, x, bias)
            return _static_int8_linear(
                x,
                meta,
                bias if bias is not None else getattr(layer, "bias", None),
                layer,
            )

        UnquantizedLinearMethod.apply = apply
        UnquantizedLinearMethod._iluvatar_static_int8_applied = True

        from vllm_fl.worker.model_runner import ModelRunnerFL

        if getattr(ModelRunnerFL, "_iluvatar_static_int8_load_patch", False):
            return
        orig_load = ModelRunnerFL.load_model

        def load_model(self, *args, **kwargs):
            _preload_cuda_ext()
            orig_load(self, *args, **kwargs)
            _quantize_loaded_model(self)

        ModelRunnerFL.load_model = load_model
        ModelRunnerFL._iluvatar_static_int8_load_patch = True
        logger.info(
            "patch_static_int8_linear_for_iluvatar: enabled (calib=%s, "
            "custom_op=%s, preloaded=1, per_layer_ws=1).",
            _CALIB,
            _static_w8a8_linear_op is not None,
        )
    except Exception as e:
        logger.warning("patch_static_int8_linear_for_iluvatar: %s", e)


def _quantize_loaded_model(worker) -> None:
    payload = torch.load(_CALIB, map_location="cpu", weights_only=False)
    alpha = float(payload.get("alpha", 0.5))
    xmax = payload["xmax"]
    by_layer: dict[int, dict[str, torch.Tensor]] = {}
    for name, tensor in xmax.items():
        parsed = _parse_hf_name(name)
        if parsed is None:
            continue
        idx, role = parsed
        by_layer.setdefault(idx, {})[role] = tensor

    model = worker.model
    n = 0
    for name, mod in model.named_modules():
        w = getattr(mod, "weight", None)
        if not isinstance(w, torch.Tensor) or w.ndim != 2:
            continue
        parsed = _parse_vllm_name(name)
        if parsed is None:
            continue
        idx, role = parsed
        roles = by_layer.get(idx)
        if not roles:
            continue
        if role == "qkv":
            xm = _stack_xmax(roles, ("q", "k", "v"), w.shape[1])
        elif role == "gate_up":
            xm = _stack_xmax(roles, ("gate", "up"), w.shape[1])
        else:
            xm = roles.get(role)
        if xm is None:
            continue
        meta = _quant_weight(w, xm, alpha)
        if meta is None:
            continue
        for k, v in meta.items():
            if torch.is_tensor(v):
                meta[k] = v.to(device=w.device)
        _apply_to_linear(mod, meta)
        n += 1
    logger.info("patch_static_int8_linear_for_iluvatar: quantized %d linears", n)


def _parse_vllm_name(name: str) -> tuple[int, str] | None:
    parts = name.split(".")
    if "layers" not in parts:
        return None
    i = parts.index("layers")
    try:
        idx = int(parts[i + 1])
    except (ValueError, IndexError):
        return None
    tail = ".".join(parts[i + 2 :])
    mapping = {
        "self_attn.qkv_proj": "qkv",
        "self_attn.q_proj": "q",
        "self_attn.k_proj": "k",
        "self_attn.v_proj": "v",
        "self_attn.o_proj": "o",
        "mlp.gate_up_proj": "gate_up",
        "mlp.gate_proj": "gate",
        "mlp.up_proj": "up",
        "mlp.down_proj": "down",
    }
    role = mapping.get(tail)
    if role is None:
        return None
    return idx, role


def _stack_xmax(roles: dict, keys: tuple[str, ...], k_dim: int) -> torch.Tensor | None:
    xs = [roles[k] for k in keys if k in roles]
    if not xs:
        return None
    xm = xs[0].float().reshape(-1)
    if xm.numel() != k_dim:
        return None
    for t in xs[1:]:
        tt = t.float().reshape(-1)
        if tt.numel() == k_dim:
            xm = torch.maximum(xm, tt)
    return xm
