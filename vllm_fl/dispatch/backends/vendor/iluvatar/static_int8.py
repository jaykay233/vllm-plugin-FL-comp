# Copyright 2026 FlagOS Contributors
"""Static W8A8 linears for Iluvatar, SmoothQuant scales from math_500 Level 3.

Uses ``torch._int_mm`` (BI-V150 int8 tensor core).  FlagGems Triton scaled_mm
is slower than bf16 here.  On by default on Iluvatar; set
``VLLM_FL_ILUVATAR_STATIC_INT8=0`` to disable.
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


def _apply_to_linear(mod: torch.nn.Module, meta: dict) -> None:
    mod._iluvatar_static_int8 = meta
    mod._iluvatar_static_int8_ws = None


def _ensure_ws(meta: dict, layer, m: int, k: int, n: int, device):
    ws = getattr(layer, "_iluvatar_static_int8_ws", None)
    if ws is None or ws["x8"].shape != (m, k) or ws["x8"].device != device:
        ws = {
            "x8": torch.empty((m, k), dtype=torch.int8, device=device),
            "acc": torch.empty((m, n), dtype=torch.int32, device=device),
            "out": torch.empty((m, n), dtype=torch.bfloat16, device=device),
        }
        layer._iluvatar_static_int8_ws = ws
    return ws


def _static_int8_linear(x: torch.Tensor, meta: dict, bias: torch.Tensor | None, layer):
    orig = x.shape
    k = orig[-1]
    x2 = x.reshape(-1, k).contiguous()
    m = x2.shape[0]
    n = meta["w8"].shape[1]
    mod = _cuda_mod()
    if mod is None:
        raise RuntimeError("static_w8a8 cuda extension is required")
    ws = _ensure_ws(meta, layer, m, k, n, x.device)
    b = bias if bias is not None else getattr(layer, "bias", None)
    if b is None:
        b = torch.empty(0, dtype=torch.bfloat16, device=x.device)
    else:
        b = b.contiguous()
    mod.static_w8a8_linear(
        x2,
        meta["w8"],
        meta["q_scale"],
        meta["out_scale"],
        ws["x8"],
        ws["acc"],
        ws["out"],
        b,
    )
    return ws["out"].reshape(*orig[:-1], n)


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
            orig_load(self, *args, **kwargs)
            _quantize_loaded_model(self)

        ModelRunnerFL.load_model = load_model
        ModelRunnerFL._iluvatar_static_int8_load_patch = True
        logger.info(
            "patch_static_int8_linear_for_iluvatar: enabled (calib=%s).", _CALIB
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
