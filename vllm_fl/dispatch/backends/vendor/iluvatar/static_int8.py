# Copyright 2026 FlagOS Contributors
"""Static W8A8 linears for Iluvatar, SmoothQuant scales from math_500 Level 3.

Uses ``torch._int_mm`` (BI-V150 int8 tensor core).  Off by default: Level 3
can fall below 0.95.  Set ``VLLM_FL_ILUVATAR_STATIC_INT8=1`` to enable.

Compile / CUDAGraph notes (2026-10-07 / 2026-10-13):
* Preload the CUDA extension so Dynamo never traces ``cpp_extension.load``.
* Use an opaque ``custom_op`` with ``mutates_args`` writing into **per-layer**
  persistent ``out`` buffers (cudagraph-safe addresses).
* ``FULL_DECODE_ONLY``: decode capture sizes are ``<=72``.  Default: bf16 for
  ``M <= _DECODE_WS_CAP`` (and while capturing) so CUDAGraph never records
  INT8.  Opt-in ``VLLM_FL_ILUVATAR_STATIC_INT8_DECODE_INT8=1`` keeps INT8 for
  decode too.
* Dynamo always traces decode-INT8 through the opaque custom_op (Python
  short-circuits before the op are dead).  Selective bf16-on-decode and
  dynamic-act must be selected **inside** the custom_op via per-layer
  ``path_mode`` (v6).
* BI-V150 RCA (2026-10-13): bare ``<<<>>>`` quant/epilogue launches used the
  default stream, so CUDAGraph capture on a side stream recorded an **empty**
  graph (stale replay → garbage tokens).  Fix: launch on
  ``at::cuda::getCurrentCUDAStream()``.  Decode-INT8 uses the CUDA ext
  (pad-to-32 + stream-aware kernels); torch-native quant/epi remains a
  fallback if the ext is missing.
* Return ``out[:m].clone()`` so inductor/CG cannot alias the persistent
  workspace into the next op.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

_CALIB_DEFAULT = Path(__file__).with_name("minicpm5_2b_l3_static_int8.pt")


def _calib_path() -> Path:
    """Optional override: decode∪prefill recalib via env path."""
    override = os.getenv("VLLM_FL_ILUVATAR_STATIC_INT8_CALIB", "").strip()
    if override:
        return Path(override)
    return _CALIB_DEFAULT


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


def _simple_int8_enabled() -> bool:
    """Plain per-channel / per-head W8A8 (no SmoothQuant, no QuaRot)."""
    return os.getenv("VLLM_FL_ILUVATAR_STATIC_INT8_SIMPLE", "0") == "1"


def _quarot_enabled() -> bool:
    """Per-GEMM randomized Hadamard (QuaRot-lite). Disabled when SIMPLE=1."""
    if _simple_int8_enabled():
        return False
    return os.getenv("VLLM_FL_ILUVATAR_STATIC_INT8_QUAROT", "0") == "1"


def _attn_head_dim() -> int:
    return int(os.getenv("VLLM_FL_ILUVATAR_STATIC_INT8_HEAD_DIM", "128"))


# Signs cache (CPU float32). Online uses CG-safe in-place FWHT + signs (not dense x@H).
_HADAMARD_SIGNS: dict[int, torch.Tensor] = {}


def _is_pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _fwht_pow2_out(src: torch.Tensor, dst: torch.Tensor) -> None:
    """Normalized FWHT last-dim pow2: write ``dst = H @ src`` (row-vector right-mul).

    In-place butterfly with explicit a/b clones — BI-V150 CUDAGraph safe
    (torch.stack/reshape allot path previously → ixErrorInvalidValue).
    """
    n = src.shape[-1]
    if dst.data_ptr() != src.data_ptr():
        dst.copy_(src)
    y = dst
    # Work in float32.
    if y.dtype != torch.float32:
        raise RuntimeError(f"fwht dst must be float32, got {y.dtype}")
    m = y.shape[0] if y.ndim >= 2 else 1
    if y.ndim == 1:
        y = y.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False
        # Flatten leading dims into rows.
        if y.ndim > 2:
            y = y.reshape(-1, n)
            m = y.shape[0]
    h = 1
    while h < n:
        yv = y.view(m, n // (2 * h), 2, h)
        a = yv[:, :, 0, :].clone()
        b = yv[:, :, 1, :].clone()
        yv[:, :, 0, :].copy_(a + b)
        yv[:, :, 1, :].copy_(a - b)
        h *= 2
    y.mul_(n**-0.5)
    if squeeze:
        dst.copy_(y.reshape(-1))


def _fwht_pow2(x: torch.Tensor) -> torch.Tensor:
    """Normalized FWHT on last dim (power-of-two) — eager/offline helper."""
    out = x.float().contiguous()
    if out.ndim == 1:
        _fwht_pow2_out(out, out)
        return out.to(dtype=x.dtype)
    flat = out.reshape(-1, out.shape[-1])
    _fwht_pow2_out(flat, flat)
    return out.reshape(x.shape).to(dtype=x.dtype)


def _fwht_last(x: torch.Tensor) -> torch.Tensor:
    """Normalized Hadamard on last dim. K=2^n direct; else block (e.g. 6144=3×2048)."""
    k = x.shape[-1]
    if _is_pow2(k):
        return _fwht_pow2(x)
    block = 1 << (k.bit_length() - 1)
    while block > 1 and k % block != 0:
        block >>= 1
    if block < 2 or k % block != 0:
        return x
    n_blk = k // block
    y = x.float().reshape(*x.shape[:-1], n_blk, block).contiguous()
    flat = y.reshape(-1, block)
    _fwht_pow2_out(flat, flat)
    return y.reshape(*x.shape).to(dtype=x.dtype)


def _quarot_signs(k: int, device, dtype) -> torch.Tensor:
    signs = _HADAMARD_SIGNS.get(k)
    if signs is None:
        g = torch.Generator(device="cpu")
        g.manual_seed(0x51A207 ^ (k * 0x9E3779B9))
        s = torch.randint(0, 2, (k,), generator=g, dtype=torch.int8)
        signs = (s * 2 - 1).to(torch.float32)
        _HADAMARD_SIGNS[k] = signs
    return signs.to(device=device, dtype=dtype)


def _hadamard_act(x: torch.Tensor, signs: torch.Tensor | None = None) -> torch.Tensor:
    """Online act rotation: ``x @ H @ diag(s)`` via FWHT + signs."""
    y = _fwht_last(x)
    s = signs if signs is not None else _quarot_signs(y.shape[-1], y.device, y.dtype)
    return y * s


def _hadamard_act_into(
    x: torch.Tensor,
    workspace: torch.Tensor,
    signs: torch.Tensor,
) -> None:
    """CG-friendly: FWHT+signs into float32 ``workspace[:m,:k]`` (no heap alloc)."""
    m, k = x.shape[0], x.shape[-1]
    ws = workspace[:m, :k]
    # bf16/fp16 → f32 via copy_ (no temporary .float() tensor — CG-safe).
    ws.copy_(x)
    if _is_pow2(k):
        _fwht_pow2_out(ws, ws)
        ws.mul_(signs[:k])
        return
    block = 1 << (k.bit_length() - 1)
    while block > 1 and k % block != 0:
        block >>= 1
    if block < 2 or k % block != 0:
        return
    n_blk = k // block
    flat = ws.view(m * n_blk, block)
    _fwht_pow2_out(flat, flat)
    ws.mul_(signs[:k])


def _fuse_quarot_weight(weight: torch.Tensor) -> torch.Tensor:
    """Fuse ``W @ S @ H`` so bf16 ``F.linear(x, W') ≡ F.linear(x@H@S, W)``."""
    w = weight.detach()
    k = w.shape[1]
    signs = _quarot_signs(k, w.device, torch.float32)
    w_s = w.float() * signs.unsqueeze(0)
    return _fwht_last(w_s).to(dtype=w.dtype)


def _per_head_w_scale(w: torch.Tensor, head_dim: int) -> torch.Tensor:
    """Per-output-head absmax/127; falls back to per-channel if N not divisible."""
    n, _k = w.shape
    if head_dim <= 0 or n % head_dim != 0:
        return (w.abs().amax(dim=1).clamp_min(1e-6) / 127.0).contiguous()
    n_heads = n // head_dim
    # [n_heads, head_dim, K] → scale per head, broadcast to channels
    w_h = w.view(n_heads, head_dim, -1)
    head_scale = w_h.abs().amax(dim=(1, 2)).clamp_min(1e-6) / 127.0  # [n_heads]
    return head_scale.repeat_interleave(head_dim).contiguous()


def _quant_weight_simple(
    weight: torch.Tensor,
    xmax: torch.Tensor,
    role: str | None = None,
) -> dict | None:
    """No SmoothQuant / no rotation.

    * Weights: per-channel; for attn ``qkv``/``o``/``q``/``k``/``v`` use **per-head**
      scales (``HEAD_DIM``, default 128).
    * Activations: static per-tensor from ``amax(xmax)`` (decode can still opt into
      per-token dynamic via ``DECODE_DYNAMIC_ACT``).
    """
    n, k = weight.shape
    xm = xmax.to(device=weight.device, dtype=torch.float32).reshape(-1)
    if xm.numel() != k:
        return None
    w = weight.detach().float()
    head_dim = _attn_head_dim()
    attn_roles = {"qkv", "o", "q", "k", "v"}
    if role in attn_roles:
        w_scale = _per_head_w_scale(w, head_dim)
    else:
        w_scale = (w.abs().amax(dim=1).clamp_min(1e-6) / 127.0).contiguous()
    w8 = (w / w_scale.unsqueeze(1)).round().clamp(-128, 127).to(torch.int8)
    a_scale = (xm.amax() / 127.0).clamp_min(1e-8)
    # Uniform act scale across K (per-tensor); keep shape [K] for existing kernels.
    q_scale = torch.full((k,), (1.0 / a_scale).item(), device=w.device, dtype=torch.float32)
    out_scale = (a_scale * w_scale).contiguous()
    # smooth=1 → dynamic-act path treats as identity channel smooth.
    smooth = torch.ones(k, device=w.device, dtype=torch.float32)
    return {
        "w8": w8.t().contiguous(),
        "q_scale": q_scale.contiguous(),
        "out_scale": out_scale,
        "smooth": smooth,
        "w_scale": w_scale.contiguous(),
    }


def _quant_weight(
    weight: torch.Tensor,
    xmax: torch.Tensor,
    alpha: float,
    role: str | None = None,
):
    if _simple_int8_enabled():
        return _quant_weight_simple(weight, xmax, role=role)
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
        # Kept for decode-time dynamic activation rescale.
        "smooth": s.contiguous(),
        "w_scale": w_scale.contiguous(),
    }


_CUDA_MOD = None
_CUDA_FAIL = False
_EXT_NAME = "fl_static_w8a8_v4"  # v4: vectorized epi (__ldg + int4/uint2)


def _clear_stale_ext_lock(ext_name: str) -> None:
    """Drop abandoned ``cpp_extension`` filelocks left by SIGKILL'd builds.

    ``torch.utils.cpp_extension.load`` waits forever on the build-dir ``lock``.
    A killed ``vllm serve`` mid-load leaves that lock held/orphan; the next
    import then hangs right after ``patch_openai_api_mbt_for_iluvatar`` (seen
    on BI-V150 when restarting 131k L3 servers).  Unlink the lock path so a
    new inode is used; safe for single-GPU bring-up.
    """
    try:
        from torch.utils.cpp_extension import _get_build_directory

        build_dir = Path(_get_build_directory(ext_name, False))
    except Exception:
        return
    lock = build_dir / "lock"
    if not lock.exists():
        return
    try:
        lock.unlink()
        logger.warning(
            "static_w8a8: removed stale torch_extensions lock %s", lock
        )
    except OSError as e:
        logger.warning("static_w8a8: could not remove stale lock %s: %s", lock, e)


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

        _clear_stale_ext_lock(_EXT_NAME)
        logger.info("static_w8a8: loading CUDA extension %s ...", _EXT_NAME)
        _CUDA_MOD = load(
            # v3: stream-aware quant/epi launches (CG-safe on BI-V150).
            name=_EXT_NAME,
            sources=[str(src_cpp), str(src_cu)],
            extra_cuda_cflags=["-O3", "-x", "ivcore"],
            extra_cflags=["-O3"],
            verbose=False,
        )
        logger.info("static_w8a8: CUDA extension %s ready", _EXT_NAME)
    except Exception as e:
        logger.warning("static_w8a8 cuda load failed: %s", e)
        _CUDA_FAIL = True
        return None
    return _CUDA_MOD


def _preload_cuda_ext() -> bool:
    return _cuda_mod() is not None


# FULL_DECODE_ONLY capture list ends at 72; keep persistent buffers to that.
_DECODE_WS_CAP = 72


def _tensor_nbytes(t: torch.Tensor | None) -> int:
    if t is None or not isinstance(t, torch.Tensor):
        return 0
    return int(t.numel() * t.element_size())


def _apply_to_linear(mod: torch.nn.Module, meta: dict) -> int:
    """Attach INT8 meta/workspaces. Returns newly allocated CUDA bytes."""
    mod._iluvatar_static_int8 = meta
    mod._iluvatar_role = meta.get("role")
    mod._iluvatar_w8 = meta["w8"]
    mod._iluvatar_q_scale = meta["q_scale"]
    mod._iluvatar_out_scale = meta["out_scale"]
    device = meta["w8"].device
    role = meta.get("role")
    mode = _path_mode_for_role(role)
    # CUDA 0-dim so custom_op / CG inputs stay on-device; .item() only runs
    # inside the opaque op during capture (branch is then baked into the graph).
    mod._iluvatar_path_mode = torch.tensor(
        mode, dtype=torch.int32, device=device
    )
    sm = meta.get("smooth")
    wscl = meta.get("w_scale")
    if sm is None:
        sm = torch.empty(0, dtype=torch.float32, device=device)
    if wscl is None:
        wscl = torch.empty(0, dtype=torch.float32, device=device)
    mod._iluvatar_smooth = sm
    mod._iluvatar_w_scale = wscl
    k = meta["w8"].shape[0]
    n = meta["w8"].shape[1]
    if _quarot_enabled():
        mod._iluvatar_quarot_signs = _quarot_signs(k, device, torch.float32)
    else:
        mod._iluvatar_quarot_signs = torch.empty(0, dtype=torch.float32, device=device)
    added = (
        _tensor_nbytes(meta["w8"])
        + _tensor_nbytes(meta["q_scale"])
        + _tensor_nbytes(meta["out_scale"])
        + _tensor_nbytes(sm)
        + _tensor_nbytes(wscl)
        + _tensor_nbytes(mod._iluvatar_path_mode)
        + _tensor_nbytes(mod._iluvatar_quarot_signs)
    )
    # Decode-INT8 / QuaRot: persistent WS (bf16-decode still writes ``out``;
    # QuaRot needs float32 ``x_rot`` for CG-safe FWHT).
    mod._iluvatar_static_int8_ws = None
    need_ws = _decode_int8_enabled() or _quarot_enabled()
    if not need_ws:
        mod._iluvatar_x_pad = None
        mod._iluvatar_x8 = None
        mod._iluvatar_acc = None
        mod._iluvatar_out = None
        mod._iluvatar_x_rot = None
        return added
    cap = _DECODE_WS_CAP
    mod._iluvatar_x_pad = torch.empty((cap, k), dtype=torch.bfloat16, device=device)
    mod._iluvatar_x8 = torch.empty((cap, k), dtype=torch.int8, device=device)
    mod._iluvatar_acc = torch.empty((cap, n), dtype=torch.int32, device=device)
    mod._iluvatar_out = torch.empty((cap, n), dtype=torch.bfloat16, device=device)
    mod._iluvatar_x_rot = (
        torch.empty((cap, k), dtype=torch.float32, device=device)
        if _quarot_enabled()
        else torch.empty(0, dtype=torch.float32, device=device)
    )
    added += (
        _tensor_nbytes(mod._iluvatar_x_pad)
        + _tensor_nbytes(mod._iluvatar_x8)
        + _tensor_nbytes(mod._iluvatar_acc)
        + _tensor_nbytes(mod._iluvatar_out)
        + _tensor_nbytes(mod._iluvatar_x_rot)
    )
    return added


def _make_ws(m: int, k: int, n: int, device, quarot: bool = False):
    ws = {
        "cap": m,
        "k": k,
        "n": n,
        "x_pad": torch.empty((m, k), dtype=torch.bfloat16, device=device),
        "x8": torch.empty((m, k), dtype=torch.int8, device=device),
        "acc": torch.empty((m, n), dtype=torch.int32, device=device),
        "out": torch.empty((m, n), dtype=torch.bfloat16, device=device),
        "x_rot": torch.empty(
            (m, k) if quarot else (0,),
            dtype=torch.float32,
            device=device,
        ),
    }
    return ws


def _ensure_ws(layer, m: int, k: int, n: int, device):
    """Persistent per-layer WS for decode-sized M; ephemeral for large prefill."""
    quarot = _quarot_enabled()
    if m > _DECODE_WS_CAP:
        return _make_ws(m, k, n, device, quarot=quarot)
    ws = getattr(layer, "_iluvatar_static_int8_ws", None)
    need_rot = quarot and (ws is None or ws.get("x_rot") is None or ws["x_rot"].numel() == 0)
    if (
        ws is None
        or ws["cap"] < m
        or ws["k"] != k
        or ws["n"] != n
        or ws["out"].device != device
        or need_rot
    ):
        # Allocate to decode cap once so all capture sizes share addresses.
        cap = _DECODE_WS_CAP
        ws = _make_ws(cap, k, n, device, quarot=quarot)
        layer._iluvatar_static_int8_ws = ws
    return ws


def _decode_int8_enabled() -> bool:
    """Opt-in INT8 inside decode CUDAGraphs (torch-native quant/epi; pad-to-32)."""
    return os.getenv("VLLM_FL_ILUVATAR_STATIC_INT8_DECODE_INT8", "0") == "1"


def _decode_bf16_roles() -> set[str]:
    """Roles that stay bf16 on decode even when decode-INT8 is on.

    Comma-separated ``VLLM_FL_ILUVATAR_STATIC_INT8_DECODE_BF16_ROLES``.
    Empty / unset → no exceptions (all decode-INT8).  Typical recovery:
    ``o,down`` (outlier-heavy projections).
    """
    raw = os.getenv("VLLM_FL_ILUVATAR_STATIC_INT8_DECODE_BF16_ROLES", "").strip()
    if not raw:
        return set()
    return {x.strip() for x in raw.split(",") if x.strip()}


def _role_bf16_on_decode(role: str | None) -> bool:
    if not role:
        return False
    return role in _decode_bf16_roles()


def _decode_dynamic_act_enabled() -> bool:
    """Per-token dynamic activation scale on decode (static SmoothQuant weights)."""
    return os.getenv("VLLM_FL_ILUVATAR_STATIC_INT8_DECODE_DYNAMIC_ACT", "0") == "1"


def _drop_bf16_enabled() -> bool:
    """Free GPU bf16 weight copies after quantize (requires decode-INT8)."""
    if not _decode_int8_enabled():
        return False
    # Selective bf16-on-decode layers still need real bf16 weights.
    if _decode_bf16_roles():
        return False
    return os.getenv("VLLM_FL_ILUVATAR_STATIC_INT8_DROP_BF16", "0") == "1"


def _drop_bf16_weights(model: torch.nn.Module) -> int:
    """Replace quantized-layer bf16 weights with a 1x1 CUDA stub; return bytes freed.

    Keeps a CUDA tensor so opaque custom_op / CG inputs stay on-device.  Shape
    lives on ``w8`` (K,N); decode-INT8 never reads bf16 weights.
    """
    freed = 0
    for _name, mod in model.named_modules():
        if getattr(mod, "_iluvatar_static_int8", None) is None:
            continue
        w = getattr(mod, "weight", None)
        if not isinstance(w, torch.Tensor) or w.numel() <= 1:
            continue
        if w.device.type != "cuda":
            continue
        freed += int(w.numel() * w.element_size())
        stub = torch.empty(1, 1, dtype=w.dtype, device=w.device)
        mod.weight = torch.nn.Parameter(stub, requires_grad=False)
    if freed:
        torch.cuda.empty_cache()
    return freed


def _use_bf16_path(m: int) -> bool:
    """Decode-sized M (and CG capture) stay on bf16 unless decode-INT8 opt-in."""
    if _decode_int8_enabled():
        return False
    if m <= _DECODE_WS_CAP:
        return True
    try:
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            return True
    except Exception:
        pass
    return False


# path_mode for custom_op v6 (baked per-layer at quantize; Dynamo-safe).
_PATH_STATIC = 0
_PATH_BF16_DECODE = 1
_PATH_DYN_ACT = 2


def _path_mode_for_role(role: str | None) -> int:
    """Select decode path; prefill (m>cap) always uses static INT8 in the op."""
    if not _decode_int8_enabled():
        return _PATH_STATIC
    if _role_bf16_on_decode(role):
        return _PATH_BF16_DECODE
    # SIMPLE = static W8A8 only — never per-token dynamic act.
    if _simple_int8_enabled():
        return _PATH_STATIC
    if _decode_dynamic_act_enabled():
        return _PATH_DYN_ACT
    return _PATH_STATIC


def _torch_w8a8_decode(
    x: torch.Tensor,
    w8: torch.Tensor,
    q_scale: torch.Tensor,
    out_scale: torch.Tensor,
    bias: torch.Tensor | None,
    x8: torch.Tensor,
    acc: torch.Tensor,
    out: torch.Tensor,
    smooth: torch.Tensor | None = None,
    w_scale: torch.Tensor | None = None,
    dynamic: bool = False,
) -> None:
    """CG-safe W8A8 decode: torch quant + pad-to-32 + ``_int_mm`` + epi.

    Optional dynamic activation: per-token ``a_dyn`` after SmoothQuant smooth,
    keeping static smoothed INT8 weights.  Must stay on ATen ops for BI-V150 CG.
    """
    m = x.shape[0]
    m_eff = 32 if m < 32 else m
    if (
        x8.shape[0] < m_eff
        or acc.shape[0] < m_eff
        or out.shape[0] < m_eff
    ):
        raise RuntimeError(
            f"static_w8a8 decode ws too small: m={m} m_eff={m_eff} "
            f"x8={tuple(x8.shape)} acc={tuple(acc.shape)} out={tuple(out.shape)}"
        )
    if (
        dynamic
        and smooth is not None
        and w_scale is not None
        and smooth.numel() > 0
        and w_scale.numel() > 0
    ):
        # x_s = x / s ; a_dyn = amax(|x_s|)/127 per row ; x8 = round(x_s/a_dyn)
        x_s = x.float() * smooth.reciprocal()
        a_dyn = x_s.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / 127.0
        x8[:m].copy_((x_s / a_dyn).round().clamp(-128, 127).to(torch.int8))
        if m_eff > m:
            x8[m:m_eff].zero_()
        torch.ops.aten._int_mm.out(x8[:m_eff], w8, out=acc[:m_eff])
        y_m = (acc[:m].float() * (a_dyn * w_scale)).to(torch.bfloat16)
        if bias is not None and bias.numel() > 0:
            y_m = y_m + bias
        out[:m].copy_(y_m)
        if m_eff > m:
            out[m:m_eff].zero_()
        return
    x8[:m].copy_((x.float() * q_scale).round().clamp(-128, 127).to(torch.int8))
    if m_eff > m:
        x8[m:m_eff].zero_()
    torch.ops.aten._int_mm.out(x8[:m_eff], w8, out=acc[:m_eff])
    y = (acc[:m_eff].float() * out_scale).to(torch.bfloat16)
    if bias is not None and bias.numel() > 0:
        y = y + bias
    out[:m_eff].copy_(y)


if hasattr(torch.library, "custom_op"):

    @torch.library.custom_op(
        "vllm_fl::static_w8a8_linear_v7",
        mutates_args={"x_pad", "x8", "acc", "out", "x_rot"},
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
        smooth: torch.Tensor,
        w_scale: torch.Tensor,
        path_mode: torch.Tensor,
        quarot_signs: torch.Tensor,
        x_rot: torch.Tensor,
    ) -> None:
        """Decode path selected by ``path_mode`` (Dynamo always hits this op).

        0=static INT8, 1=bf16 when m<=cap, 2=dynamic-act when m<=cap.
        Prefill (m>cap) always static INT8.  Global decode-INT8 off → bf16.
        QuaRot-lite: CG-safe FWHT into ``x_rot`` then quant.
        """
        m = x.shape[0]
        if out.shape[0] < m:
            raise RuntimeError(
                f"static_w8a8 workspace too small: m={m} out={tuple(out.shape)}"
            )
        if bias is None or bias.numel() == 0:
            b = None
        else:
            b = bias
        mode = int(path_mode.item()) if path_mode.numel() else _PATH_STATIC
        use_bf16 = _use_bf16_path(m) or (
            mode == _PATH_BF16_DECODE and m <= _DECODE_WS_CAP
        )
        if use_bf16:
            if w_bf16.numel() <= 1:
                raise RuntimeError(
                    "static_w8a8 bf16 path hit after drop_bf16; "
                    "enable DECODE_INT8 or disable DROP_BF16"
                )
            y = torch.nn.functional.linear(x, w_bf16, b)
            out[:m].copy_(y)
            return
        use_dyn = mode == _PATH_DYN_ACT and m <= _DECODE_WS_CAP
        m_eff = 32 if m < 32 else m
        if (
            x_pad.shape[0] < m
            or x8.shape[0] < m_eff
            or acc.shape[0] < m_eff
            or out.shape[0] < m_eff
        ):
            raise RuntimeError(
                f"static_w8a8 workspace too small: m={m} m_eff={m_eff} "
                f"x_pad={tuple(x_pad.shape)} x8={tuple(x8.shape)} "
                f"acc={tuple(acc.shape)} out={tuple(out.shape)}"
            )
        if quarot_signs.numel() > 0 and x_rot.numel() >= m * x.shape[-1]:
            _hadamard_act_into(x, x_rot, quarot_signs)
            # float32 → bf16 via copy_ (no temporary .to() alloc — CG-safe).
            x_pad[:m].copy_(x_rot[:m])
            x_q = x_pad[:m]
        else:
            x_q = x
            x_pad[:m].copy_(x)
        if use_dyn:
            _torch_w8a8_decode(
                x_q,
                w8,
                q_scale,
                out_scale,
                b,
                x8,
                acc,
                out,
                smooth=smooth,
                w_scale=w_scale,
                dynamic=True,
            )
            return
        mod = _cuda_mod()
        if mod is None:
            _torch_w8a8_decode(
                x_q, w8, q_scale, out_scale, b, x8, acc, out, dynamic=False
            )
            return
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
            x8[:m_eff],
            acc[:m_eff],
            out[:m_eff],
            bb,
        )

    @_static_w8a8_linear_op.register_fake
    def _static_w8a8_linear_op_fake(
        x,
        w8,
        w_bf16,
        q_scale,
        out_scale,
        bias,
        x_pad,
        x8,
        acc,
        out,
        smooth,
        w_scale,
        path_mode,
        quarot_signs,
        x_rot,
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
    w_bf16 = layer.weight
    # Default (decode-INT8 off): plain bf16 for decode sizes — no custom_op.
    # When decode-INT8 is on, ALWAYS go through custom_op v7 with path_mode so
    # Dynamo/CG cannot skip selective-bf16 / dynamic-act / quarot.
    if m <= _DECODE_WS_CAP and not _decode_int8_enabled():
        bb = None if b is None else b
        return torch.nn.functional.linear(x2, w_bf16, bb).reshape(*orig[:-1], n)
    if b is None:
        b = torch.empty(0, dtype=torch.bfloat16, device=x.device)
    else:
        b = b.contiguous()
    w8 = getattr(layer, "_iluvatar_w8", meta["w8"])
    q_scale = getattr(layer, "_iluvatar_q_scale", meta["q_scale"])
    out_scale = getattr(layer, "_iluvatar_out_scale", meta["out_scale"])
    smooth = getattr(layer, "_iluvatar_smooth", None)
    w_scale = getattr(layer, "_iluvatar_w_scale", None)
    path_mode = getattr(layer, "_iluvatar_path_mode", None)
    quarot_signs = getattr(layer, "_iluvatar_quarot_signs", None)
    if smooth is None:
        smooth = torch.empty(0, dtype=torch.float32, device=x.device)
    if w_scale is None:
        w_scale = torch.empty(0, dtype=torch.float32, device=x.device)
    if path_mode is None:
        path_mode = torch.zeros((), dtype=torch.int32, device=x.device)
    if quarot_signs is None:
        quarot_signs = torch.empty(0, dtype=torch.float32, device=x.device)
    if (
        m <= _DECODE_WS_CAP
        and getattr(layer, "_iluvatar_out", None) is not None
    ):
        x_pad = layer._iluvatar_x_pad
        x8 = layer._iluvatar_x8
        acc = layer._iluvatar_acc
        out = layer._iluvatar_out
        x_rot = getattr(layer, "_iluvatar_x_rot", None)
        if x_rot is None:
            x_rot = torch.empty(0, dtype=torch.float32, device=x.device)
    else:
        ws = _ensure_ws(layer, max(m, 32), k, n, x.device)
        x_pad, x8, acc, out = ws["x_pad"], ws["x8"], ws["acc"], ws["out"]
        x_rot = ws.get("x_rot")
        if x_rot is None:
            x_rot = torch.empty(0, dtype=torch.float32, device=x.device)
    if _static_w8a8_linear_op is not None:
        _static_w8a8_linear_op(
            x2,
            w8,
            w_bf16,
            q_scale,
            out_scale,
            b,
            x_pad,
            x8,
            acc,
            out,
            smooth,
            w_scale,
            path_mode,
            quarot_signs,
            x_rot,
        )
    else:
        mode = int(path_mode.item()) if path_mode.numel() else _PATH_STATIC
        if _use_bf16_path(m) or (
            mode == _PATH_BF16_DECODE and m <= _DECODE_WS_CAP
        ):
            y = torch.nn.functional.linear(
                x2, w_bf16, None if b.numel() == 0 else b
            )
            return y.reshape(*orig[:-1], n)
        if quarot_signs.numel() > 0 and x_rot.numel() >= m * k:
            _hadamard_act_into(x2, x_rot, quarot_signs)
            x_q = x_rot[:m].to(dtype=x2.dtype)
        else:
            x_q = _hadamard_act(x2) if _quarot_enabled() else x2
        m_eff = 32 if m < 32 else m
        mod = _cuda_mod()
        if mod is None or (
            mode == _PATH_DYN_ACT and m <= _DECODE_WS_CAP
        ):
            _torch_w8a8_decode(
                x_q,
                w8,
                q_scale,
                out_scale,
                None if b.numel() == 0 else b,
                x8,
                acc,
                out,
                smooth=smooth,
                w_scale=w_scale,
                dynamic=(mode == _PATH_DYN_ACT and m <= _DECODE_WS_CAP),
            )
        else:
            x_pad[:m].copy_(x_q)
            mod.static_w8a8_linear(
                x_pad[:m],
                w8,
                q_scale,
                out_scale,
                x8[:m_eff],
                acc[:m_eff],
                out[:m_eff],
                b,
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
        if os.getenv("VLLM_FL_ILUVATAR_STATIC_INT8", "0") != "1":
            logger.info(
                "patch_static_int8_linear_for_iluvatar: off by default "
                "(set VLLM_FL_ILUVATAR_STATIC_INT8=1 to enable)."
            )
            return
        calib = _calib_path()
        if not calib.is_file():
            logger.warning(
                "patch_static_int8_linear_for_iluvatar: missing calib file %s",
                calib,
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
            # model_memory_usage is set inside orig_load (bf16 only).  BI-V150
            # often keeps freed blocks inside the caching allocator, so
            # mem_get_info is a bad signal — account bytes explicitly.
            int8_added = _quantize_loaded_model(self)
            # Only rewrite weights_memory when we actually drop bf16.  Default
            # v5 keeps bf16+int8 dual occupancy but historically left
            # model_memory_usage at the bf16-only load figure; changing that
            # would shrink KV without a decode-INT8/drop opt-in.
            if _drop_bf16_enabled():
                freed = _drop_bf16_weights(self.model)
                prev = int(getattr(self, "model_memory_usage", 0) or 0)
                self.model_memory_usage = max(0, prev + int(int8_added) - int(freed))
                logger.info(
                    "patch_static_int8_linear_for_iluvatar: drop_bf16 freed=%d MiB; "
                    "model_memory_usage %d -> %d MiB (int8_added=%d MiB)",
                    freed // (1024 * 1024),
                    prev // (1024 * 1024),
                    int(self.model_memory_usage) // (1024 * 1024),
                    int(int8_added) // (1024 * 1024),
                )

        ModelRunnerFL.load_model = load_model
        ModelRunnerFL._iluvatar_static_int8_load_patch = True
        logger.info(
            "patch_static_int8_linear_for_iluvatar: enabled (calib=%s, "
            "custom_op=%s, preloaded=1, per_layer_ws=1, decode_int8=%s, "
            "drop_bf16=%s, decode_bf16_roles=%s, decode_dynamic_act=%s, "
            "quarot=%s, simple=%s).",
            calib,
            _static_w8a8_linear_op is not None,
            _decode_int8_enabled(),
            _drop_bf16_enabled(),
            sorted(_decode_bf16_roles()) or "[]",
            _decode_dynamic_act_enabled(),
            _quarot_enabled(),
            _simple_int8_enabled(),
        )
    except Exception as e:
        logger.warning("patch_static_int8_linear_for_iluvatar: %s", e)


def _quantize_loaded_model(worker) -> int:
    """Quantize linears in-place. Returns CUDA bytes newly allocated for INT8."""
    payload = torch.load(_calib_path(), map_location="cpu", weights_only=False)
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
    added = 0
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
        # QuaRot-lite (disabled when SIMPLE=1): fuse rotation into bf16 weights.
        if _quarot_enabled():
            w_rot = _fuse_quarot_weight(w)
            with torch.no_grad():
                mod.weight.data.copy_(w_rot)
            w = mod.weight
        meta = _quant_weight(w, xm, alpha, role=role)
        if meta is None:
            continue
        meta["role"] = role
        for k, v in meta.items():
            if torch.is_tensor(v):
                meta[k] = v.to(device=w.device)
        added += _apply_to_linear(mod, meta)
        n += 1
    mode_counts = {_PATH_STATIC: 0, _PATH_BF16_DECODE: 0, _PATH_DYN_ACT: 0}
    for _name, mod in model.named_modules():
        pm = getattr(mod, "_iluvatar_path_mode", None)
        if pm is None:
            continue
        mode_counts[int(pm.item())] = mode_counts.get(int(pm.item()), 0) + 1
    logger.info(
        "patch_static_int8_linear_for_iluvatar: quantized %d linears "
        "(path_mode static=%d bf16_decode=%d dyn_act=%d)",
        n,
        mode_counts.get(_PATH_STATIC, 0),
        mode_counts.get(_PATH_BF16_DECODE, 0),
        mode_counts.get(_PATH_DYN_ACT, 0),
    )
    return added


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
