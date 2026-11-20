# Copyright (c) 2026 BAAI. All rights reserved.

"""Bandwidth-bound RoPE for the inductor-compiled path (compile-safe).

STATUS (2026-11-20): **disabled by default -- it loses 1.6% of 4k throughput**.
Kernel level it is a clean win (135.1 vs 307.7 us/layer, 543 vs 245 GB/s, and
bitwise identical to the inductor lowering), but the production kernel is fused
with neighbouring pointwise work, so replacing it with an opaque op inserts a
graph boundary in both prefill and decode: 4k 6512 -> 6411 tok/s, TPOT
45.4 -> 46.3 ms (same-session control arm, fresh compile cache per arm).
Set ``VLLM_FL_ROPE_FAST=1`` to reproduce; the analysis is in
``experiments/RESULTS-...`` section 14.

With ``custom_ops=["none"]`` vLLM resolves ``RotaryEmbedding`` to
``forward_native`` and inductor lowers its pure-torch rotation into a pointwise
kernel that gathers cos/sin element-wise through the ``int64`` position tensor.
Measured on MiniCPM5-2B / BI-V150 for a 8192-token prefill chunk that kernel
moves 75.5 MB of q/k traffic in 307.7 us (**245 GB/s**), while every other
prefill elementwise kernel (silu / rms_norm / residual add) already runs at
550-605 GB/s.

This module registers ``vllm_fl_iluvatar::rope_neox_`` as a fake-able custom op
(dynamo keeps it as one opaque node, so ``fullgraph=True`` still succeeds).  The
kernel gathers cos/sin once per token and evaluates
``(x1*cos - x2*sin, x2*cos + x1*sin)`` in a single pass, writing **in place**:
the first version returned fresh tensors and the two ``empty_like`` calls per
layer cost more on the decode path (c=64 TPOT 16.8 -> 22.6 ms) than the kernel
saved.  The grid is ``(ceil(T/BT), num_heads)`` so decode (T=64) still gets 1152
programs instead of 64.

Same shapes as the inductor lowering reach 543 GB/s -> 139 us/layer (2.2x) for
prefill and 5.6 us/layer at decode size.

The hook is installed on ``RotaryEmbedding.forward_static``: ``forward_native``
is bound into ``CustomOp._forward_method`` at construction time, so patching
``forward_native`` after the modules exist would never be reached, while
``forward_static`` is looked up on every call.

Numerics match the inductor lowering exactly: inputs are widened to fp32,
combined, and rounded back to the input dtype once (bitwise identical to the
shipped kernel, verified with ``torch.equal``).

Disable with ``VLLM_FL_ROPE_FAST=0``.
"""

from __future__ import annotations

import logging
import os

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

ENV_ENABLE = "VLLM_FL_ROPE_FAST"
# Default OFF: see the 2026-11-20 finding.  The kernel is 2.2x faster in
# isolation and bitwise identical, but replacing the inductor pointwise kernel
# costs a graph boundary in both prefill and decode and measured -1.6% 4k
# throughput end to end.  Opt in with VLLM_FL_ROPE_FAST=1 to reproduce.
_ENABLED = os.getenv(ENV_ENABLE, "0").strip().lower() in ("1", "true", "yes", "on")

_DBG = os.getenv("VLLM_FL_ROPE_DEBUG", "0").strip().lower() in ("1", "true", "yes")
_seen: set[str] = set()

# Tuned on BI-V150 (16 SM): four tokens per program fills the 128-bit
# load/store path for prefill; decode prefers one token (more programs).
_BT_BIG = 4
_BT_SMALL = 1
_T_BIG = 1024
# Decode is left to the inductor lowering on purpose: putting the opaque op in
# the captured decode graph costs ~24 us/layer (4k TPOT 45.4 -> 46.4 ms) because
# it breaks inductor's pointwise fusion around the rotation, which is far more
# than the 5.6 us the kernel itself takes at T=64.  Prefill is torch-compiled
# but not graph-captured, and there the kernel is 2.2x faster than the lowering.
_MIN_TOKENS = 1024

_used = False


def _dbg(reason: str, **info) -> None:
    if not _DBG or reason in _seen or torch.compiler.is_compiling():
        return
    _seen.add(reason)
    logger.warning("[fl] rope fallback: %s | %s", reason, info)


@triton.jit
def _rope_neox_kernel(
    q_ptr, k_ptr, cos_ptr, sin_ptr, pos_ptr,
    T, stride_q, stride_k, stride_cos, stride_sin,
    NQH: tl.constexpr, NKH: tl.constexpr,
    HD: tl.constexpr, HALF: tl.constexpr, BT: tl.constexpr,
):
    pid_t = tl.program_id(0)
    h = tl.program_id(1)
    t = pid_t * BT + tl.arange(0, BT)
    d = tl.arange(0, HALF)
    mask = t < T
    pos = tl.load(pos_ptr + t, mask=mask, other=0).to(tl.int32)
    cos = tl.load(cos_ptr + pos[:, None] * stride_cos + d[None, :]).to(tl.float32)
    sin = tl.load(sin_ptr + pos[:, None] * stride_sin + d[None, :]).to(tl.float32)

    if h < NQH:
        base = q_ptr + t[:, None] * stride_q + h * HD
        mask2 = mask[:, None]
        x1 = tl.load(base + d[None, :], mask=mask2, other=0.0).to(tl.float32)
        x2 = tl.load(base + HALF + d[None, :], mask=mask2, other=0.0).to(tl.float32)
        tl.store(base + d[None, :], (x1 * cos - x2 * sin).to(q_ptr.dtype.element_ty),
                 mask=mask2)
        tl.store(base + HALF + d[None, :],
                 (x2 * cos + x1 * sin).to(q_ptr.dtype.element_ty), mask=mask2)
    else:
        hk = h - NQH
        base = k_ptr + t[:, None] * stride_k + hk * HD
        mask2 = mask[:, None]
        x1 = tl.load(base + d[None, :], mask=mask2, other=0.0).to(tl.float32)
        x2 = tl.load(base + HALF + d[None, :], mask=mask2, other=0.0).to(tl.float32)
        tl.store(base + d[None, :], (x1 * cos - x2 * sin).to(k_ptr.dtype.element_ty),
                 mask=mask2)
        tl.store(base + HALF + d[None, :],
                 (x2 * cos + x1 * sin).to(k_ptr.dtype.element_ty), mask=mask2)


def _rope_neox_impl(q: torch.Tensor, k: torch.Tensor,
                    cos: torch.Tensor, sin: torch.Tensor,
                    pos: torch.Tensor) -> None:
    """Rotate q/k in place.

    q, k: [T, heads, head_dim] with ``stride(-1) == 1`` and
    ``stride(1) == head_dim`` (a token stride other than ``heads * head_dim``
    is fine -- the qkv projection hands us a strided slice).
    cos, sin: [max_pos, head_dim // 2] views over the fused cos/sin cache.
    pos: [T] int32/int64 positions.
    """
    global _used
    if not _used:
        _used = True
        logger.warning("[fl] rope: fused neox kernel active")
    T = q.shape[0]
    bt = _BT_BIG if T >= _T_BIG else _BT_SMALL
    _rope_neox_kernel[(triton.cdiv(T, bt), q.shape[1] + k.shape[1])](
        q, k, cos, sin, pos,
        T, q.stride(0), k.stride(0), cos.stride(0), sin.stride(0),
        NQH=q.shape[1], NKH=k.shape[1],
        HD=q.shape[2], HALF=q.shape[2] // 2,
        BT=bt, num_warps=1,
    )


def _rope_neox_meta(q, k, cos, sin, pos) -> None:
    return None


# ``torch.library.custom_op`` routes mutations through ``auto_functionalized``;
# vLLM pins ``enable_auto_functionalized_v2=False``, so the v1 wrapper clones
# both inputs and copies them back -- four extra kernels per layer, which
# measured out at +1.0 ms per decode step (4k TPOT 45.4 -> 46.4 ms).  Declaring
# the schema directly keeps the op opaque to dynamo *and* tells inductor about
# the mutation through the ``Tensor(a!)`` aliases, which is how vLLM's own
# ``_C.rotary_embedding`` is registered.
_LIB = torch.library.Library("vllm_fl_iluvatar", "DEF")
_LIB.define(
    "rope_neox_(Tensor(a!) q, Tensor(b!) k, Tensor cos, Tensor sin, "
    "Tensor pos) -> ()"
)
_LIB.impl("rope_neox_", _rope_neox_impl, "CUDA")
_LIB.impl("rope_neox_", _rope_neox_meta, "Meta")


def rope_neox_(q, k, cos, sin, pos):
    torch.ops.vllm_fl_iluvatar.rope_neox_(q, k, cos, sin, pos)


def _try_static(positions, query, key, head_size, rotary_dim,
                cos_sin_cache, is_neox_style):
    """Rotate in place through the fused kernel; return (q, k) or None."""
    if not _ENABLED or key is None:
        return None
    if rotary_dim != head_size or not is_neox_style:
        _dbg("shape/style", rotary_dim=rotary_dim, head_size=head_size,
             neox=is_neox_style)
        return None
    if query.dtype not in (torch.bfloat16, torch.float16):
        _dbg("dtype", q_dtype=query.dtype)
        return None
    if key.dtype != query.dtype or not query.is_cuda or not key.is_cuda:
        _dbg("key/device", k_dtype=key.dtype, q_dev=query.device)
        return None
    if query.dim() not in (2, 3) or key.dim() not in (2, 3):
        _dbg("ndim", q_dim=query.dim(), k_dim=key.dim())
        return None

    try:
        pos = positions.flatten()
        ntok = pos.shape[0]
        q = query.view(ntok, -1, head_size)
        k = key.view(ntok, -1, head_size)
    except RuntimeError as exc:
        _dbg("view", err=repr(exc)[:200])
        return None
    if q.stride(1) != head_size or k.stride(1) != head_size:
        _dbg("head stride", q=q.stride(), k=k.stride(), hd=head_size)
        return None
    if q.stride(2) != 1 or k.stride(2) != 1:
        _dbg("inner stride", q=q.stride(), k=k.stride())
        return None
    if ntok < _MIN_TOKENS:
        _dbg("decode size", ntok=ntok)
        return None
    if not pos.is_contiguous():
        pos = pos.contiguous()
    cache = cos_sin_cache
    if cache.dim() != 2 or cache.shape[-1] != rotary_dim:
        _dbg("cache shape", cache=tuple(cache.shape), rotary_dim=rotary_dim)
        return None
    if cache.dtype != query.dtype:
        _dbg("cache dtype", cache=cache.dtype, q=query.dtype)
        return None
    if not cache.is_contiguous():
        cache = cache.contiguous()
    cos, sin = cache.chunk(2, dim=-1)
    try:
        rope_neox_(q, k, cos, sin, pos)
    except Exception as exc:  # noqa: BLE001 - never break the model
        _dbg("kernel", err=repr(exc)[:300])
        return None
    return query, key


def _install() -> None:
    if not _ENABLED:
        logger.warning(
            "[fl] rope fast path: disabled by default (set %s=1 to enable; "
            "measured -1.6%% 4k throughput when enabled)", ENV_ENABLE)
        return
    from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding

    if getattr(RotaryEmbedding, "_fl_rope_fast", False):
        return
    orig = RotaryEmbedding.forward_static

    def forward_static(positions, query, key, head_size, rotary_dim,
                       cos_sin_cache, is_neox_style):
        out = _try_static(positions, query, key, head_size, rotary_dim,
                          cos_sin_cache, is_neox_style)
        if out is not None:
            return out
        return orig(positions, query, key, head_size, rotary_dim,
                    cos_sin_cache, is_neox_style)

    RotaryEmbedding.forward_static = staticmethod(forward_static)
    RotaryEmbedding._fl_rope_fast = True
    logger.warning("[fl] rope fast path: enabled (%s=0 to disable)", ENV_ENABLE)


_install()

__all__ = ["rope_neox_"]
