# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""Fuse ``gate_up_proj + SiluAndMul`` into one MetaX kernel. Opt-in.

Why a model-level patch
-----------------------
vLLM runs the gated MLP as two ops::

    gate_up, _ = self.gate_up_proj(x)     # (M, K) @ (2N, K)^T -> (M, 2N)
    x          = self.act_fn(gate_up)      # SiluAndMul -> (M, N)

The activation dispatch (``dispatch/backends/*/impl/activation.py``) only ever
sees the finished ``(M, 2N)`` tensor, so it cannot fuse with the GEMM above it.
Fusing therefore has to happen one level up, where the gate/up weight is still in
scope.

The custom kernel
-----------------
``flag_gems.silu_and_mul_gate_up`` accumulates the gate and the up dot products
against the same A tile and applies the activation in registers, so the ``(M, 2N)``
intermediate is never written nor read back. On the C500 that round-trip is 21% of
the pair at decode (measured 2026-09-26, K=2048, N=6144; 17.2us of 80.6us).

It is **not** a strict win: keeping both B tiles resident doubles the per-stage
shared memory, and the C500 caps shared memory at 64KB, which excludes exactly the
large tiles that win at prefill. The op therefore ships with an M-based
dispatcher (``silu_and_mul_gate_up``) that falls back to the two-pass formulation
above a configurable crossover, so routing every MLP through it is safe.

What this patch swaps
---------------------
After the model is loaded, walk every module and, for those that look like a
standard gated MLP, swap exactly the two pieces the fusion covers:

* ``gate_up_proj.forward`` -> runs the fused kernel, returning
  ``(activated, None)`` under the same ``LinearBase`` contract;
* ``act_fn`` -> the identity, because the activation already happened inside the
  kernel.

The MLP's own ``forward`` (and therefore ``down_proj``, its all-reduce, tuple
unpacking, and anything else the model does) is left completely untouched. That
matters: an earlier revision replaced the whole ``forward`` and silently dropped
``down_proj``, which the real-``Qwen2MLP`` test caught.

The swap is defensive: it falls back to ``gate_up_proj``'s original ``forward``
whenever the shape, dtype, layout or bias is anything other than a plain 2-D
bf16/fp16 bias-free GEMM, and it catches exceptions from the fused path. It never
changes numerics beyond fp accumulation order, and it is a no-op for modules that
do not match.

Tensor parallelism is safe: ``MergedColumnParallelLinear`` stores the merged
weight as ``[gate_shard; up_shard]`` along dim 0, and after sharding each rank
holds exactly ``(2 * N_local, K)`` with the gate rows first - which is the layout
the kernel expects.

Enable with ``VLLM_METAX_MLP_SILU_FUSION=1`` (default: off). No change to vLLM
itself is required: everything here is plugin-side, applied at runtime from
``vendor/metax/patches``.
"""

from __future__ import annotations

import logging
import os

import torch

logger = logging.getLogger(__name__)

_ENABLED = os.environ.get("VLLM_METAX_MLP_SILU_FUSION", "0") == "1"
_STATS = {"fused_calls": 0, "fallback_calls": 0}
_MARK = "_metax_silu_mul_fused"


def _is_plain_gate_up_mlp(module) -> bool:
    """Structural check for the standard vLLM gated MLP."""
    gate_up = getattr(module, "gate_up_proj", None)
    if gate_up is None:
        return False
    weight = getattr(gate_up, "weight", None)
    if weight is None or weight.dim() != 2:
        return False
    # A bias would need an extra elementwise pass the fused kernel does not do.
    if getattr(gate_up, "bias", None) is not None:
        return False
    act_fn = getattr(module, "act_fn", None)
    if act_fn is None:
        return False
    try:
        from vllm.model_executor.layers.activation import SiluAndMul
    except Exception:  # noqa: BLE001
        return False
    return isinstance(act_fn, SiluAndMul)


class _IdentityActivation(torch.nn.Module):
    """Stand-in for the activation: the fused kernel already applied it."""

    def forward(self, x):
        return x


def _make_gate_up_forward(gate_up, original_forward):
    """Replace ``gate_up_proj.forward`` so it returns the *activated* result.

    vLLM's ``LinearBase.forward`` returns ``(output, output_bias)``; we keep that
    contract with a ``None`` bias so the caller's unpacking is unchanged.
    """

    def fused_gate_up_forward(x, *args, **kwargs):
        import flag_gems

        weight = gate_up.weight
        if (
            not isinstance(x, torch.Tensor)
            or x.dim() != 2
            or x.shape[-1] != weight.shape[1]
            or x.dtype != weight.dtype
            or x.device != weight.device
        ):
            _STATS["fallback_calls"] += 1
            return original_forward(x, *args, **kwargs)
        try:
            out = flag_gems.silu_and_mul_gate_up(x, weight)
        except Exception as exc:  # noqa: BLE001 - never break serving for an optimisation
            logger.debug(
                "MetaX MLP fusion fell back for %s: %s", type(gate_up).__name__, exc
            )
            _STATS["fallback_calls"] += 1
            return original_forward(x, *args, **kwargs)
        _STATS["fused_calls"] += 1
        return out, None

    fused_gate_up_forward.__wrapped_original__ = original_forward
    return fused_gate_up_forward


def apply(model) -> int:
    """Fuse ``gate_up_proj + SiluAndMul`` in every qualifying MLP of ``model``.

    Rather than reimplementing the MLP's ``forward`` (which would have to
    reproduce ``down_proj``, its all-reduce, tuple unpacking, ...), we only
    swap the two pieces that the fusion actually covers:

    * ``gate_up_proj.forward`` -> runs the fused kernel and returns
      ``(activated, None)`` instead of ``(pre-activation, None)``;
    * ``act_fn`` -> the identity, since the activation already happened.

    The surrounding model code - including ``down_proj`` and anything else the
    model does after the activation - therefore runs exactly as before.

    Returns the number of MLPs fused.
    """
    if model is None:
        return 0
    fused = 0
    for name, module in model.named_modules():
        if getattr(module, _MARK, False):
            continue
        if not _is_plain_gate_up_mlp(module):
            continue
        gate_up = module.gate_up_proj
        gate_up.forward = _make_gate_up_forward(gate_up, gate_up.forward)
        module.act_fn = _IdentityActivation()
        setattr(module, _MARK, True)
        fused += 1
        logger.debug("MetaX MLP fusion: fused %s (%s)", name, type(module).__name__)
    if fused:
        logger.info_once(
            "MetaX MLP fusion enabled: fused %d gated MLP module(s) "
            "(gate_up_proj + SiluAndMul).",
            fused,
        )
    return fused


def _patch_model_runner() -> None:
    """Call :func:`apply` once the weights are on the device.

    ``ModelRunner.load_model`` is the first point where ``self.model`` exists with
    real weights, so it is the only place the ``gate_up_proj.weight`` layout is
    final (quantisation and TP sharding have both been applied by then).
    """
    try:
        from vllm_fl.worker import model_runner as mr_mod
    except Exception as exc:  # noqa: BLE001
        logger.debug("MetaX MLP fusion: cannot import model_runner (%s)", exc)
        return

    runner_cls = getattr(mr_mod, "ModelRunner", None) or getattr(mr_mod, "GPUModelRunner", None)
    if runner_cls is None:
        logger.debug("MetaX MLP fusion: no ModelRunner class found")
        return
    original = runner_cls.load_model
    if getattr(original, "_metax_mlp_fusion_patched", False):
        return

    def load_model(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        try:
            apply(getattr(self, "model", None))
        except Exception as exc:  # noqa: BLE001
            logger.warning("MetaX MLP fusion: wrapping failed (%s)", exc)
        return result

    load_model._metax_mlp_fusion_patched = True
    runner_cls.load_model = load_model


if _ENABLED:
    _patch_model_runner()
