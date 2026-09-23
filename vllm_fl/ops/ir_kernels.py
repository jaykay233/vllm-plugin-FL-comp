# Copyright (c) 2026 BAAI. All rights reserved.

"""Compile-safe (torch.compile / CUDAGraph) entry points for the FL dispatch.

Why this module exists
----------------------
When inductor is the compile backend, vLLM appends ``"none"`` to
``CompilationConfig.custom_ops`` (see ``vllm/config/vllm.py``), so every
``CustomOp`` -- including the FL OOT layers -- resolves to ``forward_native``
and the chain ``forward_oot -> CachedOp -> flag_gems.modules.*`` is never
entered.  Even when a single op is force-enabled with ``+rms_norm`` the chain
still cannot run: vLLM compiles the model forward with
``torch.compile(..., fullgraph=True)`` and dynamo cannot trace a FlagGems
kernel (it aborts on the first ``logger.debug`` inside ``gems_rms_forward``,
and triton launches are untraceable anyway).

vLLM 0.24 provides a supported seam for exactly this case: ``vllm.ir`` ops.

* ``vllm_ir.rms_norm`` / ``vllm_ir.fused_add_rms_norm`` are registered as
  ``torch.ops`` custom ops with a fake impl, so dynamo keeps them as a single
  opaque node and ``fullgraph=True`` succeeds.
* A platform publishes which provider wins through
  ``Platform.get_default_ir_op_priority`` (and users can override it with
  ``--kernel-config`` / ``ir_op_priority``).
* ``VllmIRLoweringPass`` lowers the node to the selected provider impl at
  compile time, and the runtime call lands in that impl.

This module registers an FL provider named ``flagos`` for both ops.  Each
provider is a thin wrapper around an opaque ``torch.library.custom_op`` whose
*implementation* goes through :func:`vllm_fl.dispatch.call_op`, so the
existing policy / whitelist / fallback / IO-dump machinery keeps working and
``OpManager._called_ops`` still records what actually ran.

Enable/disable with ``VLLM_FL_IR_KERNELS=0``.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import torch
from torch import Tensor

logger = logging.getLogger(__name__)

ENV_ENABLE = "VLLM_FL_IR_KERNELS"
PROVIDER = "flagos"

# vllm.ir op name (== IrOpPriorityConfig field) -> FL dispatch op name.
IR_OP_TO_FL_OP = {
    "rms_norm": "rms_norm",
    "fused_add_rms_norm": "rms_norm",
}
IR_OP_NAMES = tuple(IR_OP_TO_FL_OP)

_registered = False
_HAS_CUSTOM_OP = hasattr(torch.library, "custom_op")


def ir_kernels_enabled() -> bool:
    """Global gate: FlagGems must be in use and the feature not disabled."""
    if os.environ.get(ENV_ENABLE, "1").strip().lower() not in ("1", "true"):
        return False
    try:
        from vllm_fl.utils import use_flaggems

        return use_flaggems()
    except Exception:  # pragma: no cover - defensive
        return False


class _LayerView:
    """Minimal stand-in for the vLLM layer object the FL impls expect.

    All three backends (flagos / vendor / reference) only read ``weight`` and
    ``variance_epsilon`` from the calling object.
    """

    __slots__ = ("weight", "variance_epsilon")

    def __init__(self, weight: Optional[Tensor], variance_epsilon: float) -> None:
        self.weight = weight
        self.variance_epsilon = variance_epsilon


def _dispatch_rms_norm(
    x: Tensor,
    residual: Optional[Tensor],
    weight: Optional[Tensor],
    epsilon: float,
):
    """Run ``rms_norm`` through the FL dispatch manager (policy + fallback)."""
    from vllm_fl.dispatch import call_op

    return call_op("rms_norm", _LayerView(weight, epsilon), x, residual)


if _HAS_CUSTOM_OP:

    @torch.library.custom_op("vllm_fl::rms_norm", mutates_args=())
    def rms_norm_op(
        x: Tensor,
        weight: Optional[Tensor],
        epsilon: float,
        variance_size: Optional[int] = None,
    ) -> Tensor:
        """Opaque to dynamo/inductor; dispatches to FlagGems at runtime."""
        assert variance_size is None, "FL rms_norm does not support variance_size"
        return _dispatch_rms_norm(x, None, weight, epsilon)

    @rms_norm_op.register_fake
    def _rms_norm_op_fake(x, weight, epsilon, variance_size=None):
        return torch.empty_like(x)

    @torch.library.custom_op("vllm_fl::fused_add_rms_norm", mutates_args=())
    def fused_add_rms_norm_op(
        x: Tensor,
        x_residual: Tensor,
        weight: Optional[Tensor],
        epsilon: float,
        variance_size: Optional[int] = None,
    ) -> tuple[Tensor, Tensor]:
        """Opaque to dynamo/inductor; FlagGems' fused add+norm is in-place, so
        work on clones and keep this custom op functional."""
        assert variance_size is None, "FL rms_norm does not support variance_size"
        out, residual = _dispatch_rms_norm(
            x.clone(), x_residual.clone(), weight, epsilon
        )
        return out, residual

    @fused_add_rms_norm_op.register_fake
    def _fused_add_rms_norm_op_fake(x, x_residual, weight, epsilon, variance_size=None):
        return torch.empty_like(x), torch.empty_like(x_residual)

    def rms_norm_flagos(
        x: Tensor,
        weight: Optional[Tensor],
        epsilon: float,
        variance_size: Optional[int] = None,
    ) -> Tensor:
        """``flagos`` provider for ``vllm.ir.ops.rms_norm``."""
        return rms_norm_op(x, weight, epsilon, variance_size)

    def fused_add_rms_norm_flagos(
        x: Tensor,
        x_residual: Tensor,
        weight: Optional[Tensor],
        epsilon: float,
        variance_size: Optional[int] = None,
    ) -> tuple[Tensor, Tensor]:
        """``flagos`` provider for ``vllm.ir.ops.fused_add_rms_norm``."""
        return fused_add_rms_norm_op(x, x_residual, weight, epsilon, variance_size)

    def _rms_supports(x, weight, epsilon, variance_size=None):
        return (
            variance_size is None
            and weight is not None
            and weight.dtype == x.dtype
        )

    def _fused_add_rms_supports(x, x_residual, weight, epsilon, variance_size=None):
        return (
            variance_size is None
            and weight is not None
            and weight.dtype == x.dtype
        )


def _providers_present(ir_mod) -> bool:
    """True when every IR op already carries a ``flagos`` impl."""
    try:
        return all(PROVIDER in getattr(ir_mod.ops, n).impls for n in IR_OP_NAMES)
    except Exception:  # pragma: no cover - defensive
        return False


def register_compile_safe_ops() -> bool:
    """Register the FL ``flagos`` provider for the vLLM IR ops.

    Idempotent and re-entrancy safe: importing ``vllm`` for the first time
    inside this function triggers vLLM platform-plugin discovery, which calls
    this very function again (``vllm_fl.register`` -> ``_register_ir_kernels``).
    The inner call does the work; the outer one then sees the flag/impls and
    must not register a second time (``IrOpImpl`` rejects duplicate providers).
    """
    global _registered
    if _registered:
        return True
    if not _HAS_CUSTOM_OP:
        logger.debug("torch.library.custom_op unavailable; FL IR kernels disabled")
        return False
    if not ir_kernels_enabled():
        logger.debug("FL IR kernels disabled (%s)", ENV_ENABLE)
        return False

    try:
        # NOTE: this import may re-enter (plugin activation -> our own call).
        from vllm import ir  # noqa: F401
        import vllm.ir.ops  # noqa: F401  (registers the layernorm IR ops)
    except Exception as e:  # pragma: no cover - older vLLM
        logger.debug("vllm.ir not available: %s", e)
        return False

    if _registered or _providers_present(ir):
        _registered = True
        return True

    try:
        ir.ops.rms_norm.register_impl(
            PROVIDER, supports_args=_rms_supports, supported=True
        )(rms_norm_flagos)
        # ``inplace=True`` matches the semantics of the FL/vendor fused kernel
        # (it writes x / x_residual).  Our custom op is functional -- it clones
        # the activations and returns them -- so vLLM's ``func_impl_fn`` adds one
        # extra clone on the eager path to preserve functional semantics, and
        # the inductor functionalization pass simply does not exploit the
        # donation.  Both are correct; ``inplace=False`` would also be safe and
        # save that clone.
        ir.ops.fused_add_rms_norm.register_impl(
            PROVIDER, supports_args=_fused_add_rms_supports, supported=True, inplace=True
        )(fused_add_rms_norm_flagos)
    except Exception as e:
        logger.warning("Failed to register FL IR op providers: %s", e)
        return False

    _registered = True
    logger.info(
        "vLLM IR: registered '%s' providers for %s", PROVIDER, ", ".join(IR_OP_NAMES)
    )
    return True


def default_ir_op_priority() -> dict[str, list[str]]:
    """Per-op provider priority the platform should publish.

    ``flagos`` first, then the vendor ``vllm_c`` kernels (mcoplib on MetaX),
    then the native implementation.  Ops that FlagGems should not serve are
    omitted, and users can still override the whole list through
    ``--kernel-config '{"ir_op_priority": {...}}'``.
    """
    if not register_compile_safe_ops():
        return {}

    # ``vllm.kernels`` registers the vendor ('vllm_c'/...) providers; import it
    # so the published order is complete no matter who calls us first.
    try:
        import vllm.kernels  # noqa: F401
    except Exception as e:  # pragma: no cover - defensive
        logger.debug("vllm.kernels unavailable: %s", e)

    from vllm.ir.op import IrOp
    from vllm_fl.utils import use_flaggems_op

    priority: dict[str, list[str]] = {}
    for ir_op_name, fl_op_name in IR_OP_TO_FL_OP.items():
        if not use_flaggems_op(fl_op_name):
            continue
        op = IrOp.registry[ir_op_name]
        # keep only providers that actually exist on this platform, and make
        # sure the tail is ``native`` (it always accepts every signature).
        order = [p for p in (PROVIDER, "vllm_c", "native") if p in op.impls]
        if order and order[-1] == "native":
            priority[ir_op_name] = order
    return priority


__all__ = [
    "ENV_ENABLE",
    "PROVIDER",
    "IR_OP_NAMES",
    "IR_OP_TO_FL_OP",
    "ir_kernels_enabled",
    "register_compile_safe_ops",
    "default_ir_op_priority",
]
