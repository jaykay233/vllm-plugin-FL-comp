# SPDX-License-Identifier: Apache-2.0
"""Selective FlagGems ``mm`` for vocab ``lm_head`` on MetaX C500.

Background (mcProfiler + fixed-config sweep, 2026-10-23)
--------------------------------------------------------
* qkv / o_proj / gate_up / down: FlagGems is **1.2–2.0× slower** than native
  mcblas — never route those.
* ``lm_head`` (N≈130560, K=2048): FlagGems wins only for **M ∈ [8, 64]**
  (device 0.55–0.74×; best tile ``64×128×64/s3``). M=1 ≈ parity; M≥72 loses
  ~1.3×. Prefill large-M stays on native.

vLLM never calls ``ParallelLMHead.forward`` (it raises); logits go through
``lm_head.quant_method.apply(lm_head, hidden_states)``. We wrap that ``apply``.

Routing (``should_route_lm_head_gemm``)
---------------------------------------
Use FlagGems iff all of:

1. no bias
2. ``x`` is 2-D ``(M, K)``, weight ``(N, K)``, matching dtype/device
3. ``N >= VOCAB_MIN`` (vocab-sized; skips ordinary projections)
4. ``M_MIN <= M <= M_MAX`` (default 8..64 — the measured win window)
5. dtype bf16/fp16

Else fall back to the original apply (mcblas).

Default **off**. Enable with ``VLLM_METAX_LM_HEAD_GEMM=1``.
"""

from __future__ import annotations

import logging
import os

import torch

logger = logging.getLogger("vllm_fl.metax.lm_head_gemm")

# Default off: micro win on lm_head M∈[8,64], but official 4k e2e was
# ~1.8% slower than P2 baseline (11737 vs 11957, 2026-10-23). Opt-in only.
_ENABLED = os.environ.get("VLLM_METAX_LM_HEAD_GEMM", "0") == "1"
_M_MIN = int(os.environ.get("VLLM_METAX_LM_HEAD_GEMM_M_MIN", "8"))
_M_MAX = int(os.environ.get("VLLM_METAX_LM_HEAD_GEMM_M_MAX", "64"))
_VOCAB_MIN = int(os.environ.get("VLLM_METAX_LM_HEAD_GEMM_VOCAB_MIN", "100000"))
_MARK = "_metax_lm_head_gemm_apply"


def should_route_lm_head_gemm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    m_min: int = _M_MIN,
    m_max: int = _M_MAX,
    vocab_min: int = _VOCAB_MIN,
) -> bool:
    """Return True iff FlagGems mm is the measured-faster path for this call."""
    if bias is not None:
        return False
    if not isinstance(x, torch.Tensor) or not isinstance(weight, torch.Tensor):
        return False
    if x.ndim != 2 or weight.ndim != 2:
        return False
    m, k = int(x.shape[0]), int(x.shape[1])
    n, k_w = int(weight.shape[0]), int(weight.shape[1])
    if k != k_w or k <= 0:
        return False
    if n < vocab_min:
        return False
    if m < m_min or m > m_max:
        return False
    if x.dtype != weight.dtype or x.device != weight.device:
        return False
    if x.dtype not in (torch.bfloat16, torch.float16):
        return False
    return True


def _make_apply(original_apply):
    def apply(layer, x, bias=None):
        weight = getattr(layer, "weight", None)
        if weight is not None and should_route_lm_head_gemm(x, weight, bias):
            try:
                import flag_gems

                # F.linear(x, W) == x @ W.t(); FlagGems mm_nt wants that layout.
                return flag_gems.mm(x, weight.t())
            except Exception as exc:  # noqa: BLE001
                logger.debug("lm_head FlagGems mm fell back: %s", exc)
        return original_apply(layer, x, bias)

    apply.__wrapped_original__ = original_apply
    return apply


def _is_lm_head_module(name: str, module) -> bool:
    leaf = name.rsplit(".", 1)[-1]
    if leaf in ("embed_tokens", "wte", "tok_embeddings"):
        return False
    if leaf == "lm_head" or name.endswith(".lm_head"):
        return True
    # Fallback: ParallelLMHead by type name (covers odd prefixes).
    return type(module).__name__ == "ParallelLMHead"


def apply(model) -> int:
    """Wrap ``quant_method.apply`` on every ``lm_head`` under ``model``."""
    if model is None:
        return 0
    n = 0
    for name, module in model.named_modules():
        if not _is_lm_head_module(name, module):
            continue
        qm = getattr(module, "quant_method", None)
        if qm is None or not callable(getattr(qm, "apply", None)):
            continue
        if getattr(qm, _MARK, False):
            continue
        weight = getattr(module, "weight", None)
        if weight is None or weight.ndim != 2:
            continue
        qm.apply = _make_apply(qm.apply)
        setattr(qm, _MARK, True)
        n += 1
        logger.info(
            "MetaX lm_head GEMM: wrapped %s.apply weight=%s route M∈[%d,%d] N≥%d",
            name,
            tuple(weight.shape),
            _M_MIN,
            _M_MAX,
            _VOCAB_MIN,
        )
    return n


def _patch_model_runner() -> None:
    try:
        from vllm_fl.worker import model_runner as mr_mod
    except Exception as exc:  # noqa: BLE001
        logger.debug("lm_head GEMM: cannot import model_runner (%s)", exc)
        return

    runner_cls = (
        getattr(mr_mod, "ModelRunnerFL", None)
        or getattr(mr_mod, "ModelRunner", None)
        or getattr(mr_mod, "GPUModelRunner", None)
    )
    if runner_cls is None:
        return
    original = runner_cls.load_model
    if getattr(original, "_metax_lm_head_gemm_patched", False):
        return

    def load_model(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        try:
            wrapped = apply(getattr(self, "model", None))
            if not wrapped:
                logger.warning("MetaX lm_head GEMM enabled but no lm_head matched")
        except Exception as exc:  # noqa: BLE001
            logger.warning("MetaX lm_head GEMM wrap failed: %s", exc)
        return result

    load_model._metax_lm_head_gemm_patched = True
    runner_cls.load_model = load_model
    logger.info("MetaX lm_head GEMM: patched %s.load_model", runner_cls.__name__)


if _ENABLED:
    _patch_model_runner()
