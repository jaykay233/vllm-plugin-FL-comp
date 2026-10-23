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

Default **on** (since the 2026-10-23 fair A/B below). Disable with
``VLLM_METAX_LM_HEAD_GEMM=0``.
"""

from __future__ import annotations

import logging
import os

import torch

logger = logging.getLogger(__name__)

# Default ON. Fair on/off A/B (2026-10-23, P2+light stack, silu fusion off so
# the baseline is clean), official 4k shape [4096,1024], conc=64, 2 runs each:
#   off (mcblas)       11678.40 / 11660.02 -> avg 11669.21 tok/s
#   on  (FlagGems mm)  11769.92 / 11754.16 -> avg 11762.04 tok/s   (+0.80%)
# with Mean TTFT 1941 -> 1932 ms and Mean TPOT 25.50 -> 25.30 ms, so TTFT and
# TPOT improve together (not decode-only noise). A second A/B on the silu=1
# baseline gave +0.91% (11118.61 -> 11211.69), consistent.
#
# The earlier "−1.8%" (11737 vs P2 11957) was never this change at all: the wrap
# silently failed to attach (ModelRunnerFL import raises "not support graph" on
# APIServer's UnspecifiedPlatform, and fork children inherited the no-op), so
# that number was cross-day noise against the no-light baseline. The underlying
# reason the wrap is cheap: ``weight.t()`` yields strides (1, K), so FlagGems'
# `a.stride(0) > 1 and a.stride(1) > 1` contiguous check does not fire and no
# extra ~1.9 ms buffer copy happens.
_ENABLED = os.environ.get("VLLM_METAX_LM_HEAD_GEMM", "1") == "1"
_M_MIN = int(os.environ.get("VLLM_METAX_LM_HEAD_GEMM_M_MIN", "8"))
_M_MAX = int(os.environ.get("VLLM_METAX_LM_HEAD_GEMM_M_MAX", "64"))
_VOCAB_MIN = int(os.environ.get("VLLM_METAX_LM_HEAD_GEMM_VOCAB_MIN", "100000"))
_MARK = "_metax_lm_head_gemm_apply"
_STATS = os.environ.get("VLLM_METAX_LM_HEAD_GEMM_STATS", "0") == "1"
_hit = 0
_miss = 0


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
        global _hit, _miss
        weight = getattr(layer, "weight", None)
        if weight is not None and should_route_lm_head_gemm(x, weight, bias):
            try:
                import flag_gems

                # F.linear(x, W) == x @ W.t(); FlagGems mm_nt accepts weight.t()
                # as a (K, N) view with stride (1, K) — no contiguous copy.
                out = flag_gems.mm(x, weight.t())
                if _STATS:
                    _hit += 1
                    if _hit % 256 == 0:
                        logger.info(
                            "lm_head GEMM stats: hit=%d miss=%d last_M=%d",
                            _hit,
                            _miss,
                            int(x.shape[0]),
                        )
                return out
            except Exception as exc:  # noqa: BLE001
                logger.debug("lm_head FlagGems mm fell back: %s", exc)
        if _STATS:
            _miss += 1
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


def maybe_apply_after_load(model) -> int:
    """Called from ``WorkerFL.load_model`` (MetaX only). No import-time hook."""
    if not _ENABLED:
        return 0
    try:
        wrapped = apply(model)
        if not wrapped:
            logger.warning("MetaX lm_head GEMM enabled but no lm_head matched")
        return wrapped
    except Exception as exc:  # noqa: BLE001
        logger.warning("MetaX lm_head GEMM wrap failed: %s", exc)
        return 0
