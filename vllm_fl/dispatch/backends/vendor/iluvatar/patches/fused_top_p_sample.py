# Copyright (c) 2026 BAAI. All rights reserved.

"""Iluvatar sampler patch: fused top-p filter + exponential-noise sample.

``TopKTopPSampler.forward_native`` does three full-vocabulary passes after the
top-p cut: ``logits.softmax()``, ``probs / q``, ``argmax``.  Argmax is scale
invariant, so the softmax denominator and the materialised probability tensor
can be dropped: the sampled token is

    argmax_{x >= threshold} exp(x - rowmax) / q

which ``flag_gems.top_p_sample`` evaluates in one blocked pass.  At the scored
shapes that removes ~300 us/step (rows=32) / ~600 us/step (rows=64) of the
out-of-graph decode tail.

The same patch also elides ``Sampler.forward``'s unconditional
``logits.to(torch.float32)`` when the model already produced bf16 logits and
nothing downstream needs fp32 (no penalties/processors, no logprobs, all
temperature 1).  bf16 is exactly representable in fp32, so widening inside the
kernel is exact and the selected token is unchanged; it removes a full-
vocabulary write plus half the read traffic of every later pass.

Only the top-p-only, no-logprobs, no-fp64-gumbel path is replaced.  Every
other call goes through the untouched originals.

Env: ``VLLM_FL_FUSED_TOPP_SAMPLE=0`` disables it (default on, Iluvatar only);
``VLLM_FL_FUSED_TOPP_MAX_ROWS`` bounds the fast path (default 128, matching
``top_p_threshold``'s v2 crossover).
"""

from __future__ import annotations

import logging
import os

import torch
import vllm.v1.sample.ops.topk_topp_sampler as tts
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.sampler import Sampler

from ..iluvatar import _is_iluvatar_platform

logger = logging.getLogger(__name__)

try:
    from flag_gems import top_p_sample as _top_p_sample
except ImportError:  # flag_gems build without the op
    _top_p_sample = None

_ENABLED = _is_iluvatar_platform() and os.environ.get(
    "VLLM_FL_FUSED_TOPP_SAMPLE", "1"
).strip().lower() not in ("0", "false", "off", "no")
_MAX_ROWS = int(os.environ.get("VLLM_FL_FUSED_TOPP_MAX_ROWS", "128"))
# Separate switch so the bf16 cast elision can be A/B'd against the same
# fused top-p sampler (VLLM_FL_SAMPLER_BF16=0 keeps vLLM's fp32 cast).
_BF16_ENABLED = os.environ.get("VLLM_FL_SAMPLER_BF16", "1").strip().lower() not in (
    "0", "false", "off", "no")

_ORIG_TOPP = tts.TopKTopPSampler.forward_native
_ORIG_FORWARD = Sampler.forward
_bf16_logged = False


def _fast_ok(self, logits: torch.Tensor, k, p) -> bool:
    return (
        _ENABLED
        and _top_p_sample is not None
        and k is None
        and p is not None
        and logits.dim() == 2
        and logits.dtype in (torch.float32, torch.bfloat16)
        and logits.device.type == "cuda"
        and logits.is_contiguous()
        and 0 < logits.shape[0] <= _MAX_ROWS
        and not self.use_fp64_gumbel
        and self.logprobs_mode not in ("processed_logits", "processed_logprobs")
    )


def _forward_native(self, logits, generators, k, p):
    if not _fast_ok(self, logits, k, p):
        return _ORIG_TOPP(self, logits, generators, k, p)

    # q stays fp32 even when logits are bf16.  vLLM's random_sample also uses
    # fp32 noise (it samples against fp32 probabilities), and an 8-bit mantissa
    # would change near-tie selection.
    q = torch.empty(logits.shape, dtype=torch.float32, device=logits.device)
    if len(generators) != logits.shape[0]:
        q.exponential_()
    if generators:
        for i, generator in generators.items():
            q[i].exponential_(generator=generator)

    return _top_p_sample(logits, p, q), None


def _processors_noop(md) -> bool:
    """True when the batch's logits processors cannot change the logits.

    vLLM installs MinP / MinTokens / LogitBias processors even for requests
    that leave every knob at its default.  Their ``apply`` is a plain early
    return in that state (checked in ``vllm.v1.sample.logits_processor.
    builtin``), so the bf16 path may run them as-is.  Any other processor,
    or any of these in a non-default state, falls back to vLLM's fp32 path.
    """
    for p in md.logitsprocs.all:
        name = type(p).__name__
        if name == "MinPLogitsProcessor":
            if getattr(p, "min_p_count", 0):
                return False
        elif name == "MinTokensLogitsProcessor":
            if getattr(p, "min_toks", None):
                return False
        elif name == "LogitBiasLogitsProcessor":
            if getattr(p, "biases", None):
                return False
        else:
            return False
    return True


def _bf16_ok(self, logits, md, predict_bonus_token, logprobs_mode_override) -> bool:
    if not _BF16_ENABLED or not _ENABLED or _top_p_sample is None:
        return False
    if logits.dtype != torch.bfloat16 or logits.dim() != 2:
        return False
    if logits.device.type != "cuda" or not logits.is_contiguous():
        return False
    if not (0 < logits.shape[0] <= _MAX_ROWS):
        return False
    if predict_bonus_token:
        return False
    mode = logprobs_mode_override or self.logprobs_mode
    if mode in ("processed_logits", "processed_logprobs"):
        return False
    if md.max_num_logprobs is not None or md.logprob_token_ids:
        return False
    # The bf16 path relies on B (temperature divide skipped) and on the top-p
    # fused sampler; anything else keeps vLLM's fp32 pipeline.
    if not md.all_random or not getattr(md, "_fl_skip_temperature", False):
        return False
    if md.top_p is None or md.top_k is not None:
        return False
    if not md.no_penalties or md.allowed_token_ids_mask is not None:
        return False
    if md.bad_words_token_ids:
        return False
    if md.thinking_budget_state_holder is not None:
        return False
    if not _processors_noop(md):
        return False
    return True


def _forward(self, logits, sampling_metadata, predict_bonus_token=False,
             logprobs_mode_override=None):
    global _bf16_logged
    if not _bf16_ok(self, logits, sampling_metadata, predict_bonus_token,
                    logprobs_mode_override):
        return _ORIG_FORWARD(self, logits, sampling_metadata,
                             predict_bonus_token, logprobs_mode_override)
    if not _bf16_logged:
        _bf16_logged = True
        logger.warning("[fl] sampler: bf16 logits fast path active")

    # Sampler.forward with the bf16 -> fp32 cast elided.  The bf16 values are
    # exactly representable in fp32, no processor applies and no logprobs are
    # requested, so the selected token is unchanged.
    logits = self.apply_logits_processors(
        logits, sampling_metadata, predict_bonus_token)
    sampled, _ = self.sample(logits, sampling_metadata)
    sampled = sampled.to(torch.int32)
    return SamplerOutput(
        sampled_token_ids=sampled.unsqueeze(-1), logprobs_tensors=None)


if _ENABLED:
    tts.TopKTopPSampler.forward_native = _forward_native
    if _BF16_ENABLED:
        Sampler.forward = _forward
    logger.warning(
        "[fl] fused top-p + exponential sample: enabled "
        "(VLLM_FL_FUSED_TOPP_SAMPLE=0 to disable); bf16 cast elision=%s",
        _BF16_ENABLED,
    )
