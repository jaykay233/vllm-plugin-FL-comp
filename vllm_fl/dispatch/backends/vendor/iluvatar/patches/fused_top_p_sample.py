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

Only the top-p-only, float32, no-logprobs, no-fp64-gumbel path is replaced.
Every other call goes through the untouched original.

Env: ``VLLM_FL_FUSED_TOPP_SAMPLE=0`` disables it (default on, Iluvatar only);
``VLLM_FL_FUSED_TOPP_MAX_ROWS`` bounds the fast path (default 128, matching
``top_p_threshold``'s v2 crossover).
"""

from __future__ import annotations

import logging
import os

import torch
import vllm.v1.sample.ops.topk_topp_sampler as tts

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

_ORIG = tts.TopKTopPSampler.forward_native


def _fast_ok(self, logits: torch.Tensor, k, p) -> bool:
    return (
        _ENABLED
        and _top_p_sample is not None
        and k is None
        and p is not None
        and logits.dim() == 2
        and logits.dtype == torch.float32
        and logits.device.type == "cuda"
        and logits.is_contiguous()
        and 0 < logits.shape[0] <= _MAX_ROWS
        and not self.use_fp64_gumbel
        and self.logprobs_mode not in ("processed_logits", "processed_logprobs")
    )


def _forward_native(self, logits, generators, k, p):
    if not _fast_ok(self, logits, k, p):
        return _ORIG(self, logits, generators, k, p)

    # Replicate random_sample's noise generation exactly.
    q = torch.empty_like(logits)
    if len(generators) != logits.shape[0]:
        q.exponential_()
    if generators:
        for i, generator in generators.items():
            q[i].exponential_(generator=generator)

    return _top_p_sample(logits, p, q), None


if _ENABLED:
    tts.TopKTopPSampler.forward_native = _forward_native
    logger.warning(
        "[fl] fused top-p + exponential sample: enabled "
        "(VLLM_FL_FUSED_TOPP_SAMPLE=0 to disable)"
    )
