# Copyright (c) 2026 BAAI. All rights reserved.

"""MetaX sampler patch: fused top-p filter + exponential-noise sample.

``TopKTopPSampler.forward_native`` currently materialises the top-p mask, then
runs ``softmax``, ``probs / q`` and ``argmax`` over the full vocabulary.  Argmax
is scale-invariant, so the softmax denominator and the probability tensor are
not needed:

    argmax_{x >= threshold} exp(x - rowmax) / q

``flag_gems.top_p_sample`` evaluates that in one blocked pass.  On C500 at the
served shapes it removes ~0.32 ms/step (rows=64) / ~0.26 ms/step (rows=32) from
the out-of-graph decode tail, with token-for-token identical output in the
microbenchmark.

Only the top-p-only, no-logprobs, no-fp64-gumbel path is replaced.  Every other
call goes through the untouched original, so the existing MetaX
``top_p_threshold`` fallback still applies.

Env: ``VLLM_FL_FUSED_TOPP_SAMPLE=0`` disables it (default on);
``VLLM_FL_FUSED_TOPP_MAX_ROWS`` bounds the fast path (default 128, matching
``top_p_threshold``'s v2 crossover).
"""

from __future__ import annotations

import logging
import os

import torch
import vllm.v1.sample.ops.topk_topp_sampler as tts

logger = logging.getLogger(__name__)

try:
    from flag_gems import top_p_sample as _top_p_sample
except ImportError:  # flag_gems build without the op
    _top_p_sample = None

_ENABLED = os.environ.get(
    "VLLM_FL_FUSED_TOPP_SAMPLE", "1"
).strip().lower() not in ("0", "false", "off", "no")
_MAX_ROWS = int(os.environ.get("VLLM_FL_FUSED_TOPP_MAX_ROWS", "128"))

_ORIG_TOPP = tts.TopKTopPSampler.forward_native


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
        return _ORIG_TOPP(self, logits, generators, k, p)

    # q must be generated exactly as vLLM's random_sample does: fp32
    # empty_like + exponential_(), then overwrite per-request generators.
    q = torch.empty(logits.shape, dtype=torch.float32, device=logits.device)
    if len(generators) != logits.shape[0]:
        q.exponential_()
    if generators:
        for i, generator in generators.items():
            q[i].exponential_(generator=generator)

    return _top_p_sample(logits, p, q), None


if _ENABLED and _top_p_sample is not None:
    tts.TopKTopPSampler.forward_native = _forward_native
    logger.warning(
        "[fl] fused top-p + exponential sample: enabled "
        "(VLLM_FL_FUSED_TOPP_SAMPLE=0 to disable)"
    )
