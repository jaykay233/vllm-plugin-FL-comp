# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""MetaX sampler patches for `apply_top_k_top_p`.

Two separate problems are handled here.

1. vLLM's Triton topk_topp kernel does not compile on MetaX
   (`PassManager::run failed` during `make_ttgir`), so `apply_top_k_top_p` is
   routed to `apply_top_k_top_p_pytorch`.  This is required for correctness and
   is unchanged.

2. That fallback sorts the whole vocabulary every decode step, which makes it
   the single most expensive kernel in decoding.  Profiled on MiniCPM5-2B
   (vocab 130560, eager sampler outside the CUDA graph): `aten::sort` runs
   flag_gems' radix `compute_global_hist_kernel` plus 8x `sweep`, costing 21-28
   ms per step at serving batch sizes -- 54.7% of decode GPU time.

   Top-p does not need that sort.  The fallback only uses the sorted order to
   derive a threshold, so the threshold is computed directly by
   `flag_gems.top_p_threshold` (bucketing + two linear kernels) instead.

Measured on MiniCPM5-2B, random 32-in/256-out, 48 concurrent, ignore-eos:

    TPOT            31.5 ms  ->  11.5 ms      (2.7x)
    output tput     1475     ->  3093-3803 tok/s

Measured sampler cost per call (vocab 130560, p=0.95, fp32):

    rows    radix sort (ms)   top_p_threshold (ms)
       1            2.83                   0.39
      48           21.48                   0.83
     256          113.34                   3.05

Accuracy against `apply_top_k_top_p_pytorch` on real serving logits: mask IoU
0.996-0.998, retained probability mass equal to 6 decimals.  Where they differ
it is only which tokens sit on the cut, and always in one direction -- bucketing
places the threshold on the lower edge of the deciding bucket, so retained mass
is >= p and never < p.  The retained set can be up to ~0.2% larger.

The fast path is deliberately narrow and fails closed: it only runs for the
p-only case (`k is None`, which is what serving uses) on fp32 CUDA input, and any
exception or unsupported input falls back to the reference implementation, so
behaviour can only stay the same or improve.  Set `VLLM_FL_TOPP_FAST=0` to
disable, `VLLM_FL_TOPP_FAST_VERIFY=N` to re-check the first N calls against the
sort-based reference.
"""

import atexit
import logging
import os

import torch
import vllm.v1.sample.ops.topk_topp_sampler as topk_topp_sampler

logger = logging.getLogger(__name__)

try:
    from flag_gems import top_p_threshold as _top_p_threshold
except ImportError:  # older flag_gems without the op
    _top_p_threshold = None

_ENABLED = os.environ.get("VLLM_FL_TOPP_FAST", "1") != "0"
_VERIFY_LEFT = int(os.environ.get("VLLM_FL_TOPP_FAST_VERIFY", "0"))
_STATS_PATH = os.environ.get("VLLM_FL_TOPP_FAST_STATS")
_state = {"warned": False, "verify_left": _VERIFY_LEFT, "fast": 0, "fallback": 0, "other": 0}
_next_mark = 1


def _report_counts() -> None:
    """Prove which path actually served the run: a silent fallback would make an
    accuracy pass meaningless, so always report coverage."""
    s = _state
    msg = ("[fl] topp_fast coverage: fast=%d fallback=%d other=%d enabled=%s op=%s"
           % (s["fast"], s["fallback"], s["other"], _ENABLED, _top_p_threshold is not None))
    logger.warning(msg)
    # Only a process that actually sampled is evidence.  The bench client and
    # other helpers import this module, set the same env var and hit atexit
    # without ever calling the sampler; writing their zeros would put a
    # misleading `fast=0` line after the serving process's real counts, and a
    # `tail -1` check would then wrongly conclude the fast path never ran.
    if _STATS_PATH and (s["fast"] or s["fallback"] or s["other"]):
        try:
            with open(_STATS_PATH, "a") as fh:
                fh.write(msg + "\n")
        except OSError:
            pass


def _mark() -> None:
    """Dump coverage at growing call marks.  The sampler runs in the EngineCore
    process, which is SIGKILLed at teardown and whose logs may not be forwarded,
    so atexit alone is not trustworthy evidence -- write our own file."""
    global _next_mark
    if not _STATS_PATH:
        return
    n = _state["fast"]
    if n >= _next_mark:
        _report_counts()
        _next_mark = max(n + 1, _next_mark * 4)


atexit.register(_report_counts)


def _verify(logits: torch.Tensor, p, fast_out: torch.Tensor) -> None:
    """Compare the fast result with the reference on real logits (debug only)."""
    try:
        ref = topk_topp_sampler.apply_top_k_top_p_pytorch(logits.clone(), None, p)
        keep_ref, keep_fast = torch.isfinite(ref), torch.isfinite(fast_out)
        inter = int((keep_ref & keep_fast).sum().item())
        union = int((keep_ref | keep_fast).sum().item())
        amax = logits.amax(dim=-1, keepdim=True)
        e = torch.exp(torch.nan_to_num(logits - amax, nan=float("-inf")))
        z = e.sum(dim=-1)
        mass_ref = (e * keep_ref).sum(dim=-1) / z
        mass_fast = (e * keep_fast).sum(dim=-1) / z
        logger.warning(
            "[fl] topp_fast verify rows=%d iou=%.5f mass_ref=%.6f mass_fast=%.6f "
            "mass_err_max=%.2e kept_ref=%d kept_fast=%d (fast must keep >= ref)",
            logits.shape[0], inter / union, float(mass_ref.mean()),
            float(mass_fast.mean()), float((mass_ref - mass_fast).max()),
            int(keep_ref.sum()), int(keep_fast.sum()),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("[fl] topp_fast verify failed: %s: %s",
                       type(exc).__name__, str(exc)[:120])


def _apply_top_k_top_p_no_triton(
    logits: torch.Tensor, k: torch.Tensor | None, p: torch.Tensor | None
) -> torch.Tensor:
    if p is None and k is None:
        return logits

    if _ENABLED and _top_p_threshold is not None and p is not None and k is None:
        try:
            out = _top_p_threshold(logits, p)
        except Exception as exc:  # noqa: BLE001
            if not _state["warned"]:
                _state["warned"] = True
                logger.warning(
                    "[fl] topp_fast disabled after %s: %s; using sort fallback",
                    type(exc).__name__, str(exc)[:160])
            out = None
        if out is not None:
            _state["fast"] += 1
            _mark()
            if _state["verify_left"] > 0:
                _state["verify_left"] -= 1
                _verify(logits, p, out)
            return out
        _state["fallback"] += 1
    else:
        _state["other"] += 1

    return topk_topp_sampler.apply_top_k_top_p_pytorch(logits, k, p)


# Replace the dispatch function with one that skips Triton and, when it can,
# also skips the sort.
topk_topp_sampler.apply_top_k_top_p = _apply_top_k_top_p_no_triton
