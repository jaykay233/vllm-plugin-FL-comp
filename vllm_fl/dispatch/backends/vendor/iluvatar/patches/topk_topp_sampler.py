# Copyright (c) 2026 BAAI. All rights reserved.

"""Iluvatar sampler patch: sort-free top-p.

On Iluvatar the sampler runs `TopKTopPSampler.forward_native`, which resolves
`apply_top_k_top_p` from this module's globals at call time.  For the
serving-relevant case (`k is None`, `p is not None`) that function lands on
`apply_top_k_top_p_triton`, whose top-p stage is a full-vocabulary radix sort
(`compute_global_hist_kernel` + 8x `sweep`).  Top-p does not need the sorted
order -- only a threshold -- so the threshold can be derived directly from a
logit histogram by `flag_gems.top_p_threshold` (bucketing + linear kernels).

Measured on MiniCPM5-2B (vocab 130560, p=0.95, fp32, BI-V150), per call:

    rows   vLLM path (ms)   top_p_threshold (ms)   speedup   GB/s
      1         0.610              0.252            2.4x       4
     64         6.458              0.765            8.5x      87
    128        12.848              1.297            9.9x     103
    192        19.256              3.817            5.1x      53
    256        25.785              5.080            5.1x      53
    512        51.216             10.049            5.1x      53
   1024       102.522             20.029            5.1x      53

`top_p_threshold` switches internally at `_V2_MAX_ROWS = 128` (per-row bucketing
below, histogram path above), which is why the 128 -> 192 step loses the extra
factor.  Both sub-paths win by a wide margin and the crossover the MetaX notes
report at rows=256 (L2 spill) does **not** appear on Iluvatar -- the radix sort
being replaced is simply far slower here.  Measured to 1024; the outer
`_MAX_ROWS` guard only bounds the unmeasured tail.

Correctness against `apply_top_k_top_p_pytorch` (same logits, single
application): mask IoU 0.9991-0.9997 and retained mass 0.950081 vs 0.950001 --
equal to four decimals, with per-row `mass_err_max = 0` (never below the
reference).  Bucketing places the threshold on the lower edge of the deciding
bucket, so the retained set can only come out equal or slightly larger (>= p),
never smaller.

Two measurement traps worth recording, because both silently produced a
plausible-looking wrong number:

  1. `apply_top_k_top_p_pytorch` writes back into its input (`logits.scatter_`,
     docstring says "may be updated in-place").  Reusing one logits tensor
     across benchmark iterations re-applies the mask each step -- observed
     retained tokens collapsing 130560 -> 47329 -> 577 over 26 passes.
  2. Normalising retained mass against the *same* tensor you took the mask from
     makes the ratio identically 1.0, since masked entries contribute 0 to both
     numerator and denominator.  Always measure mass against the pristine
     logits.  (Cold-start buffer reads were tested and do not affect mass.)

Unlike the MetaX patch this one does **not** redirect to
`apply_top_k_top_p_pytorch`: Triton is usable on Iluvatar, so top-k-only
sampling keeps the original fast path and only the top-p-only case is
redirected.  The fast path is deliberately narrow and fails closed -- wrong
rank/dtype/device, too many rows, or any exception falls back to the unmodified
original, so behaviour can only stay the same or improve.

Set `VLLM_FL_TOPP_FAST=0` to disable, `VLLM_FL_TOPP_FAST_VERIFY=N` to re-check
the first N calls against the sort-based reference, and
`VLLM_FL_TOPP_FAST_STATS=<path>` to record which path actually served the run --
a silent fallback would make an accuracy pass meaningless, so coverage is always
reported.
"""

from __future__ import annotations

import atexit
import logging
import os

import torch
import vllm.v1.sample.ops.topk_topp_sampler as topk_topp_sampler

logger = logging.getLogger(__name__)

try:
    from flag_gems import top_p_threshold as _top_p_threshold
except ImportError:  # flag_gems build without the op
    _top_p_threshold = None

_ENABLED = os.environ.get("VLLM_FL_TOPP_FAST", "1") != "0"
# Bounds only the unmeasured tail: the fast path was verified to rows=1024 and
# still wins 5.1x there, and `max_cudagraph_capture_size` caps real serving
# batches at 512.  Anything above this falls back to the untouched original.
_MAX_ROWS = int(os.environ.get("VLLM_FL_TOPP_FAST_MAX_ROWS", "2048"))
_VERIFY_LEFT = int(os.environ.get("VLLM_FL_TOPP_FAST_VERIFY", "0"))
_STATS_PATH = os.environ.get("VLLM_FL_TOPP_FAST_STATS")

# Keep a handle on the untouched original so every fallback preserves the
# pre-patch behaviour exactly (including the Triton path).
_ORIGINAL = topk_topp_sampler.apply_top_k_top_p

_state = {
    "warned": False,
    "verify_left": _VERIFY_LEFT,
    "fast": 0,
    "fallback": 0,
    "other": 0,
}
_next_mark = 1


def _report_counts() -> None:
    s = _state
    msg = ("[fl] topp_fast coverage: fast=%d fallback=%d other=%d enabled=%s op=%s"
           % (s["fast"], s["fallback"], s["other"], _ENABLED,
              _top_p_threshold is not None))
    logger.warning(msg)
    # Only a process that actually sampled is evidence.  The bench client and
    # other helpers import this module, set the same env var and hit atexit
    # without ever calling the sampler; writing their zeros would put a
    # misleading `fast=0` line after the serving process's real counts.
    if _STATS_PATH and (s["fast"] or s["fallback"] or s["other"]):
        try:
            with open(_STATS_PATH, "a") as fh:
                fh.write(msg + "\n")
        except OSError:
            pass


def _mark() -> None:
    """Dump coverage at growing call marks.

    The sampler runs in the EngineCore process, which is SIGKILLed at teardown
    and whose logs may not be forwarded, so atexit alone is not trustworthy
    evidence -- write our own file as calls accumulate.
    """
    global _next_mark
    if not _STATS_PATH:
        return
    n = _state["fast"]
    if n >= _next_mark:
        _report_counts()
        _next_mark = max(n + 1, _next_mark * 4)


atexit.register(_report_counts)


def _verify(logits: torch.Tensor, p, fast_out: torch.Tensor) -> None:
    """Compare the fast result against the sort reference (debug only).

    `logits` is still pristine here: `top_p_threshold` builds a fresh output
    tensor and does not write back.  The reference is handed a clone because
    `apply_top_k_top_p_pytorch` mutates its input.
    """
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
            logits.shape[0], inter / union if union else float("nan"),
            float(mass_ref.mean()), float(mass_fast.mean()),
            float((mass_ref - mass_fast).max()),
            int(keep_ref.sum()), int(keep_fast.sum()),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("[fl] topp_fast verify failed: %s: %s",
                       type(exc).__name__, str(exc)[:120])


def _apply_top_k_top_p_iluvatar(
    logits: torch.Tensor, k: torch.Tensor | None, p: torch.Tensor | None
) -> torch.Tensor:
    if p is None and k is None:
        return logits

    if (
        _ENABLED
        and _top_p_threshold is not None
        and p is not None
        and k is None
        and logits.dim() == 2
        and logits.dtype == torch.float32
        and logits.device.type == "cuda"
        and logits.shape[0] <= _MAX_ROWS
    ):
        try:
            out = _top_p_threshold(logits, p)
        except Exception as exc:  # noqa: BLE001
            if not _state["warned"]:
                _state["warned"] = True
                logger.warning(
                    "[fl] topp_fast disabled after %s: %s; using original path",
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

    return _ORIGINAL(logits, k, p)


topk_topp_sampler.apply_top_k_top_p = _apply_top_k_top_p_iluvatar
