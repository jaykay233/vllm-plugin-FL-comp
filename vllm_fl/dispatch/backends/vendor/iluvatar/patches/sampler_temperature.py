# Copyright (c) 2026 BAAI. All rights reserved.

"""Iluvatar sampler patch: skip the temperature divide when every request is T=1.

vLLM's ``Sampler.apply_temperature`` always runs ``logits.div_(temperature)``
in-place on the full ``[rows, vocab]`` float32 logits.  MiniCPM's generation
config at the scored operating point is ``temperature=1.0``, so that pass is a
bit-exact identity while still reading and writing the whole vocabulary matrix
on every decode step (measured ~169 us at rows=32, ~337 us at rows=64 on
BI-V150).

The decision is made host-side from ``InputBatch.temperature_cpu`` when the
sampling metadata is rebuilt, so no device synchronisation is added.  vLLM
clamps greedy temperatures to 1.0 before the divide, so rows with
``temperature < eps`` also count as "no scaling".  Anything else falls through
to the untouched original.

Env: ``VLLM_FL_SKIP_TEMP_DIV=0`` disables the patch (default on, Iluvatar
only).  ``VLLM_FL_SKIP_TEMP_DIV_MARK=<path>`` writes a marker file the first
time the skip is taken, for verifying that the path is live in the EngineCore
process (whose logs are not always forwarded).
"""

from __future__ import annotations

import logging
import os

import numpy as np
import torch

from vllm.v1.sample.sampler import Sampler
from vllm.v1.worker.gpu_input_batch import InputBatch

from ..iluvatar import _is_iluvatar_platform

logger = logging.getLogger(__name__)

# Matches vllm.v1.sample.sampler._SAMPLING_EPS.
_EPS = 1e-5
_ENABLED = _is_iluvatar_platform() and os.environ.get(
    "VLLM_FL_SKIP_TEMP_DIV", "1"
).strip().lower() not in ("0", "false", "off", "no")
_MARK = os.environ.get("VLLM_FL_SKIP_TEMP_DIV_MARK")

# Set for the duration of one ``Sampler.sample`` call.  The sampler is single
# threaded per engine, so a module global is enough and allocates nothing.
_skip_temperature = False
_marked = False

_ORIG_APPLY = Sampler.apply_temperature
_ORIG_SAMPLE = Sampler.sample
_ORIG_MAKE = InputBatch._make_sampling_metadata


def _apply_temperature(logits: torch.Tensor, temp, all_random: bool):
    if _skip_temperature:
        return logits
    return _ORIG_APPLY(logits, temp, all_random)


def _sample(self, logits, sampling_metadata, logprobs_mode_override=None):
    global _skip_temperature, _marked
    skip = bool(getattr(sampling_metadata, "_fl_skip_temperature", False))
    if skip and _MARK and not _marked:
        _marked = True
        try:
            with open(_MARK, "w") as fh:
                fh.write("skip temperature divide taken\n")
        except OSError:
            pass
    prev = _skip_temperature
    _skip_temperature = skip
    try:
        return _ORIG_SAMPLE(self, logits, sampling_metadata, logprobs_mode_override)
    finally:
        _skip_temperature = prev


def _make_sampling_metadata(self):
    md = _ORIG_MAKE(self)
    try:
        n = int(self.num_reqs)
        arr = self.temperature_cpu[:n]
        # vLLM maps greedy rows to 1.0 before dividing, so both are identity.
        md._fl_skip_temperature = bool(
            n > 0 and np.all((arr == 1.0) | (arr < _EPS))
        )
    except Exception:  # noqa: BLE001 — never break sampling on a metadata guess
        md._fl_skip_temperature = False
    return md


def _enable() -> None:
    Sampler.apply_temperature = staticmethod(_apply_temperature)
    Sampler.sample = _sample
    InputBatch._make_sampling_metadata = _make_sampling_metadata
    logger.warning(
        "[fl] skip temperature divide for T=1 rows: enabled "
        "(VLLM_FL_SKIP_TEMP_DIV=0 to disable)"
    )


if _ENABLED:
    _enable()
