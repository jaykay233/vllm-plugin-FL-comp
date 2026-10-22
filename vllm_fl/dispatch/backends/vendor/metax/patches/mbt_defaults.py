# SPDX-License-Identifier: Apache-2.0
"""Raise OpenAI-server default ``max_num_batched_tokens`` on MetaX C500.

vLLM's ``EngineArgs.get_batch_defaults`` gives devices under 70 GiB
``max_num_batched_tokens=2048``. MetaX C500 reports ~63.6 GiB. Raising only
the OpenAI-API default (an explicit ``--max-num-batched-tokens`` still wins)
matches the Iluvatar patch.

Default **8192** (was 4096): fair A/B with ``--no-enable-prefix-caching``
(2026-10-22, FINDINGS-mx §14) gave 4k +4.3% / 16k +2.9% vs 4096. Prefix
cache is irrelevant for the official random throughput harness.

Target from ``VLLM_FL_METAX_MAX_NUM_BATCHED_TOKENS`` (default 8192); ``0``
disables the patch.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def patch_openai_api_mbt_for_metax() -> None:
    try:
        target = int(os.getenv("VLLM_FL_METAX_MAX_NUM_BATCHED_TOKENS", "8192"))
        if target <= 0:
            return

        from vllm.engine.arg_utils import EngineArgs
        from vllm.usage.usage_lib import UsageContext

        if getattr(EngineArgs, "_metax_mbt_patch_applied", False):
            return
        _orig = EngineArgs.get_batch_defaults.__func__

        def _get_batch_defaults(cls, world_size):
            mbt, seqs = _orig(cls, world_size)
            key = UsageContext.OPENAI_API_SERVER
            if mbt.get(key, target) < target:
                mbt = dict(mbt)
                mbt[key] = target
            return mbt, seqs

        EngineArgs.get_batch_defaults = classmethod(_get_batch_defaults)
        EngineArgs._metax_mbt_patch_applied = True
        logger.info(
            "patch_openai_api_mbt_for_metax: OpenAI-server default "
            "max_num_batched_tokens -> %d.",
            target,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("patch_openai_api_mbt_for_metax: %s", e)


patch_openai_api_mbt_for_metax()
