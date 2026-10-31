# SPDX-License-Identifier: Apache-2.0
"""Raise OpenAI-server default ``max_num_batched_tokens`` on MetaX C500.

vLLM's ``EngineArgs.get_batch_defaults`` gives devices under 70 GiB
``max_num_batched_tokens=2048``. MetaX C500 reports ~63.6 GiB. Raising only
the OpenAI-API default (an explicit ``--max-num-batched-tokens`` still wins)
matches the Iluvatar patch.

History: **4096** -> **8192** was worth 4k +4.3% / 16k +2.9% (2026-10-22,
FINDINGS-mx §14). **8192** -> **16384** was rejected at the time (§27.4) because
16k lost 3.6%, but that loss came from per-chunk paged attention; with the
dense-KV prefill path (§31) MBT=16384 makes a 16k prompt a single all-dense
chunk and the sign flipped: 16k **+3.18%**, 4k **+1.27%**, both with better
TPOT (§32).

The 4k TTFT does grow (1.91 s -> 2.47 s) because the first batch waits for
more prompts to fill a chunk, which is the tradeoff §27.2 described. It stays
under the 3000 ms budget that the scoring rules impose on TTFT.

Target from ``VLLM_FL_METAX_MAX_NUM_BATCHED_TOKENS`` (default 16384); ``0``
disables the patch. Set it to 8192 to get the pre-§32 behaviour back.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def patch_openai_api_mbt_for_metax() -> None:
    try:
        # 16384 since FINDINGS-mx §32 (16k +3.18%, 4k +1.27%); 8192 for the
        # pre-§32 behaviour. Both keep 4k TTFT under the 3000 ms budget.
        target = int(os.getenv("VLLM_FL_METAX_MAX_NUM_BATCHED_TOKENS", "16384"))
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
