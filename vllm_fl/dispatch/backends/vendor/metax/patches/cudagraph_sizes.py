# SPDX-License-Identifier: Apache-2.0
"""Trim MetaX decode CUDA-graph capture sizes to the official concurrency band.

vLLM's default capture list on this stack runs to 512 (50+ sizes). Official
benches use ``--max-concurrency 64``, and the C500 16k case is already near
the KV ceiling at that concurrency, so sizes 80..512 never run but still inflate
the CUDA-graph memory estimate and shrink the KV pool.

Keep every default size ``<= 64``, plus one spare (default 72) so a short burst
above 64 still hits a graph. Override the spare with
``VLLM_FL_METAX_CUDAGRAPH_SPARE``; ``0`` disables the patch.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def patch_cudagraph_capture_sizes_for_metax() -> None:
    try:
        spare = int(os.getenv("VLLM_FL_METAX_CUDAGRAPH_SPARE", "72"))
        if spare <= 0:
            return

        from vllm.config.vllm import VllmConfig

        if getattr(VllmConfig, "_metax_cg_sizes_patch_applied", False):
            return
        _orig = VllmConfig._set_cudagraph_sizes

        def _set_cudagraph_sizes(self):
            _orig(self)
            sizes = self.compilation_config.cudagraph_capture_sizes
            if not sizes:
                return
            kept = [s for s in sizes if s <= 64]
            if spare not in kept and any(s >= spare for s in sizes):
                kept.append(spare)
            kept = sorted(set(kept))
            if kept == list(sizes):
                return
            self.compilation_config.cudagraph_capture_sizes = kept
            self.compilation_config.max_cudagraph_capture_size = kept[-1]
            self.compilation_config.post_init_cudagraph_sizes()
            logger.info(
                "patch_cudagraph_capture_sizes_for_metax: "
                "cudagraph_capture_sizes %s -> %s (spare=%d).",
                sizes,
                kept,
                spare,
            )

        VllmConfig._set_cudagraph_sizes = _set_cudagraph_sizes
        VllmConfig._metax_cg_sizes_patch_applied = True
    except Exception as e:  # noqa: BLE001
        logger.warning("patch_cudagraph_capture_sizes_for_metax: %s", e)


patch_cudagraph_capture_sizes_for_metax()
