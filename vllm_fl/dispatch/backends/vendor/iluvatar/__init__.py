# Copyright (c) 2026 BAAI. All rights reserved.

"""
ILUVATAR backend for vllm-plugin-FL dispatch.
"""

from .iluvatar import IluvatarBackend, _is_iluvatar_platform

# This package can be imported by discovery/tools outside Iluvatar serving.
# Do not patch shared vLLM sampler behavior on MetaX or other vendors.
if _is_iluvatar_platform():
    from . import patches  # noqa: F401 — apply patches only on Iluvatar

__all__ = ["IluvatarBackend"]
