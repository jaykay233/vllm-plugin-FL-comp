# Copyright (c) 2026 BAAI. All rights reserved.

"""
ILUVATAR backend for vllm-plugin-FL dispatch.
"""

from .iluvatar import IluvatarBackend
from . import patches  # noqa: F401 — apply Iluvatar kernel patches at backend load time

__all__ = ["IluvatarBackend"]
