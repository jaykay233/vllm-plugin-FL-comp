# Copyright (c) 2026 BAAI. All rights reserved.

"""Iluvatar-specific patches applied at backend load time."""

from ..iluvatar import _is_iluvatar_platform

if _is_iluvatar_platform():
    from . import topk_topp_sampler  # noqa: F401 — Iluvatar sort-free top-p
    __all__ = ["topk_topp_sampler"]
else:
    __all__ = []
