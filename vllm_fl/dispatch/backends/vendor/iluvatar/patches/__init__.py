# Copyright (c) 2026 BAAI. All rights reserved.

"""Iluvatar-specific patches applied at backend load time."""

from . import topk_topp_sampler  # noqa: F401 — sort-free top-p

__all__ = ["topk_topp_sampler"]
