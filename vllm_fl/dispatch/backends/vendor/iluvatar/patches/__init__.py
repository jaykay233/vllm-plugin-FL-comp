# Copyright (c) 2026 BAAI. All rights reserved.

"""Iluvatar-specific patches applied at backend load time."""

from ..iluvatar import _is_iluvatar_platform

if _is_iluvatar_platform():
    from . import topk_topp_sampler  # noqa: F401 — Iluvatar sort-free top-p
    from . import sampler_temperature  # noqa: F401 — skip the T=1 logits divide
    from . import fused_top_p_sample  # noqa: F401 — fuse top-p + sample tail
    from . import rope_fast  # noqa: F401 — bandwidth-bound neox RoPE
    __all__ = [
        "topk_topp_sampler",
        "sampler_temperature",
        "fused_top_p_sample",
        "rope_fast",
    ]
else:
    __all__ = []
