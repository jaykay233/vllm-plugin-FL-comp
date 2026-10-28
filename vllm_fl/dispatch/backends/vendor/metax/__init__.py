# Copyright (c) 2026 BAAI. All rights reserved.

"""
MACA backend for vllm-plugin-FL dispatch.
"""

from .metax import MacaBackend

__all__ = [
    "MacaBackend",
]

# Patches are applied from PlatformFL.import_kernels (vendor==metax), not here.
# Eager ``from . import patches`` at package import pulls cuda_wrapper/pynccl
# during plugin ``register()`` (via maca_launch_defaults) before
# ``vllm.distributed`` is ready.
