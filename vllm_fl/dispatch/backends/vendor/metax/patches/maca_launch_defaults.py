# SPDX-License-Identifier: Apache-2.0
"""Default MACA host-launch knobs for MetaX (no PDL / no GRAPH_LAUNCH_MODE=2).

4k A/B 2026-10-23 (``hostgap_opt_20261023``), P2 MiniCPM5-2B:

| arm | total tok/s | Δ |
|-----|-------------|---|
| baseline (unset) | 11760.72 | — |
| light pack below | 11896.95 | **+1.16%** |

Pack (setdefault — explicit user env always wins):

- ``MACA_DIRECT_DISPATCH=1``
- ``MACA_CPU_THREAD_POLICY=-1``  (min UMD CPU overhead)
- ``MACA_LAUNCH_MODE=4``        (light launch L1)

**Not** included:

- ``MACA_GRAPH_LAUNCH_MODE=2`` (PDL) — cudagraph capture hangs on MACA 3.7 / C500
- ``MACA_GRAPH_LAUNCH_MODE=3`` — 4k **−15.2%** vs baseline (rejected)

Disable with ``VLLM_METAX_MACA_LAUNCH_DEFAULTS=0``.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_DEFAULTS: tuple[tuple[str, str], ...] = (
    ("MACA_DIRECT_DISPATCH", "1"),
    ("MACA_CPU_THREAD_POLICY", "-1"),
    ("MACA_LAUNCH_MODE", "4"),
)


def _is_metax_runtime() -> bool:
    override = os.environ.get("VLLM_FL_PLATFORM", "").strip().lower()
    if override:
        return override == "metax"
    try:
        from vllm_fl.utils import DeviceInfo

        return DeviceInfo().vendor_name == "metax"
    except Exception:
        # Heuristic before DeviceInfo is ready: MACA install present.
        return os.path.isdir("/opt/maca") or bool(os.environ.get("MACA_PATH"))


def apply_maca_launch_defaults() -> None:
    if os.environ.get("VLLM_METAX_MACA_LAUNCH_DEFAULTS", "1") == "0":
        return
    if not _is_metax_runtime():
        return

    applied: list[str] = []
    for key, val in _DEFAULTS:
        if key not in os.environ or str(os.environ.get(key, "")).strip() == "":
            os.environ[key] = val
            applied.append(f"{key}={val}")
    if applied:
        logger.info(
            "apply_maca_launch_defaults: set %s "
            "(override any key, or VLLM_METAX_MACA_LAUNCH_DEFAULTS=0 to skip).",
            ", ".join(applied),
        )


apply_maca_launch_defaults()
