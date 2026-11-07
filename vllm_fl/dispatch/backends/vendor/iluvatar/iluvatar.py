# Copyright (c) 2026 BAAI. All rights reserved.

"""
ILUVATAR backend implementation.

This backend provides operator implementations for Iluvatar GPUs.
Iluvatar uses a CUDA-compatible architecture.
"""

from __future__ import annotations

import logging
import re
from typing import Optional, Union

import torch

from vllm_fl.dispatch.backends.base import Backend

logger = logging.getLogger(__name__)


def _is_iluvatar_platform() -> bool:
    """Return whether the active vLLM platform is Iluvatar.

    Vendor modules are auto-discovered during plugin initialization, so
    importing this backend must not apply its runtime patches on MetaX.
    Fail closed when platform detection is unavailable.
    """
    try:
        from vllm.platforms import current_platform

        vendor_name = getattr(current_platform, "vendor_name", None)
        return isinstance(vendor_name, str) and vendor_name.lower() == "iluvatar"
    except Exception:
        return False


def patch_triton_language_for_iluvatar() -> None:
    """Add make_tensor_descriptor stub to triton.language for triton < 3.3.

    triton JIT's DependencyFinder walks all @triton.jit function bodies at
    cache_key computation time (first compile, not import). Even dead-code
    branches guarded by ``USE_TD: tl.constexpr = False`` cause AttributeError
    when ``tl.make_tensor_descriptor`` does not exist, because DependencyFinder
    resolves ``tl.*`` attributes to compute a stable kernel hash.

    The _Stub below is a plain Python callable + hashable object that satisfies
    attribute lookup. It raises at call-time so accidental USE_TD=True usage is
    caught immediately.

    TODO: Remove once minimum supported triton version is >= 3.3.
    """
    if not _is_iluvatar_platform():
        logger.debug(
            "patch_triton_language_for_iluvatar: non-Iluvatar platform; skipping."
        )
        return
    try:
        import triton.language as tl

        if hasattr(tl, "make_tensor_descriptor"):
            return  # triton >= 3.3, nothing to do

        class _TensorDescriptorStub:
            """Stub for tl.make_tensor_descriptor — callable and hashable."""

            def __call__(self, *args, **kwargs):
                raise RuntimeError(
                    "tl.make_tensor_descriptor is not available on triton < 3.3. "
                    "Ensure VLLM_TRITON_ATTN_USE_TD=0 (the default on iluvatar)."
                )

            def __hash__(self):
                return hash("_tl_make_tensor_descriptor_stub")

            def __repr__(self):
                return "<tl.make_tensor_descriptor stub for triton<3.3>"

        tl.make_tensor_descriptor = _TensorDescriptorStub()
        logger.info(
            "Patched triton.language: added make_tensor_descriptor stub "
            "(triton < 3.3 detected; USE_TD=False assumed on iluvatar)."
        )
    except Exception as e:
        logger.warning("Failed to patch triton.language for iluvatar: %s", e)


def patch_triton_chained_or_for_iluvatar() -> None:
    """Rewrite chained boolean 'or' chains in vllm triton kernels.

    Iluvatar's triton version (3.2.x) raises UnsupportedLanguageConstruct for
    chained boolean operators like ``A or B or C`` inside @triton.jit functions.
    Rewrites the affected source file in-place (idempotent).

    Must be called from the main process BEFORE Worker subprocesses start,
    so the patched file is on disk when Workers import the module.

    Applies only when triton < 3.3 — later versions support chained boolean
    operators natively.

    TODO: Remove once minimum supported Iluvatar triton version is >= 3.3.
    """
    if not _is_iluvatar_platform():
        logger.debug(
            "patch_triton_chained_or_for_iluvatar: non-Iluvatar platform; skipping."
        )
        return
    import re
    import importlib.util
    import pathlib
    import sys

    # Only needed for triton < 3.3.
    try:
        import triton as _triton
        _tv = tuple(int(x) for x in _triton.__version__.split(".")[:2])
        if _tv >= (3, 3):
            return
    except Exception as e:
        logger.warning(
            "patch_triton_chained_or_for_iluvatar: cannot determine triton version, "
            "applying patch defensively: %s", e
        )

    _MARKER = "# _iluvatar_chained_or_patched"

    spec = importlib.util.find_spec("vllm.v1.attention.ops.triton_attention_helpers")
    if spec is None or spec.origin is None:
        logger.warning(
            "patch_triton_chained_or_for_iluvatar: "
            "vllm.v1.attention.ops.triton_attention_helpers not found, skipping."
        )
        return

    fpath = pathlib.Path(spec.origin)
    try:
        src = fpath.read_text()
    except Exception as e:
        logger.warning("patch_triton_chained_or_for_iluvatar: cannot read %s: %s", fpath, e)
        return

    if _MARKER in src:
        return  # already patched

    # Replace: A or B or C  →  (A or B) or C
    _OPERAND = r'(?:not\s+)?(?:\([^)]*\)|\w+)'
    pattern = re.compile(
        r'(' + _OPERAND + r')\s+or\s+(' + _OPERAND + r')\s+or\s+(' + _OPERAND + r')'
    )

    def _rewrite(m: re.Match) -> str:
        return f"({m.group(1)} or {m.group(2)}) or {m.group(3)}"

    new_src, count = re.subn(pattern, _rewrite, src)
    if count == 0:
        return  # nothing to patch

    new_src += f"\n{_MARKER}\n"
    try:
        fpath.write_text(new_src)
    except Exception as e:
        logger.warning(
            "patch_triton_chained_or_for_iluvatar: cannot write %s: %s", fpath, e
        )
        return

    # Clear pycache so Python and triton both see the patched source.
    pycache = fpath.parent / "__pycache__"
    if pycache.exists():
        import shutil
        try:
            shutil.rmtree(pycache)
        except Exception:
            pass  # non-fatal

    # Evict from sys.modules so this process reimports the patched source.
    sys.modules.pop("vllm.v1.attention.ops.triton_attention_helpers", None)

    logger.info(
        "patch_triton_chained_or_for_iluvatar: rewrote %d chained-or expression(s) in %s",
        count, fpath,
    )


def patch_triton_perf_model_for_iluvatar() -> None:
    """
    Patch triton.ops.matmul_perf_model.get_clock_rate_in_khz for iluvatar.

    On iluvatar, neither `ixsmi` nor `libnvidia-ml.so` is available, so the
    default implementation crashes. We replace it with a fixed value that
    matches typical iluvatar BI-V150 SM clock (1500 MHz = 1500000 kHz).

    The function is decorated with @functools.lru_cache so assigning a new
    callable to the module attribute is sufficient — callers import the name
    directly, so we also patch the reference in the testing helper.

    Applies only when triton < 3.3 — later versions handle missing nvsmi/nvml
    natively or remove triton.ops.matmul_perf_model entirely.

    TODO: Remove once minimum supported Iluvatar triton version is >= 3.3.
    """
    if not _is_iluvatar_platform():
        logger.debug(
            "patch_triton_perf_model_for_iluvatar: non-Iluvatar platform; skipping."
        )
        return
    try:
        import triton as _triton
        _tv = tuple(int(x) for x in _triton.__version__.split(".")[:2])
        if _tv >= (3, 3):
            return

        import triton.ops.matmul_perf_model as _mpm

        if getattr(_mpm, '_iluvatar_clock_patched', False):
            return

        import functools

        @functools.lru_cache()
        def _iluvatar_get_clock_rate_in_khz():
            # BI-V150 SM clock ~1500 MHz; used only for perf-model heuristics.
            return 1500 * 1e3

        _mpm.get_clock_rate_in_khz = _iluvatar_get_clock_rate_in_khz
        _mpm._iluvatar_clock_patched = True
        logger.info(
            "Patched triton.ops.matmul_perf_model.get_clock_rate_in_khz "
            "for iluvatar (fixed 1500 MHz, no nvsmi/nvml available)."
        )
    except Exception as e:
        logger.warning(
            "Failed to patch triton perf model for iluvatar: %s", e
        )

patch_triton_chained_or_for_iluvatar()
patch_triton_language_for_iluvatar()
patch_triton_perf_model_for_iluvatar()


def patch_sampler_compile_for_iluvatar() -> None:
    if not _is_iluvatar_platform():
        logger.debug(
            "patch_sampler_compile_for_iluvatar: non-Iluvatar platform; skipping."
        )
        return
    # Disable torch.compile on vllm sampler ops for Iluvatar.
    # flagtree triton only supports Iluvatar backend, not cuda target.
    # TODO: Remove once flagtree triton supports cuda inductor target.
    try:
        import importlib
        _tts = importlib.import_module('vllm.v1.sample.ops.topk_topp_sampler')
        if not getattr(_tts, '_iluvatar_compile_patched', False):
            _orig = _tts.compiled_random_sample
            _tts.compiled_random_sample = getattr(_orig, '__wrapped__', _orig)
            _tts._iluvatar_compile_patched = True
            logger.info('patch_sampler_compile_for_iluvatar: unwrapped topk_topp_sampler.compiled_random_sample')
    except Exception as e:
        logger.warning('patch_sampler_compile_for_iluvatar (topk_topp): %s', e)
    try:
        import importlib
        _lp = importlib.import_module('vllm.v1.sample.ops.logprobs')
        if not getattr(_lp, '_iluvatar_compile_patched', False):
            _orig = _lp.batched_count_greater_than
            _lp.batched_count_greater_than = getattr(_orig, '__wrapped__', _orig)
            _lp._iluvatar_compile_patched = True
            logger.info('patch_sampler_compile_for_iluvatar: unwrapped logprobs.batched_count_greater_than')
    except Exception as e:
        logger.warning('patch_sampler_compile_for_iluvatar (logprobs): %s', e)

patch_sampler_compile_for_iluvatar()


def patch_torch_inductor_for_iluvatar() -> None:
    """Patch torch._inductor GPUTarget to use the correct triton backend name.

    torch._inductor always passes device_type='cuda' to GPUTarget, but
    flagtree triton only registers an 'iluvatar' (3.6.x) or 'corex' (3.2.x)
    backend -- never 'cuda'.

    Fix: subclass GPUTarget to intercept 'cuda' and remap it to whatever
    backend name flagtree triton actually registered.

    Compatibility: safe to call when not using flagtree -- if triton
    has a 'cuda' backend or no backends at all, the patch is skipped.

    Hardware gate: Iluvatar only.
    TODO: Remove once torch._inductor or flagtree natively handles this.
    """
    if not _is_iluvatar_platform():
        logger.debug(
            "patch_torch_inductor_for_iluvatar: non-Iluvatar platform; skipping."
        )
        return
    try:
        import triton.backends as _tb
        _registered = list(getattr(_tb, 'backends', {}).keys())
        # If 'cuda' is already registered as a backend name, no patch needed
        if 'cuda' in _registered or not _registered:
            logger.debug('patch_torch_inductor_for_iluvatar: triton has cuda backend or no backends, skipping')
            return
        # Probe the actual target string expected by supports_target().
        # Different flagtree triton versions use different conventions:
        #   - 3.2.x: backend name 'iluvatar', supports_target checks == 'cuda'
        #   - 3.6.x: backend name 'iluvatar', supports_target checks == 'corex'
        # We must discover the correct target string at runtime.
        _target = None
        try:
            from triton.backends.compiler import GPUTarget as _GPUTarget
            # Include 'cuda' in probes — some backends accept 'cuda' as target
            for _probe in ('cuda', 'corex', 'iluvatar') + tuple(_registered):
                try:
                    _t = object.__new__(_GPUTarget)
                    _GPUTarget.__init__(_t, _probe, 90, False)
                    for _bname in _registered:
                        if _tb.backends[_bname].compiler.supports_target(_t):
                            _target = _probe
                            break
                except Exception:
                    pass
                if _target:
                    break
        except Exception:
            pass
        if not _target:
            _target = 'corex'  # safe default for all known flagtree versions

        # If the discovered target is already 'cuda', no remapping needed —
        # inductor naturally passes 'cuda' to GPUTarget.
        if _target == 'cuda':
            logger.info(
                "patch_torch_inductor_for_iluvatar: backend '%s' already "
                "supports target='cuda', no GPUTarget remap needed.",
                _registered,
            )
            return
    except Exception:
        logger.debug('patch_torch_inductor_for_iluvatar: cannot inspect triton backends, skipping')
        return

    try:
        import torch._inductor.runtime.triton_heuristics as _th
        from triton.backends.compiler import GPUTarget as _OrigGPUTarget

        if getattr(_th, '_iluvatar_gputarget_patched', False):
            return

        _remap_target = _target

        class _IluvatarGPUTarget(_OrigGPUTarget):
            """GPUTarget wrapper that remaps 'cuda' to the flagtree backend."""
            def __new__(cls, backend, *args, **kwargs):
                if backend == 'cuda':
                    backend = _remap_target
                return super().__new__(cls)

            def __init__(self, backend, *args, **kwargs):
                if backend == 'cuda':
                    backend = _remap_target
                super().__init__(backend, *args, **kwargs)

        _th.GPUTarget = _IluvatarGPUTarget
        _th._iluvatar_gputarget_patched = True
        logger.info(
            "Patched torch._inductor GPUTarget: 'cuda' -> '%s' (iluvatar)",
            _remap_target,
        )
    except Exception as e:
        logger.warning(
            "Failed to patch torch._inductor for iluvatar triton backend: %s", e
        )

patch_torch_inductor_for_iluvatar()


def patch_triton_unified_attention_prefill_for_iluvatar() -> None:
    """Retile the unified-attention prefill path for BI-V150.

    vLLM's default prefill tiling is inherited from a decode-oriented layout:
    ``BLOCK_M=16`` (two query rows per 8:1 GQA group) and four warps.  On the
    16-SM BI-V150 this creates tens of thousands of tiny CTAs and leaves the
    tensor cores mostly idle.  A kernel sweep on the competition shape
    (16 query heads / 2 KV heads / head_size 128 / bf16) measured:

        BM16  T32 w4:       5.293 ms/layer   3.2 TFLOPS
        BM128 T32 w8:       1.278 ms/layer  13.4 TFLOPS
        BM128 T64 w8 s2:    0.801 ms/layer  ~21 TFLOPS   ← default

    The patch rewrites the wrapper source at runtime and swaps the caller's
    imported symbol, so only this process changes.  It applies only to
    Iluvatar, only to the 2D prefill path, and only to the tested 128-wide GQA
    shape.  Decode keeps vLLM's original tiling.

    The same wrapper has a latent 2D online-softmax bug on older Triton builds:
    ``L`` starts at 1.0 even though the first tile should establish it, which
    scaled every 2D prefill row by ``S/(S+1)``.  The kernel source is rewritten
    with the companion fix (L starts at zero; sinks retain the deliberate unit
    contribution) and the 2D epilogue guards the all-masked row before dividing.
    flagtree 0.7.0+iluvatar3.6 fixes this in the compiler, so on that build the
    kernel is left untouched; the rewrite is only a fallback for older compilers
    and is applied best-effort so it can never block the tiling patch.

    Set ``VLLM_FL_ILUVATAR_PREFILL_TILING=0`` to restore vLLM's original
    prefill tiling and softmax initialization for controlled experiments.
    Knobs (defaults = measured best):

    * ``VLLM_FL_ILUVATAR_PREFILL_BLOCK_M``   64 / 128 / 256   (default 128)
    * ``VLLM_FL_ILUVATAR_PREFILL_TILE``      32 / 64 / 128    (default 64)
    * ``VLLM_FL_ILUVATAR_PREFILL_NUM_WARPS`` 4 / 8            (default 4 on
      flagtree >= 0.7.0, 8 on older builds; 8 with 2 stages is nondeterministic
      on flagtree >= 0.7.0)
    * ``VLLM_FL_ILUVATAR_PREFILL_NUM_STAGES`` 0 / 2 / 3       (default 2; 0 = Triton default)
    * ``VLLM_FL_ILUVATAR_PREFILL_L0_FIX``    auto / 1 / 0     (default auto:
      on for flagtree < 0.7.0, off for newer compilers that already fix it)
    """
    try:
        from vllm.platforms import current_platform

        vendor_name = getattr(current_platform, "vendor_name", None)
        if not isinstance(vendor_name, str) or vendor_name.lower() != "iluvatar":
            logger.info(
                "patch_triton_unified_attention_prefill_for_iluvatar: "
                "current platform is %r, leaving attention unchanged.",
                vendor_name,
            )
            return

        import os

        if os.getenv("VLLM_FL_ILUVATAR_PREFILL_TILING", "1") == "0":
            logger.info(
                "patch_triton_unified_attention_prefill_for_iluvatar: "
                "disabled by VLLM_FL_ILUVATAR_PREFILL_TILING=0."
            )
            return

        def _env_int(name: str, default: int, allowed: tuple[int, ...]) -> int:
            raw = os.getenv(name, str(default))
            try:
                value = int(raw)
            except ValueError:
                logger.warning(
                    "patch_triton_unified_attention_prefill_for_iluvatar: "
                    "invalid %s=%r; using %s.",
                    name,
                    raw,
                    default,
                )
                return default
            if value not in allowed:
                logger.warning(
                    "patch_triton_unified_attention_prefill_for_iluvatar: "
                    "unsupported %s=%s; allowed %s; using %s.",
                    name,
                    value,
                    allowed,
                    default,
                )
                return default
            return value

        def _flagtree_version() -> tuple[int, ...]:
            """Parse the numeric part of the installed flagtree version."""
            try:
                from importlib.metadata import version as _pkg_version

                raw = _pkg_version("flagtree")
            except Exception:
                return ()
            # "0.7.0+iluvatar3.6" -> "0.7.0"; the local tag must not leak into
            # the numeric parts ("0+iluvatar3" would parse as 3).
            public = raw.split("+", 1)[0]
            parts: list[int] = []
            for chunk in public.split(".")[:3]:
                # Leading digits only: "2a3" is a pre-release of 2, not 23.
                match = re.match(r"\d+", chunk)
                parts.append(int(match.group()) if match else 0)
            while len(parts) < 3:
                parts.append(0)
            return tuple(parts[:3])

        flagtree_ver = _flagtree_version()
        # flagtree 0.7.0 is the first build that folds the first softmax tile
        # correctly, but it also made the 8-warp / 2-stage prefill pipeline
        # nondeterministic (see the L0 note below and FINDINGS 17.52).  Both
        # facts only matter to this patch, and they point in opposite
        # directions for the defaults:
        #
        #   0.6.2:  w8 s2 = 23.1 TFLOPS, deterministic  -> keep 8 warps
        #   0.7.0:  w8 s2 = 32.3 TFLOPS but WRONG       -> must use 4 warps
        #            w4 s2 = 26.9 TFLOPS, deterministic
        #
        # so the default depends on the compiler.  An explicit knob always wins.
        _default_warps = 4 if (not flagtree_ver or flagtree_ver >= (0, 7, 0)) else 8

        prefill_block_m = _env_int(
            "VLLM_FL_ILUVATAR_PREFILL_BLOCK_M", 128, (64, 128, 256)
        )
        prefill_tile = _env_int(
            "VLLM_FL_ILUVATAR_PREFILL_TILE", 64, (32, 64, 128)
        )
        prefill_num_warps = _env_int(
            "VLLM_FL_ILUVATAR_PREFILL_NUM_WARPS", _default_warps, (4, 8)
        )
        prefill_num_stages = _env_int(
            "VLLM_FL_ILUVATAR_PREFILL_NUM_STAGES", 2, (0, 2, 3)
        )
        stages_literal = (
            "None" if prefill_num_stages == 0 else str(prefill_num_stages)
        )

        import inspect

        import vllm.v1.attention.ops.triton_unified_attention as _uam

        if getattr(_uam, "_iluvatar_prefill_patch_applied", False):
            return

        kernel = _uam.kernel_unified_attention
        kernel_src = getattr(kernel, "src", None)
        if not isinstance(kernel_src, str):
            logger.warning(
                "patch_triton_unified_attention_prefill_for_iluvatar: "
                "kernel source is unavailable, skipping."
            )
            return

        def _replace_once(source: str, old: str, new: str, label: str) -> str:
            count = source.count(old)
            if count != 1:
                raise RuntimeError(
                    f"expected one {label} site, found {count}; "
                    "vLLM attention source changed"
                )
            return source.replace(old, new)

        # -- L0 softmax fix: compiler workaround, now optional ----------------
        #
        # vLLM's 2D online softmax starts L at 1.0 and relies on the first
        # tile's exp(-inf - m) = 0 to clear it.  On flagtree 0.6.2a3+iluvatar3.6
        # that fold did not happen, so every 2D prefill row came out scaled by
        # S/(S+1) -- q_len=1/kv_len=1 returned exactly 0.5*v0 (FINDINGS 16.4).
        # The kernel is rewritten here to start L at 0.0 and guard the rows that
        # stay empty.
        #
        # flagtree 0.7.0+iluvatar3.6 fixes the compiler: the stock kernel now
        # returns 0.0 error on 1/1, so the rewrite is redundant there.  It is
        # still applied for older compilers, but only best-effort -- if vLLM
        # ever edits those lines, the tiling patch below must still land,
        # because losing tiling costs 21.5 -> 3.2 TFLOPS while losing this fix
        # costs nothing on a fixed compiler.
        l0_mode = os.getenv(
            "VLLM_FL_ILUVATAR_PREFILL_L0_FIX", "auto"
        ).strip().lower()
        if l0_mode in ("0", "off", "false", "no"):
            apply_l0 = False
            l0_reason = "disabled by VLLM_FL_ILUVATAR_PREFILL_L0_FIX"
        elif l0_mode in ("1", "on", "true", "yes"):
            apply_l0 = True
            l0_reason = "forced by VLLM_FL_ILUVATAR_PREFILL_L0_FIX"
        else:
            # 0.7.0 is the first flagtree that folds the first tile correctly.
            # When the version cannot be read, keep the rewrite: it is harmless
            # on a fixed compiler and correct on a broken one, whereas skipping
            # it on a broken compiler silently scales every prefill row.
            apply_l0 = not flagtree_ver or flagtree_ver < (0, 7, 0)
            l0_reason = (
                f"auto: flagtree {'.'.join(map(str, flagtree_ver)) or 'unknown'}"
            )

        if apply_l0:
            try:
                l_init_old = "    L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)"
                l_init_new = (
                    "    # iluvatar_softmax_l0_fix: the first tile establishes L.\n"
                    "    L = tl.full([BLOCK_M], 0.0, dtype=tl.float32)\n"
                    "    if USE_SINKS:\n"
                    "        L = tl.where(query_mask_1, 1.0, 0.0)"
                )
                patched_src = _replace_once(
                    kernel_src, l_init_old, l_init_new, "softmax denominator init"
                )

                div_old = "        acc = acc / L[:, None]"
                div_new = (
                    "        # iluvatar_softmax_l0_fix: an empty row keeps a zero "
                    "numerator.\n"
                    "        acc = acc / tl.where(L == 0.0, 1.0, L)[:, None]"
                )
                patched_src = _replace_once(
                    patched_src, div_old, div_new, "2D softmax denominator write"
                )
            except RuntimeError as e:
                # Never let this kill the tiling patch.
                logger.warning(
                    "patch_triton_unified_attention_prefill_for_iluvatar: "
                    "L0 softmax fix skipped (%s); assuming the compiler handles "
                    "it and continuing with the tiling patch.",
                    e,
                )
            else:
                kernel_src = patched_src
                kernel._unsafe_update_src(kernel_src)
                if hasattr(kernel, "device_caches"):
                    kernel.device_caches.clear()
                logger.info(
                    "patch_triton_unified_attention_prefill_for_iluvatar: "
                    "L0 softmax denominator fix applied (%s).",
                    l0_reason,
                )
        else:
            logger.info(
                "patch_triton_unified_attention_prefill_for_iluvatar: "
                "L0 softmax denominator fix not applied (%s); "
                "flagtree >= 0.7.0 supplies the fix itself.",
                l0_reason,
            )

        # 8 warps with an explicit 2-stage pipeline is nondeterministic on
        # flagtree 0.7.x: identical calls return different results and the error
        # at ns=64/q=4096 reaches 1.5e-01, ~29x the bf16 rounding floor.  It is
        # also faster (32 vs 27 TFLOPS), i.e. the race drops real work.  4 warps
        # keeps the same tile and is deterministic on both compilers at roughly
        # the old cost, so it is the default; warn if anyone forces the pair.
        if (
            prefill_num_warps == 8
            and prefill_num_stages == 2
            and (not flagtree_ver or flagtree_ver >= (0, 7, 0))
        ):
            logger.warning(
                "patch_triton_unified_attention_prefill_for_iluvatar: "
                "num_warps=8 with num_stages=2 is nondeterministic on "
                "flagtree >= 0.7.0 (results vary run to run and the ns=64 "
                "error reaches 1.5e-01). Use VLLM_FL_ILUVATAR_PREFILL_NUM_WARPS=4."
            )

        wrapper_src = inspect.getsource(_uam.unified_attention)
        block_m_old = (
            "    BLOCK_M = (\n"
            "        16 if num_queries_per_kv <= 16 else "
            "triton.next_power_of_2(num_queries_per_kv)\n"
            "    )"
        )
        block_m_new = (
            "    _iluvatar_prefill_tiling = (\n"
            "        max_seqlen_q > 1\n"
            "        and not use_td\n"
            "        and head_size == 128\n"
            "        and 2 <= num_queries_per_kv <= 16\n"
            "    )\n"
            "    if _iluvatar_prefill_tiling:\n"
            f"        BLOCK_M = {prefill_block_m}\n"
            "    else:\n"
            "        BLOCK_M = (\n"
            "            16 if num_queries_per_kv <= 16 else "
            "triton.next_power_of_2(num_queries_per_kv)\n"
            "        )"
        )
        wrapper_src = _replace_once(
            wrapper_src, block_m_old, block_m_new, "prefill BLOCK_M selection"
        )

        tile_old = (
            "    if tuned_large_head:\n"
            "        TILE_SIZE_PREFILL = 128"
        )
        tile_new = (
            "    if tuned_large_head:\n"
            "        TILE_SIZE_PREFILL = 128\n"
            "    if _iluvatar_prefill_tiling:\n"
            f"        TILE_SIZE_PREFILL = {prefill_tile}"
        )
        wrapper_src = _replace_once(
            wrapper_src, tile_old, tile_new, "prefill TILE_SIZE_PREFILL selection"
        )

        warps_old = (
            "    launch_num_warps: int | None = None\n"
            "    launch_num_stages: int | None = None"
        )
        warps_new = (
            "    launch_num_warps: int | None = (\n"
            f"        {prefill_num_warps} if _iluvatar_prefill_tiling else None\n"
            "    )\n"
            "    launch_num_stages: int | None = (\n"
            f"        {stages_literal} if _iluvatar_prefill_tiling else None\n"
            "    )"
        )
        wrapper_src = _replace_once(
            wrapper_src, warps_old, warps_new, "prefill launch num_warps/stages"
        )

        exec(compile(wrapper_src, _uam.__file__, "exec"), _uam.__dict__)
        import vllm.v1.attention.backends.triton_attn as _ta

        _ta.unified_attention = _uam.unified_attention
        _uam._iluvatar_prefill_patch_applied = True
        logger.info(
            "patch_triton_unified_attention_prefill_for_iluvatar: enabled "
            "BLOCK_M=%s / TILE=%s / num_warps=%s / num_stages=%s for "
            "128-wide GQA prefill.",
            prefill_block_m,
            prefill_tile,
            prefill_num_warps,
            stages_literal,
        )
    except Exception as e:
        logger.warning(
            "patch_triton_unified_attention_prefill_for_iluvatar: %s", e
        )


def patch_triton_attn_segments_for_iluvatar() -> None:
    """Lower NUM_PAR_SOFTMAX_SEGMENTS to match BI-V150's SM count.

    vLLM hardcodes the 3D (parallel-softmax) attention path to 16 segments:

        vllm/v1/attention/backends/triton_attn.py:55
            NUM_PAR_SOFTMAX_SEGMENTS = 16

    That is tuned for datacenter parts with 100+ SMs.  BI-V150 has 16 SMs, so 16
    segments over-partitions: each decode step launches
    grid = (num_q_blocks, num_kv_heads, segments) = (64, 2, 16) = 2048 CTAs whose
    per-segment partials must then be LSE-merged, and the merge traffic grows with
    the segment count while there is no spare parallelism to win.

    Measured on BI-V150, conc=64, bf16, triton_unified_attention:

        ctx     2D kernel   segm=16 (default)   segm=1 (optimal)   raw-read ceiling
        1024       24.20         21.74              17.24              9.27   ms/step
        4096       94.37         71.01              64.71             32.27   ms/step

    so segm=1 beats the shipped default by 4.5 ms/step (-13%) at ctx=1024 and
    6.3 ms/step at ctx=4096.  End-to-end (conc=64, ctx=1024, 3 rounds):
    34.01 -> 30.39 ms/step (-10.6%).

    Numerically equivalent (kernel-level, vs an fp32 dense reference):
    segm=1 diverges from segm=16 by 6.1e-05 max abs at ctx=1024 -- one bf16 ULP
    at this magnitude -- and is no less accurate than the default against fp32.

    The value is read once, as a module global, in
    ``TritonAttentionMetadataBuilder.__init__``:
        self.num_par_softmax_segments = NUM_PAR_SOFTMAX_SEGMENTS
    so assigning the attribute before the engine is constructed is sufficient --
    no source edits, no file patching.  This module-level call runs on import,
    which is earlier than any engine build.

    Hardware gate: Iluvatar only, and skipped if the symbol is absent (other
    vLLM versions may not have it). The explicit runtime platform check keeps
    MetaX/MX on vLLM's original value even if this module is imported directly.
    TODO: Remove once vLLM derives this from SM count.
    """
    try:
        from vllm.platforms import current_platform

        vendor_name = getattr(current_platform, "vendor_name", None)
        if not isinstance(vendor_name, str) or vendor_name.lower() != "iluvatar":
            logger.info(
                "patch_triton_attn_segments_for_iluvatar: current platform is "
                "%r, leaving NUM_PAR_SOFTMAX_SEGMENTS unchanged.",
                vendor_name,
            )
            return

        import vllm.v1.attention.backends.triton_attn as _ta

        if not hasattr(_ta, "NUM_PAR_SOFTMAX_SEGMENTS"):
            logger.debug(
                "patch_triton_attn_segments_for_iluvatar: "
                "NUM_PAR_SOFTMAX_SEGMENTS not present, skipping."
            )
            return

        _old = _ta.NUM_PAR_SOFTMAX_SEGMENTS
        # Default to one segment for BI-V150; expose a bounded per-process
        # override so serving workloads can be tuned without affecting other
        # vendors or changing global vLLM defaults.
        import os

        raw_target = os.getenv("VLLM_FL_ILUVATAR_NUM_PAR_SOFTMAX_SEGMENTS", "1")
        try:
            _target = int(raw_target)
        except ValueError:
            logger.warning(
                "patch_triton_attn_segments_for_iluvatar: invalid "
                "VLLM_FL_ILUVATAR_NUM_PAR_SOFTMAX_SEGMENTS=%r; using 1.",
                raw_target,
            )
            _target = 1
        if _target not in (1, 2, 4, 8, 16):
            logger.warning(
                "patch_triton_attn_segments_for_iluvatar: unsupported value "
                "%s; allowed values are 1, 2, 4, 8, 16; using 1.",
                _target,
            )
            _target = 1
        if _old <= _target:
            logger.debug(
                "patch_triton_attn_segments_for_iluvatar: already %s, nothing to do.",
                _old,
            )
            return
        _ta.NUM_PAR_SOFTMAX_SEGMENTS = _target
        logger.info(
            "patch_triton_attn_segments_for_iluvatar: NUM_PAR_SOFTMAX_SEGMENTS "
            "%s -> %s (BI-V150 has 16 SMs; 16 segments over-partitions the "
            "3D attention path).",
            _old,
            _target,
        )
    except Exception as e:
        logger.warning(
            "patch_triton_attn_segments_for_iluvatar: %s", e
        )


def select_hnd_kv_cache_layout_for_iluvatar() -> None:
    """Lay the KV cache out as contiguous HND pages on Iluvatar only.

    ixformer's paged kernels require contiguous HND
    ``[blocks, kv_heads, block_size, head_dim]``.  Two pieces are needed:

    1. ``VLLM_KV_CACHE_LAYOUT=HND`` / ``set_kv_cache_layout("HND")`` so the
       model runner picks an HND stride order instead of NHD.
    2. A **KV-first** stride order ``(1, 0, 3, 2, 4)``.  Upstream's HND order
       ``(0, 1, 3, 2, 4)`` keeps K/V interleaved under ``dim=1``, so after
       ``unbind(1)`` the HND view still has a ``*2`` block stride and is not
       contiguous -- the ixformer path silently falls back to Triton.  Putting
       the K/V axis outermost makes each of K and V a free contiguous HND
       page tensor.

    Logical shape ``(num_blocks, 2, block_size, kv_heads, head_dim)`` and the
    ``unbind(1)`` call site are unchanged, so Triton fallback still works via
    strides.  MetaX / other vendors never enter this function
    (``_is_iluvatar_platform`` gate).  Disable with
    ``VLLM_FL_ILUVATAR_KV_CACHE_HND=0``.
    """
    try:
        if not _is_iluvatar_platform():
            return

        import os

        if os.getenv("VLLM_FL_ILUVATAR_KV_CACHE_HND", "1") == "0":
            logger.info(
                "select_hnd_kv_cache_layout_for_iluvatar: disabled by "
                "VLLM_FL_ILUVATAR_KV_CACHE_HND=0."
            )
            return

        user_layout = os.getenv("VLLM_KV_CACHE_LAYOUT")
        if user_layout and user_layout != "HND":
            logger.info(
                "select_hnd_kv_cache_layout_for_iluvatar: VLLM_KV_CACHE_LAYOUT="
                "%s was set explicitly, leaving it alone.",
                user_layout,
            )
            return

        from vllm.v1.attention.backends.utils import set_kv_cache_layout
        import vllm.v1.attention.backends.triton_attn as _ta

        os.environ["VLLM_KV_CACHE_LAYOUT"] = "HND"
        set_kv_cache_layout("HND")

        # Upstream HND: (0, 1, 3, 2, 4) -> phys (nb, 2, nkv, bs, hd) -- K/V
        # interleaved, HND view after unbind is NOT contiguous.
        # Iluvatar:      (1, 0, 3, 2, 4) -> phys (2, nb, nkv, bs, hd) -- each of
        # K/V is contiguous HND after unbind(1).
        _KV_FIRST_HND_STRIDE = (1, 0, 3, 2, 4)

        def _iluvatar_kv_cache_stride_order(
            include_num_layers_dimension: bool = False,
        ):
            if include_num_layers_dimension:
                # Match upstream HND+layers intent but keep K/V outermost so
                # per-layer unbind still yields contiguous HND pages.
                # logical (nb, layers, 2, bs, nkv, hd) is not used on the
                # default non-layered path; fall back to upstream if asked.
                return (1, 4, 0, 2, 3, 5)
            return _KV_FIRST_HND_STRIDE

        _ta.TritonAttentionBackend.get_kv_cache_stride_order = staticmethod(
            _iluvatar_kv_cache_stride_order
        )
        logger.info(
            "select_hnd_kv_cache_layout_for_iluvatar: KV cache layout=HND with "
            "KV-first stride_order=%s (contiguous HND after unbind); set "
            "VLLM_FL_ILUVATAR_KV_CACHE_HND=0 to restore vLLM's default.",
            _KV_FIRST_HND_STRIDE,
        )
    except Exception as e:
        logger.warning("select_hnd_kv_cache_layout_for_iluvatar: %s", e)


def patch_triton_attn_reorder_split_for_iluvatar() -> None:
    """Let TRITON_ATTN reorder decode rows to the front and publish the split.

    vLLM already knows which requests are decoding, but ``TRITON_ATTN`` never
    opts in: ``TritonAttentionMetadataBuilder`` leaves the base class's
    ``reorder_batch_threshold = None``, so decode and chunked-prefill rows stay
    interleaved and a mixed batch can only be handed to one kernel.  FlashInfer
    and the MLA backends do opt in.

    Opting in gives two things the fused gate needs: decode rows become a
    contiguous prefix (sliceable, no gather), and
    ``split_decodes_and_prefills`` yields the boundary from
    ``query_start_loc_cpu`` on the host, so reading it costs no sync.  The
    boundary is published per step; the model's attention layers consume it.

    On by default for Iluvatar after §17.11 (once-per-step metadata).  Disable
    with ``VLLM_FL_ILUVATAR_IXFORMER_SPLIT=0``.  The builder prebuilds flashinfer
    page metadata and FA ``cu_seqlens`` once per step so the 42 attention layers
    reuse them; without that, FlagGems turns the ~20 pointwise ops into a
    host-bound mixed step (§17.10).
    """
    try:
        if not _is_iluvatar_platform():
            logger.info(
                "patch_triton_attn_reorder_split_for_iluvatar: "
                "current platform is not iluvatar, leaving batch order unchanged."
            )
            return

        import os

        if os.getenv("VLLM_FL_ILUVATAR_IXFORMER_SPLIT", "1") == "0":
            logger.info(
                "patch_triton_attn_reorder_split_for_iluvatar: disabled by "
                "VLLM_FL_ILUVATAR_IXFORMER_SPLIT=0."
            )
            return
        if os.getenv("VLLM_FL_ILUVATAR_IXFORMER_ATTN", "1") == "0":
            logger.info(
                "patch_triton_attn_reorder_split_for_iluvatar: ixformer attention "
                "is off, so the split would have no consumer; leaving order unchanged."
            )
            return

        import vllm.v1.attention.backends.triton_attn as _ta
        from vllm.v1.attention.backends.utils import split_decodes_and_prefills

        if getattr(_ta, "_iluvatar_reorder_split_patch_applied", False):
            return

        try:
            from flag_gems.runtime.backend._iluvatar.fused import (
                clear_batch_split,
                prebuild_split_meta,
            )
        except Exception as import_err:
            logger.warning(
                "patch_triton_attn_reorder_split_for_iluvatar: FlagGems fused "
                "ops lack the batch-split hooks (%s); leaving order unchanged.",
                import_err,
            )
            return

        builder = _ta.TritonAttentionMetadataBuilder
        _orig_init = builder.__init__
        _orig_build = builder.build

        def _init(self, *args, **kwargs):
            _orig_init(self, *args, **kwargs)
            self._init_reorder_batch_threshold(1)

        def _build(self, common_prefix_len, common_attn_metadata, *args, **kwargs):
            try:
                num_decodes, _, num_decode_tokens, _ = split_decodes_and_prefills(
                    common_attn_metadata, decode_threshold=1
                )
                # Once per step: flashinfer pages + FA cu_seqlens.  Rebuilding
                # them in every attention layer is what made the split path
                # host-bound under FlagGems (§17.10).
                prebuild_split_meta(
                    num_decodes,
                    num_decode_tokens,
                    common_attn_metadata.num_actual_tokens,
                    common_attn_metadata.block_table_tensor,
                    common_attn_metadata.seq_lens,
                    common_attn_metadata.query_start_loc,
                    self.block_size,
                    # Host cu_seqlens: FA shape gate reads min(q_len) with no
                    # GPU D2H (avoids per-layer aten::min + .item() sync).
                    query_start_loc_cpu=common_attn_metadata.query_start_loc_cpu,
                )
            except Exception:
                # A stale split is worse than none: the gate would slice a batch
                # on a boundary that does not belong to it.
                clear_batch_split()
            return _orig_build(
                self, common_prefix_len, common_attn_metadata, *args, **kwargs
            )

        builder.__init__ = _init
        builder.build = _build
        _ta._iluvatar_reorder_split_patch_applied = True
        logger.info(
            "patch_triton_attn_reorder_split_for_iluvatar: enabled decode-first "
            "batch reorder and once-per-step split metadata "
            "(VLLM_FL_ILUVATAR_IXFORMER_SPLIT=1)."
        )
    except Exception as e:
        logger.warning("patch_triton_attn_reorder_split_for_iluvatar: %s", e)


def patch_triton_unified_attention_ixformer_for_iluvatar() -> None:
    """Dispatch TRITON_ATTN through FlagGems-comp ixformer fused ops.

    When eligible (bf16/fp16, head 64/128, no softcap/sinks/alibi/fp8/window,
    contiguous HND paged KV):

    * ``max_seqlen_q == 1`` → ``paged_attention_decode`` (flashinfer HND)
    * pure prefill admitted by ``(n_seq, min_q, kv_len)`` shape gate
      (see FlagGems ``_fa_prefill_shape_ok``; cheap reject floor
      ``VLLM_FL_ILUVATAR_IXFORMER_PREFILL_MIN_Q``, default 256) →
      ``flash_attn_prefill``
    * **mixed** (some q_len==1) or shapes outside the gate → Triton tile64

    Ops live under
    ``flag_gems.runtime.backend._iluvatar.fused`` (FlagGems-comp).  Gate with
    ``VLLM_FL_ILUVATAR_IXFORMER_ATTN=0`` to force Triton-only.  KV layout is
    selected by ``select_hnd_kv_cache_layout_for_iluvatar``.
    """
    try:
        if not _is_iluvatar_platform():
            logger.info(
                "patch_triton_unified_attention_ixformer_for_iluvatar: "
                "current platform is not iluvatar, leaving attention unchanged."
            )
            return

        import os

        if os.getenv("VLLM_FL_ILUVATAR_IXFORMER_ATTN", "1") == "0":
            logger.info(
                "patch_triton_unified_attention_ixformer_for_iluvatar: "
                "disabled by VLLM_FL_ILUVATAR_IXFORMER_ATTN=0."
            )
            return

        import vllm.v1.attention.backends.triton_attn as _ta
        import vllm.v1.attention.ops.triton_unified_attention as _uam

        if getattr(_ta, "_iluvatar_ixformer_attn_patch_applied", False):
            return

        try:
            from flag_gems.runtime.backend._iluvatar.fused import (
                get_dispatch_stats,
                try_ixformer_unified_attention,
            )
        except Exception as import_err:
            logger.warning(
                "patch_triton_unified_attention_ixformer_for_iluvatar: "
                "FlagGems fused ixformer ops unavailable (%s); keeping Triton.",
                import_err,
            )
            return

        _orig = _ta.unified_attention
        _call_n = {"n": 0}

        def _unified_attention_ixformer(*args, **kwargs):
            _call_n["n"] += 1
            try:
                hit = try_ixformer_unified_attention(*args, **kwargs)
                if _call_n["n"] == 1 or _call_n["n"] % 256 == 0:
                    stats = get_dispatch_stats()
                    logger.warning(
                        "ixformer_attn_probe call=%d hit=%s stats=%s",
                        _call_n["n"],
                        hit,
                        stats,
                    )
                if hit:
                    return
            except Exception as e:
                logger.warning(
                    "patch_triton_unified_attention_ixformer_for_iluvatar: "
                    "ixformer path failed (%s); falling back to Triton.",
                    e,
                )
            return _orig(*args, **kwargs)

        _ta.unified_attention = _unified_attention_ixformer
        _uam.unified_attention = _unified_attention_ixformer
        _ta._iluvatar_ixformer_attn_patch_applied = True

        # Silent fallback is the failure mode that cost a whole benchmark
        # round, so make the dispatch split visible at process exit.
        import atexit

        def _report_dispatch() -> None:
            stats = get_dispatch_stats()
            if not any(stats.values()):
                return
            logger.info(
                "ixformer attention dispatch: decode=%d prefill=%d split=%d "
                "spec_mtp=%d mixed_skip=%d short_skip=%d decode_unsafe_skip=%d "
                "split_unsafe_skip=%d "
                "fallback_unsupported=%d fallback_non_hnd_layout=%d",
                stats.get("decode", 0),
                stats.get("prefill", 0),
                stats.get("split", 0),
                stats.get("spec_mtp", 0),
                stats.get("mixed_skip", 0),
                stats.get("short_skip", 0),
                stats.get("decode_unsafe_skip", 0),
                stats.get("split_unsafe_skip", 0),
                stats.get("unsupported", 0),
                stats.get("layout", 0),
            )

        atexit.register(_report_dispatch)
        logger.info(
            "patch_triton_unified_attention_ixformer_for_iluvatar: enabled "
            "decode + large pure-prefill FA; mixed/short chunks stay on "
            "Triton tile64. Set VLLM_FL_ILUVATAR_IXFORMER_ATTN=0 to disable."
        )
    except Exception as e:
        logger.warning(
            "patch_triton_unified_attention_ixformer_for_iluvatar: %s", e
        )


def patch_gpu_mem_util_cli_for_iluvatar() -> None:
    """Add ``--iluvatar-gpu-memory-utilization`` to ``vllm serve``.

    The 16k decode concurrency is pinned by the KV pool: at the eval-harness
    default ``--gpu-memory-utilization 0.85`` the pool holds 522,576 tokens,
    i.e. 30.02 concurrent 17,408-token sequences, and 128 prompts take 5
    waves.  At 0.90 the pool holds 562,528 tokens (32.31 sequences) and the
    same workload takes 4 waves: measured 5241 tok/s vs 5151 (+1.75%,
    §17.45).  Shrinking the CUDA-graph reservation cannot buy those 34,480
    tokens (§17.45.3: the spare knob moves the pool by 768 tokens), so the
    only way to reach it is a higher utilisation.

    The eval harness hardcodes ``--gpu-memory-utilization 0.85``, so this adds
    an explicit opt-in flag that overrides it *only when passed*.  Nothing is
    changed by default: unset, behaviour is byte-identical to stock vLLM.
    """
    try:
        if not _is_iluvatar_platform():
            return

        import sys

        from vllm.engine.arg_utils import EngineArgs

        if getattr(EngineArgs, "_iluvatar_util_cli_patch_applied", False):
            return

        opt_name = "--iluvatar-gpu-memory-utilization"
        dest = "iluvatar_gpu_memory_utilization"

        # ``vllm serve`` builds its argument parser *before* general plugins are
        # loaded, so registering the option via ``EngineArgs.add_cli_args`` is
        # too late there (verified: the wrapper is installed but never called,
        # and argparse rejects the flag as unrecognised).  Consume it from
        # ``sys.argv`` instead, which works no matter which order the plugin
        # loads in, and still register it on the parser for the cases where the
        # plugin does load first (offline ``LLM(...)``, ``--help``).
        override = None
        for i, argv in enumerate(sys.argv):
            if argv == opt_name and i + 1 < len(sys.argv):
                try:
                    override = float(sys.argv[i + 1])
                except ValueError:
                    logger.warning(
                        "%s expects a float, got %r; ignoring.",
                        opt_name,
                        sys.argv[i + 1],
                    )
                del sys.argv[i : i + 2]
                break
            if argv.startswith(opt_name + "="):
                try:
                    override = float(argv.split("=", 1)[1])
                except ValueError:
                    logger.warning(
                        "%s expects a float, got %r; ignoring.",
                        opt_name,
                        argv.split("=", 1)[1],
                    )
                del sys.argv[i]
                break

        _orig_add_cli_args = EngineArgs.add_cli_args
        _orig_from_cli_args = EngineArgs.from_cli_args.__func__

        def add_cli_args(parser):
            parser = _orig_add_cli_args(parser)
            try:
                import argparse

                parser.add_argument(
                    opt_name,
                    dest=dest,
                    type=float,
                    # SUPPRESS so the key only exists when the flag is actually
                    # passed.  Serve rebuilds a parser to diff the Namespace
                    # against the real one (get_non_default_args); a phantom
                    # key with default=None makes that diff raise
                    # AttributeError on the real Namespace.
                    default=argparse.SUPPRESS,
                    help=(
                        "Iluvatar only: override --gpu-memory-utilization for "
                        "this run. A larger KV pool raises the 16k decode "
                        "concurrency (0.85 -> 30 sequences / 5 waves, "
                        "0.90 -> 32 / 4 waves). Unset keeps the CLI value."
                    ),
                )
            except Exception as e:
                # Already registered (double import / repeated parser build).
                # Log rather than swallow: a silent failure here means the flag
                # is simply absent at runtime, which is impossible to debug
                # from a serve log later.
                logger.warning(
                    "patch_gpu_mem_util_cli_for_iluvatar: could not register "
                    "%s (%s: %s); falling back to sys.argv parsing only.",
                    opt_name,
                    type(e).__name__,
                    e,
                )
            return parser

        def from_cli_args(cls, args):
            engine_args = _orig_from_cli_args(cls, args)
            # ``args`` carries the value when the plugin loaded early enough for
            # add_cli_args to have run; otherwise fall back to the value taken
            # out of sys.argv at import time.
            value = getattr(args, dest, None)
            if value is None:
                value = override
            if value is None:
                return engine_args
            if not 0.0 < value < 1.0:
                logger.warning(
                    "%s=%s is outside (0, 1); ignoring.", opt_name, value
                )
                return engine_args
            previous = engine_args.gpu_memory_utilization
            if abs(value - previous) > 1e-9:
                logger.warning(
                    "%s=%s overrides --gpu-memory-utilization=%s.",
                    opt_name,
                    value,
                    previous,
                )
            engine_args.gpu_memory_utilization = value
            return engine_args

        EngineArgs.add_cli_args = staticmethod(add_cli_args)
        EngineArgs.from_cli_args = classmethod(from_cli_args)
        EngineArgs._iluvatar_util_cli_patch_applied = True
        logger.info(
            "patch_gpu_mem_util_cli_for_iluvatar: %s registered "
            "(requested=%s; default unset -> stock --gpu-memory-utilization).",
            opt_name,
            override,
        )
    except Exception as e:
        logger.warning("patch_gpu_mem_util_cli_for_iluvatar: %s", e)


def patch_openai_api_mbt_for_iluvatar() -> None:
    """Raise the OpenAI-server default ``max_num_batched_tokens`` on Iluvatar.

    vLLM gives devices under 70 GB ``max_num_batched_tokens=2048`` for the API
    server, so a 4096-token prompt is prefilled in two chunks, and this vLLM
    rejects concurrent partial prefills.  Only the default is raised (an
    explicit ``--max-num-batched-tokens`` still wins), and only when it is
    smaller than the target.  Target from
    ``VLLM_FL_ILUVATAR_MAX_NUM_BATCHED_TOKENS`` (default 4096); ``0`` disables.
    """
    try:
        if not _is_iluvatar_platform():
            return

        import os

        target = int(os.getenv("VLLM_FL_ILUVATAR_MAX_NUM_BATCHED_TOKENS", "4096"))
        if target <= 0:
            return

        from vllm.engine.arg_utils import EngineArgs
        from vllm.usage.usage_lib import UsageContext

        if getattr(EngineArgs, "_iluvatar_mbt_patch_applied", False):
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
        EngineArgs._iluvatar_mbt_patch_applied = True
        logger.info(
            "patch_openai_api_mbt_for_iluvatar: OpenAI-server default "
            "max_num_batched_tokens -> %d.",
            target,
        )
    except Exception as e:
        logger.warning("patch_openai_api_mbt_for_iluvatar: %s", e)


def patch_async_scheduling_default_for_iluvatar() -> None:
    """Keep async scheduling on by default on Iluvatar (no serve-arg change).

    vLLM already auto-enables when ``async_scheduling is None`` and the stack
    is compatible.  We pin the EngineArgs default to ``True`` so an unset
    flag stays on.  ``VLLM_FL_ILUVATAR_ASYNC_SCHEDULING=0`` restores auto
    (``None``).  Explicit ``--async-scheduling`` / ``--no-async-scheduling``
    still win.  CPU n-gram speculative decode is incompatible; the n-gram
    opt-in patch clears this default when it enables ngram.
    """
    try:
        if not _is_iluvatar_platform():
            return

        import os

        from vllm.engine.arg_utils import EngineArgs

        if getattr(EngineArgs, "_iluvatar_async_sched_patch_applied", False):
            return

        if os.getenv("VLLM_FL_ILUVATAR_ASYNC_SCHEDULING", "1") == "0":
            logger.info(
                "patch_async_scheduling_default_for_iluvatar: leave auto "
                "(VLLM_FL_ILUVATAR_ASYNC_SCHEDULING=0)."
            )
            EngineArgs._iluvatar_async_sched_patch_applied = True
            return

        _orig = EngineArgs.create_engine_config

        def create_engine_config(self, *args, **kwargs):
            # Only fill the unset default; respect an explicit CLI bool.
            if getattr(self, "async_scheduling", None) is None:
                self.async_scheduling = True
            return _orig(self, *args, **kwargs)

        EngineArgs.create_engine_config = create_engine_config
        EngineArgs._iluvatar_async_sched_patch_applied = True
        logger.info(
            "patch_async_scheduling_default_for_iluvatar: "
            "async_scheduling default True "
            "(set VLLM_FL_ILUVATAR_ASYNC_SCHEDULING=0 for vLLM auto)."
        )
    except Exception as e:
        logger.warning("patch_async_scheduling_default_for_iluvatar: %s", e)


def patch_ngram_speculative_for_iluvatar() -> None:
    """Opt-in n-gram speculative decoding without a serve-arg change.

    Off by default: random-bench accept rate is too low and CPU n-gram
    disables async scheduling, so e2e throughput regresses vs bf16 baseline.
    Set ``VLLM_FL_ILUVATAR_NGRAM=1`` to enable (k=1, window 5). Verification
    for uniform q=2 goes through ``paged_attention_mtp``. An explicit
    speculative config from the caller is left alone.
    """
    try:
        if not _is_iluvatar_platform():
            return

        import os

        from vllm.engine.arg_utils import EngineArgs

        if getattr(EngineArgs, "_iluvatar_ngram_patch_applied", False):
            return
        _orig = EngineArgs.create_speculative_config

        def create_speculative_config(
            self, target_model_config, target_parallel_config
        ):
            cfg = _orig(self, target_model_config, target_parallel_config)
            if cfg is not None:
                return cfg
            if os.getenv("VLLM_FL_ILUVATAR_NGRAM", "0") != "1":
                return None
            from vllm.config import SpeculativeConfig

            k = int(os.getenv("VLLM_FL_ILUVATAR_NGRAM_K", "1"))
            n = int(os.getenv("VLLM_FL_ILUVATAR_NGRAM_N", "5"))
            # CPU ngram + explicit async_scheduling=True hard-fails in vLLM.
            if getattr(self, "async_scheduling", None) is not False:
                self.async_scheduling = False
            logger.info(
                "patch_ngram_speculative_for_iluvatar: ngram k=%d window=%d. "
                "async scheduling forced off (CPU ngram).",
                k,
                n,
            )
            return SpeculativeConfig(
                method="ngram",
                num_speculative_tokens=k,
                prompt_lookup_min=n,
                prompt_lookup_max=n,
                target_model_config=target_model_config,
                target_parallel_config=target_parallel_config,
            )

        EngineArgs.create_speculative_config = create_speculative_config
        EngineArgs._iluvatar_ngram_patch_applied = True
        if os.getenv("VLLM_FL_ILUVATAR_NGRAM", "0") != "1":
            logger.info(
                "patch_ngram_speculative_for_iluvatar: hook installed, "
                "default off (set VLLM_FL_ILUVATAR_NGRAM=1 to enable)"
            )
    except Exception as e:
        logger.warning("patch_ngram_speculative_for_iluvatar: %s", e)


def patch_scheduler_kv_admit_for_iluvatar() -> None:
    """Opt-in scheduler knobs for chunked-prefill / KV admission (no serve CLI).

    All unset by default (no behavior change).  Set any of:

    * ``VLLM_FL_ILUVATAR_MAX_NUM_PARTIAL_PREFILLS`` — concurrent partial prefills
    * ``VLLM_FL_ILUVATAR_MAX_LONG_PARTIAL_PREFILLS`` — concurrent *long* partials
    * ``VLLM_FL_ILUVATAR_LONG_PREFILL_TOKEN_THRESHOLD`` — long-prompt cutoff
    * ``VLLM_FL_ILUVATAR_SCHED_WATERMARK`` — fraction of KV blocks kept free
      when admitting (``0``..``<1``; empty = leave default ``0.0``)
    * ``VLLM_FL_ILUVATAR_RESERVE_FULL_ISL`` — ``0``/``1`` toggle
      ``scheduler_reserve_full_isl`` (admit only if full ISL fits)

    Applied in ``EngineArgs.create_engine_config`` after other Iluvatar hooks.
    Explicit CLI scheduler flags still win when already set on EngineArgs.
    """
    try:
        if not _is_iluvatar_platform():
            return

        import os

        keys = (
            "VLLM_FL_ILUVATAR_MAX_NUM_PARTIAL_PREFILLS",
            "VLLM_FL_ILUVATAR_MAX_LONG_PARTIAL_PREFILLS",
            "VLLM_FL_ILUVATAR_LONG_PREFILL_TOKEN_THRESHOLD",
            "VLLM_FL_ILUVATAR_SCHED_WATERMARK",
            "VLLM_FL_ILUVATAR_RESERVE_FULL_ISL",
        )
        if not any(os.getenv(k) not in (None, "") for k in keys):
            return

        from vllm.engine.arg_utils import EngineArgs

        if getattr(EngineArgs, "_iluvatar_sched_kv_admit_patch_applied", False):
            return

        _orig = EngineArgs.create_engine_config

        def create_engine_config(self, *args, **kwargs):
            cfg = _orig(self, *args, **kwargs)
            sc = cfg.scheduler_config
            applied = []

            def _int_env(name: str):
                raw = os.getenv(name)
                if raw in (None, ""):
                    return None
                return int(raw)

            def _float_env(name: str):
                raw = os.getenv(name)
                if raw in (None, ""):
                    return None
                return float(raw)

            v = _int_env("VLLM_FL_ILUVATAR_MAX_NUM_PARTIAL_PREFILLS")
            if v is not None and v >= 1:
                sc.max_num_partial_prefills = v
                applied.append(f"max_num_partial_prefills={v}")

            v = _int_env("VLLM_FL_ILUVATAR_MAX_LONG_PARTIAL_PREFILLS")
            if v is not None and v >= 1:
                sc.max_long_partial_prefills = v
                applied.append(f"max_long_partial_prefills={v}")

            v = _int_env("VLLM_FL_ILUVATAR_LONG_PREFILL_TOKEN_THRESHOLD")
            if v is not None and v >= 0:
                sc.long_prefill_token_threshold = v
                applied.append(f"long_prefill_token_threshold={v}")

            w = _float_env("VLLM_FL_ILUVATAR_SCHED_WATERMARK")
            if w is not None and 0.0 <= w < 1.0:
                sc.watermark = w
                applied.append(f"watermark={w}")

            raw = os.getenv("VLLM_FL_ILUVATAR_RESERVE_FULL_ISL")
            if raw in ("0", "1"):
                sc.scheduler_reserve_full_isl = raw == "1"
                applied.append(f"reserve_full_isl={sc.scheduler_reserve_full_isl}")

            if applied:
                logger.info(
                    "patch_scheduler_kv_admit_for_iluvatar: %s",
                    ", ".join(applied),
                )
            return cfg

        EngineArgs.create_engine_config = create_engine_config
        EngineArgs._iluvatar_sched_kv_admit_patch_applied = True
    except Exception as e:
        logger.warning("patch_scheduler_kv_admit_for_iluvatar: %s", e)


def patch_fp8_kv_cache_for_iluvatar() -> None:
    """Opt-in FP8 KV *storage* with software quant/dequant (no native FP8 ALU).

    BI-V150 reports compute capability 7.1 and has no fp8e4nv tensor cores.
    Torch can still allocate ``float8_e4m3fn``; Triton reshape-and-cache and
    ``unified_attention`` quantize on store and dequantize to bf16 on load.
    That halves KV bytes (~2× token pool) at the cost of falling off the
    ixformer FA path (feature gate rejects quantized KV).

    Off by default (``VLLM_FL_ILUVATAR_KV_FP8=0`` / unset): no cache_dtype
    change and no FP8 KV patches.  Set ``VLLM_FL_ILUVATAR_KV_FP8=1`` before
    serve starts to force ``cache_config.cache_dtype=fp8_e4m3`` without a
    serve-arg change.  Optional ``VLLM_FL_ILUVATAR_KV_FP8_CALC_SCALES=1``
    (default on when FP8 is on) enables one-shot dynamic ``k_scale``/
    ``v_scale`` (vLLM's deprecated ``calculate_kv_scales``).  Explicit
    ``--kv-cache-dtype`` from the caller is left alone.
    """
    try:
        if not _is_iluvatar_platform():
            return

        import os

        if os.getenv("VLLM_FL_ILUVATAR_KV_FP8", "0") != "1":
            logger.info(
                "patch_fp8_kv_cache_for_iluvatar: skipped "
                "(VLLM_FL_ILUVATAR_KV_FP8 default 0; set =1 to enable)."
            )
            return

        from vllm.engine.arg_utils import EngineArgs

        # Allow Triton reshape-and-cache to accept fp8_* on this device.
        # Upstream gates on SM89 / native fp8e4nv; we only need the software
        # cast path (probe: store+dequant roundtrip works on BI-V150).
        try:
            import vllm.v1.attention.ops.triton_reshape_and_cache_flash as _rac

            _orig_ok = _rac._is_supported_kv_cache_dtype

            def _iluvatar_kv_dtype_ok(kv_cache_dtype: str) -> bool:
                if _orig_ok(kv_cache_dtype):
                    return True
                if not isinstance(kv_cache_dtype, str):
                    return False
                # Software FP8 storage only; per-token-head INT8 untested here.
                return kv_cache_dtype in ("fp8", "fp8_e4m3", "fp8_e5m2")

            _rac._is_supported_kv_cache_dtype = _iluvatar_kv_dtype_ok
            logger.info(
                "patch_fp8_kv_cache_for_iluvatar: Triton reshape-and-cache "
                "allows software FP8 KV on Iluvatar (no native fp8e4nv)."
            )
        except Exception as e:
            logger.warning(
                "patch_fp8_kv_cache_for_iluvatar: reshape-and-cache gate: %s", e
            )

        # q/k/v_range are CPU float32 tensors; FlagGems true_divide(CUDA, CPU)
        # dies with "Pointer argument cannot be accessed from Triton".
        # Keep absmax/div on-device via aten (no D2H for the math). The
        # ``_q/k/v_scale_float`` mirrors still need a one-shot ``.item()`` —
        # calc_kv_scales runs once per layer then disables itself, not hot path.
        try:
            from vllm.model_executor.layers.attention.attention import Attention

            if not getattr(Attention, "_iluvatar_calc_kv_scales_patched", False):
                def _calc_kv_scales_device(self, query, key, value):
                    def _apply(dst, t, rng):
                        if torch.is_tensor(rng):
                            if rng.device != t.device or rng.dtype != torch.float32:
                                rng = rng.to(
                                    device=t.device, dtype=torch.float32
                                )
                        else:
                            rng = torch.tensor(
                                float(rng), device=t.device, dtype=torch.float32
                            )
                        amax = torch.ops.aten.abs.default(t).amax()
                        dst.copy_(torch.ops.aten.div.Tensor(amax, rng))

                    _apply(self._q_scale, query, self.q_range)
                    _apply(self._k_scale, key, self.k_range)
                    _apply(self._v_scale, value, self.v_range)
                    self._q_scale_float = float(self._q_scale.item())
                    self._k_scale_float = float(self._k_scale.item())
                    self._v_scale_float = float(self._v_scale.item())
                    self.calculate_kv_scales = False

                Attention.calc_kv_scales = _calc_kv_scales_device
                Attention._iluvatar_calc_kv_scales_patched = True
                logger.info(
                    "patch_fp8_kv_cache_for_iluvatar: calc_kv_scales on-device "
                    "aten absmax/div (range moved to CUDA; one-shot .item() "
                    "only for scale_float mirrors)."
                )
        except Exception as e:
            logger.warning(
                "patch_fp8_kv_cache_for_iluvatar: calc_kv_scales patch: %s", e
            )

        if getattr(EngineArgs, "_iluvatar_kv_fp8_patch_applied", False):
            return

        _orig = EngineArgs.create_engine_config

        def create_engine_config(self, *args, **kwargs):
            cfg = _orig(self, *args, **kwargs)
            # Respect an explicit non-auto dtype from CLI / caller.
            cur = getattr(cfg.cache_config, "cache_dtype", "auto")
            if cur not in (None, "auto"):
                logger.info(
                    "patch_fp8_kv_cache_for_iluvatar: leaving cache_dtype=%s "
                    "(explicit; VLLM_FL_ILUVATAR_KV_FP8 ignored).",
                    cur,
                )
                return cfg
            cfg.cache_config.cache_dtype = "fp8_e4m3"
            # One-shot absmax scales beat scale=1.0 (heavy clamp). Default on.
            if os.getenv("VLLM_FL_ILUVATAR_KV_FP8_CALC_SCALES", "1") == "1":
                cfg.cache_config.calculate_kv_scales = True
            logger.info(
                "patch_fp8_kv_cache_for_iluvatar: cache_dtype=fp8_e4m3 "
                "calculate_kv_scales=%s (storage-only; attn dequants to bf16; "
                "ixformer FA disabled for quantized KV).",
                cfg.cache_config.calculate_kv_scales,
            )
            return cfg

        EngineArgs.create_engine_config = create_engine_config
        EngineArgs._iluvatar_kv_fp8_patch_applied = True
    except Exception as e:
        logger.warning("patch_fp8_kv_cache_for_iluvatar: %s", e)


def patch_cudagraph_capture_sizes_for_iluvatar() -> None:
    """Drop decode CUDA-graph sizes above 64 except a single spare (72).

    Official benches use ``--max-concurrency 64``.  On this 32 GB card the 16k
    case is KV-bound at ~30 concurrent sequences, so the 24 capture sizes from
    80..256 never run but still inflate the CUDA-graph memory estimate and
    shrink the KV pool.  Keep every default size ``<= 64``, plus one size
    ``72`` so a short burst above 64 still hits a graph.  Override the spare
    with ``VLLM_FL_ILUVATAR_CUDAGRAPH_SPARE`` (default 72); ``0`` disables.
    """
    try:
        if not _is_iluvatar_platform():
            return

        import os

        spare = int(os.getenv("VLLM_FL_ILUVATAR_CUDAGRAPH_SPARE", "72"))
        if spare <= 0:
            return

        from vllm.config.vllm import VllmConfig

        if getattr(VllmConfig, "_iluvatar_cg_sizes_patch_applied", False):
            return
        _orig = VllmConfig._set_cudagraph_sizes

        def _set_cudagraph_sizes(self):
            _orig(self)
            sizes = self.compilation_config.cudagraph_capture_sizes
            if not sizes:
                return
            kept = [s for s in sizes if s <= 64]
            # Only add the spare when the original list would have covered it
            # (or anything larger); otherwise leave a smaller max alone.
            if spare not in kept and any(s >= spare for s in sizes):
                kept.append(spare)
            # Speculative verify schedules (1 + k) queries per request.
            # Official concurrency is 64, so the hot decode graph is
            # 64 * (1 + k) tokens. The trim above drops that size.
            n_spec = int(getattr(self, "num_speculative_tokens", 0) or 0)
            if n_spec > 0:
                spec_tokens = 64 * (1 + n_spec)
                if spec_tokens not in kept and any(s >= spec_tokens for s in sizes):
                    kept.append(spec_tokens)
            kept = sorted(set(kept))
            if kept == list(sizes):
                return
            self.compilation_config.cudagraph_capture_sizes = kept
            self.compilation_config.max_cudagraph_capture_size = kept[-1]
            self.compilation_config.post_init_cudagraph_sizes()
            logger.info(
                "patch_cudagraph_capture_sizes_for_iluvatar: "
                "cudagraph_capture_sizes %s -> %s (spare=%d).",
                sizes,
                kept,
                spare,
            )

        VllmConfig._set_cudagraph_sizes = _set_cudagraph_sizes
        VllmConfig._iluvatar_cg_sizes_patch_applied = True
    except Exception as e:
        logger.warning("patch_cudagraph_capture_sizes_for_iluvatar: %s", e)


patch_triton_attn_segments_for_iluvatar()
patch_triton_unified_attention_prefill_for_iluvatar()
select_hnd_kv_cache_layout_for_iluvatar()
patch_triton_attn_reorder_split_for_iluvatar()
patch_triton_unified_attention_ixformer_for_iluvatar()
patch_openai_api_mbt_for_iluvatar()
patch_gpu_mem_util_cli_for_iluvatar()
patch_async_scheduling_default_for_iluvatar()
patch_ngram_speculative_for_iluvatar()
patch_cudagraph_capture_sizes_for_iluvatar()
patch_scheduler_kv_admit_for_iluvatar()
patch_fp8_kv_cache_for_iluvatar()
from vllm_fl.dispatch.backends.vendor.iluvatar.static_int8 import (
    patch_static_int8_linear_for_iluvatar,
)

patch_static_int8_linear_for_iluvatar()


class IluvatarBackend(Backend):
    """
    Iluvatar backend for operator implementations.

    This backend uses Iluvatar libraries to provide high-performance
    operator implementations for Iluvatar GPUs.
    """

    _available: Optional[bool] = None

    @property
    def name(self) -> str:
        return "iluvatar"

    @property
    def vendor(self) -> Optional[str]:
        return "iluvatar"

    def is_available(self) -> bool:
        """
        Check if Iluvatar hardware and libraries are available.

        This method uses the platform's vendor information to determine
        if the device is an Iluvatar GPU.
        """
        if IluvatarBackend._available is None:
            try:
                from vllm.platforms import current_platform
                # Iluvatar GPUs should be detected via vendor_name
                if hasattr(current_platform, 'vendor_name') and current_platform.vendor_name == "iluvatar":
                    IluvatarBackend._available = True
                else:
                    # Fallback: check if CUDA is available with iluvatar device
                    if torch.cuda.is_available():
                        # Try to detect Iluvatar GPU
                        # Iluvatar GPUs typically expose CUDA-compatible interface
                        # We can check device name if available
                        device_name = torch.cuda.get_device_name(0)
                        if "iluvatar" in device_name.lower():
                            IluvatarBackend._available = True
                        else:
                            IluvatarBackend._available = False

                    else:
                        IluvatarBackend._available = False
            except Exception:
                IluvatarBackend._available = False
        return IluvatarBackend._available

    # ==================== Operator Implementations ====================

    def silu_and_mul(self, obj, x: torch.Tensor) -> torch.Tensor:
        """
        SiLU activation followed by element-wise multiplication.

        Args:
            obj: The calling obj (for interface consistency)
            x: Input tensor of shape [..., 2*d]

        Returns:
            Output tensor of shape [..., d]
        """
        from .impl.activation import silu_and_mul_iluvatar

        return silu_and_mul_iluvatar(obj, x)

    def rms_norm(
        self,
        obj,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        RMS normalization.

        Args:
            obj: The calling obj (e.g., RMSNorm layer)
            x: Input tensor
            residual: Optional residual tensor

        Returns:
            Normalized tensor, or tuple of (normalized, residual) if residual is provided
        """
        from .impl.normalization import rms_norm_iluvatar

        return rms_norm_iluvatar(obj, x, residual)

    def rotary_embedding(
        self,
        obj,
        query: torch.Tensor,
        key: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        position_ids: torch.Tensor,
        rotary_interleaved: bool = False,
        inplace: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary position embedding.

        Args:
            obj: The calling obj (for interface consistency)
            query: Query tensor
            key: Key tensor
            cos: Cosine cache
            sin: Sine cache
            position_ids: Position indices
            rotary_interleaved: Whether to use interleaved rotary
            inplace: Whether to modify tensors in-place

        Returns:
            Tuple of (embedded_query, embedded_key)
        """
        from .impl.rotary import rotary_embedding_iluvatar

        return rotary_embedding_iluvatar(
            obj,
            query,
            key,
            cos,
            sin,
            position_ids,
            rotary_interleaved=rotary_interleaved,
            inplace=inplace,
        )

    def attention_backend(self, use_mla: bool = False, use_sparse: bool = False) -> str:
        """
        Get the attention backend class path for Iluvatar.

        Args:
            use_mla: Whether to use Multi-head Latent Attention (MLA)
            use_sparse: Whether to use Deepseek Sparse Attention (DSA)

        Returns:
            Fully qualified class path string
        """
        from vllm.v1.attention.backends.registry import AttentionBackendEnum

        if use_mla:
            if use_sparse:
                return AttentionBackendEnum.FLASHMLA_SPARSE.get_path()
            return AttentionBackendEnum.FLASHMLA.get_path()

        # flash_attn is not available on iluvatar. Use TRITON_ATTN (the vllm
        # default). The tl.make_tensor_descriptor stub and perf model patches
        # are already applied at module level (above), so triton JIT's
        # DependencyFinder can hash the kernel without AttributeError on
        # triton < 3.3.  The kernel itself uses USE_TD=False at runtime, so
        # the stub is never called.
        return AttentionBackendEnum.TRITON_ATTN.get_path()
