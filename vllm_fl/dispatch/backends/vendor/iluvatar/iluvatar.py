# Copyright (c) 2026 BAAI. All rights reserved.

"""
ILUVATAR backend implementation.

This backend provides operator implementations for Iluvatar GPUs.
Iluvatar uses a CUDA-compatible architecture.
"""

from __future__ import annotations

import logging
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

    The same wrapper has a latent 2D online-softmax bug on this Triton
    version: ``L`` starts at 1.0 even though the first tile should establish
    it.  The kernel source is rewritten with the companion fix (L starts at
    zero; sinks retain the deliberate unit contribution), and the 2D epilogue
    guards the all-masked row before dividing.

    Set ``VLLM_FL_ILUVATAR_PREFILL_TILING=0`` to restore vLLM's original
    prefill tiling and softmax initialization for controlled experiments.
    Knobs (defaults = measured best):

    * ``VLLM_FL_ILUVATAR_PREFILL_BLOCK_M``   64 / 128 / 256   (default 128)
    * ``VLLM_FL_ILUVATAR_PREFILL_TILE``      32 / 64 / 128    (default 64)
    * ``VLLM_FL_ILUVATAR_PREFILL_NUM_WARPS`` 4 / 8            (default 8)
    * ``VLLM_FL_ILUVATAR_PREFILL_NUM_STAGES`` 0 / 2 / 3       (default 2; 0 = Triton default)
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

        prefill_block_m = _env_int(
            "VLLM_FL_ILUVATAR_PREFILL_BLOCK_M", 128, (64, 128, 256)
        )
        prefill_tile = _env_int(
            "VLLM_FL_ILUVATAR_PREFILL_TILE", 64, (32, 64, 128)
        )
        prefill_num_warps = _env_int(
            "VLLM_FL_ILUVATAR_PREFILL_NUM_WARPS", 8, (4, 8)
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

        l_init_old = "    L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)"
        l_init_new = (
            "    # iluvatar_softmax_l0_fix: the first tile establishes L.\n"
            "    L = tl.full([BLOCK_M], 0.0, dtype=tl.float32)\n"
            "    if USE_SINKS:\n"
            "        L = tl.where(query_mask_1, 1.0, 0.0)"
        )
        kernel_src = _replace_once(
            kernel_src, l_init_old, l_init_new, "softmax denominator init"
        )

        div_old = "        acc = acc / L[:, None]"
        div_new = (
            "        # iluvatar_softmax_l0_fix: an empty row keeps a zero numerator.\n"
            "        acc = acc / tl.where(L == 0.0, 1.0, L)[:, None]"
        )
        kernel_src = _replace_once(
            kernel_src, div_old, div_new, "2D softmax denominator write"
        )
        kernel._unsafe_update_src(kernel_src)
        if hasattr(kernel, "device_caches"):
            kernel.device_caches.clear()

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


patch_triton_attn_segments_for_iluvatar()
patch_triton_unified_attention_prefill_for_iluvatar()


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
