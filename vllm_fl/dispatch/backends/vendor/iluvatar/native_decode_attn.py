# Copyright (c) 2026 BAAI. All rights reserved.
"""Native split-KV decode attention for Iluvatar BI-V150 (NOT on by default).

Status: correct, validated, and slower than the kernel it would replace, so it
stays behind ``VLLM_FL_ILUVATAR_NATIVE_DECODE=1``.

What it is: a hand-written replacement for the decode step's attention, in
``decode_attn.h`` -- the same kernel the tuning harness in
``experiments/attn_kernel`` compiles and checks against an fp32 reference.
Numerics are good: worst relative error 1.9e-03 over nine shapes from 1x256 to
130x2048, no element over 5% of its row's peak.

Why it is not the default.  Measured on the same machine, same shapes, same
session (503.3 MB moved in the 16k case):

    shape        vendor FA        Triton FD        this kernel
    30 x 16384   1.617 ms / 311   0.816 ms / 617   1.522 ms / 331   GB/s
    64 x 16384   --               1.671 ms / 642   (declines)
    32 x 4096    --               0.231 ms / 581   0.443 ms / 303

Triton FlashDecoding is the incumbent in the delivered configuration
(``VLLM_FL_ILUVATAR_DECODE_TRITON=1``) and it is 1.9x this kernel.  The vendor
kernel this was originally aimed at is both slower *and* numerically broken at
few sequences and long context -- at 30 x 16384 it lands 2.327 relative error
against fp32 where this kernel lands 4.8e-03 -- but vLLM never calls it on that
shape, because FlagGems routes around it (``_decode_is_unsafe``) and the padded
fallback keeps the batch at 32.

The 1.9x is instruction issue, not bandwidth.  Per 16-position page this kernel
issues roughly 216 wavefront instructions -- 8 global loads for K, 16 shared
stores plus 8 shared loads to stage it, 32 two-byte gathers for V, 16 MMAs, 32
shuffles plus ~40 ALU for the online softmax -- for 8 KB moved.  At a 64-lane
wavefront issuing over two cycles that is close to the issue floor, and it
measures 2.75 cycles per instruction, i.e. ~73% issue efficiency.  Memory alone
with the same pattern reaches 598 GB/s (``probe_decode_split.cu``), so what is
left is the instruction stream.  Triton gets the same bytes in about half the
instructions by taking 64-128 positions per tile instead of 16, which amortises
the softmax reduction and lets V arrive as wide vectors.

The catch is that the prize is small: decode attention is 1.671 ms of the 84.0 ms
16k decode step (2.0%) and 0.231 ms of the 46 ms 4k step (0.5%), and Triton is
already at 78% of peak bandwidth there.  Making decode attention free would move
16k from 5412 to ~5524 tok/s.

Env: ``VLLM_FL_ILUVATAR_NATIVE_DECODE=1`` turns the hook on.  Even then it only
takes shapes with at most 64 work units, where the vendor kernel starves.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

PAGE = 16
HD = 128
# Warps per CTA.  8 was the best of {4, 8, 16} at 16 warps/SM.
W = 8
# The probe's sweet spot: 480 warps = 60 units x 8 splits on the 16k shape.
TARGET_WARPS = 480

_EXT_NAME = "fl_native_decode_attn_v1"
_CUDA_MOD = None
_CUDA_FAIL = False
_BUFS: dict[tuple, tuple] = {}
_SIG = None
_HITS = {"n": 0}
_MISS: dict[str, int] = {}


def enabled() -> bool:
    # Off by default: this kernel is 1.9x slower than the Triton FlashDecoding path
    # that the delivered config already uses.  See the module docstring.
    return os.getenv("VLLM_FL_ILUVATAR_NATIVE_DECODE", "0").strip().lower() not in (
        "0",
        "false",
        "off",
        "no",
        "",
    )


def _clear_stale_lock(ext_name: str) -> None:
    try:
        from torch.utils.cpp_extension import _get_build_directory

        lock = Path(_get_build_directory(ext_name, False)) / "lock"
        if lock.exists():
            lock.unlink()
    except Exception:
        pass


def _cuda_mod():
    global _CUDA_MOD, _CUDA_FAIL
    if _CUDA_FAIL:
        return None
    if _CUDA_MOD is not None:
        return _CUDA_MOD
    here = Path(__file__).resolve().parent
    # decode_attn.h sits beside this file: it is the one copy of the kernel, shared
    # with the harness in experiments/attn_kernel (which compiles it from here).
    try:
        from torch.utils.cpp_extension import load

        _clear_stale_lock(_EXT_NAME)
        logger.info("native_decode_attn: building %s ...", _EXT_NAME)
        _CUDA_MOD = load(
            name=_EXT_NAME,
            sources=[str(here / "native_decode_attn.cpp"), str(here / "native_decode_attn.cu")],
            extra_include_paths=[str(here)],
            extra_cuda_cflags=["-O3", "-x", "ivcore", "-DDEC_MINB=2"],
            extra_cflags=["-O3"],
            verbose=False,
        )
        logger.info("native_decode_attn: %s ready", _EXT_NAME)
    except Exception as e:
        logger.warning("native_decode_attn build failed: %s", e)
        _CUDA_FAIL = True
        return None
    return _CUDA_MOD


def _bufs(rows: int, nkv: int, nh: int, split: int, device, maxblk: int):
    key = (rows, nkv, nh, split, device.index, maxblk)
    b = _BUFS.get(key)
    if b is None:
        nhk = nh // nkv
        n = rows * nkv * split * nhk
        b = (
            torch.empty((n, HD), dtype=torch.float32, device=device),
            torch.empty(n, dtype=torch.float32, device=device),
            torch.empty(n, dtype=torch.float32, device=device),
            torch.arange(rows, dtype=torch.int32, device=device),
        )
        _BUFS[key] = b
    return b


def _signature_names():
    global _SIG
    if _SIG is None:
        try:
            import inspect

            import vllm.v1.attention.ops.triton_unified_attention as _uam

            fn = getattr(_uam, "_iluvatar_stock_unified_attention", None) or _uam.unified_attention
            _SIG = list(inspect.signature(fn).parameters)
        except Exception:
            _SIG = []
    return _SIG


def _bind(args, kwargs):
    if not args:
        return kwargs
    names = _signature_names()
    if not names:
        return None
    d = dict(kwargs)
    for name, val in zip(names, args):
        d.setdefault(name, val)
    return d


def _bail(why: str) -> bool:
    _MISS[why] = _MISS.get(why, 0) + 1
    return False


def try_native_decode_attention(args, kwargs) -> bool:
    """Take the decode step if every shape and feature this kernel needs holds."""
    if not enabled():
        return False
    # This runs on every attention call of every layer, so the q_len check comes before
    # any binding work: prefill and spec-decode calls are the majority and must cost
    # one comparison, not a kwargs copy.
    msq = kwargs.get("max_seqlen_q")
    if msq is None and len(args) > 5:
        msq = args[5]
    if msq is None or int(msq) != 1:
        return False
    d = _bind(args, kwargs)
    if d is None:
        return _bail("unbindable-call")
    if d.get("softcap") or d.get("sinks") is not None or d.get("alibi_slopes") is not None:
        return _bail("softcap-sinks-alibi")
    if d.get("q_descale") is not None or d.get("k_descale") is not None:
        return _bail("quantized")
    if d.get("output_scale") is not None or d.get("qq_bias") is not None:
        return _bail("output-scale")
    if d.get("mm_prefix_range") is not None:
        return _bail("mm-prefix")
    if int(d.get("chunk_lookback") or -1) != -1:
        return _bail("chunk-lookback")
    win = d.get("window_size")
    if win is not None and tuple(win) not in ((-1, -1),):
        return _bail("sliding-window")
    if d.get("use_td"):
        return _bail("use-td")
    kvq = d.get("kv_quant_mode")
    if kvq is not None and int(getattr(kvq, "value", kvq)) != 0:
        return _bail("kv-quant")

    q = d.get("q")
    k = d.get("k")
    v = d.get("v")
    out = d.get("out")
    bt = d.get("block_table")
    seqused = d.get("seqused_k")
    if not all(isinstance(t, torch.Tensor) for t in (q, k, v, out, bt, seqused)):
        return _bail("non-tensor")
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16:
        return _bail("dtype")
    if q.shape[2] != HD or q.shape[1] % k.shape[1] != 0 or q.shape[1] // k.shape[1] > 8:
        return _bail("head-geometry")
    if k.shape[2] != PAGE or k.shape[3] != HD or k.stride(-1) != 1:
        return _bail("kv-layout")
    rows = q.shape[0]
    if bt.shape[0] != rows or seqused.numel() != rows:
        return _bail("rows-vs-blocktable")
    if not (q.is_contiguous() and out.is_contiguous() and k.is_contiguous()):
        return _bail("strided")
    if bt.dtype != torch.int32 or seqused.dtype != torch.int32:
        return _bail("index-dtype")

    mod = _cuda_mod()
    if mod is None:
        return False

    nkv = k.shape[1]
    nh = q.shape[1]
    maxblk = bt.shape[1]
    nunit = rows * nkv
    # Crossover.  This kernel reaches ~350 GB/s whatever the shape, because per warp
    # per page it issues 40 global and 24 shared-memory instructions for 8 KB; the
    # vendor kernel goes about 6% faster per byte once it has enough work units but
    # starves below ~64 of them (60 units measured 311 GB/s against this kernel's 332).
    # Measured, same session: 60 units 1.23x faster, 64 units 1.40x, 128 units 0.81x,
    # 260 units 0.81x.  So take the step only where the vendor is the one that starves.
    if nunit > 64:
        return False
    pages = -(-int(d.get("max_seqlen_k") or 0) // PAGE)
    # 480 warps is the count probe_decode_split.cu liked best.  On the 16k shape that
    # is 60 units x 8, and 8 was the best of {4, 6, 8, 10, 12} there by 12%.
    split = max(1, min(16, (TARGET_WARPS + nunit - 1) // max(nunit, 1)))
    if pages:
        split = min(split, max(pages, 1))
    Op, Mp, Lp, qrow = _bufs(rows, nkv, nh, split, q.device, maxblk)
    try:
        mod.native_decode(q, k, v, out, bt, seqused, qrow, Op, Mp, Lp,
                          float(d.get("softmax_scale") or 0.0), split, W)
    except Exception as e:  # a shape this kernel did not expect: keep the vendor path
        logger.warning("native_decode_attn: launch failed (%s); falling back.", e)
        return False
    _HITS["n"] += 1
    n = _HITS["n"]
    if n == 1 or n % 512 == 0:
        logger.warning(
            "native_decode_attn: rows=%d nkv=%d split=%d maxblk=%d hits=%d miss=%s",
            rows, nkv, split, maxblk, n, dict(list(_MISS.items())[:6]),
        )
    return True
