# Copyright (c) 2026 BAAI. All rights reserved.
"""ivcore u8 decode drop-in for Iluvatar BI-V150.

CUDA-graph compatible path:
  * persistent full-pool u8 workspace (fixed shape = phys KV blocks)
  * single CUDA pack kernel (no host unique / dynamic remap)
  * INT8-K + INT8-V with per-token scales (hibyte-K fails on real MiniCPM KV)

Opt-in via ``VLLM_FL_ILUVATAR_IVCORE_DECODE=1``.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

PAGE = 16
HD = 128
GROUP = 8
_EXT_NAME = "fl_ivcore_u8_decode_v44"
_CUDA_MOD = None
_CUDA_FAIL = False
# Full-pool pack workspace: (n_kv, device) -> (ku8, vu8, ksc, vsc, blocks)
_PACK: dict[tuple, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]] = {}
_MISS: dict[str, int] = {}
_DEBUG_COS_LEFT = -1  # lazy init from env
_DUMPED = False


def enabled() -> bool:
    return os.getenv("VLLM_FL_ILUVATAR_IVCORE_DECODE", "0").strip().lower() not in (
        "0",
        "false",
        "off",
        "no",
        "",
    )


def _v_mode() -> int:
    # 0=hibyte-V, 1=fp8-V, 2=int8*scale-V (default). K is always int8+scale.
    try:
        return int(os.getenv("VLLM_FL_ILUVATAR_IVCORE_VMODE", "2"))
    except ValueError:
        return 2


def _force_split() -> int | None:
    raw = os.getenv("VLLM_FL_ILUVATAR_IVCORE_SPLIT", "").strip()
    if not raw:
        return None
    try:
        s = int(raw)
        return s if 1 <= s <= 8 else None
    except ValueError:
        return None


def _debug_cos_budget() -> int:
    global _DEBUG_COS_LEFT
    if _DEBUG_COS_LEFT < 0:
        try:
            _DEBUG_COS_LEFT = int(os.getenv("VLLM_FL_ILUVATAR_IVCORE_DEBUG_COS", "0"))
        except ValueError:
            _DEBUG_COS_LEFT = 0
    return _DEBUG_COS_LEFT


def _miss(reason: str) -> bool:
    _MISS[reason] = _MISS.get(reason, 0) + 1
    n = _MISS[reason]
    if n <= 3 or n % 256 == 0:
        logger.warning("ivcore miss [%s] count=%d totals=%s", reason, n, dict(_MISS))
    return False


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
    try:
        from torch.utils.cpp_extension import load

        _clear_stale_lock(_EXT_NAME)
        logger.info("ivcore_u8_decode: building %s ...", _EXT_NAME)
        _CUDA_MOD = load(
            name=_EXT_NAME,
            sources=[
                str(here / "ivcore_u8_decode.cpp"),
                str(here / "ivcore_u8_decode.cu"),
            ],
            extra_cuda_cflags=["-O3", "-x", "ivcore"],
            extra_cflags=["-O3"],
            verbose=False,
        )
        logger.info("ivcore_u8_decode: %s ready", _EXT_NAME)
    except Exception as e:
        logger.warning("ivcore_u8_decode build failed: %s", e)
        _CUDA_FAIL = True
        return None
    return _CUDA_MOD


def usable(query: torch.Tensor, key_cache: torch.Tensor, out: torch.Tensor) -> bool:
    if query.dtype != torch.bfloat16 or out.dtype != torch.bfloat16:
        return False
    if query.dim() != 3 or key_cache.dim() != 4:
        return False
    if int(query.shape[-1]) != HD:
        return False
    n_qo = int(query.shape[1])
    if int(key_cache.shape[-1]) != HD:
        return False
    if int(key_cache.shape[1]) == PAGE:
        n_kv = int(key_cache.shape[2])
    elif int(key_cache.shape[2]) == PAGE:
        n_kv = int(key_cache.shape[1])
    else:
        return False
    if n_kv <= 0 or n_qo != n_kv * GROUP:
        return False
    return True


def _to_phys(cache: torch.Tensor) -> torch.Tensor:
    """BNDH [blocks,PAGE,n_kv,HD] or already-phys → phys [blocks,n_kv,PAGE,HD]."""
    if int(cache.shape[1]) == PAGE and int(cache.shape[-1]) == HD:
        return cache.permute(0, 2, 1, 3)
    return cache


def _pack_bufs(blocks: int, n_kv: int, device: torch.device):
    """Fixed full-pool workspace — shape never changes after first alloc (cudagraph)."""
    key = (n_kv, str(device))
    ws = _PACK.get(key)
    if ws is not None:
        ku8, vu8, ksc, vsc, nb = ws
        if nb == blocks:
            return ku8, vu8, ksc, vsc
        _PACK.pop(key, None)
        del ku8, vu8, ksc, vsc
        torch.cuda.empty_cache()
    try:
        ku8 = torch.empty(blocks, n_kv, HD, PAGE, device=device, dtype=torch.uint8)
        vu8 = torch.empty(blocks, n_kv, PAGE, HD, device=device, dtype=torch.uint8)
        ksc = torch.empty(blocks, n_kv, PAGE, device=device, dtype=torch.float32)
        vsc = torch.empty(blocks, n_kv, PAGE, device=device, dtype=torch.float32)
    except Exception as e:
        logger.warning("ivcore pack buf alloc failed blocks=%d: %s", blocks, e)
        return None
    _PACK[key] = (ku8, vu8, ksc, vsc, blocks)
    return ku8, vu8, ksc, vsc


def _pick_split(batch: int, n_kv: int, max_pages: int) -> int:
    # Cap pages: block_table is often padded to max_model_len/PAGE (e.g. 8192).
    # Using the pad width over-splits short decodes; idle splits used to read
    # uninitialized partials (fixed by zeros in the extension) and also waste.
    pages = min(int(max_pages), 512)
    work = batch * n_kv * pages
    if work <= 64 * 2 * 128:
        return 2
    if work <= 64 * 2 * 320:
        return 4
    return 8


def paged_decode_ivcore(
    out: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    seqused_k: torch.Tensor,
    softmax_scale: float,
) -> bool:
    """Drop-in decode. ``True`` = wrote ``out``. CUDA-graph safe."""
    if not enabled():
        return False
    if not usable(query, key_cache, out):
        return _miss("usable")
    mod = _cuda_mod()
    if mod is None:
        return _miss("no_cuda_mod")

    batch, n_qo, _ = (int(x) for x in query.shape)
    k_phys = _to_phys(key_cache)
    v_phys = _to_phys(value_cache)
    if int(k_phys.shape[2]) != PAGE or int(k_phys.shape[3]) != HD:
        return _miss("phys_shape")
    blocks = int(k_phys.shape[0])
    n_kv = int(k_phys.shape[1])
    if n_qo != n_kv * GROUP:
        return _miss("gqa")

    # Contiguous views — shapes fixed for a given engine (cudagraph OK).
    if not k_phys.is_contiguous():
        k_phys = k_phys.contiguous()
    if not v_phys.is_contiguous():
        v_phys = v_phys.contiguous()

    bt = block_table if block_table.dtype == torch.int32 else block_table.to(torch.int32)
    sl = seqused_k if seqused_k.dtype == torch.int32 else seqused_k.to(torch.int32)
    if not bt.is_contiguous():
        bt = bt.contiguous()
    if not sl.is_contiguous():
        sl = sl.contiguous()

    bufs = _pack_bufs(blocks, n_kv, query.device)
    if bufs is None:
        return _miss("buf_oom")
    ku8, vu8, ksc, vsc = bufs

    # Fixed-grid CUDA pack over batch rows; unused pages early-exit.
    try:
        mod.ivcore_pack_bt(k_phys, v_phys, ku8, vu8, ksc, vsc, bt, sl)
    except Exception as e:
        logger.warning("ivcore pack failed: %s", e)
        return _miss(f"pack_exc:{type(e).__name__}")

    forced = _force_split()
    split = forced if forced is not None else _pick_split(batch, n_kv, int(bt.shape[1]))
    vmode = _v_mode()
    vsc_arg = vsc if vmode == 2 else None
    try:
        result = mod.ivcore_u8_paged_decode(
            query,
            ku8,
            vu8,
            bt,
            sl,
            float(softmax_scale),
            split,
            vmode,
            ksc,
            vsc_arg,
        )
    except Exception as e:
        logger.warning("ivcore decode failed: %s", e)
        return _miss(f"decode_exc:{type(e).__name__}")

    # Eager-only FA reference cos (skip under cudagraph capture/replay Python).
    budget = _debug_cos_budget()
    if budget > 0:
        try:
            capturing = bool(torch.cuda.is_current_stream_capturing())
        except Exception:
            capturing = False
        if not capturing:
            global _DEBUG_COS_LEFT, _DUMPED
            _DEBUG_COS_LEFT = budget - 1
            try:
                from flag_gems.runtime.backend._iluvatar.fused.paged_attention_decode import (
                    paged_attention_decode,
                )

                ref = torch.empty_like(out)
                max_k = int(sl.max().item()) if sl.numel() else 0
                paged_attention_decode(
                    out=ref,
                    query=query,
                    key_cache=key_cache,
                    value_cache=value_cache,
                    block_table=block_table,
                    seqused_k=seqused_k,
                    softmax_scale=softmax_scale,
                    max_seqlen_k=max(max_k, 1),
                )
                a = result.float().reshape(-1)
                b = ref.float().reshape(-1)
                if sl.numel() == batch:
                    active = (sl > 0).nonzero(as_tuple=False).flatten()
                    if active.numel() > 0:
                        a = result[active].float().reshape(-1)
                        b = ref[active].float().reshape(-1)
                denom = (a.norm() * b.norm()).clamp_min(1e-12)
                cos = float((a * b).sum() / denom)
                if not _DUMPED:
                    _DUMPED = True
                    try:
                        import pathlib

                        d = pathlib.Path("/root/bench_results/ivcore_debug_dump")
                        d.mkdir(parents=True, exist_ok=True)
                        pages = int((int(sl[0].item()) + 15) // 16)
                        ids = bt[0, :pages].detach().cpu()
                        torch.save(
                            {
                                "q": query.detach().float().cpu(),
                                "result": result.detach().float().cpu(),
                                "ref": ref.detach().float().cpu(),
                                "sl": seqused_k.detach().cpu(),
                                "scale": float(softmax_scale),
                                "split": split,
                                "vmode": vmode,
                                "cos": cos,
                            },
                            d / "call0_v42.pt",
                        )
                        logger.warning(
                            "ivcore_debug_dump wrote %s cos=%.6f",
                            d / "call0_v42.pt",
                            cos,
                        )
                    except Exception as e:
                        logger.warning("ivcore_debug_dump failed: %s", e)
                logger.warning(
                    "ivcore_debug_cos cos=%.6f batch=%d n_qo=%d n_kv=%d "
                    "blocks=%d max_pages=%d max_seqlen=%d split=%d vmode=%d "
                    "scale=%.6f out_norm=%.4f ref_norm=%.4f left=%d",
                    cos,
                    batch,
                    n_qo,
                    n_kv,
                    blocks,
                    int(bt.shape[1]),
                    max_k,
                    split,
                    vmode,
                    float(softmax_scale),
                    float(a.norm()),
                    float(b.norm()),
                    _DEBUG_COS_LEFT,
                )
            except Exception as e:
                logger.warning("ivcore_debug_cos failed: %s", e)

    out.copy_(result)
    return True
