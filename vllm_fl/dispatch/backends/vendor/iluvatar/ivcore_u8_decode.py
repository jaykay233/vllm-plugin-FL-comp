# Copyright (c) 2026 BAAI. All rights reserved.
"""ivcore u8 decode drop-in for Iluvatar BI-V150.

CUDA-graph compatible path:
  * persistent full-pool u8 workspace (fixed shape = phys KV blocks)
  * single CUDA pack kernel (no host unique / dynamic remap)
  * INT8-K + INT8-V with per-token scales (hibyte-K fails on real MiniCPM KV)

Default OFF (``VLLM_FL_ILUVATAR_IVCORE_DECODE=1`` to enable). Shared-workspace
last-page broke Math L3; full-pack e2e lost ~3600 tok/s vs stock Triton.
Pairs with ``VLLM_FL_ILUVATAR_KV_K_HD_MAJOR`` asym layout. CUDA-graph safe when
``warm_for_cudagraph`` runs before capture (plugin hooks this at KV init).
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
_EXT_NAME = "fl_ivcore_u8_decode_v53"
_CUDA_MOD = None
_CUDA_FAIL = False
# Full-pool pack workspace: (n_kv, device) -> (ku8, vu8, ksc, vsc, blocks)
_PACK: dict[tuple, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]] = {}
# Keys that have received a full pack (enables last-page incremental after).
_PACK_FULL_DONE: set[tuple] = set()
# True when bf16 KV may have pages not reflected in the u8 workspace
# (warm zeros, or a prefill that wrote many pages). Eager ivcore full-packs;
# FULL CUDA-graph replay runs ``sync_pack_from_kv`` once before the graph
# (last-page stays frozen in the graph for speed).
_U8_STALE: bool = False
_MISS: dict[str, int] = {}
_DEBUG_COS_LEFT = -1  # lazy init from env
_DUMPED = False
_SYNC_LOG_LEFT = 8


def _pack_last_enabled() -> bool:
    """Last-page incremental pack (DEFAULT OFF).

    Shared u8 workspace is overwritten per layer; last-page after another
    layer's pack leaves intermediate pages as the wrong layer's KV and
    corrupts decode (Math L3 garbage). Only safe with per-layer workspaces.
    """
    return os.getenv("VLLM_FL_ILUVATAR_IVCORE_PACK_LAST", "0").strip().lower() not in (
        "0",
        "false",
        "off",
        "no",
        "",
    )


def invalidate_pack_full_done() -> None:
    """Mark u8 workspace stale after prefill (or any multi-page KV write)."""
    global _U8_STALE
    _U8_STALE = True


def last_page_ready(key_cache: torch.Tensor) -> bool:
    """True when pack workspace is fresh enough for last-page incremental."""
    if not _pack_last_enabled() or not _PACK_FULL_DONE or _U8_STALE:
        return False
    try:
        if _is_k_hd_major(key_cache):
            blocks = int(key_cache.shape[0])
            n_kv = int(key_cache.shape[1])
        elif int(key_cache.shape[1]) == PAGE and int(key_cache.shape[-1]) == HD:
            blocks = int(key_cache.shape[0])
            n_kv = int(key_cache.shape[2])
        elif int(key_cache.shape[2]) == PAGE and int(key_cache.shape[-1]) == HD:
            blocks = int(key_cache.shape[0])
            n_kv = int(key_cache.shape[1])
        else:
            return False
        return (n_kv, str(key_cache.device), blocks) in _PACK_FULL_DONE
    except Exception:
        return False


def enabled() -> bool:
    # Default OFF — opt-in only; stock e2e (~6212) is BNDH+Triton without ivcore.
    return os.getenv("VLLM_FL_ILUVATAR_IVCORE_DECODE", "0").strip().lower() not in (
        "0",
        "false",
        "off",
        "no",
        "",
    )


def _capturing() -> bool:
    try:
        return bool(torch.cuda.is_current_stream_capturing())
    except Exception:
        return False


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


def _use_sme() -> int:
    # Trial SME G2S double-buffer. Default OFF — enable with =1 after A/B win.
    return int(
        os.getenv("VLLM_FL_ILUVATAR_IVCORE_SME", "0").strip().lower()
        not in ("0", "false", "off", "no", "")
    )


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


def _is_k_hd_major(k: torch.Tensor) -> bool:
    """K HD-major: [blocks, n_kv, HD, PAGE] contiguous innermost PAGE."""
    return (
        k.dim() == 4
        and int(k.shape[2]) == HD
        and int(k.shape[3]) == PAGE
        and int(k.stride(3)) == 1
        and int(k.stride(2)) == PAGE
    )


def _is_v_bndh(v: torch.Tensor) -> bool:
    """V BNDH: [blocks, PAGE, n_kv, HD]."""
    return (
        v.dim() == 4
        and int(v.shape[1]) == PAGE
        and int(v.shape[3]) == HD
        and int(v.stride(3)) == 1
    )


def usable(query: torch.Tensor, key_cache: torch.Tensor, out: torch.Tensor) -> bool:
    if query.dtype != torch.bfloat16 or out.dtype != torch.bfloat16:
        return False
    if query.dim() != 3 or key_cache.dim() != 4:
        return False
    if int(query.shape[-1]) != HD:
        return False
    n_qo = int(query.shape[1])
    if _is_k_hd_major(key_cache):
        n_kv = int(key_cache.shape[1])
    elif int(key_cache.shape[-1]) != HD:
        return False
    elif int(key_cache.shape[1]) == PAGE:
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
        _PACK_FULL_DONE.discard((n_kv, str(device), nb))
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
    _PACK_FULL_DONE.discard((n_kv, str(device), blocks))
    return ku8, vu8, ksc, vsc


def _pick_split(batch: int, n_kv: int, max_pages: int) -> int:
    # Cap pages: block_table is often padded to max_model_len/PAGE (e.g. 8192).
    # Target ~48 pages/split — BI-V150 sweet spot for i8-MMA QK (B=64 CTX=4k
    # → split=6 ≈444µs; split=4/8 are ~40–100µs slower).
    del batch, n_kv  # occupancy follows from pages; grid = B*n_kv*split
    pages = min(int(max_pages), 512)
    if pages <= 64:
        return 2
    sp = max(1, (pages + 47) // 48)
    if sp <= 4:
        return 4
    if sp <= 6:
        return 6
    return 8


def warm_for_cudagraph(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    batch_sizes: list[int] | None = None,
    max_pages: int | None = None,
    softmax_scale: float | None = None,
) -> None:
    """Allocate pack + decode workspaces and full-pack once before CG capture.

    Must run in eager mode. Records last-page pack into subsequent graphs when
    ``PACK_LAST`` is on.
    """
    if not enabled():
        return
    if _capturing():
        logger.warning("ivcore warm_for_cudagraph skipped (stream capturing)")
        return
    mod = _cuda_mod()
    if mod is None:
        return

    asym = _is_k_hd_major(key_cache) and _is_v_bndh(value_cache)
    if asym:
        k_in = key_cache if key_cache.is_contiguous() else key_cache.contiguous()
        v_in = value_cache if value_cache.is_contiguous() else value_cache.contiguous()
        blocks = int(k_in.shape[0])
        n_kv = int(k_in.shape[1])
    else:
        k_phys = _to_phys(key_cache)
        v_phys = _to_phys(value_cache)
        if int(k_phys.shape[2]) != PAGE or int(k_phys.shape[3]) != HD:
            return
        k_in = k_phys if k_phys.is_contiguous() else k_phys.contiguous()
        v_in = v_phys if v_phys.is_contiguous() else v_phys.contiguous()
        blocks = int(k_in.shape[0])
        n_kv = int(k_in.shape[1])

    device = key_cache.device
    bufs = _pack_bufs(blocks, n_kv, device)
    if bufs is None:
        return
    ku8, vu8, ksc, vsc = bufs
    pack_key = (n_kv, str(device), blocks)
    # Zero workspace, then touch decode so CG capture can freeze last-page.
    # Mark FULL_DONE (for capture) + STALE (live must sync/full-pack first).
    global _U8_STALE
    ku8.zero_()
    vu8.zero_()
    ksc.fill_(1.0)
    vsc.fill_(1.0)
    _PACK_FULL_DONE.add(pack_key)
    _U8_STALE = True

    n_qo = n_kv * GROUP
    scale = float(HD**-0.5 if softmax_scale is None else softmax_scale)
    sizes = batch_sizes or [1, 2, 4, 8, 16, 32, 64]
    mp = int(max_pages) if max_pages is not None else min(blocks, 512)

    # Touch decode workspace for each cudagraph batch size (grow-only).
    for bsz in sorted({int(x) for x in sizes if int(x) > 0}):
        q = torch.zeros(bsz, n_qo, HD, device=device, dtype=torch.bfloat16)
        out = torch.empty_like(q)
        bt = torch.zeros(bsz, max(mp, 1), device=device, dtype=torch.int32)
        sl = torch.ones(bsz, device=device, dtype=torch.int32)
        ok = paged_decode_ivcore(out, q, key_cache, value_cache, bt, sl, scale)
        if not ok:
            logger.warning(
                "ivcore warm decode miss batch=%d totals=%s", bsz, dict(_MISS)
            )
            break
    # Re-assert: capture may last-page; live must sync before trusting u8.
    _PACK_FULL_DONE.add(pack_key)
    _U8_STALE = True
    logger.info(
        "ivcore warm_for_cudagraph: blocks=%d n_kv=%d batches=%s "
        "pack_full=%s stale=%s",
        blocks,
        n_kv,
        sizes,
        pack_key in _PACK_FULL_DONE,
        _U8_STALE,
    )


def paged_decode_ivcore(
    out: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    seqused_k: torch.Tensor,
    softmax_scale: float,
) -> bool:
    """Drop-in decode. ``True`` = wrote ``out``. CUDA-graph safe after warm."""
    if not enabled():
        return False
    if not usable(query, key_cache, out):
        return _miss("usable")
    mod = _cuda_mod()
    if mod is None:
        return _miss("no_cuda_mod")

    capturing = _capturing()
    batch, n_qo, _ = (int(x) for x in query.shape)
    asym = _is_k_hd_major(key_cache) and _is_v_bndh(value_cache)
    if asym:
        if not (key_cache.is_contiguous() and value_cache.is_contiguous()):
            if capturing:
                return _miss("cg_noncontig_kv")
            k_in = key_cache if key_cache.is_contiguous() else key_cache.contiguous()
            v_in = value_cache if value_cache.is_contiguous() else value_cache.contiguous()
        else:
            k_in, v_in = key_cache, value_cache
        blocks = int(k_in.shape[0])
        n_kv = int(k_in.shape[1])
    else:
        k_phys = _to_phys(key_cache)
        v_phys = _to_phys(value_cache)
        if int(k_phys.shape[2]) != PAGE or int(k_phys.shape[3]) != HD:
            return _miss("phys_shape")
        blocks = int(k_phys.shape[0])
        n_kv = int(k_phys.shape[1])
        if not (k_phys.is_contiguous() and v_phys.is_contiguous()):
            if capturing:
                return _miss("cg_noncontig_phys")
            k_phys = k_phys.contiguous()
            v_phys = v_phys.contiguous()
        k_in, v_in = k_phys, v_phys
    if n_qo != n_kv * GROUP:
        return _miss("gqa")
    if not query.is_contiguous() or not out.is_contiguous():
        if capturing:
            return _miss("cg_noncontig_qout")
        if not query.is_contiguous():
            query = query.contiguous()
        if not out.is_contiguous():
            return _miss("out_noncontig")

    if block_table.dtype != torch.int32 or not block_table.is_contiguous():
        if capturing:
            return _miss("cg_bt")
        bt = block_table.to(torch.int32).contiguous()
    else:
        bt = block_table
    if seqused_k.dtype != torch.int32 or not seqused_k.is_contiguous():
        if capturing:
            return _miss("cg_sl")
        sl = seqused_k.to(torch.int32).contiguous()
    else:
        sl = seqused_k

    if capturing:
        ws = _PACK.get((n_kv, str(query.device)))
        if ws is None or int(ws[4]) != blocks:
            return _miss("cg_no_pack_ws")
        ku8, vu8, ksc, vsc, _nb = ws
    else:
        bufs = _pack_bufs(blocks, n_kv, query.device)
        if bufs is None:
            return _miss("buf_oom")
        ku8, vu8, ksc, vsc = bufs

    # Always full-pack from *this layer's* K/V into the shared u8 workspace.
    # Last-page is opt-in (PACK_LAST=1) and unsafe across layers with a shared
    # workspace — disabled by default after Math L3 corruption.
    try:
        global _U8_STALE
        pack_key = (n_kv, str(query.device), blocks)
        use_last = (
            _pack_last_enabled()
            and (not _U8_STALE)
            and pack_key in _PACK_FULL_DONE
            and (not capturing)  # never freeze last-page into CG
            and hasattr(
                mod,
                "ivcore_pack_bt_asym_last" if asym else "ivcore_pack_bt_last",
            )
        )
        if asym:
            if use_last:
                mod.ivcore_pack_bt_asym_last(k_in, v_in, ku8, vu8, ksc, vsc, bt, sl)
            else:
                mod.ivcore_pack_bt_asym(k_in, v_in, ku8, vu8, ksc, vsc, bt, sl)
                if not capturing:
                    _PACK_FULL_DONE.add(pack_key)
                    _U8_STALE = False
        else:
            if use_last:
                mod.ivcore_pack_bt_last(k_in, v_in, ku8, vu8, ksc, vsc, bt, sl)
            else:
                mod.ivcore_pack_bt(k_in, v_in, ku8, vu8, ksc, vsc, bt, sl)
                if not capturing:
                    _PACK_FULL_DONE.add(pack_key)
                    _U8_STALE = False
    except Exception as e:
        logger.warning("ivcore pack failed: %s", e)
        return _miss(f"pack_exc:{type(e).__name__}")

    forced = _force_split()
    split = forced if forced is not None else _pick_split(batch, n_kv, int(bt.shape[1]))
    vmode = _v_mode()
    vsc_arg = vsc if vmode == 2 else None
    try:
        mod.ivcore_u8_paged_decode(
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
            out,
            _use_sme(),
        )
    except Exception as e:
        logger.warning("ivcore decode failed: %s", e)
        return _miss(f"decode_exc:{type(e).__name__}")

    # Eager-only FA reference cos (skip under cudagraph capture/replay).
    budget = _debug_cos_budget()
    if budget > 0 and not capturing:
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
            a = out.float().reshape(-1)
            b = ref.float().reshape(-1)
            denom = (a.norm() * b.norm()).clamp_min(1e-12)
            cos = float((a * b).sum() / denom)
            logger.warning(
                "ivcore_debug_cos cos=%.6f batch=%d split=%d vmode=%d left=%d",
                cos,
                batch,
                split,
                vmode,
                _DEBUG_COS_LEFT,
            )
        except Exception as e:
            logger.warning("ivcore_debug_cos failed: %s", e)

    return True


def _resolve_kv_pair(sample) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Best-effort (K, V) from a runner kv_caches entry."""
    if isinstance(sample, (tuple, list)) and len(sample) >= 2:
        k, v = sample[0], sample[1]
        if isinstance(k, torch.Tensor) and isinstance(v, torch.Tensor):
            return k, v
        return None
    if not isinstance(sample, torch.Tensor) or sample.dim() < 5:
        return None
    # KV-first HND phys: [2, blocks, n_kv, PAGE, HD]
    if int(sample.shape[0]) == 2:
        return sample[0], sample[1]
    # Legacy interleaved: [blocks, 2, ...]
    if int(sample.shape[1]) == 2:
        return sample[:, 0], sample[:, 1]
    return None


def sync_pack_from_kv(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    seqused_k: torch.Tensor,
) -> bool:
    """Eager full-pack of ``block_table`` pages into the persistent u8 workspace.

    Call before FULL CUDA-graph replay when ``_U8_STALE`` (after warm/prefill).
    Clears stale and marks FULL_DONE on success.
    """
    global _U8_STALE, _SYNC_LOG_LEFT
    if not enabled():
        return False
    if _capturing():
        return False
    mod = _cuda_mod()
    if mod is None:
        return False
    try:
        asym = _is_k_hd_major(key_cache) and _is_v_bndh(value_cache)
        if asym:
            k_in = key_cache if key_cache.is_contiguous() else key_cache.contiguous()
            v_in = value_cache if value_cache.is_contiguous() else value_cache.contiguous()
            blocks = int(k_in.shape[0])
            n_kv = int(k_in.shape[1])
        else:
            k_phys = _to_phys(key_cache)
            v_phys = _to_phys(value_cache)
            if int(k_phys.shape[2]) != PAGE or int(k_phys.shape[3]) != HD:
                return False
            k_in = k_phys if k_phys.is_contiguous() else k_phys.contiguous()
            v_in = v_phys if v_phys.is_contiguous() else v_phys.contiguous()
            blocks = int(k_in.shape[0])
            n_kv = int(k_in.shape[1])
        bufs = _pack_bufs(blocks, n_kv, key_cache.device)
        if bufs is None:
            return False
        ku8, vu8, ksc, vsc = bufs
        bt = block_table
        if bt.dtype != torch.int32 or not bt.is_contiguous():
            bt = bt.to(torch.int32).contiguous()
        sl = seqused_k
        if sl.dtype != torch.int32 or not sl.is_contiguous():
            sl = sl.to(torch.int32).contiguous()
        # Trim padded CG rows with seqused==0 if present.
        if sl.numel() > 0:
            n_live = int(sl.numel())
            bt = bt[:n_live]
            sl = sl[:n_live]
        if asym:
            mod.ivcore_pack_bt_asym(k_in, v_in, ku8, vu8, ksc, vsc, bt, sl)
        else:
            mod.ivcore_pack_bt(k_in, v_in, ku8, vu8, ksc, vsc, bt, sl)
        pack_key = (n_kv, str(key_cache.device), blocks)
        _PACK_FULL_DONE.add(pack_key)
        _U8_STALE = False
        if _SYNC_LOG_LEFT > 0:
            _SYNC_LOG_LEFT -= 1
            logger.info(
                "ivcore sync_pack_from_kv: batch=%d max_pages=%d asym=%s",
                int(bt.shape[0]),
                int(bt.shape[1]) if bt.dim() > 1 else 0,
                asym,
            )
        return True
    except Exception as e:
        logger.warning("ivcore sync_pack_from_kv failed: %s", e)
        return False


def _attn_meta_bt_sl(attn_metadata):
    """Pull (block_table, seq_lens) from v1 forward-context attn metadata."""
    md = attn_metadata
    if md is None:
        return None, None
    if isinstance(md, list):
        md = md[0] if md else None
    if isinstance(md, dict):
        md = next(iter(md.values())) if md else None
    if md is None:
        return None, None
    bt = getattr(md, "block_table", None)
    sl = getattr(md, "seq_lens", None)
    if not isinstance(bt, torch.Tensor) or not isinstance(sl, torch.Tensor):
        return None, None
    return bt, sl


def maybe_sync_before_cudagraph(runner) -> None:
    """If u8 is stale and this step is FULL CG, full-pack once before replay."""
    if not enabled() or not _U8_STALE:
        return
    try:
        from vllm.config.compilation import CUDAGraphMode
        from vllm.forward_context import get_forward_context
    except Exception:
        return
    try:
        ctx = get_forward_context()
    except Exception:
        return
    if ctx is None:
        return
    mode = getattr(ctx, "cudagraph_runtime_mode", None)
    try:
        if mode != CUDAGraphMode.FULL:
            return
    except Exception:
        return
    bt, sl = _attn_meta_bt_sl(getattr(ctx, "attn_metadata", None))
    if bt is None or sl is None:
        return
    k = v = None
    try:
        rc = getattr(runner, "kv_caches", None)
        if isinstance(rc, (list, tuple)) and rc:
            k_v = _resolve_kv_pair(rc[0])
            if k_v is not None:
                k, v = k_v
        elif isinstance(rc, dict) and rc:
            k_v = _resolve_kv_pair(next(iter(rc.values())))
            if k_v is not None:
                k, v = k_v
    except Exception as e:
        logger.warning("ivcore cg-sync resolve KV failed: %s", e)
        return
    if k is None or v is None:
        return
    sync_pack_from_kv(k, v, bt, sl)


def _warm_from_runner(runner, kv_caches) -> None:
    """Allocate pack/decode workspaces using a sample layer cache."""
    if not enabled():
        return
    k = v = None
    try:
        if isinstance(kv_caches, dict) and kv_caches:
            k_v = _resolve_kv_pair(next(iter(kv_caches.values())))
            if k_v is not None:
                k, v = k_v
        elif isinstance(kv_caches, (list, tuple)) and kv_caches:
            k_v = _resolve_kv_pair(kv_caches[0])
            if k_v is not None:
                k, v = k_v
        if (k is None or v is None) and getattr(runner, "kv_caches", None):
            rc = runner.kv_caches
            if isinstance(rc, (list, tuple)) and rc:
                k_v = _resolve_kv_pair(rc[0])
                if k_v is not None:
                    k, v = k_v
    except Exception as e:
        logger.warning("ivcore warm resolve cache failed: %s", e)
        return
    if k is None or v is None:
        logger.warning("ivcore warm: no KV sample; skip")
        return
    sizes = None
    try:
        sizes = list(runner.compilation_config.cudagraph_capture_sizes or [])
    except Exception:
        sizes = None
    if not sizes:
        sizes = [1, 2, 4, 8, 16, 32, 64]
    max_pages = None
    try:
        sched = getattr(runner, "scheduler_config", None) or getattr(
            getattr(runner, "vllm_config", None), "scheduler_config", None
        )
        if sched is not None:
            mml = int(getattr(sched, "max_model_len", 0) or 0)
            if mml > 0:
                max_pages = (mml + 15) // 16
    except Exception:
        max_pages = None
    try:
        warm_for_cudagraph(k, v, batch_sizes=sizes, max_pages=max_pages)
    except Exception as e:
        logger.warning("ivcore warm_for_cudagraph failed: %s", e)


def patch_ivcore_warm_for_iluvatar() -> None:
    """Warm ivcore at KV init when HD-major is off (HD path warms itself)."""
    if not enabled():
        logger.info("ivcore warm patch skipped (IVCORE_DECODE off)")
        return
    try:
        from vllm_fl.dispatch.backends.vendor.iluvatar.kv_k_hd_major import (
            enabled as _hd_on,
        )

        if _hd_on():
            logger.info(
                "ivcore warm patch skipped (HD-major owns warm at KV init)"
            )
            return
    except Exception:
        pass

    try:
        from vllm.platforms import current_platform

        vendor = getattr(current_platform, "vendor_name", None)
        if not (isinstance(vendor, str) and vendor.lower() == "iluvatar"):
            return
    except Exception:
        return

    try:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    except Exception as e:
        logger.warning("ivcore warm patch: GPUModelRunner import failed: %s", e)
        return

    runners: list[type] = [GPUModelRunner]
    for fl_name in ("ModelRunnerFL", "GPUModelRunner"):
        try:
            import vllm_fl.worker.model_runner as fl_mr

            fl_cls = getattr(fl_mr, fl_name, None)
            if isinstance(fl_cls, type) and fl_cls not in runners:
                runners.append(fl_cls)
        except Exception:
            continue

    def _wrap(cls: type) -> None:
        if getattr(cls, "_iluvatar_ivcore_warm_patched", False):
            return
        _orig_init = cls.initialize_kv_cache_tensors

        def initialize_kv_cache_tensors(self, kv_cache_config, kernel_block_sizes):
            kv_caches = _orig_init(self, kv_cache_config, kernel_block_sizes)
            _warm_from_runner(self, kv_caches)
            return kv_caches

        cls.initialize_kv_cache_tensors = initialize_kv_cache_tensors

        if hasattr(cls, "_model_forward") and not getattr(
            cls, "_iluvatar_ivcore_cg_sync_patched", False
        ):
            _orig_fwd = cls._model_forward

            def _model_forward(self, *args, **kwargs):
                maybe_sync_before_cudagraph(self)
                return _orig_fwd(self, *args, **kwargs)

            cls._model_forward = _model_forward
            cls._iluvatar_ivcore_cg_sync_patched = True

        cls._iluvatar_ivcore_warm_patched = True

    for cls in runners:
        _wrap(cls)
    logger.info(
        "ivcore warm+cg-sync: wrapped initialize_kv_cache_tensors / "
        "_model_forward on %s (stock BNDH/HND, no HD-major)",
        ", ".join(c.__name__ for c in runners),
    )
