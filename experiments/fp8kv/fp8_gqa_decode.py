# Copyright (c) 2026 BAAI. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""
FP8-KV GQA paged-decode attention prototype for MetaX C500.

Context (see experiments/FINDINGS-mx.md sections 16.3, 18.2, 18.4):

  * bf16 decode attention already runs at ~90% of measured HBM peak on C500
    (batch 64, seq 4096, 42 layers: 11.27 GB / 8.80 ms / 1280 GB/s). No tiling
    work can "beat FA" while K/V stay bf16 -- the bytes are the floor.
  * The only lever is to move fewer bytes: fp8_e4m3 K/V halves the traffic.
  * The vendor path cannot do it:
      - `_C_cache_ops::reshape_and_cache_flash` / `_C::paged_attention_v1`
        reject every fp8 dtype string. `"auto"` is NOT a workaround: it
        reinterprets the bf16 bit pattern as fp8 (1.5 -> -2.0 / 1.875).
      - `paged_attention_v1` also indexes GQA as `head_idx / num_queries_per_kv`,
        so one KV head is re-read once per query head -- 8x traffic for this
        8:1 model, which cancels the fp8 win several times over.
      - `flash_attn_with_kvcache` requires `q.dtype == k.dtype`; the MACA build
        has no fp8 path (`flash_attn_supports_fp8() -> False`).

Design notes that matter for performance on this chip:

  1. GQA grouping -- one program owns (seq, kv_head) and all `kv_group_num`
     query heads of that group, so K/V are read exactly once per KV head.
  2. Page-contiguous loads -- K/V are read a whole page at a time
     (`BLOCK_N == page_size`), never token-by-token. A per-token gather costs
     ~9x bandwidth on C500 (measured: 136 GB/s vs 1226 GB/s for FA).
  3. fp8 storage + dequant on load. Decode is bandwidth bound, so dot products
     stay bf16. Note there is NO working fp8 tensor-core path to borrow here:
     `_C::cutlass_scaled_mm` silently returns garbage, Triton cannot lower an
     fp8 `tl.dot` ("PassManager::run failed"), and the mctlassEx blockwise fp8
     GEMM never writes its output buffer. See fp8_mctlass_gemm.py.
  4. CUDA-graph safe: no host sync, no `.item()`, all shapes constexpr
     (attention is replayed from a FULL decode CUDA graph).

MEASURED VERDICT -- fp8 K/V through Triton is a NET LOSS on C500.

  Isolation (bw_probe.py / fp8_load_probe.py, batch 64, seq 4096, GQA 8:1).
  bw_probe passes K and V as *distinct* buffers; aliasing them makes every V
  read hit cache and inflates the figure past the 1416 GB/s DRAM peak.

      load-only, bf16 K/V          1521 GB/s    0.176 ms/layer
      load-only, fp8  K/V           228 GB/s    0.587 ms/layer
      ------------------------------------------------ 
      vendor flash_attn (bf16)     1519 GB/s    0.177 ms/layer

  Two things fall out of that table:

  1. bf16 load-only (0.176 ms) == full flash attention (0.177 ms). FA is
     already sitting exactly on the load bound, with all of its matmul and
     online-softmax work fully overlapped. There is no headroom left in the
     load pattern; it is a *compute pipeline* problem from here on.
  2. fp8 load-only (0.587 ms) is 3.3x slower than FA's entire attention
     kernel. Even a hypothetical fp8 kernel that did zero math would lose.
     This is why fp8 K/V is dead here, independent of how good the kernel is.

  Where that 3.3x comes from (fp8_load_probe.py, contiguous, no paging):

      bf16 load + widen to f32        1254 GB/s
      fp8  load + widen to f32         283 GB/s   <-- 4.4x slower
      fp8  load + widen to bf16        304 GB/s
      fp8  load + bitcast (no widen)   890 GB/s   <-- 3x faster than widening

  The load is fine; the fp8 -> float *numeric conversion* is what costs 3-4x,
  and Triton emulates it in software on this backend (the compiled asm does
  contain 81 `mma` instructions, so this is not an operator-fallback story).
  Per element the fp8 path is ~2.2x slower than bf16, so the 2x byte saving is
  more than eaten by the 4.4x conversion.

  Paged async-transpose detail: K is now loaded as [PAGE, D] (page-contiguous,
  D innermost) and transposed in the MMA descriptor, matching bw_probe.py's
  addressing instead of reading [D, PAGE]. That took bf16 from 3.377 to 2.942
  ms/layer (-13%), with correctness unchanged (maxabs 5e-4). It does NOT close
  the gap to FA (0.177 ms): at 46 GB/s vs 1521 GB/s, ~97% of the deficit is the
  compute pipeline -- online softmax, an MMA tile padded from 8 to 16 query
  heads (50% waste), and no async pipelining to hide dot latency behind loads.

  What *would* work: an fp8 attention kernel in MACA C++/mctlass, where the
  fp8 -> fp16 conversion happens inside the MMA pipeline rather than as a
  separate elementwise step. Caveat: nothing in this stack currently proves an
  fp8 MMA is reachable -- every fp8 GEMM entry point on this box is broken or
  absent (see fp8_mctlass_gemm.py) -- so even that is speculative. It would
  also have to beat the 0.177 ms bf16 bound after paying the conversion.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _fp8_gqa_decode_stage1(
    Q,
    K,
    V,
    BlockTable,
    B_Seqlen,
    Mid_Out,
    Mid_Lse,
    k_scale,
    v_scale,
    sm_scale,
    stride_qb,
    stride_qh,
    stride_kp,
    stride_kb,
    stride_kh,
    stride_vp,
    stride_vb,
    stride_vh,
    stride_btb,
    stride_mb,
    stride_mh,
    stride_ms,
    kv_group_num: tl.constexpr,
    q_head_num: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_H: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
    IS_FP8: tl.constexpr,
    Lk: tl.constexpr,
    Lv: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_kv_head = tl.program_id(1)
    split_id = tl.program_id(2)

    # GQA group: the BLOCK_H query heads sharing this KV head. BLOCK_H is
    # padded to >= 16 because dot needs a 16-row tile here; spare rows masked.
    cur_head = cur_kv_head * kv_group_num + tl.arange(0, BLOCK_H)
    mask_h = (tl.arange(0, BLOCK_H) < kv_group_num) & (cur_head < q_head_num)

    offs_d = tl.arange(0, BLOCK_D)
    offs_dv = tl.arange(0, BLOCK_DV)
    offs_n = tl.arange(0, PAGE_SIZE)
    mask_d = offs_d < Lk
    mask_dv = offs_dv < Lv

    offs_q = cur_batch * stride_qb + cur_head[:, None] * stride_qh + offs_d[None, :]
    q = tl.load(Q + offs_q, mask=mask_h[:, None] & mask_d[None, :], other=0.0)

    seq_len = tl.load(B_Seqlen + cur_batch)
    num_pages = tl.cdiv(seq_len, PAGE_SIZE)
    pages_per_split = tl.cdiv(num_pages, NUM_KV_SPLITS)
    page_start = split_id * pages_per_split
    page_end = tl.minimum(page_start + pages_per_split, num_pages)

    m_i = tl.full([BLOCK_H], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_DV], dtype=tl.float32)

    for pi in range(page_start, page_end):
        page = tl.load(BlockTable + cur_batch * stride_btb + pi).to(tl.int64)
        mask_n = (pi * PAGE_SIZE + offs_n) < seq_len

        # One whole page, contiguous in the head-dim: coalesced.
        # Load K paged as [PAGE, D] -- page-contiguous, D innermost -- exactly
        # like bw_probe.py's load_only. Addressing it as [D, PAGE] instead (the
        # shape tl.dot wants) makes the innermost axis stride by `stride_kb`, so
        # every lane straddles a different row. The transpose is free in the MMA
        # descriptor; the uncoalesced load is not.
        offs_k = (
            page * stride_kp
            + offs_n[:, None] * stride_kb
            + cur_kv_head * stride_kh
            + offs_d[None, :]
        )
        k = tl.load(K + offs_k, mask=mask_n[:, None] & mask_d[None, :], other=0.0)
        if IS_FP8:
            # Widen for the scale, return to bf16: dot must stay bf16, and
            # `bf16 * fp32_scalar` would silently promote to fp32.
            k = (k.to(tl.float32) * k_scale).to(tl.bfloat16)

        qk = tl.dot(q, tl.trans(k)) * sm_scale
        qk = tl.where(mask_h[:, None] & mask_n[None, :], qk, float("-inf"))

        m_new = tl.maximum(tl.max(qk, 1), m_i)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])

        offs_v = (
            page * stride_vp
            + offs_n[:, None] * stride_vb
            + cur_kv_head * stride_vh
            + offs_dv[None, :]
        )
        v = tl.load(V + offs_v, mask=mask_n[:, None] & mask_dv[None, :], other=0.0)
        if IS_FP8:
            v = (v.to(tl.float32) * v_scale).to(tl.bfloat16)

        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_new

    offs_mid = (
        cur_batch * stride_mb
        + cur_head[:, None] * stride_mh
        + split_id * stride_ms
        + offs_dv[None, :]
    )
    tl.store(
        Mid_Out + offs_mid,
        acc / l_i[:, None],
        mask=mask_h[:, None] & mask_dv[None, :],
    )
    offs_lse = cur_batch * stride_mb + cur_head * stride_mh + split_id * stride_ms + Lv
    tl.store(Mid_Lse + offs_lse, m_i + tl.log(l_i), mask=mask_h)


@triton.jit
def _fp8_gqa_decode_combine(
    Mid,
    Out,
    stride_mb,
    stride_mh,
    stride_ms,
    stride_ob,
    stride_oh,
    NUM_KV_SPLITS: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    Lv: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    offs_dv = tl.arange(0, BLOCK_DV)
    offs_s = tl.arange(0, NUM_KV_SPLITS)

    base = cur_batch * stride_mb + cur_head * stride_mh
    lse = tl.load(Mid + base + offs_s * stride_ms + Lv)
    lse_max = tl.max(lse, 0)
    w = tl.exp(lse - lse_max)
    denom = tl.sum(w, 0)

    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)
    for i in tl.static_range(NUM_KV_SPLITS):
        wi = tl.sum(tl.where(offs_s == i, w, 0.0), 0)
        o = tl.load(
            Mid + base + i * stride_ms + offs_dv, mask=offs_dv < Lv, other=0.0
        )
        acc += o.to(tl.float32) * wi

    tl.store(
        Out + cur_batch * stride_ob + cur_head * stride_oh + offs_dv,
        (acc / denom).to(tl.bfloat16),
        mask=offs_dv < Lv,
    )


def fp8_gqa_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    sm_scale: float,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    num_kv_splits: int = 8,
    num_warps: int = 4,
    num_stages: int = 1,
) -> torch.Tensor:
    """Paged GQA decode attention reading fp8_e4m3 K/V.

    q:           [B, H, D]                 bf16
    k_cache:     [num_pages, P, HKV, D]    fp8_e4m3 (bf16 also accepted)
    v_cache:     same
    block_table: [B, max_pages]            int32
    seq_lens:    [B]                       int32
    """
    B, H, D = q.shape
    HKV = k_cache.shape[2]
    P = k_cache.shape[1]
    kv_group_num = H // HKV
    is_fp8 = k_cache.dtype == torch.float8_e4m3fn

    block_h = max(16, triton.next_power_of_2(kv_group_num))
    block_d = triton.next_power_of_2(D)

    mid = torch.empty((B, H, num_kv_splits, D + 1), device=q.device, dtype=torch.float32)
    out = torch.empty_like(q)

    _fp8_gqa_decode_stage1[(B, HKV, num_kv_splits)](
        q,
        k_cache,
        v_cache,
        block_table,
        seq_lens,
        mid,
        mid,
        k_scale,
        v_scale,
        sm_scale,
        q.stride(0),
        q.stride(1),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        block_table.stride(0),
        mid.stride(0),
        mid.stride(1),
        mid.stride(2),
        kv_group_num=kv_group_num,
        q_head_num=H,
        PAGE_SIZE=P,
        BLOCK_D=block_d,
        BLOCK_DV=block_d,
        BLOCK_H=block_h,
        NUM_KV_SPLITS=num_kv_splits,
        IS_FP8=is_fp8,
        Lk=D,
        Lv=D,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    if num_kv_splits == 1:
        out.copy_(mid[..., :D].squeeze(2).to(torch.bfloat16))
        return out

    _fp8_gqa_decode_combine[(B, H)](
        mid,
        out,
        mid.stride(0),
        mid.stride(1),
        mid.stride(2),
        out.stride(0),
        out.stride(1),
        NUM_KV_SPLITS=num_kv_splits,
        BLOCK_DV=block_d,
        Lv=D,
        num_warps=1,
        num_stages=1,
    )
    return out


# ---------------------------------------------------------------------------
# reference + benchmark
# ---------------------------------------------------------------------------


def _reference(q, k_cache, v_cache, block_table, seq_lens, sm_scale, k_scale, v_scale):
    B, H, D = q.shape
    HKV = k_cache.shape[2]
    g = H // HKV
    P = k_cache.shape[1]
    out = torch.empty_like(q, dtype=torch.float32)
    kd = k_cache.to(torch.float32) * k_scale
    vd = v_cache.to(torch.float32) * v_scale
    for b in range(B):
        n = int(seq_lens[b])
        npg = (n + P - 1) // P
        pages = block_table[b, :npg].long()
        k = kd[pages].reshape(-1, HKV, D)[:n]
        v = vd[pages].reshape(-1, HKV, D)[:n]
        k = k.repeat_interleave(g, dim=1).permute(1, 0, 2)
        v = v.repeat_interleave(g, dim=1).permute(1, 0, 2)
        s = torch.einsum("hd,hnd->hn", q[b].float(), k) * sm_scale
        p = torch.softmax(s, dim=-1)
        out[b] = torch.einsum("hn,hnd->hd", p, v)
    return out.to(torch.bfloat16)


def _make_case(B, SEQ, HKV, D, P, dev="cuda", fp8=True):
    q = torch.randn(B, 16, D, device=dev, dtype=torch.bfloat16)
    npages_per = (SEQ + P - 1) // P
    total = B * npages_per
    kv = torch.randn(total, P, HKV, D, device=dev, dtype=torch.bfloat16) * 0.5
    bt = torch.arange(total, device=dev, dtype=torch.int32).reshape(B, npages_per)
    lens = torch.full((B,), SEQ, device=dev, dtype=torch.int32)
    if fp8:
        kv = kv.to(torch.float8_e4m3fn)
    return q, kv, kv.clone(), bt, lens


def main():
    torch.manual_seed(0)
    dev = "cuda"
    H, HKV, D = 16, 2, 128
    LAYERS = 42

    # ---- correctness ----
    Bc, SEQc, Pc = 4, 512, 64
    q, kf, vf, bt, lens = _make_case(Bc, SEQc, HKV, D, Pc, fp8=True)
    kb = kf.to(torch.bfloat16)
    ref_fp8 = _reference(q, kf, vf, bt, lens, D**-0.5, 1.0, 1.0)
    ref_bf16 = _reference(q, kb, kb, bt, lens, D**-0.5, 1.0, 1.0)
    got_fp8 = fp8_gqa_decode(q, kf, vf, bt, lens, D**-0.5, num_kv_splits=8)
    got_fp8_s1 = fp8_gqa_decode(q, kf, vf, bt, lens, D**-0.5, num_kv_splits=1)
    got_bf16 = fp8_gqa_decode(q, kb, kb, bt, lens, D**-0.5, num_kv_splits=8)

    def rel(a, b):
        return float((a.float() - b.float()).abs().max())

    print("CORR fp8 s8  vs fp8ref  maxabs=%.5f" % rel(got_fp8, ref_fp8))
    print("CORR fp8 s1  vs fp8ref  maxabs=%.5f" % rel(got_fp8_s1, ref_fp8))
    print("CORR bf16 s8 vs bf16ref maxabs=%.5f" % rel(got_bf16, ref_bf16))
    print("CORR fp8 vs bf16 (quant) maxabs=%.5f" % rel(got_fp8, ref_bf16))

    # ---- bandwidth at the official shape ----
    B, SEQ, P = 64, 4096, 64
    q, kf, vf, bt, lens = _make_case(B, SEQ, HKV, D, P, fp8=True)
    _, kb, vb, _, _ = _make_case(B, SEQ, HKV, D, P, fp8=False)

    def bench(fn, iters=20):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        for _ in range(iters):
            fn()
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e) / iters

    kb1 = B * SEQ * HKV * D * 2
    kb2 = kb1 * 2

    best = None
    for nw in (2, 4, 8):
        for sp in (1, 2, 4, 8, 16, 32):
            try:
                t = bench(
                    lambda w_=nw, s_=sp: fp8_gqa_decode(
                        q, kf, vf, bt, lens, D**-0.5, num_kv_splits=s_, num_warps=w_
                    )
                )
            except Exception as exc:
                print("BENCH fp8 w=%d sp=%-2d FAIL %s" % (nw, sp, type(exc).__name__))
                continue
            print(
                "BENCH fp8  w=%d sp=%-2d %7.3f ms/layer %6.0f GB/s  x42=%6.2f ms"
                % (nw, sp, t, kb1 / (t * 1e6), t * LAYERS)
            )
            if best is None or t < best[0]:
                best = (t, nw, sp)
    print("BEST fp8: ms/layer=%.3f warps=%d splits=%d" % best)

    t = bench(lambda: fp8_gqa_decode(q, kb, vb, bt, lens, D**-0.5, num_kv_splits=best[2],
                                     num_warps=best[1]))
    print("BENCH my bf16 same conf %7.3f ms/layer %6.0f GB/s  x42=%6.2f ms"
          % (t, kb2 / (t * 1e6), t * LAYERS))

    try:
        from flash_attn import flash_attn_with_kvcache

        qd = q.unsqueeze(1)
        kcb = kb.float().to(torch.bfloat16)
        t = bench(lambda: flash_attn_with_kvcache(
            qd, kcb, kcb, block_table=bt, cache_seqlens=lens, causal=True))
        print("BENCH vendor FA bf16    %7.3f ms/layer %6.0f GB/s  x42=%6.2f ms"
              % (t, kb2 / (t * 1e6), t * LAYERS))
        t = bench(lambda: flash_attn_with_kvcache(
            qd, kcb, kcb, block_table=bt, cache_seqlens=lens, causal=True, num_splits=16))
        print("BENCH vendor FA bf16 ns16 %5.3f ms/layer %6.0f GB/s  x42=%6.2f ms"
              % (t, kb2 / (t * 1e6), t * LAYERS))
        print("VERDICT fp8 %.3f ms vs FA %.3f ms -> %.1fx SLOWER. "
              "fp8 KV via Triton is a net loss: the fp8->float widening is "
              "software-emulated here (283 GB/s) and costs more than the 2x "
              "byte saving buys back."
              % (best[0], t, best[0] / t))
    except Exception as exc:
        print("BENCH vendor FA skipped:", type(exc).__name__, str(exc)[:120])


if __name__ == "__main__":
    main()
