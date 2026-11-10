"""Isolate the achievable bandwidth of the paged GQA KV access pattern
(no dot, no softmax) so we can tell whether the fp8 decode kernel is
load-bound or compute-bound on C500."""

import torch, triton, triton.language as tl


@triton.jit
def load_only(K, V, BlockTable, B_Seqlen, Out, k_scale,
              stride_kp, stride_kb, stride_kh,
              stride_vp, stride_vb, stride_vh, stride_btb,
              kv_group_num: tl.constexpr, PAGE_SIZE: tl.constexpr,
              BLOCK_D: tl.constexpr, NUM_KV_SPLITS: tl.constexpr):
    b = tl.program_id(0)
    kh = tl.program_id(1)
    sp = tl.program_id(2)
    offs_d = tl.arange(0, BLOCK_D)
    offs_n = tl.arange(0, PAGE_SIZE)
    seq = tl.load(B_Seqlen + b)
    npg = tl.cdiv(seq, PAGE_SIZE)
    per = tl.cdiv(npg, NUM_KV_SPLITS)
    ps = sp * per
    pe = tl.minimum(ps + per, npg)
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    for pi in range(ps, pe):
        page = tl.load(BlockTable + b * stride_btb + pi).to(tl.int64)
        mn = (pi * PAGE_SIZE + offs_n) < seq
        k = tl.load(K + page * stride_kp + offs_n[:, None] * stride_kb
                    + kh * stride_kh + offs_d[None, :], mask=mn[:, None], other=0.0)
        v = tl.load(V + page * stride_vp + offs_n[:, None] * stride_vb
                    + kh * stride_vh + offs_d[None, :], mask=mn[:, None], other=0.0)
        acc += tl.sum(k.to(tl.float32), 0) * k_scale + tl.sum(v.to(tl.float32), 0)
    tl.store(Out + (b * 64 + kh) * BLOCK_D + offs_d + sp * 0, acc)


def bw(K, V, bt, lens, H, HKV, D, P, splits, warps, stages):
    B = lens.numel()
    out = torch.empty(B * 64 * D * 64, device=K.device, dtype=torch.float32)
    fn = lambda: load_only[(B, HKV, splits)](
        K, V, bt, lens, out, 1.0,
        K.stride(0), K.stride(1), K.stride(2),
        V.stride(0), V.stride(1), V.stride(2), bt.stride(0),
        kv_group_num=H // HKV, PAGE_SIZE=P, BLOCK_D=D,
        NUM_KV_SPLITS=splits, num_warps=warps, num_stages=stages)
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(20):
        fn()
    e.record()
    torch.cuda.synchronize()
    t = s.elapsed_time(e) / 20
    nb = B * int(lens[0]) * HKV * D * 2 * K.element_size()
    return t, nb / (t * 1e6)


def main():
    dev = "cuda"
    B, SEQ, H, HKV, D = 64, 4096, 16, 2, 128
    print("# NOTE: K and V are DISTINCT buffers. Passing the same tensor for both")
    print("# makes every V read hit cache and inflates the number ~2x (it even")
    print("# exceeds the 1416 GB/s measured DRAM peak).")
    for P in (64, 128):
        npg = (SEQ + P - 1) // P
        total = B * npg
        kb = torch.randn(total, P, HKV, D, device=dev, dtype=torch.bfloat16) * 0.5
        vb = torch.randn(total, P, HKV, D, device=dev, dtype=torch.bfloat16) * 0.5
        bt = torch.arange(total, device=dev, dtype=torch.int32).reshape(B, npg)
        lens = torch.full((B,), SEQ, device=dev, dtype=torch.int32)
        for tag, K, V in (
            ("bf16", kb, vb),
            ("fp8 ", kb.to(torch.float8_e4m3fn), vb.to(torch.float8_e4m3fn)),
        ):
            best = None
            for warps in (1, 2, 4, 8):
                for splits in (1, 4, 8, 16):
                    for stages in (1, 3):
                        try:
                            t, g = bw(K, V, bt, lens, H, HKV, D, P, splits, warps, stages)
                        except Exception as exc:
                            print("LOAD P=%d %s w=%d sp=%-2d st=%d FAIL %s"
                                  % (P, tag, warps, splits, stages, type(exc).__name__))
                            continue
                        if best is None or g > best[0]:
                            best = (g, t, warps, splits, stages)
            print("LOAD P=%-3d %s BEST %6.0f GB/s  %.3f ms/layer  w=%d sp=%d st=%d"
                  % (P, tag, best[0], best[1], best[2], best[3], best[4]))


if __name__ == "__main__":
    main()
