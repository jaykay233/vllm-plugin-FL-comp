import torch
import triton
import triton.language as tl

from vllm.model_executor.layers.rotary_embedding.base import ApplyRotaryEmb

dev = "cuda"
HD = 128
HALF = HD // 2
MAXP = 16384


@triton.jit
def rope_kernel(
    q_ptr, k_ptr, cs_ptr, pos_ptr,
    stride_qt, stride_kt, stride_c,
    NQH: tl.constexpr, NKH: tl.constexpr,
    HD: tl.constexpr, HALF: tl.constexpr, BT: tl.constexpr,
):
    pid = tl.program_id(0)
    t = pid * BT + tl.arange(0, BT)
    d = tl.arange(0, HALF)
    pos = tl.load(pos_ptr + t)
    cbase = cs_ptr + pos[:, None] * stride_c + d[None, :]
    cos = tl.load(cbase).to(tl.float32)
    sin = tl.load(cbase + HALF).to(tl.float32)

    for hb in tl.static_range(NQH):
        base = q_ptr + t[:, None] * stride_qt + hb * HD
        x1 = tl.load(base + d[None, :]).to(tl.float32)
        x2 = tl.load(base + HALF + d[None, :]).to(tl.float32)
        tl.store(base + d[None, :], (x1 * cos - x2 * sin).to(q_ptr.dtype.element_ty))
        tl.store(base + HALF + d[None, :], (x2 * cos + x1 * sin).to(q_ptr.dtype.element_ty))

    for hb in tl.static_range(NKH):
        base = k_ptr + t[:, None] * stride_kt + hb * HD
        x1 = tl.load(base + d[None, :]).to(tl.float32)
        x2 = tl.load(base + HALF + d[None, :]).to(tl.float32)
        tl.store(base + d[None, :], (x1 * cos - x2 * sin).to(k_ptr.dtype.element_ty))
        tl.store(base + HALF + d[None, :], (x2 * cos + x1 * sin).to(k_ptr.dtype.element_ty))


def make(T, HQ, HK, stride_qt=None, stride_kt=None):
    pos = torch.randint(0, MAXP, (T,), device=dev)
    cache = (torch.randn(MAXP, HD, device=dev, dtype=torch.float32) * 0.5).to(torch.bfloat16)
    sq = stride_qt or HQ * HD
    sk = stride_kt or HK * HD
    qb = torch.randn(T, sq, device=dev, dtype=torch.bfloat16)
    kb = torch.randn(T, sk, device=dev, dtype=torch.bfloat16)
    return pos, cache, qb, kb, sq, sk


def native(q, k, cache, pos, HQ, HK):
    c, s = cache.chunk(2, dim=-1)
    c = c.index_select(0, pos)
    s = s.index_select(0, pos)
    qr = ApplyRotaryEmb.forward_static(q[:, : HQ * HD].view(-1, HQ, HD), c, s, True)
    kr = ApplyRotaryEmb.forward_static(k[:, : HK * HD].view(-1, HK, HD), c, s, True)
    return qr, kr


def timed(build, replays=5, nlay=42):
    build()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(nlay):
            build()
    torch.cuda.synchronize()
    s = torch.cuda.Event(True)
    e = torch.cuda.Event(True)
    s.record()
    for _ in range(replays):
        g.replay()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / replays / nlay * 1000.0


def run(T, HQ, HK, stride_qt=None, stride_kt=None, tag=""):
    pos, cache, q, k, sq, sk = make(T, HQ, HK, stride_qt, stride_kt)
    nbytes = 2 * (T * HQ * HD + T * HK * HD) * 2

    qc = q[:, : HQ * HD].contiguous()
    kc = k[:, : HK * HD].contiguous()
    pos_t, cache_t = pos, cache
    nq = HQ
    nk = HK

    def ind():
        c, s = cache_t.chunk(2, dim=-1)
        c = c.index_select(0, pos_t)
        s = s.index_select(0, pos_t)
        ApplyRotaryEmb.forward_static(qc.view(T, nq, HD), c, s, True)
        ApplyRotaryEmb.forward_static(kc.view(T, nk, HD), c, s, True)

    ind_c = torch.compile(ind, fullgraph=True)
    t_ind = timed(ind_c)
    print(f"\n--- {tag} T={T} HQ={HQ} stride_qt={sq} ({nbytes / 1e6:.1f} MB) ---")
    print(f"inductor             {t_ind:8.1f} us  {nbytes / t_ind / 1e3:6.1f} GB/s")

    q0, k0 = q.clone(), k.clone()
    ref_out = native(q0, k0, cache, pos, HQ, HK)

    best = None
    for BT, nw in [(1, 1), (2, 1), (4, 1), (8, 1), (2, 2), (4, 2), (8, 2), (16, 2), (4, 4), (8, 4)]:
        grid = (triton.cdiv(T, BT),)

        def fn(BT=BT, nw=nw, grid=grid):
            rope_kernel[grid](q, k, cache, pos, sq, sk, cache.stride(0),
                              NQH=HQ, NKH=HK, HD=HD, HALF=HALF, BT=BT, num_warps=nw)

        q.copy_(q0)
        k.copy_(k0)
        try:
            fn()
            torch.cuda.synchronize()
        except Exception as exc:  # noqa: BLE001
            print(f"  BT={BT} w={nw} failed: {exc!r}")
            continue
        err = max((q[:, : HQ * HD].view(-1, HQ, HD).float() - ref_out[0].float()).abs().max().item(),
                  (k[:, : HK * HD].view(-1, HK, HD).float() - ref_out[1].float()).abs().max().item())
        us = timed(fn)
        eff = nbytes / us / 1e3
        star = ""
        if best is None or us < best[0]:
            best = (us, BT, nw)
            star = " *"
        print(f"  BT={BT:2d} w={nw}        {us:8.1f} us  {eff:6.1f} GB/s  maxdiff={err:.3f}{star}")
    return best


run(8192, 16, 2, 2560, 2560, "prefill qkv-split")
run(8192, 16, 2, None, None, "prefill contiguous")
run(64, 16, 2, 2560, 2560, "decode qkv-split")
