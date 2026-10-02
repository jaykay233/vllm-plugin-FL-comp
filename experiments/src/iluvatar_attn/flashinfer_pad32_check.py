"""Can padding the decode slice to 32 sequences avoid the wrong-sequence read?

Mixed batches run eager, so the split path's decode slice gets no shelter from
graph capture, and below 32 sequences flashinfer reads sequence 0's KV for every
row.  The failure boundary tracks sequence count alone, so adding dummy rows up
to 32 should push the kernel back onto the correct path.  Rows in a varlen batch
are independent, so the padded outputs are simply discarded.

Checks (a) that padding restores correct reads, (b) that the real rows are
unchanged by the padding, and (c) what it costs.
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("VLLM_PLUGINS", "fl")
sys.path.insert(0, "/root/bench_results")

import torch  # noqa: E402

N_QH, N_KVH, HD, BLOCK = 16, 2, 128, 16
DT, DEV = torch.bfloat16, "cuda"
CH_SEQ, CH_ONES = 0, 2
PAD_TO = 32


def timeit(fn, warm=5, reps=20):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(reps):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / reps


def build(n, ctx, marked=True):
    """Pool with each sequence's identity written into V channel 0."""
    nb = ctx // BLOCK
    pool = n * nb + PAD_TO * nb
    k_hnd = torch.randn(pool, N_KVH, BLOCK, HD, dtype=DT, device=DEV) * 0.5
    v_hnd = torch.randn(pool, N_KVH, BLOCK, HD, dtype=DT, device=DEV) * 0.5
    used = torch.arange(n * nb, dtype=torch.int32, device=DEV).reshape(n, nb)
    if marked:
        k_hnd.zero_()
        v_hnd.zero_()
        for s in range(n):
            v_hnd[used[s].long(), :, :, CH_SEQ] = float(s + 1)
            v_hnd[used[s].long(), :, :, CH_ONES] = 1.0
    return k_hnd, v_hnd, used, pool


def call(q, k_hnd, v_hnd, bt, seqused, ctx):
    from ixformer.inference.functions.paged_attention import (
        paged_attention_flashinfer,
    )

    n, nb = bt.shape
    pages = (seqused + BLOCK - 1) // BLOCK
    indptr = torch.zeros(n + 1, dtype=torch.int32, device=DEV)
    torch.cumsum(pages.to(torch.int32), 0, out=indptr[1:])
    idx = torch.arange(nb, device=DEV)
    keep = idx.unsqueeze(0) < pages.unsqueeze(1)
    indices = bt[keep].contiguous().to(torch.int32)
    last = torch.where(
        seqused % BLOCK == 0,
        torch.full_like(seqused, BLOCK),
        seqused % BLOCK,
    ).to(torch.int32)
    out = torch.empty_like(q)
    paged_attention_flashinfer(
        out, q, (k_hnd, v_hnd), indptr, indices, last,
        1.0 / (HD ** 0.5), max_seq_len=ctx, kv_cache_format="HND",
    )
    return out


def seq_ids(out):
    mass = out[:, :, CH_ONES].clamp_min(1e-6)
    return (out[:, :, CH_SEQ] / mass)[:, 0]


def main() -> int:
    import vllm_fl.dispatch.backends.vendor.iluvatar  # noqa: F401

    lines = []

    def log(s=""):
        print(s, flush=True)
        lines.append(s)

    ctx = 4096
    nb = ctx // BLOCK

    log("(a) does padding to 32 rows restore correct sequence reads?")
    log(f"{'n_real':>7} {'padded':>7}  sequence ids read by the real rows")
    for n in (4, 8, 16, 24, 31):
        k_hnd, v_hnd, used, pool = build(n, ctx)
        q = torch.zeros(n, N_QH, HD, dtype=DT, device=DEV)
        bt = used
        sk = torch.full((n,), ctx, dtype=torch.int32, device=DEV)
        plain = seq_ids(call(q, k_hnd, v_hnd, bt, sk, ctx))

        # Pad with dummy rows: valid pages, minimal context.
        pad = PAD_TO - n
        bt_p = torch.cat([bt, bt[:1].expand(pad, nb).contiguous()], 0)
        sk_p = torch.cat([sk, torch.full((pad,), BLOCK, dtype=torch.int32,
                                         device=DEV)], 0)
        q_p = torch.zeros(PAD_TO, N_QH, HD, dtype=DT, device=DEV)
        padded = seq_ids(call(q_p, k_hnd, v_hnd, bt_p, sk_p, ctx))[:n]

        torch.cuda.synchronize()
        want = torch.arange(1, n + 1, device=DEV, dtype=torch.float32)
        ok_plain = (plain - want).abs().max().item() < 0.05
        ok_pad = (padded - want).abs().max().item() < 0.05
        log(f"{n:7d} {'no':>7}  {[round(x, 1) for x in plain.tolist()[:8]]} "
            f"{'OK' if ok_plain else 'WRONG'}")
        log(f"{n:7d} {'yes':>7}  {[round(x, 1) for x in padded.tolist()[:8]]} "
            f"{'OK' if ok_pad else 'WRONG'}")

    log()
    log("(b) with real data, does padding change the real rows' output?")
    log(f"{'n':>4} {'err_plain':>11} {'err_padded':>11}  (vs fp32 reference)")
    for n in (8, 16, 24):
        torch.manual_seed(0)
        k_hnd, v_hnd, used, pool = build(n, ctx, marked=False)
        q = torch.randn(n, N_QH, HD, dtype=DT, device=DEV) * 0.5
        bt, sk = used, torch.full((n,), ctx, dtype=torch.int32, device=DEV)

        rep = N_QH // N_KVH
        outs = []
        for i in range(n):
            pages = bt[i].long()
            kk = k_hnd[pages].permute(1, 0, 2, 3).reshape(N_KVH, -1, HD).float()
            vv = v_hnd[pages].permute(1, 0, 2, 3).reshape(N_KVH, -1, HD).float()
            kk = kk.repeat_interleave(rep, 0)
            vv = vv.repeat_interleave(rep, 0)
            s = torch.matmul(q[i].float().unsqueeze(1), kk.transpose(1, 2))
            s = s / (HD ** 0.5)
            outs.append(torch.matmul(torch.softmax(s, -1), vv).squeeze(1))
        ref = torch.stack(outs)

        o1 = call(q, k_hnd, v_hnd, bt, sk, ctx)
        pad = PAD_TO - n
        bt_p = torch.cat([bt, bt[:1].expand(pad, nb).contiguous()], 0)
        sk_p = torch.cat([sk, torch.full((pad,), BLOCK, dtype=torch.int32,
                                         device=DEV)], 0)
        q_p = torch.cat([q, torch.zeros(pad, N_QH, HD, dtype=DT, device=DEV)], 0)
        o2 = call(q_p, k_hnd, v_hnd, bt_p, sk_p, ctx)[:n]
        torch.cuda.synchronize()
        log(f"{n:4d} {(o1.float() - ref).abs().max().item():11.2e} "
            f"{(o2.float() - ref).abs().max().item():11.2e}")

    log()
    log("(c) cost of the padding")
    for n in (8, 24):
        torch.manual_seed(0)
        k_hnd, v_hnd, used, pool = build(n, ctx, marked=False)
        q = torch.randn(n, N_QH, HD, dtype=DT, device=DEV) * 0.5
        bt, sk = used, torch.full((n,), ctx, dtype=torch.int32, device=DEV)
        pad = PAD_TO - n
        bt_p = torch.cat([bt, bt[:1].expand(pad, nb).contiguous()], 0)
        sk_p = torch.cat([sk, torch.full((pad,), BLOCK, dtype=torch.int32,
                                         device=DEV)], 0)
        q_p = torch.cat([q, torch.zeros(pad, N_QH, HD, dtype=DT, device=DEV)], 0)
        t1 = timeit(lambda: call(q, k_hnd, v_hnd, bt, sk, ctx))
        t2 = timeit(lambda: call(q_p, k_hnd, v_hnd, bt_p, sk_p, ctx))
        log(f"  n={n:3d}: unpadded={t1:.3f} ms  padded_to_32={t2:.3f} ms "
            f"({t2 / t1:.2f}x)")

    with open("/root/bench_results/flashinfer_pad32_check.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
