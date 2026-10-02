"""Is the small-batch flashinfer error a split-K combine bug, or my calling?

Two competing explanations for the n<32 / ctx>=2048 error:

  (A) split-K combine.  Too few sequences to fill 16 SMs, so the kernel splits
      the KV length, computes a partial softmax per shard, and rescales by
      exp(local_max - global_max) when combining.  A bug there is sensitive to
      how much the per-shard maxima differ, i.e. to softmax sharpness, and
      vanishes when attention is nearly uniform.

  (B) my wrapper builds bad page metadata (indptr / indices / last_page_len).
      That would corrupt *which* KV is read, so it would not care about softmax
      sharpness, and it would not spare n>=32 using the same code path.

Test 1 bypasses my wrapper entirely with hand-built, trivially correct metadata.
Test 2 sweeps softmax sharpness.  Test 3 varies last_page_len (ctx not a
multiple of the page size), which is the metadata field most likely to be wrong.
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("VLLM_PLUGINS", "fl")
sys.path.insert(0, "/root/bench_results")

import torch  # noqa: E402

N_QH, N_KVH, HD, BLOCK = 16, 2, 128, 16
DT, DEV = torch.bfloat16, "cuda"


def reference(q, k_hnd, v_hnd, bt, kv_len, scale):
    n = q.shape[0]
    nb = (kv_len + BLOCK - 1) // BLOCK
    rep = N_QH // N_KVH
    outs = []
    for i in range(n):
        pages = bt[i, :nb].long()
        k = k_hnd[pages].permute(1, 0, 2, 3).reshape(N_KVH, nb * BLOCK, HD)[:, :kv_len]
        v = v_hnd[pages].permute(1, 0, 2, 3).reshape(N_KVH, nb * BLOCK, HD)[:, :kv_len]
        k = k.float().repeat_interleave(rep, dim=0)
        v = v.float().repeat_interleave(rep, dim=0)
        s = torch.matmul(q[i].float().unsqueeze(1), k.transpose(1, 2)) * scale
        outs.append(torch.matmul(torch.softmax(s, -1), v).squeeze(1))
    return torch.stack(outs)


def make(n, ctx, seed=0, q_scale=1.0):
    torch.manual_seed(seed)
    nb = (ctx + BLOCK - 1) // BLOCK
    num_blocks = n * nb
    k_hnd = torch.randn(num_blocks, N_KVH, BLOCK, HD, dtype=DT, device=DEV) * 0.5
    v_hnd = torch.randn(num_blocks, N_KVH, BLOCK, HD, dtype=DT, device=DEV) * 0.5
    q = (torch.randn(n, N_QH, HD, dtype=DT, device=DEV) * 0.5 * q_scale).to(DT)
    # Sequential pages: the simplest possible block table.
    bt = torch.arange(num_blocks, dtype=torch.int32, device=DEV).reshape(n, nb)
    return k_hnd, v_hnd, q, bt, nb


def direct_flashinfer(q, k_hnd, v_hnd, bt, ctx, scale):
    """Call ixformer directly, with metadata built inline and by hand."""
    from ixformer.inference.functions.paged_attention import (
        paged_attention_flashinfer,
    )

    n, nb = bt.shape
    full_pages = ctx // BLOCK
    last = ctx - full_pages * BLOCK
    if last == 0:
        pages_per_seq, last = full_pages, BLOCK
    else:
        pages_per_seq = full_pages + 1

    indptr = torch.arange(n + 1, dtype=torch.int32, device=DEV) * pages_per_seq
    indices = bt[:, :pages_per_seq].reshape(-1).contiguous().to(torch.int32)
    last_page_len = torch.full((n,), last, dtype=torch.int32, device=DEV)

    out = torch.empty_like(q)
    paged_attention_flashinfer(
        out, q, (k_hnd, v_hnd), indptr, indices, last_page_len,
        float(scale), max_seq_len=ctx, kv_cache_format="HND",
    )
    return out


def main() -> int:
    import vllm_fl.dispatch.backends.vendor.iluvatar  # noqa: F401

    scale = 1.0 / (HD ** 0.5)
    lines = []

    def log(s=""):
        print(s, flush=True)
        lines.append(s)

    log("TEST 1: direct ixformer call, metadata built by hand (my wrapper bypassed)")
    log(f"{'n':>5} {'ctx':>6} {'err':>10}")
    for n, ctx in [(8, 4096), (16, 4096), (31, 4096), (32, 4096), (64, 4096),
                   (8, 1024), (8, 8192)]:
        k_hnd, v_hnd, q, bt, _ = make(n, ctx)
        ref = reference(q, k_hnd, v_hnd, bt, ctx, scale)
        out = direct_flashinfer(q, k_hnd, v_hnd, bt, ctx, scale)
        torch.cuda.synchronize()
        log(f"{n:5d} {ctx:6d} {(out.float() - ref).abs().max().item():10.2e}")

    log()
    log("TEST 2: softmax sharpness at n=8, ctx=4096")
    log("  flat softmax -> all shard maxima nearly equal -> a combine bug cancels")
    log(f"{'q_scale':>8} {'err':>10} {'max_softmax_p':>14}")
    for qs in (0.0, 0.05, 0.2, 1.0, 4.0, 16.0):
        k_hnd, v_hnd, q, bt, _ = make(8, 4096, q_scale=qs)
        ref = reference(q, k_hnd, v_hnd, bt, 4096, scale)
        out = direct_flashinfer(q, k_hnd, v_hnd, bt, 4096, scale)
        torch.cuda.synchronize()
        # How peaked is the attention distribution?
        pages = bt[0].long()
        k0 = k_hnd[pages].permute(1, 0, 2, 3).reshape(N_KVH, -1, HD)[:, :4096].float()
        k0 = k0.repeat_interleave(N_QH // N_KVH, dim=0)
        s = torch.matmul(q[0].float().unsqueeze(1), k0.transpose(1, 2)) * scale
        peak = torch.softmax(s, -1).max().item()
        err = (out.float() - ref).abs().max().item()
        log(f"{qs:8.2f} {err:10.2e} {peak:14.4f}")

    log()
    log("TEST 3: partial last page at n=8 (exercises last_page_len)")
    log(f"{'ctx':>6} {'last_page':>10} {'err':>10}")
    for ctx in (4096, 4090, 4080, 4000, 2056):
        k_hnd, v_hnd, q, bt, _ = make(8, ctx)
        ref = reference(q, k_hnd, v_hnd, bt, ctx, scale)
        out = direct_flashinfer(q, k_hnd, v_hnd, bt, ctx, scale)
        torch.cuda.synchronize()
        lp = ctx % BLOCK or BLOCK
        log(f"{ctx:6d} {lp:10d} {(out.float() - ref).abs().max().item():10.2e}")

    log()
    log("TEST 4: does the error live in one shard? n=8 ctx=4096,")
    log("  compare against references truncated to KV prefixes")
    k_hnd, v_hnd, q, bt, _ = make(8, 4096)
    out = direct_flashinfer(q, k_hnd, v_hnd, bt, 4096, scale).float()
    best = None
    for frac in (0.25, 0.5, 0.75, 1.0):
        kv = int(4096 * frac)
        ref = reference(q, k_hnd, v_hnd, bt[:, : kv // BLOCK], kv, scale)
        e = (out - ref).abs().max().item()
        log(f"  vs reference over first {kv:5d} KV: err={e:.2e}")
        if best is None or e < best[1]:
            best = (kv, e)
    log(f"  closest prefix: {best[0]} (full ctx is 4096)")

    with open("/root/bench_results/flashinfer_splitk_probe.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
