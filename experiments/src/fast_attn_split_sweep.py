#!/usr/bin/env python3
"""FAST kernel-level sweep of decode attention `num_splits` on MetaX.

The end-to-end harness (bench_attn_num_splits.py) answers the right question but
costs ~4-5 min per config, almost all of it model load + torch.compile + CUDA
graph capture + generating 3x1024 tokens -- none of which is attention.

This times the attention kernel itself. No model, no engine, no graph capture:
just flash_attn_with_kvcache() over a (seqlen x batch x num_splits) grid with
CUDA events. Milliseconds per point.

Kernel names produced:
  num_splits == 1  -> flash_fwd_kernel                       (no reduction)
  num_splits >= 2  -> flash_fwd_splitkv_kernel
                      + flash_fwd_splitkv_combine_kernel     (the reduction)
  num_splits == 0  -> MACA heuristic picks one of the above

Per-decode-step attention cost = per_call_us * 42 layers.

Run:  conda activate mx && python /root/src/fast_attn_split_sweep.py
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path("/root/bench_results/attn_splits_fast")
OUT.mkdir(parents=True, exist_ok=True)

NUM_HEADS = 16      # MiniCPM5-2B
NUM_KV_HEADS = 2
HEAD_DIM = 128
N_LAYERS = 42
DT = None  # set in main


def build(seqlen: int, batch: int, block_size: int, dev, dtype):
    import torch

    max_blocks = (seqlen + block_size - 1) // block_size
    nblocks = max_blocks * batch
    q = torch.randn(batch, 1, NUM_HEADS, HEAD_DIM, dtype=dtype, device=dev) * 0.1
    k_cache = torch.randn(nblocks, block_size, NUM_KV_HEADS, HEAD_DIM,
                          dtype=dtype, device=dev) * 0.1
    v_cache = torch.randn(nblocks, block_size, NUM_KV_HEADS, HEAD_DIM,
                          dtype=dtype, device=dev) * 0.1
    block_table = torch.arange(nblocks, dtype=torch.int32, device=dev).view(
        batch, max_blocks)
    cache_seqlens = torch.full((batch,), seqlen, dtype=torch.int32, device=dev)
    return q, k_cache, v_cache, block_table, cache_seqlens


def call(fn, q, kc, vc, bt, cs, num_splits):
    return fn(
        q=q, k_cache=kc, v_cache=vc,
        block_table=bt, cache_seqlens=cs,
        softmax_scale=HEAD_DIM ** -0.5,
        causal=True, num_splits=num_splits,
    )


def timeit(f, warmup=30, iters=100):
    """DEPRECATED for tiny kernels.

    CUDA-event wall time around a ~30us kernel measures host launch cost, not
    GPU execution (back-to-back eager launches serialize with enqueue). That is
    why this disagreed with the end-to-end numbers by ~30x. Use gpu_time()
    below, which reads the profiler's device_time.
    """
    import torch
    for _ in range(warmup):
        f()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        f()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1000.0


def gpu_time(f, warmup=30, iters=50):
    """Pure GPU microseconds per call, plus a per-kernel breakdown.

    In production the decode step runs from a replayed CUDA graph, so host
    launch cost is zero and only GPU execution matters. Summing the profiler's
    device_time over steady-state iterations reproduces that.
    """
    import torch
    for _ in range(warmup):
        f()
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as p:
        for _ in range(iters):
            f()
        torch.cuda.synchronize()
    total = 0.0
    per_kernel: dict[str, float] = {}
    for ev in p.events():
        if ev.device_type != torch.autograd.DeviceType.CUDA or not ev.name:
            continue
        total += ev.device_time
        key = ("combine" if "combine" in ev.name
               else "splitkv" if "splitkv" in ev.name
               else "other")
        per_kernel[key] = per_kernel.get(key, 0.0) + ev.device_time
    return (total / iters,
            {k: v / iters for k, v in per_kernel.items()})


def kernels_of(f):
    import torch
    f()
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as p:
        f()
        torch.cuda.synchronize()
    out = []
    for ev in p.events():
        if ev.device_type == torch.autograd.DeviceType.CUDA and ev.name:
            out.append((ev.name, ev.count))
    return out


def main() -> None:
    import torch
    import vllm  # noqa: F401  bring MACA runtime up first (bare flash_attn segfaults)

    from flash_attn import flash_attn_with_kvcache as fa

    global DT
    DT = torch.bfloat16
    dev = "cuda"

    seqlens = [32, 64, 128, 256, 640, 1024, 1536, 2048]
    batches = [1, 8, 32]
    spl = [0, 1, 2, 4, 8, 16, 32, 64]

    print("=" * 108)
    print("FAST DECODE-ATTENTION num_splits SWEEP  (flash_attn_with_kvcache, "
          "MiniCPM5-2B shapes)")
    print("  metric = PURE GPU time per call (profiler device_time), i.e. what a")
    print("  replayed decode CUDA graph actually pays. CUDA-event timing of these")
    print("  ~30us kernels measures launch overhead instead, and misled an earlier")
    print("  version of this script by ~30x.")
    print("=" * 108)
    print(f"  heads={NUM_HEADS} kv_heads={NUM_KV_HEADS} head_dim={HEAD_DIM} "
          f"layers={N_LAYERS} dtype=bfloat16")

    # --- find a block size the paged kernel accepts -----------------------
    blk = None
    for cand in (16, 32, 64, 128, 256):
        try:
            q, kc, vc, bt, cs = build(512, 1, cand, dev, DT)
            call(fa, q, kc, vc, bt, cs, 1)
            torch.cuda.synchronize()
            blk = cand
            break
        except Exception as exc:  # noqa: BLE001
            print(f"  block_size={cand}: {type(exc).__name__}: {str(exc)[:70]}")
    if blk is None:
        print("  no block size worked")
        return
    print(f"  using block_size={blk}\n")

    resid = {}
    for batch in batches:
        print("=" * 108)
        print(f"batch = {batch}      (values = pure GPU us/call; "
              f"per-step cost = x{N_LAYERS} layers)")
        print("=" * 108)
        head = (f"  {'seqlen':>7} | " +
                " ".join(f"{('ns=' + str(s)):>8}" for s in spl) +
                f" | {'best':>5} {'ns=0':>8} {'gain%':>7} {'comb@0':>7} {'comb@best':>9}")
        print(head)
        print("  " + "-" * (len(head) - 2))
        for seqlen in seqlens:
            q, kc, vc, bt, cs = build(seqlen, batch, blk, dev, DT)
            row = {}
            brk = {}
            for ns in spl:
                try:
                    us, parts = gpu_time(lambda ns=ns: call(fa, q, kc, vc, bt, cs, ns))
                    row[ns] = us
                    brk[ns] = parts
                except Exception:  # noqa: BLE001
                    row[ns] = None
                    brk[ns] = {}
            valid = {k: v for k, v in row.items() if v}
            best_ns = min(valid, key=valid.get)
            heur = row.get(0)
            bestv = valid[best_ns]
            gain = (heur - bestv) / heur * 100 if heur else 0.0
            cells = " ".join(
                (f"{row[s]:>8.1f}" if row.get(s) else f"{'--':>8}") for s in spl)
            print(f"  {seqlen:>7} | {cells} | {best_ns:>5} {heur:>8.1f} "
                  f"{gain:>6.2f}% "
                  f"{brk.get(0, {}).get('combine', 0.0):>7.1f} "
                  f"{brk.get(best_ns, {}).get('combine', 0.0):>9.1f}")
            resid.setdefault(str(batch), {})[str(seqlen)] = {
                "us": {str(k): v for k, v in row.items()},
                "breakdown": {str(k): v for k, v in brk.items()},
                "best_num_splits": best_ns,
                "heuristic_us": heur,
                "best_us": bestv,
                "gain_pct": gain,
            }
        print()

    # --- what kernels does each regime launch? ---------------------------
    print("=" * 108)
    print("KERNELS LAUNCHED (batch=1, seqlen=1024)")
    print("=" * 108)
    q, kc, vc, bt, cs = build(1024, 1, blk, dev, DT)
    for ns in (0, 1, 4, 16):
        ks = kernels_of(lambda ns=ns: call(fa, q, kc, vc, bt, cs, ns))
        print(f"  num_splits={ns:<3} -> " +
              ", ".join(f"{n.split('(')[0]}x{c}" for n, c in ks))

    print()
    print("=" * 108)
    print("VERDICT  (per-decode-step = us/call x 42 layers)")
    print("=" * 108)
    print("  gain% = how much better the best num_splits is than the MACA")
    print("          heuristic (num_splits=0) at that (batch, seqlen).")
    for b, rows in resid.items():
        print(f"\n    batch={b}")
        for sl, d in sorted(rows.items(), key=lambda kv: int(kv[0])):
            step_save = (d["heuristic_us"] - d["best_us"]) * N_LAYERS
            print(f"      seqlen={sl:>5}  best ns={d['best_num_splits']:>3}  "
                  f"heur {d['heuristic_us']:6.1f} -> {d['best_us']:6.1f} us/call  "
                  f"gain {d['gain_pct']:5.2f}%  "
                  f"saves {step_save:6.1f} us/step")

    (OUT / "fast_sweep.json").write_text(json.dumps(resid, indent=2))
    print(f"\n  wrote {OUT / 'fast_sweep.json'}")
    print("FAST_ATTN_SWEEP_DONE", flush=True)


if __name__ == "__main__":
    main()
