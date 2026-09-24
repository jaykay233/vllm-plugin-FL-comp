#!/usr/bin/env python3
"""Task 2: per-op attribution for MiniCPM5-2B's 42 layers.

Measures the PLUGIN dispatch path (`CachedOp(op)` -> backend impl -> kernel)
against FlagGems' own kernel and the reference torch path, at the two shapes
that actually occur:

  decode  T=1    (one token per step, 42x per generated token)
  prefill T=512  (chunked prefill, 42x per 512-token prompt)

Then scales by calls-per-layer x 42 layers to attribute time to each op.

Using CachedOp directly (rather than instantiating the vLLM layer classes)
keeps this independent of CompilationConfig while measuring the same code the
OOT layers invoke from `forward_oot`.

Run:  conda activate mx && python /root/src/profile_hot_ops.py
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import torch

OUT = Path("/root/bench_results/hot_op_attribution")
OUT.mkdir(parents=True, exist_ok=True)

# ---- MiniCPM5-2B geometry (from config.json) ----
HIDDEN = 2048
INTER = 6144
NUM_LAYERS = 42
NUM_Q_HEADS = 16
NUM_KV_HEADS = 2
HEAD_DIM = 128
ROPE_THETA = 5_000_000.0
RMS_EPS = 1e-6

RMS_CALLS_PER_LAYER = 2     # input_layernorm + post_attention_layernorm
SILU_CALLS_PER_LAYER = 1
ROPE_CALLS_PER_LAYER = 1

# Reference budgets measured on this box (TTFT/TPOT at 512-in / 128-out, c1)
BUDGETS = {
    "decode_T1": ("TPOT", 10.13),    # ms, compile mode (vllm bench serve)
    "prefill_T512": ("TTFT", 70.62),  # ms
}


class _StubRMS:
    """Minimal stand-in carrying the attrs the FlagGems impl reads off `obj`."""

    def __init__(self, weight, eps):
        self.weight = weight
        self.variance_epsilon = eps


def bench(fn, n_warmup=20, n_iters=100):
    """Return (wall_ms_per_call, gpu_ms_per_call).

    wall: amortized host-side per-call cost (~launch bound at small shapes).
    gpu : pure device time measured with CUDA events (excludes host sync).
    Both are amortized over n_iters so the per-call fixed sync cost of a naive
    `sync(); t0; fn(); sync()` loop does not dominate small shapes.
    """
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_iters):
        fn()
    torch.cuda.synchronize()
    wall = (time.perf_counter() - t0) / n_iters * 1e3

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    gpu = start.elapsed_time(end) / n_iters

    return wall, gpu


def main() -> None:
    dev, dt = "cuda", torch.bfloat16
    print(f"device: {torch.cuda.get_device_name(0)}")

    from vllm_fl.dispatch import CachedOp
    from vllm_fl.dispatch import get_default_manager

    mgr = get_default_manager()
    mgr.ensure_initialized()

    op_rms = CachedOp("rms_norm")
    op_silu = CachedOp("silu_and_mul")
    op_rope = CachedOp("rotary_embedding")

    # report which backend each dispatch op resolves to
    print("\n-- dispatch resolution --")
    for name in ("rms_norm", "silu_and_mul", "rotary_embedding"):
        try:
            impl = mgr.resolve(name)
            print(f"  {name:20s} -> {getattr(impl, '__name__', impl)}")
        except Exception as e:
            print(f"  {name:20s} -> <error {e}>")

    from flag_gems.modules.normalization import gems_rms_forward
    from flag_gems.modules.activation import gems_silu_and_mul
    from flag_gems.modules.rotary_embedding import gems_rope_forward
    import torch.nn.functional as F

    results: dict[str, dict] = {}

    for tag, T in (("decode_T1", 1), ("prefill_T512", 512)):
        print("\n" + "=" * 78)
        print(f"shape group: {tag} (tokens={T})")
        print("=" * 78)
        grp: dict[str, dict] = {}

        # ---------- rms_norm ----------
        x = torch.randn(T, HIDDEN, device=dev, dtype=dt)
        res = torch.randn(T, HIDDEN, device=dev, dtype=dt)
        w = torch.randn(HIDDEN, device=dev, dtype=dt)
        stub = _StubRMS(w, RMS_EPS)

        grp["rms_norm"] = {}
        for k, fn in (
            ("dispatch", lambda: op_rms(stub, x, res)),
            ("dispatch_no_res", lambda: op_rms(stub, x, None)),
            ("raw_flaggems", lambda: gems_rms_forward(x, res.clone(), w, RMS_EPS)),
            ("torch_ref", lambda: (lambda v: w * (v * torch.rsqrt(
                v.pow(2).mean(-1, keepdim=True) + RMS_EPS)))(x + res)),
        ):
            wl, gp = bench(fn)
            grp["rms_norm"][f"{k}_wall_us"] = wl * 1e3
            grp["rms_norm"][f"{k}_gpu_us"] = gp * 1e3

        # ---------- silu_and_mul ----------
        xi = torch.randn(T, 2 * INTER, device=dev, dtype=dt)
        h = INTER
        grp["silu_and_mul"] = {}
        for k, fn in (
            ("dispatch", lambda: op_silu(None, xi)),
            ("raw_flaggems", lambda: gems_silu_and_mul(xi[..., :h], xi[..., h:])),
            ("torch_ref", lambda: F.silu(xi[..., :h]) * xi[..., h:]),
        ):
            wl, gp = bench(fn)
            grp["silu_and_mul"][f"{k}_wall_us"] = wl * 1e3
            grp["silu_and_mul"][f"{k}_gpu_us"] = gp * 1e3

        # ---------- rotary_embedding ----------
        # FlagGems contract: q (*, q_heads, head_dim), k (*, k_heads, head_dim),
        # cos/sin (max_seq_len, head_dim // 2), position_ids == q.shape[:-2].
        q = torch.randn(T, NUM_Q_HEADS, HEAD_DIM, device=dev, dtype=dt)
        k = torch.randn(T, NUM_KV_HEADS, HEAD_DIM, device=dev, dtype=dt)
        pos = torch.arange(T, device=dev, dtype=torch.long)
        cos = torch.randn(T, HEAD_DIM // 2, device=dev, dtype=dt)
        sin = torch.randn(T, HEAD_DIM // 2, device=dev, dtype=dt)

        grp["rotary_embedding"] = {}
        for kk, fn in (
            ("dispatch", lambda: op_rope(None, q.clone(), k.clone(), cos, sin, pos, False, True)),
            ("raw_flaggems", lambda: gems_rope_forward(
                q.clone(), k.clone(), cos, sin,
                position_ids=pos, rotary_interleaved=False, inplace=True)),
        ):
            wl, gp = bench(fn)
            grp["rotary_embedding"][f"{kk}_wall_us"] = wl * 1e3
            grp["rotary_embedding"][f"{kk}_gpu_us"] = gp * 1e3

        results[tag] = grp
        print(f"  {'op':20s} {'variant':18s} {'wall us':>9s} {'gpu us':>9s}")
        for op, m in grp.items():
            for variant in ("dispatch", "dispatch_no_res", "raw_flaggems", "torch_ref"):
                wk = f"{variant}_wall_us"
                if wk in m:
                    print(f"  {op:20s} {variant:18s} {m[wk]:9.2f} {m[f'{variant}_gpu_us']:9.2f}")

    # ---------------- aggregation ----------------
    print("\n" + "=" * 78)
    print("attribution: x calls-per-layer x 42 layers")
    print("=" * 78)

    calls = {
        "rms_norm": RMS_CALLS_PER_LAYER,
        "silu_and_mul": SILU_CALLS_PER_LAYER,
        "rotary_embedding": ROPE_CALLS_PER_LAYER,
    }
    agg: dict[str, dict] = {}
    for tag in ("decode_T1", "prefill_T512"):
        label, budget = BUDGETS[tag]
        print(f"\n[{tag}]  budget = measured {label} = {budget} ms")
        print(f"  {'op':20s} {'calls':>6s} | "
              f"{'wall ms':>8s} {'share':>6s} | {'gpu ms':>8s} {'share':>6s} | "
              f"{'flagos/direct':>13s}")
        tot_w = tot_g = 0.0
        rows = {}
        for op, per in calls.items():
            m = results[tag][op]
            n = per * NUM_LAYERS
            wall = m["dispatch_wall_us"] * n / 1e3
            gpu = m["dispatch_gpu_us"] * n / 1e3
            raw_gpu = m["raw_flaggems_gpu_us"]
            disp_gpu = m["dispatch_gpu_us"]
            ovh = disp_gpu - raw_gpu
            tot_w += wall
            tot_g += gpu
            rows[op] = {
                "calls_per_step": n,
                "dispatch_wall_ms": wall, "dispatch_gpu_ms": gpu,
                "raw_flaggems_gpu_ms": raw_gpu * n / 1e3,
                "overhead_us_per_call": ovh,
            }
            # how much slower/faster the FlagGems kernel is vs the torch reference
            ratio = ""
            if "torch_ref_gpu_us" in m and m["torch_ref_gpu_us"] > 0:
                r = raw_gpu / m["torch_ref_gpu_us"]
                ratio = f"{r:6.2f}x"
            print(f"  {op:20s} {n:6d} | {wall:8.3f} {wall/budget*100:5.1f}% | "
                  f"{gpu:8.3f} {gpu/budget*100:5.1f}% | {ratio:>13s}")
        print(f"  {'SUM':20s} {'':6s} | {tot_w:8.3f} {tot_w/budget*100:5.1f}% | "
              f"{tot_g:8.3f} {tot_g/budget*100:5.1f}% |")
        rows["_totals"] = {"dispatch_wall_ms": tot_w, "dispatch_gpu_ms": tot_g,
                           "budget_ms": budget,
                           "share_wall_pct": tot_w / budget * 100,
                           "share_gpu_pct": tot_g / budget * 100}
        agg[tag] = rows

    print("\n" + "=" * 78)
    print("interpretation")
    print("=" * 78)
    print("  wall = host-amortized per-call cost; at T=1 this is launch/dispatch bound.")
    print("  gpu  = device time from CUDA events; compare against the step budget.")
    print("  `flagos/direct` = raw FlagGems kernel GPU time vs the torch reference,")
    print("  >1 means the FlagGems kernel is SLOWER than plain torch on this chip.")
    print(f"  these 3 ops: {agg['decode_T1']['_totals']['share_gpu_pct']:.1f}% of TPOT (gpu view), "
          f"{agg['prefill_T512']['_totals']['share_gpu_pct']:.1f}% of TTFT (gpu view)")

    (OUT / "attribution.json").write_text(json.dumps(
        {"results": results, "aggregation": agg, "budgets": BUDGETS,
         "geometry": {"hidden": HIDDEN, "inter": INTER, "layers": NUM_LAYERS,
                      "q_heads": NUM_Q_HEADS, "kv_heads": NUM_KV_HEADS,
                      "head_dim": HEAD_DIM}}, indent=2))
    print(f"\nWrote {OUT / 'attribution.json'}")
    print("HOT_OP_ATTRIBUTION_DONE", flush=True)


if __name__ == "__main__":
    main()
