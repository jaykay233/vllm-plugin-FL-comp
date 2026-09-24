#!/usr/bin/env python3
"""Aggregate an mcTracer (Perfetto) trace inside the marker-bracketed region.

Windows on the `mcprof_begin_marker` / `mcprof_end_marker` kernels so that model
loading and warmup are excluded, then attributes GPU kernel time by family.

Run:  python /root/src/mctrace_analyze.py <trace.json> [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict

# Kernel-name -> family. MACA names triton kernels "<fn>_kernel[_rank_N]",
# so match on substrings rather than exact names.
FAMILY_RULES: list[tuple[str, str]] = [
    # FlagGems triton kernels
    ("silu_and_mul_kernel", "silu_and_mul (GEMS triton)"),
    ("fused_add_rms_norm_kernel", "rms_norm (GEMS triton)"),
    ("rms_norm_kernel", "rms_norm (GEMS triton)"),
    ("apply_rotary_pos_emb_inplace_kernel", "rope (GEMS triton)"),
    ("apply_rotary_pos_emb_kernel", "rope (GEMS triton)"),
    ("linear_kernel", "GEMM linear (GEMS triton)"),
    ("mm_kernel", "GEMM mm (GEMS triton)"),
    # vendor / MACA
    ("flash_fwd_splitkv_combine_kernel", "attention flash (vendor)"),
    ("flash_fwd_splitkv_kernel", "attention flash (vendor)"),
    ("flash_fwd_kernel", "attention flash (vendor)"),
    ("reshape_and_cache_flash_kernel", "kvcache (vendor)"),
    # misc triton elementwise from FlagGems
    ("_copy_kernel_kernel_rank", "copy (GEMS triton)"),
    ("fill_scalar_func_kernel", "fill (GEMS triton)"),
    ("true_div_func_kernel", "div (GEMS triton)"),
    ("lt_func_kernel", "lt (GEMS triton)"),
    ("le_func_kernel", "le (GEMS triton)"),
    ("argmax_kernel", "argmax (GEMS triton)"),
    ("softmax_kernel", "softmax (GEMS triton)"),
    ("cat_kernel", "cat (GEMS triton)"),
    ("sort_kernel", "sort (GEMS triton)"),
    ("gather_kernel", "gather (GEMS triton)"),
    ("scatter_kernel", "scatter (GEMS triton)"),
    ("cumsum_kernel", "cumsum (GEMS triton)"),
    ("add_func_kernel", "add (GEMS triton)"),
    ("mul_func_kernel", "mul (GEMS triton)"),
    ("where_kernel", "where (GEMS triton)"),
    ("copy_kernel", "copy (GEMS triton)"),
    # host / framework noise that should be excluded from GPU attribution
    ("mcGetDevice", "HOST api"),
    ("mcLaunchKernel", "HOST api"),
    ("mcModuleLaunchKernel", "HOST api"),
    ("mcMemcpyAsync", "HOST api"),
    ("mcDeviceSynchronize", "HOST api"),
    ("mcStreamSynchronize", "HOST api"),
    ("mcEvent", "HOST api"),
    ("mcPointer", "HOST api"),
    ("mcMalloc", "HOST api"),
    ("mcFree", "HOST api"),
    ("mcStream", "HOST api"),
    ("mcDevice", "HOST api"),
    ("mcCtx", "HOST api"),
    ("mcFunc", "HOST api"),
    ("mcModule", "HOST api"),
    ("sweep", "autotune"),
    ("zero_persistent_kernel", "autotune/zero-persist"),
    ("full_kernel_scale", "autotune/scale"),
    ("compute_global_hist_kernel", "autotune/hist"),
    ("_scatter_jit_function", "autotune/scatter"),
    ("zeros_kernel", "zeros"),
    ("memcpy", "memcpy"),
    ("Launch", "HOST api"),
]


def family_of(name: str) -> str:
    low = name.lower()
    for key, fam in FAMILY_RULES:
        if key.lower() in low:
            return fam
    return "other"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--json", dest="out", default=None)
    ap.add_argument("--top", type=int, default=30)
    args = ap.parse_args()

    print(f"loading {args.trace} ...", flush=True)
    with open(args.trace) as f:
        d = json.load(f)
    ev = d.get("traceEvents", [])
    print(f"{len(ev)} events", flush=True)

    # locate marker kernels
    begins, ends = [], []
    for e in ev:
        n = e.get("name", "")
        if "mcprof_begin_marker" in n and isinstance(e.get("ts"), int):
            begins.append(e["ts"])
        elif "mcprof_end_marker" in n and isinstance(e.get("ts"), int):
            ends.append(e["ts"])

    if not begins or not ends:
        # mcTracer sometimes drops the final kernel when it flushes at process
        # exit. The measured region is still well defined: it ends at the last
        # traced event, which is the last model kernel.
        if not begins:
            print("ERROR: begin marker not found; cannot window the trace")
            return
        all_ts = [e["ts"] for e in ev if isinstance(e.get("ts"), int)]
        ends = [max(all_ts)]
        print("NOTE: end marker missing; windowing on trace end instead")
    lo, hi = min(begins), max(ends)
    # mcTracer emits ts and dur in NANOSECONDS.
    window_ms = (hi - lo) / 1e6
    print(f"marker window: {window_ms:.3f} ms  ({lo} -> {hi})", flush=True)

    # aggregate duration events inside the window
    fam_tot: dict[str, float] = defaultdict(float)
    fam_cnt: dict[str, int] = defaultdict(int)
    raw: list[dict] = []
    for e in ev:
        ts = e.get("ts")
        dur = e.get("dur")
        if e.get("ph") != "X" or not isinstance(ts, int) or not isinstance(dur, (int, float)):
            continue
        if not (lo <= ts <= hi):
            continue
        name = e.get("name", "")
        fam = family_of(name)
        if fam == "HOST api":
            continue
        fam_tot[fam] += float(dur)
        fam_cnt[fam] += 1
        raw.append({"kernel": name, "family": fam, "dur_ns": float(dur), "count": 1, "ts": ts})

    total_ns = sum(fam_tot.values())
    total_ms = total_ns / 1e6
    print(f"\n-- GPU kernel time inside window: {total_ms:.3f} ms "
          f"(window {window_ms:.3f} ms => {total_ms / window_ms * 100:.1f}% busy) --")
    print(f"  {'family':34s} {'total_ms':>10s} {'share':>7s} {'calls':>7s} {'avg_us':>10s}")
    for fam, tot in sorted(fam_tot.items(), key=lambda kv: -kv[1]):
        print(f"  {fam:34s} {tot / 1e6:10.3f} {tot / total_ns * 100:6.1f}% "
              f"{fam_cnt[fam]:7d} {tot / fam_cnt[fam] / 1e3:10.2f}")

    # per-kernel rollup
    kb: dict[str, dict] = defaultdict(lambda: {"ns": 0.0, "n": 0})
    for r in raw:
        kb[r["kernel"]]["ns"] += r["dur_ns"]
        kb[r["kernel"]]["n"] += 1
    print(f"\n-- top {args.top} individual kernels by total time --")
    print(f"  {'total_ms':>10s} {'share':>7s} {'calls':>7s} {'avg_us':>9s}  kernel")
    for k, v in sorted(kb.items(), key=lambda kv: -kv[1]["ns"])[: args.top]:
        print(f"  {v['ns'] / 1e6:10.3f} {v['ns'] / total_ns * 100:6.1f}% "
              f"{v['n']:7d} {v['ns'] / v['n'] / 1e3:9.2f}  {k[:78]}")

    if args.out:
        rec = {
            "trace": args.trace,
            "window_ms": window_ms,
            "gpu_total_ms": total_ms,
            "gpu_busy_pct": total_ms / window_ms * 100,
            "by_family": {k: {"ms": v / 1e6, "calls": fam_cnt[k]}
                          for k, v in sorted(fam_tot.items(), key=lambda kv: -kv[1])},
            "by_kernel": {k: {"ms": v["ns"] / 1e6, "calls": v["n"]}
                          for k, v in sorted(kb.items(), key=lambda kv: -kv[1]["ns"])},
        }
        with open(args.out, "w") as f:
            json.dump(rec, f, indent=2)
        print(f"\nwrote {args.out}")

    print("\nMCTRACE_ANALYZE_DONE", flush=True)


if __name__ == "__main__":
    main()
