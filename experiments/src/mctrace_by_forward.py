#!/usr/bin/env python3
"""Segment the mcTracer window into forwards using `embedding_kernel` as the
forward boundary (it launches exactly once per forward), then attribute per-op
device time to prefill vs decode.

This replaces gap-based segmentation, which merges/splits forwards incorrectly.

Run:  python /root/src/mctrace_by_forward.py <trace.json>
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

FAMILY_RULES: list[tuple[str, str]] = [
    ("silu_and_mul_kernel", "silu_and_mul"),
    ("fused_add_rms_norm_kernel", "rms_norm(fused_add)"),
    ("rms_norm_kernel", "rms_norm"),
    ("apply_rotary_pos_emb", "rope"),
    ("linear_kernel", "GEMM linear"),
    ("mm_kernel", "GEMM mm"),
    ("flash_fwd_splitkv_combine_kernel", "attention"),
    ("flash_fwd_splitkv_kernel", "attention"),
    ("flash_fwd_kernel", "attention"),
    ("reshape_and_cache_flash_kernel", "kvcache_write"),
    ("argmax_kernel", "argmax(sampler)"),
    ("embedding_kernel", "BOUNDARY"),
    ("_copy_kernel_kernel_rank", "copy"),
    ("_to_copy_func_kernel", "copy"),
    ("fill_scalar_func_kernel", "fill"),
    ("add_func_kernel", "add"),
    ("sub_func", "sub"),
    ("zero_persistent_kernel", "splitk_zero"),
    ("_compute_slot_mapping_kernel", "slot_mapping"),
    ("_index_jit_function", "index"),
    ("reduce_then_scan", "prefill_scan"),
    ("memcpy", "memcpy"),
]


def family_of(name: str) -> str:
    low = name.lower()
    for key, fam in FAMILY_RULES:
        if key.lower() in low:
            return fam
    if low.startswith("mc"):
        return "HOST"
    return "other"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--json", dest="out", default=None)
    args = ap.parse_args()

    with open(args.trace) as f:
        ev = json.load(f).get("traceEvents", [])

    begins = [e["ts"] for e in ev
              if "mcprof_begin_marker" in e.get("name", "") and isinstance(e.get("ts"), int)]
    if not begins:
        print("begin marker missing")
        return
    lo = min(begins)
    hi = max(e["ts"] for e in ev if isinstance(e.get("ts"), int))

    ks = []
    for e in ev:
        ts, dur = e.get("ts"), e.get("dur")
        if e.get("ph") != "X" or not isinstance(ts, int) or not isinstance(dur, (int, float)):
            continue
        if not (lo <= ts <= hi):
            continue
        fam = family_of(e.get("name", ""))
        if fam == "HOST":
            continue
        ks.append((ts, float(dur), fam, e.get("name", "")))
    ks.sort()

    bounds = [i for i, k in enumerate(ks) if k[2] == "BOUNDARY"]
    print(f"window {(hi - lo) / 1e6:.1f} ms, {len(ks)} kernels, "
          f"{len(bounds)} forward boundaries\n")

    segs = []
    for a, b in zip(bounds, bounds[1:]):
        segs.append(ks[a:b])
    if bounds:
        segs.append(ks[bounds[-1]:])

    print(f"  {'#':>3s} {'t_start_ms':>10s} {'span_ms':>9s} {'gpu_ms':>8s} {'nkern':>6s} "
          f"{'nGEMM':>6s} {'max GEMM us':>12s} {'rope us':>8s} {'silu us':>8s}  phase")
    for i, seg in enumerate(segs):
        if not seg:
            continue
        span = (seg[-1][0] + seg[-1][1] - seg[0][0]) / 1e6
        gpu = sum(s[1] for s in seg) / 1e6
        g = [s[1] for s in seg if s[2].startswith("GEMM")]
        ro = [s[1] for s in seg if s[2] == "rope"]
        si = [s[1] for s in seg if s[2] == "silu_and_mul"]
        mx = max(g) / 1e3 if g else 0
        phase = "PREFILL" if mx > 300 else "decode"
        print(f"  {i:3d} {(seg[0][0] - lo) / 1e6:10.1f} {span:9.2f} {gpu:8.2f} {len(seg):6d} "
              f"{len(g):6d} {mx:12.1f} "
              f"{(sum(ro) / len(ro) / 1e3 if ro else 0):8.2f} "
              f"{(sum(si) / len(si) / 1e3 if si else 0):8.2f}  {phase}")

    # aggregate
    agg = {}
    for phase in ("PREFILL", "decode"):
        sel = [s for s in segs if s and
               ("PREFILL" if max([k[1] for k in s if k[2].startswith("GEMM")] or [0]) / 1e3 > 300
                else "decode") == phase]
        if not sel:
            continue
        ft: dict[str, float] = defaultdict(float)
        fn: dict[str, int] = defaultdict(int)
        for s in sel:
            for _, d, f, _n in s:
                ft[f] += d
                fn[f] += 1
        n = len(sel)
        agg[phase] = {"forwards": n,
                      "span_ms_avg": sum((s[-1][0] + s[-1][1] - s[0][0]) / 1e6 for s in sel) / n,
                      "gpu_ms_avg": sum(sum(k[1] for k in s) for s in sel) / 1e6 / n,
                      "by_family": {k: {"ms_per_fwd": v / 1e6 / n,
                                        "calls_per_fwd": fn[k] / n,
                                        "us_each": v / fn[k] / 1e3}
                                    for k, v in ft.items()}}

    for phase, a in agg.items():
        print("\n" + "=" * 90)
        print(f"{phase}: {a['forwards']} fwd | span {a['span_ms_avg']:.2f} ms/fwd | "
              f"GPU {a['gpu_ms_avg']:.2f} ms/fwd "
              f"({a['gpu_ms_avg'] / a['span_ms_avg'] * 100:.1f}% busy)")
        print("=" * 90)
        tot = sum(v["ms_per_fwd"] for v in a["by_family"].values())
        print(f"  {'family':22s} {'ms/fwd':>9s} {'share':>7s} {'calls/fwd':>10s} {'us each':>9s}")
        for fam, v in sorted(a["by_family"].items(), key=lambda kv: -kv[1]["ms_per_fwd"]):
            print(f"  {fam:22s} {v['ms_per_fwd']:9.3f} {v['ms_per_fwd'] / tot * 100:6.1f}% "
                  f"{v['calls_per_fwd']:10.1f} {v['us_each']:9.2f}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"by_phase": agg}, f, indent=2)
        print(f"\nwrote {args.out}")
    print("\nBY_FORWARD_DONE")


if __name__ == "__main__":
    main()
