#!/usr/bin/env python3
"""Split the mcTracer marker window into individual forward passes and
attribute per-op device time separately for prefill and decode.

Motivation: the profile window contains 1 prefill + 15 decode forwards. Mixing
them hides that prefill (T=512) and decode (T=1) have completely different cost
structure -- which is exactly what determines whether a fusion pays off.

Segmentation: each forward launches the same op multiset (42 rope, 42 silu,
84 norm, 168 GEMM ...), and prefill kernels are ~100x longer than decode ones.
So we cut the timeline at the *largest* time gaps between consecutive kernel
launches; the resulting segments are the individual forwards.

Run:  python /root/src/mctrace_forwards.py <trace.json>
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
    ("reshape_and_cache_flash_kernel", "kvcache_write"),
    ("argmax_kernel", "argmax(sampler)"),
    ("embedding_kernel", "embedding"),
    ("_copy_kernel_kernel_rank", "copy"),
    ("_to_copy_func_kernel", "copy"),
    ("fill_scalar_func_kernel", "fill"),
    ("add_func_kernel", "add"),
    ("sub_func_tensor_scalar_kernel", "sub"),
    ("zero_persistent_kernel", "splitk_zero"),
    ("_compute_slot_mapping_kernel", "slot_mapping"),
    ("_index_jit_function", "index"),
    ("memcpy", "memcpy"),
    ("mcprof_", "MARKER"),
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
    args = ap.parse_args()

    with open(args.trace) as f:
        ev = json.load(f).get("traceEvents", [])

    begins, ends = [], []
    for e in ev:
        n = e.get("name", "")
        if isinstance(e.get("ts"), int):
            if "mcprof_begin_marker" in n:
                begins.append(e["ts"])
            elif "mcprof_end_marker" in n:
                ends.append(e["ts"])
    if not begins or not ends:
        print("markers not found")
        return
    lo, hi = min(begins), max(ends)

    # all GPU kernels inside the window, sorted by launch time
    ks = []
    for e in ev:
        ts, dur = e.get("ts"), e.get("dur")
        if e.get("ph") != "X" or not isinstance(ts, int) or not isinstance(dur, (int, float)):
            continue
        if not (lo <= ts <= hi):
            continue
        fam = family_of(e.get("name", ""))
        if fam in ("MARKER", "other") or e.get("name", "").startswith("mc"):
            continue
        ks.append((ts, float(dur), fam, e.get("name", "")))
    ks.sort()
    print(f"{len(ks)} GPU kernel launches in window ({(hi - lo) / 1e6:.2f} ms)")

    # cut at the largest inter-launch gaps -> individual forwards
    gaps = [(ks[i + 1][0] - ks[i][0], i) for i in range(len(ks) - 1)]
    gaps.sort(reverse=True)

    N_FWD = 16
    cuts = sorted(i for _, i in gaps[: N_FWD - 1])
    bounds = [-1] + cuts + [len(ks) - 1]

    segs = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        seg = ks[a + 1: b + 1]
        if not seg:
            continue
        span = (seg[-1][0] + seg[-1][1] - seg[0][0]) / 1e6
        gpu = sum(s[1] for s in seg) / 1e6
        n_lin = sum(1 for s in seg if s[2].startswith("GEMM"))
        segs.append({"span_ms": span, "gpu_ms": gpu, "nkern": len(seg),
                     "n_gemm": n_lin, "kernels": seg})

    print(f"\ncut into {len(segs)} segments by largest launch gaps")
    print(f"  {'#':>3s} {'span_ms':>9s} {'gpu_ms':>8s} {'nkern':>6s} {'nGEMM':>6s} "
          f"{'avg GEMM us':>12s}  guess")
    for i, s in enumerate(segs):
        gemms = [k[1] for k in s["kernels"] if k[2].startswith("GEMM")]
        avg = sum(gemms) / len(gemms) / 1e3 if gemms else 0
        phase = "PREFILL" if avg > 200 else "decode"
        s["phase"] = phase
        s["avg_gemm_us"] = avg
        print(f"  {i:3d} {s['span_ms']:9.2f} {s['gpu_ms']:8.2f} {s['nkern']:6d} "
              f"{s['n_gemm']:6d} {avg:12.1f}  {phase}")

    # aggregate per phase
    agg: dict[str, dict[str, dict]] = {}
    for phase in ("PREFILL", "decode"):
        sel = [s for s in segs if s["phase"] == phase]
        if not sel:
            continue
        fam_t: dict[str, float] = defaultdict(float)
        fam_n: dict[str, int] = defaultdict(int)
        for s in sel:
            for _, dur, fam, _nm in s["kernels"]:
                fam_t[fam] += dur
                fam_n[fam] += 1
        n = len(sel)
        agg[phase] = {"forwards": n,
                      "span_ms_avg": sum(s["span_ms"] for s in sel) / n,
                      "gpu_ms_avg": sum(s["gpu_ms"] for s in sel) / n,
                      "by_family": {k: {"ms_per_fwd": v / 1e6 / n,
                                        "calls_per_fwd": fam_n[k] / n,
                                        "us_each": v / fam_n[k] / 1e3}
                                    for k, v in fam_t.items()}}

    for phase, a in agg.items():
        print("\n" + "=" * 88)
        print(f"{phase}: {a['forwards']} forward(s) | span {a['span_ms_avg']:.2f} ms/fwd "
              f"| GPU {a['gpu_ms_avg']:.2f} ms/fwd "
              f"({a['gpu_ms_avg'] / a['span_ms_avg'] * 100:.1f}% busy)")
        print("=" * 88)
        print(f"  {'family':22s} {'ms/fwd':>9s} {'share':>7s} {'calls/fwd':>10s} {'us each':>9s}")
        tot = sum(v["ms_per_fwd"] for v in a["by_family"].values())
        for fam, v in sorted(a["by_family"].items(), key=lambda kv: -kv[1]["ms_per_fwd"]):
            print(f"  {fam:22s} {v['ms_per_fwd']:9.3f} {v['ms_per_fwd'] / tot * 100:6.1f}% "
                  f"{v['calls_per_fwd']:10.1f} {v['us_each']:9.2f}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"segments": [{k: v for k, v in s.items() if k != "kernels"}
                                    for s in segs], "by_phase": agg}, f, indent=2)
        print(f"\nwrote {args.out}")

    print("\nMCTRACE_FORWARDS_DONE")


if __name__ == "__main__":
    main()
