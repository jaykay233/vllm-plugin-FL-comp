"""Per-step breakdown of a vLLM worker torch-profiler trace.

Usage: analyze.py <trace.json> [<trace.json> ...]

Steps are delimited by the worker's "execute_new_*_cached_*" annotation.  The
forward is the span of "Call CompiledFxGraph" ranges inside the step; host time
before it is preprocess (input prep, metadata build, batch reorder), after it
is postprocess (sampling, output).  Steps are classed by which attention
kernels they launch.
"""

from __future__ import annotations

import bisect
import collections
import json
import sys

SYNC = ("cudaStreamSynchronize", "cudaEventSynchronize", "cudaDeviceSynchronize")


def kclass(name: str) -> str:
    n = name.lower()
    if "unified_attention" in n:
        return "attn_triton"
    if "compute_flash_attn" in n or "flash_attn" in n:
        return "attn_fa"
    if "paged" in n or "flashinfer" in n or "batch_decode" in n:
        return "attn_flashinfer"
    if "mm_kernel" in n or "gemm" in n or "linear" in n:
        return "gemm"
    return "other"


def analyze(path):
    with open(path) as f:
        ev = json.load(f)["traceEvents"]
    ev = [e for e in ev if e.get("ph") == "X"]
    steps = sorted((e for e in ev if e.get("cat") == "user_annotation"
                    and e["name"].startswith("execute_new_")), key=lambda e: e["ts"])
    fx = sorted((e for e in ev if e.get("cat") == "user_annotation"
                 and "CompiledFxGraph" in e["name"]), key=lambda e: e["ts"])
    kernels = sorted((e for e in ev if e.get("cat") in ("kernel", "gpu_memcpy")),
                     key=lambda e: e["ts"])
    rt = sorted((e for e in ev if e.get("cat") in ("cuda_runtime", "cuda_driver")),
                key=lambda e: e["ts"])
    cpu_ops = sorted((e for e in ev if e.get("cat") == "cpu_op"), key=lambda e: e["ts"])
    # Kernels belong to the step whose host launch they came from; correlate by
    # the launch's correlation id rather than by GPU timestamp.
    launch_ts = {}
    for e in rt:
        cid = e.get("args", {}).get("correlation")
        if cid is not None:
            launch_ts[cid] = e["ts"]

    def sl(lst, t0, t1):
        ts = [e["ts"] for e in lst]
        return lst[bisect.bisect_left(ts, t0):bisect.bisect_left(ts, t1)]

    k_by_launch = sorted(
        ((launch_ts.get(k.get("args", {}).get("correlation"), k["ts"]), k) for k in kernels),
        key=lambda x: x[0])
    k_ts = [x[0] for x in k_by_launch]

    groups = collections.defaultdict(list)
    for s in steps:
        t0, t1 = s["ts"], s["ts"] + s["dur"]
        ks = [x[1] for x in k_by_launch[bisect.bisect_left(k_ts, t0):bisect.bisect_left(k_ts, t1)]]
        cls = collections.Counter()
        names = set()
        for k in ks:
            c = kclass(k["name"])
            cls[c] += k["dur"]
            names.add(c)
        if "attn_triton" in names and ("attn_fa" in names or "attn_flashinfer" in names):
            kind = "mixed: triton + ixformer (?)"
        elif "attn_flashinfer" in names and "attn_fa" in names:
            kind = "mixed: split (flashinfer + FA)"
        elif "attn_triton" in names:
            kind = "mixed: triton"
        elif "attn_fa" in names:
            kind = "pure prefill: FA"
        elif not ks or len(ks) < 50:
            kind = "decode: cuda graph"
        else:
            kind = "decode: eager"
        f = sl(fx, t0, t1)
        if f:
            f0, f1 = f[0]["ts"], max(e["ts"] + e["dur"] for e in f)
        else:
            f0 = f1 = t0
        r = sl(rt, t0, t1)
        syncs = [e for e in r if e["name"] in SYNC]
        gpu_span = (max(k["ts"] + k["dur"] for k in ks) - min(k["ts"] for k in ks)) if ks else 0
        groups[kind].append({
            "wall": s["dur"], "pre": f0 - t0, "fwd": f1 - f0, "post": t1 - f1,
            "gpu": sum(k["dur"] for k in ks), "gpu_span": gpu_span,
            "launches": sum(1 for e in r if "Launch" in e["name"]),
            "cls": cls, "sync_n": len(syncs), "sync_us": sum(e["dur"] for e in syncs),
            "sync_in_fwd": sum(e["dur"] for e in syncs if f0 <= e["ts"] < f1),
            "t0": t0, "t1": t1,
        })

    print(f"==== {path}  ({len(steps)} steps)")
    for kind in sorted(groups):
        g = groups[kind]
        n = len(g)
        a = lambda k: sum(x[k] for x in g) / n / 1e3  # noqa: E731
        cls = collections.Counter()
        for x in g:
            cls.update(x["cls"])
        print(f"-- {kind}: {n} steps")
        print(f"   wall {a('wall'):7.1f} | pre {a('pre'):6.1f}  fwd {a('fwd'):6.1f}  "
              f"post {a('post'):6.1f}   (host ms)")
        print(f"   gpu busy {a('gpu'):7.1f}  gpu span {a('gpu_span'):7.1f}  "
              f"launches {sum(x['launches'] for x in g) / n:6.0f}")
        print("   gpu by class: " + "  ".join(
            f"{c}={us / n / 1e3:.1f}" for c, us in cls.most_common()))
        print(f"   sync calls {sum(x['sync_n'] for x in g) / n:4.1f}/step  "
              f"{a('sync_us'):6.1f} ms  (inside forward {a('sync_in_fwd'):6.1f} ms)")
    return groups


def top_cpu(path, kind_filter, groups, top=15):
    """Host ops (self time) inside the preprocess window of the given steps."""
    with open(path) as f:
        ev = json.load(f)["traceEvents"]
    ops = [e for e in ev if e.get("ph") == "X" and e.get("cat") in ("cpu_op", "python_function")]
    agg = collections.Counter()
    for kind, g in groups.items():
        if kind_filter not in kind:
            continue
        for x in g:
            t0, t1 = x["t0"], x["t1"]
            for e in ops:
                if t0 <= e["ts"] < t1:
                    agg[e["name"]] += e["dur"]
    print(f"   top host ops in '{kind_filter}' steps (inclusive us, summed):")
    for name, us in agg.most_common(top):
        print(f"      {us / 1e3:9.1f} ms  {name[:90]}")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        analyze(p)
