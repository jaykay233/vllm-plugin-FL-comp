#!/usr/bin/env python3
"""Second pass: separate cudagraph-replayed work from eager work in a vLLM trace.

Decode steps run inside a CUDA graph replay (`execute_new_0_cached_<bs>`), so the
inner kernels do not appear individually. This script counts replays and their
mean duration (= GPU time per decode step) and reports the eager kernels (prefill
+ non-graphed ops) separately, so the two regimes are not mixed.
"""
from __future__ import annotations

import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent


def bucket(name: str) -> str:
    n = name.lower()
    if "flash_fwd" in n or "flash_attn" in n or "attention" in n or "fmha" in n:
        return "attention"
    if any(k in n for k in ("mcblas", "gemm", "gemv", "mm_kernel", "linear")):
        return "gemm"
    if "rms_norm" in n or "rmsnorm" in n or "layer_norm" in n:
        return "rms_norm"
    if "rotary" in n or "rope" in n:
        return "rope"
    if "silu" in n or "swiglu" in n:
        return "silu"
    if any(k in n for k in ("top_p", "topp", "softmax", "argmax", "sort", "cumsum", "exponential", "multinomial", "grid_stride")):
        return "sampler"
    if "reshape_and_cache" in n or "slot_mapping" in n:
        return "kv_write"
    if "copy" in n or "memcpy" in n or "cat" in n or "fill" in n:
        return "copy/fill"
    if "add" in n or "mul" in n or "elementwise" in n:
        return "elementwise"
    return "other"


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else max(
        HERE.joinpath("trace").glob("rank0*.gz"), key=lambda p: p.stat().st_mtime
    )
    with gzip.open(path, "rt") as f:
        ev = json.load(f)["traceEvents"]

    graphs = defaultdict(list)
    eager = defaultdict(float)
    eager_cnt = defaultdict(int)
    for e in ev:
        if e.get("ph") != "X":
            continue
        if e.get("cat") not in ("kernel", "Kernel", "gpu_user_annotation"):
            continue
        name = e.get("name", "")
        dur = e.get("dur") or 0
        if name.startswith("execute_new_0_cached") or "cached_" in name and name.startswith("execute"):
            graphs[name].append(dur)
        elif not name.startswith("execute_"):
            eager[bucket(name)] += dur
            eager_cnt[bucket(name)] += 1

    print("=== cudagraph replays (decode) ===")
    tot_graph = 0.0
    for name, ds in sorted(graphs.items(), key=lambda kv: -sum(kv[1])):
        tot_graph += sum(ds)
        print(f"  {name:38s} n={len(ds):5d}  mean={sum(ds)/len(ds)/1000:7.3f} ms  total={sum(ds)/1000:8.1f} ms")
    print(f"  graph total {tot_graph/1000:.1f} ms")

    tot_eager = sum(eager.values())
    print(f"\n=== eager kernels total {tot_eager/1000:.1f} ms (prefill + non-graphed) ===")
    for b, v in sorted(eager.items(), key=lambda kv: -kv[1]):
        print(f"  {b:12s} {v/1000:9.1f} ms  {100*v/max(tot_eager,1):5.1f}%  n={eager_cnt[b]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
