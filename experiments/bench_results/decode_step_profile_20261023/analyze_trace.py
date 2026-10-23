#!/usr/bin/env python3
"""Summarise a vLLM torch trace: where does decode-step time actually go?

Groups CUDA kernels into buckets (gemm / attention / norm / rope / silu /
sampler / other) by name pattern, and reports the decode-step total from the
step markers so the buckets can be checked against TPOT.
"""
from __future__ import annotations

import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load(path: Path) -> dict:
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as f:
            return json.load(f)
    return json.loads(path.read_text())


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
    if "silu" in n or "swiglu" in n or "act" in n and "fused" in n:
        return "silu"
    if any(k in n for k in ("top_p", "topp", "softmax", "argmax", "sort", "cumsum", "exponential", "multinomial")):
        return "sampler"
    if "copy" in n or "memcpy" in n or "cat" in n or "fill" in n:
        return "copy/fill"
    if "elementwise" in n or "vectorized" in n or "add" in n or "mul" in n:
        return "elementwise"
    return "other"


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else max(
        HERE.joinpath("trace").glob("rank0*.gz"), key=lambda p: p.stat().st_mtime
    )
    print("trace:", path.name, flush=True)
    tr = load(path)
    ev = tr.get("traceEvents", tr.get("events", []))

    cats = defaultdict(float)
    names = defaultdict(float)
    total_gpu_us = 0.0
    for e in ev:
        if e.get("ph") != "X":
            continue
        dur = e.get("dur") or 0
        if "cuda_time" in e:
            dur = e["cuda_time"]
        cat = e.get("cat", "")
        if cat not in ("kernel", "Kernel", "gpu_user_annotation", "cuda_runtime", "Kernel"):
            continue
        if cat in ("cuda_runtime",):
            continue
        name = e.get("name", "")
        b = bucket(name)
        cats[b] += dur
        names[(b, name)] += dur
        total_gpu_us += dur

    print(f"\n=== GPU kernel total {total_gpu_us/1000:.1f} ms ===")
    for b, v in sorted(cats.items(), key=lambda kv: -kv[1]):
        print(f"  {b:12s} {v/1000:9.2f} ms  {100*v/total_gpu_us:5.1f}%")

    print("\n=== top kernels per bucket ===")
    for b in sorted(cats, key=lambda k: -cats[k]):
        rows = sorted(((n, v) for (bb, n), v in names.items() if bb == b), key=lambda kv: -kv[1])[:5]
        print(f"[{b}]")
        for n, v in rows:
            print(f"   {v/1000:8.2f} ms  {100*v/total_gpu_us:5.1f}%  {n[:88]}")

    # iteration boundaries, if present
    iters = [(e["ts"], e["dur"]) for e in ev if e.get("name") in ("execute_context",) or "EngineCoreExecute" in str(e.get("name"))]
    print(f"\nfound {len(iters)} iteration-like spans")
    return 0


if __name__ == "__main__":
    sys.exit(main())
