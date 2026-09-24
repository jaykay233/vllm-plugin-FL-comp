#!/usr/bin/env python3
"""Does the MetaX GEMV fast path for `linear` move end-to-end, in each mode?

The upstream commit verified the kernel (1.2-2.6x at M=1) but measured e2e only
in `vllm serve` (compile mode), where inductor forces custom_ops=["none"] so
*only* lm_head reaches flag_gems.linear -> x0.992.

Eager mode is the untested regime. An mcTracer profile of eager decode shows
169 GEMMs per forward (42 layers x 4 + lm_head) ALL running the generic
`linear_kernel`, i.e. every one of them routes through flag_gems.linear and can
be intercepted by this override. So eager is where the override has maximum
reach -- but eager is also launch-bound (GPU ~16% busy), so reach and payoff may
decouple. This script measures both.

env:  MODE=eager|compile   GEMV=0|1
Run:  MODE=eager GEMV=1 python /root/src/gemv_eager_ab.py
"""

from __future__ import annotations

import json
import os
import statistics
import time
from collections import defaultdict
from pathlib import Path

MODE = os.environ.get("MODE", "eager")
GEMV = os.environ.get("GEMV", "1")
OUT = Path("/root/bench_results/gemv_regime_ab")
OUT.mkdir(parents=True, exist_ok=True)

os.environ["FLAG_GEMS_METAX_GEMV"] = GEMV
os.environ.setdefault("VLLM_FL_PREFER", "flagos")
os.environ.setdefault("USE_FLAGGEMS", "1")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

TRACE = OUT / f"shapes_{MODE}_gemv{GEMV}.txt"
os.environ["FLAG_GEMS_METAX_GEMV_TRACE"] = str(TRACE)
if TRACE.exists():
    TRACE.unlink()

MODEL = "/root/models/MiniCPM5-2B"
STEPS = 32
REPS = 3

GEMM_NAMES = ("_gemv_kernel", "linear_kernel", "mm_kernel")


def gemm_kind(name: str) -> str | None:
    low = name.lower()
    if "gemv" in low:
        return "GEMV (new)"
    if "linear_kernel" in low:
        return "linear_kernel (generic)"
    if "mm_kernel" in low:
        return "mm_kernel (MetaX mm)"
    return None


def main() -> None:
    import torch
    from torch.profiler import ProfilerActivity, profile

    from vllm import LLM, SamplingParams

    kwargs = dict(
        model=MODEL,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=2048,
        gpu_memory_utilization=0.85,
        enable_prefix_caching=False,
    )
    if MODE == "eager":
        kwargs["enforce_eager"] = True

    print(f"\n{'#' * 78}\n### MODE={MODE}  FLAG_GEMS_METAX_GEMV={GEMV}\n{'#' * 78}", flush=True)

    import flag_gems
    lin_mod = getattr(getattr(flag_gems, "linear", None), "__module__", "?")
    print(f"flag_gems.linear -> {lin_mod}", flush=True)

    t0 = time.perf_counter()
    llm = LLM(**kwargs)
    init_s = time.perf_counter() - t0

    sp = SamplingParams(max_tokens=STEPS, temperature=0.0, ignore_eos=True)
    prompt = "Explain the theory of relativity in detail. " * 60
    warm = "Summarize the history of computing in one paragraph. " * 60

    for _ in range(2):
        llm.generate([warm], SamplingParams(max_tokens=4, temperature=0.0, ignore_eos=True))
    torch.cuda.synchronize()

    times = []
    n_out = 0
    for _ in range(REPS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        res = llm.generate([prompt], sp)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        n_out = len(res[0].outputs[0].token_ids)
    best = min(times)
    med = statistics.median(times)

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        llm.generate([prompt], sp)
        torch.cuda.synchronize()

    gemm_t: dict[str, float] = defaultdict(float)
    gemm_n: dict[str, int] = defaultdict(int)
    all_gpu_us = 0.0
    for e in prof.key_averages():
        t = e.self_device_time_total
        if t <= 0:
            continue
        if e.key.lower().startswith("mc"):
            continue
        all_gpu_us += t
        k = gemm_kind(e.key)
        if k:
            gemm_t[k] += t
            gemm_n[k] += e.count

    gemm_total = sum(gemm_t.values())
    print(f"init={init_s:.1f}s  n_out={n_out}", flush=True)
    print(f"wall best={best:.3f}s  median={med:.3f}s  "
          f"=> {n_out / best:.2f} tok/s (best)", flush=True)
    print(f"GPU total (1 fwd) = {all_gpu_us / 1e3:.2f} ms", flush=True)
    print(f"GEMM total        = {gemm_total / 1e3:.2f} ms "
          f"({gemm_total / all_gpu_us * 100:.1f}% of GPU)", flush=True)
    for k, t in sorted(gemm_t.items(), key=lambda kv: -kv[1]):
        print(f"    {k:28s} {t / 1e3:8.2f} ms  x{gemm_n[k]}  "
              f"{t / gemm_n[k]:6.2f} us each", flush=True)

    shapes = TRACE.read_text().splitlines() if TRACE.exists() else []
    via = defaultdict(set)
    for ln in shapes:
        parts = ln.split()
        v = parts[0]
        d = dict(p.split("=") for p in parts[1:])
        via[v].add((int(d["M"]), int(d["N"]), int(d["K"])))
    print(f"shapes reaching metax linear:", flush=True)
    for v in sorted(via):
        print(f"    via {v}: {len(via[v])} distinct", flush=True)
        for s in sorted(via[v]):
            print(f"      M={s[0]} N={s[1]} K={s[2]}", flush=True)

    rec = {
        "mode": MODE, "gemv": GEMV, "linear_impl": lin_mod,
        "init_s": init_s, "n_out": n_out,
        "wall_best_s": best, "wall_median_s": med,
        "tok_per_s_best": n_out / best,
        "wall_all_s": times,
        "gpu_total_ms": all_gpu_us / 1e3,
        "gemm_total_ms": gemm_total / 1e3,
        "gemm_by_kind": {k: {"ms": v / 1e3, "calls": gemm_n[k]}
                         for k, v in sorted(gemm_t.items(), key=lambda kv: -kv[1])},
        "shapes_via": {v: sorted(list(s)) for v, s in via.items()},
    }
    (OUT / f"{MODE}_gemv{GEMV}.json").write_text(json.dumps(rec, indent=2))
    print(f"wrote {OUT / f'{MODE}_gemv{GEMV}.json'}", flush=True)
    print(f"GEMV_REGIME_DONE {MODE} gemv={GEMV} tok/s={n_out / best:.2f} "
          f"wall={best:.3f} gemm_ms={gemm_total / 1e3:.2f}", flush=True)


if __name__ == "__main__":
    main()
