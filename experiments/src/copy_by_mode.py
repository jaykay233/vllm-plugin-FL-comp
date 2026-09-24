#!/usr/bin/env python3
"""Do the 168 `aten::clone` -> `_copy_kernel_kernel_rank_2` copies survive
without torch.compile?

Eager attribution pinned the plugin's single biggest non-GEMM overhead:

    _copy_kernel_kernel_rank_2   168/step  (4 per layer)  <- aten::clone
    _copy_kernel_kernel_rank_3    42/step  (1 per layer)  <- unified_attention output write

168/step is exactly the per-layer `aten::mm` count (4 projections x 42 layers),
i.e. one clone per matmul, so the leading hypothesis is that inductor inserts
them for layout/aliasing reasons. If so they vanish when compile is off, and the
`nocompile_cg` configuration the FL plugin already supports is worth ~7% TPOT.

This measures the copy kernels and TPOT for each configuration:

    eager            enforce_eager=True            (no compile, no cudagraph)
    nocompile        cc.mode=NONE                  (no compile, no cudagraph)
    nocompile_cg     cc.mode=NONE, cudagraph FULL_DECODE_ONLY
    compile          default                       (inductor + cudagraph)

Run:  MODE=nocompile python /root/src/copy_by_mode.py
"""

from __future__ import annotations

import json
import os
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

# MUST be set before importing vllm: without it the engine runs in a child
# process and the parent-side profiler sees zero kernels.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

MODE = os.environ.get("MODE", "nocompile")
MODEL = "/root/models/MiniCPM5-2B"
OUT = Path("/root/bench_results/copy_by_mode")
OUT.mkdir(parents=True, exist_ok=True)
WARM, REPEAT, GEN = 2, 5, 128


def build_kwargs(mode: str) -> dict:
    kw: dict = dict(
        model=MODEL,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=2048,
        gpu_memory_utilization=0.30,
    )
    if mode == "eager":
        kw["enforce_eager"] = True
    elif mode == "nocompile":
        kw["compilation_config"] = {"mode": "NONE"}
    elif mode == "nocompile_cg":
        kw["compilation_config"] = {
            "mode": "NONE",
            "cudagraph_mode": "FULL_DECODE_ONLY",
        }
    elif mode == "compile":
        pass
    else:
        raise SystemExit(f"unknown MODE={mode}")
    return kw


def main() -> None:
    import torch
    from torch.profiler import ProfilerActivity, profile
    from vllm import LLM, SamplingParams

    kw = build_kwargs(MODE)
    print(f"########## MODE={MODE} ##########", flush=True)
    llm = LLM(**kw)
    try:
        cc = llm.llm_engine.vllm_config.compilation_config
        print(f"  mode={cc.mode}  cudagraph_mode={cc.cudagraph_mode}  "
              f"custom_ops={cc.custom_ops}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"  config introspect failed: {e}", flush=True)

    # short prompt + long output == the MATH-500 shape
    prompt = "The theory of general relativity describes gravity as"
    sp = SamplingParams(max_tokens=GEN, temperature=0.0, ignore_eos=True)

    for _ in range(WARM):
        llm.generate([prompt], sp, use_tqdm=False)

    # ---- timing ------------------------------------------------------
    times = []
    for _ in range(REPEAT):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = llm.generate([prompt], sp, use_tqdm=False)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    n = len(out[0].outputs[0].token_ids)
    med, best = statistics.median(times), min(times)
    ids = list(out[0].outputs[0].token_ids)
    print(f"  tok/s median={n / med:8.1f}  best={n / best:8.1f}   "
          f"({n} tokens, {med * 1e6 / n:.0f} us/token median)", flush=True)

    # ---- copy kernels -------------------------------------------------
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        llm.generate([prompt], sp, use_tqdm=False)
        torch.cuda.synchronize()

    steps = n - 1  # first token is prefill; the rest are decode steps
    cnt: Counter[str] = Counter()
    dur: Counter[str] = Counter()
    tot = defaultdict(float)
    for e in prof.events():
        if e.device_type != torch.autograd.DeviceType.CUDA or not e.name:
            continue
        if "==" in e.name or e.name.startswith("Memcpy") or e.name.startswith("Memset"):
            continue
        cnt[e.name] += e.count
        dur[e.name] += e.device_time
    print(f"  --- kernels (per decode step, {steps} steps) ---", flush=True)
    for name, d in dur.most_common(16):
        mark = "  <<< COPY" if "_copy_kernel" in name else ""
        print(f"    {d / steps:8.1f} us/step  x{cnt[name] / steps:7.1f}/step  "
              f"{name[:60]}{mark}", flush=True)

    copy_us = sum(v for k, v in dur.items() if "_copy_kernel" in k) / steps
    copy_n = sum(v for k, v in cnt.items() if "_copy_kernel" in k) / steps
    print(f"  COPY TOTAL: {copy_us:.1f} us/step, {copy_n:.1f} launches/step", flush=True)

    rec = {
        "mode": MODE,
        "tok_per_s_median": n / med,
        "us_per_token_median": med * 1e6 / n,
        "n_tokens": n,
        "copy_us_per_step": copy_us,
        "copy_launches_per_step": copy_n,
        "token_ids_head": ids[:16],
        "kernels": {k: {"us_per_step": dur[k] / steps, "launches_per_step": cnt[k] / steps}
                    for k in dur},
    }
    (OUT / f"{MODE}.json").write_text(json.dumps(rec, indent=2))
    print(f"COPY_BY_MODE_DONE {MODE} copy_us_per_step={copy_us:.1f}", flush=True)


if __name__ == "__main__":
    main()
