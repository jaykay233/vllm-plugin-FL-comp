#!/usr/bin/env python3
"""Can CUDAGraph be used *without* torch.compile, and what is each one worth?

`enforce_eager=True` disables both (`config/vllm.py`: "Enforce eager set,
disabling torch.compile and CUDAGraphs"), so the earlier eager/compile
comparison cannot separate the two contributions.  This runs the missing
configurations, all on the same workload and the same FlagGems settings:

    eager        enforce_eager=True          (no compile, no cudagraph)   [baseline]
    nocompile    -cc.mode=NONE               (no compile, no cudagraph)
    nocompile_cg -cc.mode=NONE, cg=FULL_DECODE_ONLY   <- the question
    nocompile_cg_full -cc.mode=NONE, cg=FULL
    compile      default                     (inductor + FULL_AND_PIECEWISE) [reference]

`mode=NONE` also flips `custom_ops` from ['none'] to ['all'], so the FL OOT
chain (forward_oot -> CachedOp -> flag_gems) becomes reachable again -- i.e.
this configuration is a candidate for "FlagGems without the IR-op bridge".

Run:  MODE=nocompile_cg python /root/src/cudagraph_no_compile.py
"""

from __future__ import annotations

import json
import os
import statistics
import time
from pathlib import Path

MODE = os.environ.get("MODE", "nocompile_cg")
# PROMPT=long -> the ~500 token repeated sentence; PROMPT=short -> a normal prompt
PROMPT_KIND = os.environ.get("PROMPT", "long")
IR = os.environ.get("VLLM_FL_IR_KERNELS", "1")
# GPU_MEM lowers the reservation so a run still starts when another job holds
# part of the device (this box has concurrent users; 0.85 needs 54 of 64 GiB).
GPU_MEM = float(os.environ.get("GPU_MEM", "0.85"))
MODEL = "/root/models/MiniCPM5-2B"
OUT = Path("/root/bench_results/cg_no_compile")
OUT.mkdir(parents=True, exist_ok=True)


def build_kwargs(mode: str) -> dict:
    kwargs: dict = dict(
        model=MODEL,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=2048,
        gpu_memory_utilization=GPU_MEM,
    )
    if mode == "eager":
        kwargs["enforce_eager"] = True
    elif mode == "nocompile":
        kwargs["compilation_config"] = {"mode": "NONE"}
    elif mode == "nocompile_cg":
        kwargs["compilation_config"] = {
            "mode": "NONE",
            "cudagraph_mode": "FULL_DECODE_ONLY",
        }
    elif mode == "nocompile_cg_full":
        kwargs["compilation_config"] = {"mode": "NONE", "cudagraph_mode": "FULL"}
    elif mode == "compile":
        pass
    else:
        raise SystemExit(f"unknown MODE={mode}")
    return kwargs


def main() -> None:
    import torch
    from vllm import LLM, SamplingParams

    kwargs = build_kwargs(MODE)
    print(f"\n########## MODE={MODE} PROMPT={PROMPT_KIND} IR={IR} ##########", flush=True)
    print(f"vllm kwargs: { {k: v for k, v in kwargs.items() if k != 'model'} }", flush=True)

    t0 = time.perf_counter()
    llm = LLM(**kwargs)
    init_s = time.perf_counter() - t0

    # ---- what did vLLM actually decide? ----
    try:
        cc = llm.llm_engine.vllm_config.compilation_config
        print(f"  mode            = {cc.mode}", flush=True)
        print(f"  cudagraph_mode  = {cc.cudagraph_mode}", flush=True)
        print(f"  custom_ops      = {cc.custom_ops}", flush=True)
        print(f"  ir op priority  = "
              f"{llm.llm_engine.vllm_config.kernel_config.ir_op_priority}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"  config introspect failed: {type(e).__name__}: {e}", flush=True)

    sp = SamplingParams(max_tokens=128, temperature=0.0, ignore_eos=True)
    prompt = (
        "Explain the theory of relativity in detail. " * 40
        if PROMPT_KIND == "long"
        else "The capital of France is"
    )

    llm.generate([prompt], SamplingParams(max_tokens=8, temperature=0.0, ignore_eos=True))

    # ---- tokens: correctness across configurations must be identical ----
    out = llm.generate([prompt], sp)
    ids = list(out[0].outputs[0].token_ids)
    print(f"  token_ids[:12] = {ids[:12]}", flush=True)
    print(f"  uniq={len(set(ids))}  text={out[0].outputs[0].text[:120]!r}", flush=True)

    # ---- which FL implementations ran? ----
    try:
        from vllm_fl.dispatch import get_default_manager

        mgr = get_default_manager()
        print(f"  FL dispatch ops used: {dict(mgr._called_ops)}", flush=True)
        for op in ("rms_norm", "silu_and_mul", "rotary_embedding"):
            try:
                impl = mgr._resolve_impl(op)
                print(f"    {op:18s} -> {impl.impl_id}", flush=True)
            except Exception:  # noqa: BLE001
                pass
    except Exception as e:  # noqa: BLE001
        print(f"  dispatch introspect failed: {type(e).__name__}: {e}", flush=True)

    # ---- throughput ----
    times = []
    for _ in range(7):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        llm.generate([prompt], sp)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    med = statistics.median(times)
    best = min(times)
    n_out = len(ids)
    print(f"  init={init_s:.1f}s  median gen={med:.3f}s  best={best:.3f}s  "
          f"=> {n_out / med:.1f} (median) / {n_out / best:.1f} (best) tok/s",
          flush=True)
    print(f"  all runs: {[f'{t:.3f}' for t in times]}", flush=True)

    # ---- kernel profile ----
    fam_ms: dict[str, float] = {}
    fam_calls: dict[str, int] = {}
    try:
        from collections import defaultdict

        from torch.profiler import ProfilerActivity, profile

        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            llm.generate([prompt], sp)
            torch.cuda.synchronize()
        acc = defaultdict(float)
        cnt = defaultdict(int)
        for e in prof.key_averages():
            t = e.self_device_time_total
            if t <= 0:
                continue
            low = e.key.lower()
            if "fused_add_rms_norm" in low or "rms_norm" in low:
                f = "rms_norm"
            elif "silu" in low:
                f = "silu_and_mul"
            elif "rotary" in low:
                f = "rope"
            elif low.startswith("triton_"):
                f = "inductor-triton"
            elif "gemm" in low or "linear" in low or "mm" in low:
                f = "gemm"
            else:
                f = "other"
            acc[f] += t / 1e3
            cnt[f] += e.count
        print("  --- kernel families ---", flush=True)
        for f, ms in sorted(acc.items(), key=lambda kv: -kv[1]):
            print(f"    {f:18s} {ms:9.3f}ms  x{cnt[f]}", flush=True)
        fam_ms = dict(acc)
        fam_calls = dict(cnt)
    except Exception as e:  # noqa: BLE001
        print(f"  profile failed: {type(e).__name__}: {e}", flush=True)

    rec = {
        "mode": MODE,
        "init_s": init_s,
        "n_out": n_out,
        "median_gen_s": med,
        "best_gen_s": best,
        "tok_per_s": n_out / med,
        "tok_per_s_best": n_out / best,
        "runs": times,
        "token_ids": ids,
        "kernel_families": {k: {"ms": v, "calls": fam_calls[k]} for k, v in fam_ms.items()},
        "kwargs": {k: str(v) for k, v in kwargs.items()},
    }
    (OUT / f"{MODE}.json").write_text(json.dumps(rec, indent=2))
    print(f"CG_NO_COMPILE_DONE {MODE} tok_per_s={n_out / med:.1f}", flush=True)


if __name__ == "__main__":
    main()
