#!/usr/bin/env python3
"""Kernel-level attribution on real MiniCPM5-2B inference (MACA / MetaX C500).

Synthetic microbenchmarks are misleading here: at T=1 the host launch path
dominates, and dispatch/stub calls never appear in the GPU timeline. This
script profiles the ACTUAL model with torch.profiler (MACA exposes kineto, so
it reports real device time per kernel) and aggregates by kernel name.

VLLM_ENABLE_V1_MULTIPROCESSING=0 keeps EngineCore in this process so the
profiler sees the model's kernels instead of an empty timeline.

Two workloads:
  prefill : 1 prompt of ~512 tokens, 1 output token  -> TTFT composition
  decode  : 1 prompt of 8 tokens, 64 output tokens  -> steady-state TPOT

Run:  conda activate mx && python /root/src/kernel_profile.py
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

MODEL = "/root/models/MiniCPM5-2B"
OUT = Path("/root/bench_results/kernel_profile")
OUT.mkdir(parents=True, exist_ok=True)

EAGER = os.environ.get("EAGER", "1") == "1"


def classify(name: str) -> str:
    """Bucket a kernel name into an operator family."""
    n = name.lower()
    rules = [
        ("rms_norm", ["rms_norm", "fused_add_rms"]),
        ("silu/gelu+act", ["silu", "gelu", "sigmoid", "act_and_mul"]),
        ("rope", ["rotary", "rope"]),
        ("attention", ["flash", "attention", "attn", "paged", "mha"]),
        ("gemm/linear", ["gemm", "cutlass", "mmpy", "gemv", "matmul", "linear",
                         "hgemm", "sgemm", "wgrad", "tensorop"]),
        ("norm-other", ["layer_norm", "layernorm", "group_norm"]),
        ("elementwise", ["elementwise", "vectorized_elementwise", "add", "mul",
                         "copy", "fill", "cast", "index", "broadcast"]),
        ("softmax/topk", ["softmax", "topk", "sort", "argmax", "cumsum"]),
        ("reduction", ["reduce", "sum", "mean", "max", "min"]),
        ("quant", ["quant", "fp8", "int8", "scale"]),
    ]
    for fam, keys in rules:
        if any(k in n for k in keys):
            return fam
    return "other"


def profile_run(llm, prompt, sp, tag: str) -> dict:
    import torch
    from torch.profiler import ProfilerActivity, profile

    # warmup outside the profile
    llm.generate([prompt], sp)
    torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof:
        llm.generate([prompt], sp)
        torch.cuda.synchronize()

    evts = prof.key_averages()
    total_cuda = sum(e.self_device_time_total for e in evts)

    per_kernel = []
    per_family: dict[str, float] = defaultdict(float)
    for e in evts:
        t = e.self_device_time_total
        if t <= 0:
            continue
        per_kernel.append({
            "kernel": e.key,
            "calls": e.count,
            "cuda_us_total": t,
            "cuda_us_avg": t / e.count if e.count else 0.0,
            "family": classify(e.key),
        })
        per_family[classify(e.key)] += t

    per_kernel.sort(key=lambda r: -r["cuda_us_total"])
    fam_sorted = sorted(per_family.items(), key=lambda kv: -kv[1])

    print(f"\n{'=' * 82}")
    print(f"[{tag}]  total CUDA time = {total_cuda / 1e3:.3f} ms")
    print("=" * 82)
    print(f"  {'family':16s} {'ms':>9s} {'share':>7s}")
    for fam, t in fam_sorted:
        print(f"  {fam:16s} {t / 1e3:9.3f} {t / total_cuda * 100:6.1f}%")

    print(f"\n  top kernels:")
    for r in per_kernel[:18]:
        nm = r["kernel"][:58]
        print(f"    {r['cuda_us_total'] / 1e3:9.3f}ms "
              f"{r['cuda_us_total'] / total_cuda * 100:6.1f}%  x{r['calls']:<5d} "
              f"{r['cuda_us_avg']:8.2f}us  {nm}")

    # FlagGems coverage signal: triton kernels carry a distinctive name
    flaggems_like = [r for r in per_kernel
                     if any(k in r["kernel"].lower()
                            for k in ("triton", "gems", "flag_gems"))]
    print(f"\n  FlagGems/Triton-looking kernels: {len(flaggems_like)}")
    for r in flaggems_like[:10]:
        print(f"    {r['cuda_us_total'] / 1e3:9.3f}ms  x{r['calls']:<5d} {r['kernel'][:60]}")

    return {
        "tag": tag,
        "total_cuda_ms": total_cuda / 1e3,
        "by_family_ms": {k: v / 1e3 for k, v in fam_sorted},
        "top_kernels": per_kernel[:60],
        "flaggems_kernel_count": len(flaggems_like),
        "flaggems_kernels": flaggems_like[:20],
    }


def main() -> None:
    from vllm import LLM, SamplingParams

    print(f"### kernel_profile  EAGER={EAGER}  "
          f"multiproc={os.environ.get('VLLM_ENABLE_V1_MULTIPROCESSING')} ###")
    llm = LLM(
        model=MODEL, dtype="bfloat16", trust_remote_code=True,
        max_model_len=2048, gpu_memory_utilization=0.85, enforce_eager=EAGER,
    )

    results = []
    # ~512-token prompt
    prefill_prompt = "Explain the theory of relativity in detail. " * 40
    results.append(profile_run(
        llm, prefill_prompt,
        SamplingParams(max_tokens=1, temperature=0.0, ignore_eos=True),
        "prefill_512",
    ))
    results.append(profile_run(
        llm, "The capital of France is",
        SamplingParams(max_tokens=64, temperature=0.0, ignore_eos=True),
        "decode_64",
    ))

    (OUT / f"kernel_profile_eager{int(EAGER)}.json").write_text(
        json.dumps(results, indent=2))

    print("\n" + "=" * 82)
    print("SUMMARY")
    print("=" * 82)
    for r in results:
        print(f"  {r['tag']:14s} total={r['total_cuda_ms']:8.3f}ms  "
              f"flaggems_kernels={r['flaggems_kernel_count']}")
        top = list(r["by_family_ms"].items())[:4]
        print("      " + "  ".join(f"{k}={v:.3f}ms" for k, v in top))

    print(f"\nWrote {OUT}/kernel_profile_eager{int(EAGER)}.json")
    print(f"KERNEL_PROFILE_DONE eager={EAGER}", flush=True)


if __name__ == "__main__":
    main()
