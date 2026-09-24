#!/usr/bin/env python3
"""Decisive test: in eager mode (the only mode where FlagGems actually runs),
does restricting FlagGems to the fused ops recover performance?

Keyword: MACA names triton kernels "<fn>_kernel_rank_N", so a substring search
for 'triton' misses them. We match FlagGems' real kernel names explicitly.

Configs driven by env (set by the runner):
  VLLM_FL_FLAGOS_WHITELIST   -> flag_gems.only_enable(include=...)
  (unset)                    -> flag_gems.enable()  (all ops, incl. triton GEMM)

Run:  conda activate mx && python /root/src/eager_wl_test.py
"""

from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_FL_PREFER", "flagos")
os.environ.setdefault("USE_FLAGGEMS", "1")

MODEL = "/root/models/MiniCPM5-2B"
OUT = Path("/root/bench_results/eager_wl")
OUT.mkdir(parents=True, exist_ok=True)

# FlagGems triton kernels, as MACA reports them (substring -> family)
GEMS_KERNELS = {
    "linear_kernel": "GEMM(flagGems triton)",
    "fused_add_rms_norm_kernel": "rms_norm(flagGems)",
    "rms_norm_kernel": "rms_norm(flagGems)",
    "silu_and_mul_kernel": "silu_and_mul(flagGems)",
    "apply_rotary_pos_emb_inplace_kernel": "rope(flagGems)",
    "apply_rotary_pos_emb_kernel": "rope(flagGems)",
    "mm_kernel": "matmul(flagGems)",
}

VENDOR_HINTS = {
    "flash_fwd_splitkv": "attention(vendor)",
    "reshape_and_cache_flash": "kvcache(vendor)",
    "elementwise_kernel": "elementwise(vendor)",
    "vectorized_elementwise": "elementwise(vendor)",
    "catArrayBatched": "misc(vendor)",
}


def label_kernel(name: str) -> str:
    low = name.lower()
    for key, fam in GEMS_KERNELS.items():
        if key in low:
            return fam
    for key, fam in VENDOR_HINTS.items():
        if key in low:
            return fam
    return "other"


def main() -> None:
    import torch
    from vllm import LLM, SamplingParams
    from torch.profiler import ProfilerActivity, profile

    wl = os.environ.get("VLLM_FL_FLAGOS_WHITELIST", "")
    tag = os.environ.get("RUN_TAG", "all")
    print(f"\n{'#' * 80}\n### RUN_TAG={tag}  whitelist={wl or '<none: enable ALL>'}\n{'#' * 80}",
          flush=True)

    llm = LLM(model=MODEL, dtype="bfloat16", trust_remote_code=True,
              max_model_len=2048, gpu_memory_utilization=0.85, enforce_eager=True)

    prompt = "Explain the theory of relativity in detail. " * 40  # ~512 tokens
    sp = SamplingParams(max_tokens=64, temperature=0.0, ignore_eos=True)

    llm.generate([prompt], SamplingParams(max_tokens=8, temperature=0.0, ignore_eos=True))
    torch.cuda.synchronize()

    times = []
    for _ in range(3):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = llm.generate([prompt], sp)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    n_out = len(out[0].outputs[0].token_ids)
    best = min(times)
    print(f"  gen best={best:.3f}s for {n_out} tok => {n_out / best:.1f} tok/s", flush=True)

    # kernel profile
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        llm.generate([prompt], sp)
        torch.cuda.synchronize()

    fam_tot: dict[str, float] = defaultdict(float)
    fam_calls: dict[str, int] = defaultdict(int)
    raw: list[dict] = []
    total = 0.0
    for e in prof.key_averages():
        t = e.self_device_time_total
        if t <= 0:
            continue
        f = label_kernel(e.key)
        fam_tot[f] += t
        fam_calls[f] += e.count
        total += t
        raw.append({"kernel": e.key, "family": f, "calls": e.count,
                    "ms": t / 1e3, "us_avg": t / e.count if e.count else 0.0})
    raw.sort(key=lambda r: -r["ms"])

    print(f"  total CUDA {total / 1e3:.1f} ms", flush=True)
    for f, t in sorted(fam_tot.items(), key=lambda kv: -kv[1]):
        print(f"    {f:28s} {t / 1e3:9.2f}ms {t / total * 100:5.1f}%  x{fam_calls[f]}", flush=True)

    print("\n  --- top 25 raw kernels ---", flush=True)
    for r in raw[:25]:
        print(f"    {r['ms']:8.2f}ms {r['ms'] / (total / 1e3) * 100:5.1f}% "
              f"x{r['calls']:<6d} {r['us_avg']:8.2f}us  {r['kernel'][:64]}", flush=True)

    rec = {
        "tag": tag, "whitelist": wl, "tok_per_s": n_out / best,
        "n_out": n_out, "best_gen_s": best,
        "total_cuda_ms": total / 1e3,
        "by_family": {k: {"ms": v / 1e3, "calls": fam_calls[k]}
                      for k, v in sorted(fam_tot.items(), key=lambda kv: -kv[1])},
        "top_kernels": raw[:40],
    }
    (OUT / f"{tag}.json").write_text(json.dumps(rec, indent=2))
    print(f"EAGER_WL_DONE tag={tag} tok_per_s={n_out / best:.1f}", flush=True)


if __name__ == "__main__":
    main()
