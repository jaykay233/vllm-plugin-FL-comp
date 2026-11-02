#!/usr/bin/env python3
"""Prefill-shaped workload for torch.profiler on the *-comp stack.

Mirrors the official 4k/16k cases as closely as an offline LLM call can:
concurrency 64, prompts of exactly IN_LEN tokens, OUTPUT_LEN=1 so the traced
window is one big chunked-prefill wave with essentially no decode.

Env:
  IN_LEN=4096          prompt length in tokens
  CONC=64              concurrent sequences
  MODE=prefill|decode  prefill: out=1 ; decode: 1-token prompt + out=1024
  OUT=<dir>            trace dir
Run:
  python experiments/src/prefill_trace_comp.py
"""
from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from pathlib import Path

IN_LEN = int(os.environ.get("IN_LEN", "4096"))
CONC = int(os.environ.get("CONC", "64"))
MODE = os.environ.get("MODE", "prefill")
OUT = Path(os.environ.get("OUT", f"/root/bench_results/trace/{MODE}_{IN_LEN}"))
OUT.mkdir(parents=True, exist_ok=True)

FAMILY_RULES = [
    ("silu_and_mul_kernel", "silu_and_mul"),
    ("silu_and_mul_gate_up", "silu_gate_up(fused)"),
    ("fused_add_rms_norm_kernel", "rms_norm_fused_add"),
    ("rms_norm_kernel", "rms_norm"),
    ("apply_rotary_pos_emb", "rope"),
    ("linear_kernel", "gemm_linear(gems)"),
    ("mm_kernel", "gemm_mm(gems)"),
    ("gemv", "gemv"),
    ("flash_fwd_splitkv_combine_kernel", "attn_combine"),
    ("flash_fwd_splitkv_kernel", "attn_decode_splitkv"),
    ("flash_fwd_kernel", "attn_prefill"),
    ("flash_fwd_", "attn_other"),
    ("reshape_and_cache_flash_kernel", "kvcache_write"),
    ("reshape_and_cache", "kvcache_write"),
    ("argmax_kernel", "sampler_argmax"),
    ("top_p", "sampler_topp"),
    ("histogram", "sampler_hist"),
    ("softmax_kernel", "sampler_softmax"),
    ("_copy_kernel_kernel_rank", "copy(gems)"),
    ("_to_copy_func_kernel", "copy(gems)"),
    ("fill_scalar_func_kernel", "fill(gems)"),
    ("add_func_kernel", "add(gems)"),
    ("mul_func_kernel", "mul(gems)"),
    ("_compute_slot_mapping_kernel", "slot_mapping"),
    ("_index_jit_function", "index"),
    ("embedding_kernel", "embedding"),
    ("memcpy", "memcpy"),
    ("mcblas", "mcblas"),
    ("gemm", "gemm_mcblas"),
    ("triton_poi_fused", "inductor_pointwise"),
    ("triton_red_fused", "inductor_reduction"),
]


def family_of(name: str) -> str:
    low = name.lower()
    for key, fam in FAMILY_RULES:
        if key in low:
            return fam
    return "other"


def main() -> None:
    import torch
    from vllm import LLM, SamplingParams

    tok_prompt = "Explain the theory of relativity in detail. "  # ~10 tokens

    llm = LLM(
        model="/workspace/MiniCPM5-2B",
        dtype="bfloat16",
        max_model_len=max(2048, IN_LEN + 2048),
        gpu_memory_utilization=0.85,
        max_num_batched_tokens=8192,
        # Every prompt below is byte-identical, so prefix caching would turn 63
        # of the 64 prefills into cache hits and the traced window into a
        # single-token decode. Distinct prefixes + disabled caching keeps the
        # prefill work real.
        enable_prefix_caching=os.environ.get("PREFIX_CACHE", "0") == "1",
    )

    tokenizer = llm.get_tokenizer()
    unit = tokenizer(tok_prompt)["input_ids"]
    per_seq = IN_LEN // len(unit) + 1
    prompts = []
    if MODE == "decode":
        ctx = int(os.environ.get("DECODE_CTX", "1"))
        for i in range(CONC):
            ids = (unit * (ctx // len(unit) + 1))[:ctx]
            ids[0] = 100 + i
            prompts.append({"prompt_token_ids": ids})
    else:
        for i in range(CONC):
            ids = (unit * per_seq)[:IN_LEN]
            ids[0] = 100 + i  # unique first token: no shared prefix
            prompts.append({"prompt_token_ids": ids})
    out_len = 1 if MODE == "prefill" else int(os.environ.get("DECODE_STEPS", "1024"))
    sp = SamplingParams(max_tokens=out_len, temperature=0.0, ignore_eos=True)

    # warmup: same shape, 1 token, keeps cudagraph capture out of the window
    llm.generate(prompts, SamplingParams(max_tokens=1, temperature=0.0, ignore_eos=True))
    torch.cuda.synchronize()
    print(f"warmup done: mode={MODE} in_len={IN_LEN} conc={CONC}", flush=True)

    from torch.profiler import ProfilerActivity, profile

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        llm.generate(prompts, sp)
        torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    print(f"traced wall = {wall:.2f} s", flush=True)

    rows = []
    total = 0.0
    for ev in prof.key_averages():
        if ev.self_device_time_total <= 0:
            continue
        rows.append((ev.key, ev.count, ev.self_device_time_total))
        total += ev.self_device_time_total
    rows.sort(key=lambda r: -r[2])

    fam: dict[str, list[float]] = defaultdict(lambda: [0.0, 0])
    for name, cnt, us in rows:
        f = fam[family_of(name)]
        f[0] += us
        f[1] += cnt

    print(f"\n=== device time total {total/1e6:.2f} s over wall {wall:.2f} s ===")
    print(f"{'family':26s} {'ms':>10s} {'%':>7s} {'calls':>7s}")
    for name, (us, cnt) in sorted(fam.items(), key=lambda kv: -kv[1][0]):
        print(f"{name:26s} {us/1e3:10.1f} {100*us/total:6.1f}% {cnt:7d}")

    print(f"\n--- top 20 kernels ---")
    for name, cnt, us in rows[:20]:
        print(f"{us/1e3:9.1f}ms {100*us/total:5.1f}% n={cnt:6d}  {name[:90]}")

    (OUT / "families.json").write_text(json.dumps(
        {"wall_s": wall, "total_device_s": total / 1e6,
         "families": {k: {"ms": v[0] / 1e3, "calls": v[1]} for k, v in fam.items()},
         "top": [{"name": n, "count": c, "ms": u / 1e3} for n, c, u in rows[:80]]},
        indent=2))
    prof.export_chrome_trace(str(OUT / "trace.json"))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
