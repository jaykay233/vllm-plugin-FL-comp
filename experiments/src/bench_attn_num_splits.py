#!/usr/bin/env python3
"""A/B MetaX decode-time FlashAttention num_splits.

Root cause
----------
Decode with paged KV goes through flash_attn_with_kvcache(), whose `num_splits`
defaults to 0 == "let the MACA heuristic decide". On MiniCPM5-2B (42 layers,
16 q-heads / 2 kv-heads, head_dim 128) the heuristic picks an *excessive and
sequence-length-agnostic* split count, so every decode step pays for a
cross-split reduction pass:

    flash_fwd_splitkv_kernel          ~357 us/step  (42 layers)
    flash_fwd_splitkv_combine_kernel ~1008 us/step  (42 layers)  <-- pure overhead

The combine cost is ~the same at 13 and at 1201 KV tokens, i.e. the heuristic
does not react to the real sequence length. Attention runs INSIDE the FULL
decode CUDA graph (verified: only 42 capture-time calls, all with a frozen
max_seqlen), so `num_splits` is a capture-time constant -- it cannot be chosen
per step. Hence we sweep it and pick per workload regime.

The plugin forwards FL_METAX_ATTN_NUM_SPLITS to the decode call.

Usage:
    conda activate mx
    python bench_attn_num_splits.py --splits-list 0,1,4 --ctx 1200 --batch 16
    python bench_attn_num_splits.py --splits 1 --ctx 1200 --batch 16   # worker
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

MODEL = "/root/models/MiniCPM5-2B"
BASE_SENTENCE = "The theory of general relativity describes gravity as the curvature of spacetime"
DECODE_TOKENS_DEFAULT = 64
OUT = Path("/root/bench_results/attn_num_splits")
OUT.mkdir(parents=True, exist_ok=True)

ATTN_BUCKETS = {
    "attn_nosplit": ("flash_fwd_kernel",),
    "attn_splitkv": ("flash_fwd_splitkv_kernel",),
    "attn_combine": ("flash_fwd_splitkv_combine",),
    "kv_cache_write": ("reshape_and_cache",),
}


def classify(name: str) -> str | None:
    for bucket, pats in ATTN_BUCKETS.items():
        if any(p in name for p in pats):
            return bucket
    return None


def build_prompts(target_tokens: int, batch: int) -> list[str]:
    if batch <= 1:
        if target_tokens <= 0:
            return [BASE_SENTENCE]
        return [" ".join([BASE_SENTENCE] * max(1, target_tokens // 15))]
    out = []
    for i in range(batch):
        body = " ".join([BASE_SENTENCE] * max(1, target_tokens // 15))
        # distinct prefixes so prefix caching cannot merge the batch
        out.append(f"[req {i}] " + body)
    return out


def measure(num_splits: int, ctx: int, batch: int, out_tokens: int) -> dict:
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    if num_splits >= 0:
        os.environ["FL_METAX_ATTN_NUM_SPLITS"] = str(num_splits)
    else:
        # -1 == leave the env var unset so the adaptive batch rule applies
        os.environ.pop("FL_METAX_ATTN_NUM_SPLITS", None)

    import torch
    from vllm import LLM, SamplingParams

    from vllm_fl.dispatch.backends.vendor.metax.impl.attention import (
        flash_attn as _vfa,
    )

    effective = _vfa._decode_num_splits(int(batch))

    llm = LLM(
        model=MODEL,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=2048,
        gpu_memory_utilization=0.30,
    )

    prompts = build_prompts(ctx, batch)
    tok = llm.get_tokenizer()
    real_prompt_tokens = max(len(tok(p)["input_ids"]) for p in prompts)

    def gen(n: int) -> tuple[float, list[list[int]]]:
        sp = SamplingParams(max_tokens=n, temperature=0.0, ignore_eos=True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = llm.generate(prompts, sp, use_tqdm=False)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        return dt, [list(o.outputs[0].token_ids) for o in out]

    gen(out_tokens)  # warmup: compile + cudagraph capture
    gen(out_tokens)  # settle

    # TPOT: repeat and take the MIN. Interference from other GPU work can only
    # add time, so the minimum is the cleanest estimate of the true cost.
    # A single sample was noisy enough to contradict the profiler by 600us.
    tpot_samples = []
    for _ in range(3):
        t1, ids1 = gen(1)
        t2, ids64 = gen(out_tokens)
        steps = max(1, len(ids64[0]) - len(ids1[0]))
        tpot_samples.append((t2 - t1) / steps * 1e6)
    tpot_us = min(tpot_samples)

    sp = SamplingParams(max_tokens=out_tokens, temperature=0.0, ignore_eos=True)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        out = llm.generate(prompts, sp, use_tqdm=False)
        torch.cuda.synchronize()
    n_out = len(out[0].outputs[0].token_ids)
    n_steps = max(1, n_out - 1)

    counts: Counter[str] = Counter()
    dur: Counter[str] = Counter()
    attn_counts: Counter[str] = Counter()
    attn_dur: Counter[str] = Counter()
    for e in prof.events():
        if e.device_type != torch.autograd.DeviceType.CUDA or not e.name:
            continue
        counts[e.name] += e.count
        dur[e.name] += e.device_time
        b = classify(e.name)
        if b:
            attn_counts[b] += e.count
            attn_dur[b] += e.device_time

    total_us = sum(dur.values())
    return {
        "num_splits": num_splits,
        "ctx_target": ctx,
        "batch": batch,
        "out_tokens": out_tokens,
        "prompt_tokens": real_prompt_tokens,
        "module_effective_num_splits": effective,
        "tpot_us": tpot_us,
        "tpot_samples_us": tpot_samples,
        "decode_steps": n_steps,
        "tokens_per_step": batch,
        "tpot_per_token_us": tpot_us / batch,
        "total_launches": sum(counts.values()),
        "gpu_us_per_step": total_us / n_steps,
        "attn": {
            b: {
                "launches": attn_counts[b],
                "total_ms": attn_dur[b] / 1000,
                "per_step_us": attn_dur[b] / n_steps,
                "per_layer_us": attn_dur[b] / n_steps / 42,
            }
            for b in ATTN_BUCKETS
            if attn_counts.get(b)
        },
        "attn_total_per_step_us": sum(attn_dur.values()) / n_steps,
        "token_ids_head": ids64[0][:12],
        "top_kernels": [
            {"name": n, "ms": dur[n] / 1000, "launches": counts[n]}
            for n, _ in dur.most_common(15)
        ],
    }


def driver(split_list: list[int], ctx: int, batch: int, out_tokens: int) -> None:
    tag = f"ctx{ctx}_b{batch}_o{out_tokens}"
    print("=" * 108)
    print(f"DECODE ATTENTION num_splits A/B   MiniCPM5-2B / MetaX C500 / "
          f"batch={batch} / prompt ~{ctx} tok / generate {out_tokens} tok")
    print("=" * 108)
    results = []
    for ns in split_list:
        log = OUT / f"worker_ns{ns}_{tag}.log"
        print(f"\n>>> FL_METAX_ATTN_NUM_SPLITS={ns}  {tag} ...", flush=True)
        env = dict(os.environ)
        if ns >= 0:
            env["FL_METAX_ATTN_NUM_SPLITS"] = str(ns)
        else:
            env.pop("FL_METAX_ATTN_NUM_SPLITS", None)
        env.pop("PYTHONPATH", None)
        with log.open("w") as fh:
            rc = subprocess.call(
                [sys.executable, __file__, "--splits", str(ns),
                 "--ctx", str(ctx), "--batch", str(batch),
                 "--out", str(out_tokens)],
                stdout=fh, stderr=subprocess.STDOUT, env=env,
            )
        rf = OUT / f"result_ns{ns}_{tag}.json"
        if rc != 0 or not rf.exists():
            print(f"     worker failed (rc={rc}), see {log}")
            continue
        res = json.loads(rf.read_text())
        results.append(res)
        print(f"     prompt={res['prompt_tokens']} tok  TPOT={res['tpot_us']:.1f} us "
              f"({res['tpot_per_token_us']:.1f} us/token)  "
              f"attn={res['attn_total_per_step_us']:.1f} us/step  "
              f"combine={'YES' if 'attn_combine' in res['attn'] else 'no'}")

    if not results:
        print("\nno results")
        return

    base = next((r for r in results if r["num_splits"] == 0), results[0])
    print()
    print("=" * 108)
    print(f"SUMMARY  baseline num_splits={base['num_splits']}  "
          f"prompt={base['prompt_tokens']} tok  batch={batch}  "
          f"out={out_tokens} tok  "
          f"TPOT={base['tpot_us']:.1f} us  ({base['tpot_per_token_us']:.1f} us/token)")
    print("=" * 108)
    hdr = (f"  {'splits':>7} {'TPOT us':>9} {'us/token':>9} {'vs base':>8} "
           f"{'gain%':>7} {'splitkv':>9} {'combine':>9} {'attn/step':>10} {'launches':>9}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in sorted(results, key=lambda x: x["num_splits"]):
        a = r["attn"]
        g = (base["tpot_us"] - r["tpot_us"]) / base["tpot_us"] * 100
        print(f"  {r['num_splits']:>7} {r['tpot_us']:>9.1f} "
              f"{r['tpot_per_token_us']:>9.1f} "
              f"{base['tpot_us'] / r['tpot_us']:>7.3f}x {g:>6.2f}% "
              f"{a.get('attn_splitkv', {}).get('per_step_us', 0):>9.1f} "
              f"{a.get('attn_combine', {}).get('per_step_us', 0):>9.1f} "
              f"{r['attn_total_per_step_us']:>10.1f} "
              f"{r['total_launches']:>9}")

    print()
    print("  correctness (first 12 greedy token ids of request 0):")
    for r in sorted(results, key=lambda x: x["num_splits"]):
        print(f"    splits={r['num_splits']:<3} {r['token_ids_head']}")
    ref = results[0]["token_ids_head"]
    print(f"    -> {'ALL IDENTICAL' if all(r['token_ids_head'] == ref for r in results) else 'MISMATCH!'}")

    (OUT / f"summary_{tag}.json").write_text(json.dumps(results, indent=2))
    print(f"\n  wrote {OUT / f'summary_{tag}.json'}")
    print("ATTN_NUM_SPLITS_DONE", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=int, default=None)
    ap.add_argument("--splits-list", type=str, default="0,1")
    ap.add_argument("--ctx", type=int, default=0)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--out", type=int, default=DECODE_TOKENS_DEFAULT)
    args = ap.parse_args()

    if args.splits is not None:
        res = measure(args.splits, args.ctx, args.batch, args.out)
        (OUT / f"result_ns{args.splits}_ctx{args.ctx}_b{args.batch}_o{args.out}.json"
         ).write_text(json.dumps(res, indent=2))
        print(json.dumps({k: v for k, v in res.items() if k != "top_kernels"},
                         indent=2))
        return

    driver([int(x) for x in args.splits_list.split(",") if x.strip()],
           args.ctx, args.batch, args.out)


if __name__ == "__main__":
    main()
