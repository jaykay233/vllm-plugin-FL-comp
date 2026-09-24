#!/usr/bin/env python3
"""Separate prefill and decode cost structure with torch.profiler.

The mcTracer trace mixes both phases and the window cannot be cut cleanly, so
this measures them independently instead:

    max_tokens=1  -> essentially one 512-token prefill forward  (TTFT side)
    max_tokens=64 with a 1-token prompt -> pure decode forwards (TPOT side)

Reports per-kernel-family device time for each, which is what decides whether a
fusion is worth building.

Run:  python /root/src/prefill_decode_profile.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

OUT = Path("/root/bench_results/prefill_decode")
OUT.mkdir(parents=True, exist_ok=True)

os_env = {
    "VLLM_FL_PREFER": "flagos",
    "USE_FLAGGEMS": "1",
    "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
}

FAMILY_RULES = [
    ("silu_and_mul_kernel", "silu_and_mul (GEMS)"),
    ("fused_add_rms_norm_kernel", "rms_norm fused_add (GEMS)"),
    ("rms_norm_kernel", "rms_norm (GEMS)"),
    ("apply_rotary_pos_emb", "rope (GEMS)"),
    ("linear_kernel", "GEMM linear (GEMS)"),
    ("mm_kernel", "GEMM mm (MetaX)"),
    ("flash_fwd_splitkv_combine_kernel", "attention (vendor)"),
    ("flash_fwd_splitkv_kernel", "attention (vendor)"),
    ("flash_fwd_kernel", "attention (vendor)"),
    ("reshape_and_cache_flash_kernel", "kvcache_write (vendor)"),
    ("argmax_kernel", "argmax sampler (GEMS)"),
    ("embedding_kernel", "embedding"),
    ("_copy_kernel_kernel_rank", "copy (GEMS)"),
    ("_to_copy_func_kernel", "copy (GEMS)"),
    ("fill_scalar_func_kernel", "fill (GEMS)"),
    ("add_func_kernel", "add (GEMS)"),
    ("sub_func", "sub (GEMS)"),
    ("_compute_slot_mapping_kernel", "slot_mapping"),
    ("_index_jit_function", "index"),
    ("reduce_then_scan", "prefill_scan"),
    ("memcpy", "memcpy"),
    ("triton_red_fused", "inductor fused reduction"),
    ("triton_poi_fused", "inductor fused pointwise"),
]


def family_of(name: str) -> str:
    low = name.lower()
    for key, fam in FAMILY_RULES:
        if key.lower() in low:
            return fam
    if low.startswith("mc"):
        return "HOST api"
    return "other"


def main() -> None:
    import os

    for k, v in os_env.items():
        os.environ.setdefault(k, v)

    import torch
    from torch.profiler import ProfilerActivity, profile
    from vllm import LLM, SamplingParams

    llm = LLM(
        model="/root/models/MiniCPM5-2B",
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=4096,
        gpu_memory_utilization=0.85,
        enforce_eager=True,
        enable_prefix_caching=False,
    )

    long_prompt = "Explain the theory of relativity in detail. " * 60   # ~480 tok
    short_prompt = "Hi"

    # warmup both shapes
    llm.generate([long_prompt], SamplingParams(max_tokens=1, temperature=0.0, ignore_eos=True))
    llm.generate([short_prompt], SamplingParams(max_tokens=8, temperature=0.0, ignore_eos=True))
    torch.cuda.synchronize()

    cases = {
        "prefill_only": (long_prompt, 1),
        "decode_only": (short_prompt, 64),
    }

    out = {"driver": torch.cuda.get_device_name(0), "cases": {}}
    for tag, (prompt, mt) in cases.items():
        sp = SamplingParams(max_tokens=mt, temperature=0.0, ignore_eos=True)

        # wall time
        torch.cuda.synchronize()
        import time
        t0 = time.perf_counter()
        res = llm.generate([prompt], sp)
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        n_in = len(res[0].prompt_token_ids)
        n_out = len(res[0].outputs[0].token_ids)

        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            llm.generate([prompt], sp)
            torch.cuda.synchronize()

        fam_t: dict[str, float] = defaultdict(float)
        fam_n: dict[str, int] = defaultdict(int)
        raw: list[dict] = []
        for e in prof.key_averages():
            t = e.self_device_time_total
            if t <= 0:
                continue
            f = family_of(e.key)
            if f == "HOST api":
                continue
            fam_t[f] += t
            fam_n[f] += e.count
            raw.append({"kernel": e.key, "family": f, "calls": e.count,
                        "ms": t / 1e3, "us_each": t / e.count})
        raw.sort(key=lambda r: -r["ms"])
        tot = sum(fam_t.values())

        print("\n" + "=" * 90)
        print(f"{tag}: {n_in} in / {n_out} out | wall {wall * 1e3:.1f} ms | "
              f"GPU {tot / 1e3:.2f} ms")
        print("=" * 90)
        print(f"  {'family':28s} {'ms':>9s} {'share':>7s} {'calls':>8s} {'us each':>9s}")
        for f, t in sorted(fam_t.items(), key=lambda kv: -kv[1]):
            print(f"  {f:28s} {t / 1e3:9.3f} {t / tot * 100:6.1f}% "
                  f"{fam_n[f]:8d} {t / fam_n[f]:9.2f}")
        print(f"  {'TOTAL':28s} {tot / 1e3:9.3f}")

        print(f"\n  -- top 12 kernels --")
        for r in raw[:12]:
            print(f"    {r['ms']:8.3f}ms {r['ms'] / (tot / 1e3) * 100:5.1f}% "
                  f"x{r['calls']:<6d} {r['us_each']:8.2f}us  {r['kernel'][:60]}")

        out["cases"][tag] = {
            "wall_ms": wall * 1e3, "gpu_ms": tot / 1e3,
            "n_in": n_in, "n_out": n_out,
            "by_family": {k: {"ms": v / 1e3, "calls": fam_n[k]}
                          for k, v in sorted(fam_t.items(), key=lambda kv: -kv[1])},
            "top_kernels": raw[:25],
        }

    (OUT / "prefill_decode.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {OUT / 'prefill_decode.json'}")
    print("\nPREFILL_DECODE_PROFILE_DONE", flush=True)


if __name__ == "__main__":
    main()
