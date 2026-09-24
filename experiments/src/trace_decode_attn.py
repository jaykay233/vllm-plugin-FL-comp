#!/usr/bin/env python3
"""Is decode attention eager (outside the CUDA graph), or captured inside it?

This decides whether `num_splits` can be chosen PER STEP from the real sequence
length, or has to be a single constant baked in at graph-capture time.

Evidence that it may be eager:
  compilation_config.splitting_ops contains 'vllm::unified_attention_with_output'
  -> vLLM splits the (piecewise) graph around attention.

Method: wrap `flash_attn_with_kvcache` in the vendor attention module namespace
(that is the symbol decode actually calls -- the earlier trace wrongly wrapped
flash_attn_varlen_func, which decode never touches). During a generation of N
tokens the KV length grows monotonically, so:
  * eager  -> one call per layer per step, cache_seqlens.max() increases 13,14,...
  * graphed-> calls only during capture/warmup, cache_seqlens looks frozen

Run:  conda activate mx && python /root/src/trace_decode_attn.py
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

MODEL = "/root/models/MiniCPM5-2B"
PROMPT = "The theory of general relativity describes gravity as"
N_TOKENS = 16
OUT = Path("/root/bench_results/attn_path")
OUT.mkdir(parents=True, exist_ok=True)

CALLS: list[dict] = []


def instrument() -> None:
    import importlib

    mod = importlib.import_module(
        "vllm_fl.dispatch.backends.vendor.metax.impl.attention.flash_attn"
    )
    orig = mod.flash_attn_with_kvcache
    print(f"  wrapping {mod.__name__}.flash_attn_with_kvcache -> {orig}")

    def wrapper(*args, **kwargs):
        rec = {"kwargs": sorted(kwargs)}
        cs = kwargs.get("cache_seqlens")
        if cs is not None and hasattr(cs, "max"):
            try:
                rec["max_seqlen"] = int(cs.max().item())
                rec["min_seqlen"] = int(cs.min().item())
                rec["batch"] = int(cs.shape[0])
            except Exception as exc:  # noqa: BLE001
                rec["seqlen_err"] = type(exc).__name__
        q = kwargs.get("q")
        if hasattr(q, "shape"):
            rec["q_shape"] = tuple(q.shape)
        rec["num_splits_passed"] = kwargs.get("num_splits", "<absent>")
        rec["called_from_impl"] = True
        CALLS.append(rec)
        return orig(*args, **kwargs)

    wrapper._traced = True
    mod.flash_attn_with_kvcache = wrapper
    # the module also bound it into the class closure? verify at runtime below
    return


def main() -> None:
    import vllm  # noqa: F401  (brings MACA runtime up)

    print("=" * 100)
    print("TRACING decode attention (flash_attn_with_kvcache)")
    print("=" * 100)
    instrument()

    import torch
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=MODEL,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=2048,
        gpu_memory_utilization=0.5,
    )

    sp = SamplingParams(max_tokens=N_TOKENS, temperature=0.0, ignore_eos=True)
    CALLS.clear()

    # also try wrapping the class-level binding, in case the impl closed over it
    from vllm_fl.dispatch.backends.vendor.metax.impl.attention import (
        flash_attn as vfa,
    )

    cls = vfa.FlashAttentionImpl
    print(f"  FlashAttentionImpl.forward is {cls.forward.__qualname__}")

    out = llm.generate([PROMPT], sp, use_tqdm=False)
    torch.cuda.synchronize()
    n_out = len(out[0].outputs[0].token_ids)

    print()
    print("=" * 100)
    print(f"CALLS RECORDED: {len(CALLS)}   (generated {n_out} tokens)")
    print("=" * 100)
    if not CALLS:
        print("  NONE -> the impl does not call module-global flash_attn_with_kvcache")
        print("        (it may be captured inside a CUDA graph, or bound elsewhere)")
    else:
        print(f"  layers/step implied: {len(CALLS) / max(1, n_out):.1f}")
        print(f"  num_splits passed  : {CALLS[0]['num_splits_passed']!r}")
        print(f"  q_shape            : {CALLS[0].get('q_shape')}")
        print(f"  batch              : {CALLS[0].get('batch')}")
        print()
        print("  first 20 calls (max_seqlen should CLIMB if attention is eager):")
        for i, r in enumerate(CALLS[:20]):
            print(f"    call {i:3d}  max_seqlen={r.get('max_seqlen')}  "
                  f"batch={r.get('batch')}")
        seqs = [r.get("max_seqlen") for r in CALLS if "max_seqlen" in r]
        print()
        print(f"  distinct max_seqlen values: {sorted(set(seqs))[:25]}")
        print(f"  -> {'EAGER (per-step, adaptive num_splits is possible)'
                     if len(set(seqs)) > 1 else 'FROZEN (captured in CUDA graph)'}")

    print("\nTRACE_DECODE_ATTN_DONE", flush=True)


if __name__ == "__main__":
    main()
