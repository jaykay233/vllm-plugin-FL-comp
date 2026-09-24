#!/usr/bin/env python3
"""Task 2 + 3 combined: instrument every candidate implementation with counters,
then run real inference in eager and compiled mode.

For each layer op we count how many times each *actual* implementation ran:
  * the plugin OOT entry   (RMSNormFL.forward_oot / SiluAndMulFL / RotaryEmbeddingFL)
  * the FlagGems gems_* kernels
  * the reference torch path
This tells us both WHO is used (coverage) and HOW MANY times (attribution input).

Run:  EAGER=1 python /root/src/probe_paths.py
      EAGER=0 python /root/src/probe_paths.py
"""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

EAGER = os.environ.get("EAGER", "1") == "1"
MODEL = "/root/models/MiniCPM5-2B"
OUT = Path("/root/bench_results/path_counts")
OUT.mkdir(parents=True, exist_ok=True)

COUNTS: Counter = Counter()
_PATCHED = []


def patch(target, name, label):
    """Wrap target.name so each call increments COUNTS[label]."""
    orig = getattr(target, name, None)
    if orig is None:
        print(f"  [skip] {label}: {target}.{name} not found")
        return

    def wrapper(*a, **kw):
        COUNTS[label] += 1
        return orig(*a, **kw)

    wrapper.__wrapped__ = orig
    setattr(target, name, wrapper)
    _PATCHED.append((target, name, orig, label))
    print(f"  [ok] {label}")


def main() -> None:
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=MODEL, dtype="bfloat16", trust_remote_code=True,
        max_model_len=2048, gpu_memory_utilization=0.85, enforce_eager=EAGER,
    )

    print("\n--- patching implementations (post model load) ---")
    # OOT layer entry points actually installed on the layers
    from vllm_fl.ops.layernorm import RMSNormFL
    from vllm_fl.ops.activation import SiluAndMulFL
    from vllm_fl.ops.rotary_embedding import RotaryEmbeddingFL

    patch(RMSNormFL, "forward_oot", "OOT RMSNormFL.forward_oot")
    patch(SiluAndMulFL, "forward_oot", "OOT SiluAndMulFL.forward_oot")
    patch(RotaryEmbeddingFL, "forward_oot", "OOT RotaryEmbeddingFL.forward_oot")

    # FlagGems kernels
    import flag_gems.modules.normalization as fn
    import flag_gems.modules.activation as fa
    import flag_gems.modules.rotary_embedding as fr
    import flag_gems

    patch(fn, "gems_rms_forward", "GEMS gems_rms_forward")
    patch(fa, "gems_silu_and_mul", "GEMS gems_silu_and_mul")
    patch(fr, "gems_rope_forward", "GEMS gems_rope_forward")
    patch(flag_gems, "rms_norm", "GEMS flag_gems.rms_norm")

    # vLLM native fused kernels (does compiled mode use these instead?)
    try:
        import vllm._custom_ops as cops  # type: ignore

        for cand in ("rms_norm", "fused_add_rms_norm", "silu_and_mul"):
            if hasattr(cops, cand):
                patch(cops, cand, f"VLLM_NATIVE {cand}")
    except Exception as e:
        print(f"  [info] vllm._custom_ops not patchable: {e}")

    # dispatch layer itself
    from vllm_fl.dispatch import CachedOp

    patch(CachedOp, "__call__", "PLUGIN_DISPATCH CachedOp.__call__")

    # ---------------- run inference ----------------
    COUNTS.clear()
    out = llm.generate(
        ["The capital of France is", "Explain what a GPU is in one sentence."],
        SamplingParams(max_tokens=16, temperature=0.0, ignore_eos=True),
    )
    n_tokens = sum(len(o.outputs[0].token_ids) for o in out)

    print(f"\n########## EAGER={EAGER}  ({n_tokens} output tokens, 2 prompts) ##########")
    for label, n in sorted(COUNTS.items(), key=lambda kv: -kv[1]):
        print(f"  {n:8d}  {label}")

    rec = {"eager": EAGER, "output_tokens": n_tokens, "counts": dict(COUNTS)}
    (OUT / f"counts_eager{int(EAGER)}.json").write_text(json.dumps(rec, indent=2))

    # restore
    for target, name, orig, _ in _PATCHED:
        setattr(target, name, orig)

    print(f"\nPROBE_PATHS_DONE eager={EAGER}", flush=True)


if __name__ == "__main__":
    main()
