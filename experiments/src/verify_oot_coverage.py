#!/usr/bin/env python3
"""Task 3: does the OOT layer path actually reach the plugin dispatch?

Evidence so far: in `vllm serve` logs the dispatch manager only ever logged
`Op 'attention_backend' using 'vendor.metax'`. If the 42-layer hot ops went
through the OOT -> CachedOp path, we'd also see rms_norm / silu_and_mul /
rotary_embedding. This script runs real inference offline and then reads
OpManager._called_ops, which records exactly which dispatch ops were invoked.

Run:  conda activate mx && python /root/src/verify_oot_coverage.py
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

os.environ["VLLM_FL_DISPATCH_DEBUG"] = "1"

OUT = Path("/root/bench_results/oot_coverage")
OUT.mkdir(parents=True, exist_ok=True)

# make the plugin's plain-logger INFO lines visible (vllm_fl.ops.custom_ops etc.)
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
)

MODEL = "/root/models/MiniCPM5-2B"


def main() -> None:
    import torch
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=MODEL,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=2048,
        gpu_memory_utilization=0.85,
        enforce_eager=True,
    )

    out = llm.generate(
        ["The capital of France is"],
        SamplingParams(max_tokens=8, temperature=0.0, ignore_eos=True),
    )
    print("\n--- generated:", repr(out[0].outputs[0].text[:60]))

    # ---------------- which dispatch ops were actually invoked? ----------------
    from vllm_fl.dispatch import get_default_manager

    mgr = get_default_manager()
    called = dict(mgr._called_ops)

    print("\n" + "=" * 74)
    print("dispatch ops ACTUALLY USED during inference")
    print("=" * 74)
    if not called:
        print("  <none>  -> the plugin dispatch was never called")
    for op, impl in sorted(called.items()):
        print(f"  {op:28s} -> {impl}")

    # ---------------- which layer classes did the model actually get? ----------
    print("\n" + "=" * 74)
    print("actual layer classes in the loaded model")
    print("=" * 74)
    model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    layer0 = model.model.layers[0]

    probe = {
        "input_layernorm": layer0.input_layernorm,
        "post_attention_layernorm": layer0.post_attention_layernorm,
        "mlp.act_fn": getattr(layer0.mlp, "act_fn", None),
        "self_attn.rotary_emb": getattr(layer0.self_attn, "rotary_emb", None),
    }
    layers = {}
    for name, obj in probe.items():
        if obj is None:
            layers[name] = None
            print(f"  {name:28s} = <None>")
            continue
        cls = type(obj)
        mro = [c.__name__ for c in cls.__mro__[:4]]
        is_fl = cls.__module__.startswith("vllm_fl")
        layers[name] = {"class": cls.__name__, "module": cls.__module__, "is_fl": is_fl}
        print(f"  {name:28s} = {cls.__module__}.{cls.__name__}  FL={is_fl}")
        print(f"      mro: {' > '.join(mro)}")

    # does the OOT class define forward_oot that we expect to be called?
    for name, obj in probe.items():
        if obj is None:
            continue
        fwd = getattr(obj, "_forward_method", None)
        print(f"  {name:28s} _forward_method = {getattr(fwd, '__name__', fwd)}")

    verdict = {
        "hot_ops_reached_dispatch": [
            op for op in ("rms_norm", "silu_and_mul", "rotary_embedding")
            if op in called
        ],
        "all_dispatch_ops_used": called,
        "layer_classes": layers,
    }
    (OUT / "coverage.json").write_text(json.dumps(verdict, indent=2, default=str))

    print("\n" + "=" * 74)
    print("VERDICT")
    print("=" * 74)
    reached = verdict["hot_ops_reached_dispatch"]
    print(f"  hot ops reaching plugin dispatch: {reached or '<NONE>'}")
    print(f"  total dispatch ops used        : {len(called)}")
    print(f"\nWrote {OUT / 'coverage.json'}")
    print("OOT_COVERAGE_DONE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback

        traceback.print_exc()
        print(f"\nOOT_COVERAGE_FAILED: {type(e).__name__}: {e}", flush=True)
        raise
