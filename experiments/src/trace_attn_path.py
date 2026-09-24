#!/usr/bin/env python3
"""Which attention code path actually runs at decode time?

Static reading is contradictory:
  * metax.yaml forces `attention_backend: vendor:metax`
  * vendor/.../attention/flash_attn.py -> FlashAttentionImpl.forward calls
    flash_attn_varlen_func(out=..., seqused_k=..., num_splits=0, fa_version=...)
  * but vendor .../attention/utils/fa_utils.py just re-exports MetaX's
    flash_attn.flash_attn_interface.flash_attn_varlen_func, whose signature is
      (q,k,v,cu_seqlens_q,cu_seqlens_k,max_seqlen_q,max_seqlen_k,...,block_table,s_aux,...)
    i.e. NO out/seqused_k/num_splits/fa_version.

So one of those two is dead code. Wrapping the *module globals* that the
AttentionImpl classes actually close over tells us which one is live.

NOTE: importing flash_attn standalone segfaults (MACA runtime not up), so vllm
must be imported first to bring the platform plugin up.

Run:  conda activate mx && python /root/src/trace_attn_path.py
"""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

MODEL = "/root/models/MiniCPM5-2B"
PROMPT = "The theory of general relativity describes gravity as"
N_TOKENS = 24
OUT = Path("/root/bench_results/attn_path")
OUT.mkdir(parents=True, exist_ok=True)

RECORDS: Counter = Counter()
SHAPES: dict = {}


def _shp(t):
    try:
        return tuple(t.shape)
    except Exception:  # noqa: BLE001
        return None


def _rec(name: str, kwargs: dict):
    sig = (name, tuple(sorted(kwargs.keys())))
    RECORDS[sig] += 1
    if sig not in SHAPES:
        info = {}
        for k, v in kwargs.items():
            if hasattr(v, "shape"):
                info[k] = _shp(v)
            elif isinstance(v, (int, float, str, bool, type(None))):
                info[k] = v
            else:
                info[k] = type(v).__name__
        SHAPES[sig] = info


def _wrap(obj, attr: str, label: str) -> bool:
    fn = getattr(obj, attr, None)
    if fn is None or not callable(fn):
        return False
    if getattr(fn, "_traced", False):
        return True

    def wrapper(*args, **kwargs):
        # most of these are called with kwargs by vLLM; recover names otherwise
        _rec(label, kwargs if kwargs else {"_args": args})
        return fn(*args, **kwargs)

    wrapper._traced = True
    setattr(obj, attr, wrapper)
    return True


def instrument() -> None:
    """Wrap every attention entry point reachable by the impls."""
    targets: list[tuple[str, str, str]] = [
        # ---- the vendor (MACA) impl's binding
        ("vllm_fl.dispatch.backends.vendor.metax.impl.attention.flash_attn",
         "flash_attn_varlen_func", "VENDOR.flash_attn_varlen_func"),
        # ---- the flaggems impl's binding
        ("vllm_fl.dispatch.backends.flaggems.impl.attention",
         "flash_attn_varlen_func", "FLAGGEMS.flash_attn_varlen_func"),
        # ---- raw FA namespace
        ("flash_attn.flash_attn_interface", "flash_attn_varlen_func",
         "RAW.flash_attn_varlen_func"),
        ("flash_attn.flash_attn_interface", "flash_attn_with_kvcache",
         "RAW.flash_attn_with_kvcache"),
    ]
    import importlib

    for mod_name, attr, label in targets:
        try:
            mod = importlib.import_module(mod_name)
        except Exception as exc:  # noqa: BLE001
            print(f"  [skip] {mod_name}: {type(exc).__name__}")
            continue
        ok = _wrap(mod, attr, label)
        print(f"  [{'ok ' if ok else 'miss'}] {label}  ({mod_name})")


def report_impls(llm) -> None:
    try:
        from vllm.config import get_layers_from_vllm_config
        from vllm.model_executor.layers.attention.attention import Attention

        cfg = llm.llm_engine.vllm_config
        layers = get_layers_from_vllm_config(cfg, Attention)
        kinds = Counter()
        sample = None
        for name, layer in layers.items():
            kinds[f"{type(layer.impl).__module__}.{type(layer.impl).__name__}"] += 1
            if sample is None:
                sample = (name, layer.impl)
        print("  AttentionImpl classes:")
        for k, n in kinds.items():
            print(f"    {n:4d} layers  {k}")
        if sample is not None:
            impl = sample[1]
            print(f"    sample layer: {sample[0]}")
            for attr in ("max_num_splits", "vllm_flash_attn_version",
                         "num_heads", "num_kv_heads", "head_size",
                         "aot_schedule", "use_full_cuda_graph"):
                if hasattr(impl, attr):
                    print(f"      impl.{attr} = {getattr(impl, attr)}")
    except Exception as exc:  # noqa: BLE001
        print(f"  (could not enumerate impls: {type(exc).__name__}: {exc})")


def main() -> None:
    import vllm  # noqa: F401  -- brings MACA runtime + platform plugin up

    print("=" * 100)
    print("INSTRUMENTING ATTENTION ENTRY POINTS")
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

    print()
    print("=" * 100)
    print("ATTENTION IMPL ACTUALLY INSTANTIATED")
    print("=" * 100)
    report_impls(llm)

    sp = SamplingParams(max_tokens=N_TOKENS, temperature=0.0, ignore_eos=True)
    out = llm.generate([PROMPT], sp, use_tqdm=False)
    torch.cuda.synchronize()
    n_out = len(out[0].outputs[0].token_ids)

    print()
    print("=" * 100)
    print(f"ATTENTION FUNCTIONS CALLED  ({n_out} tokens: 1 prefill + {n_out - 1} decode)")
    print("=" * 100)
    for (name, keys), count in RECORDS.most_common():
        print(f"\n  {name}   x{count}")
        print(f"    kwargs: {list(keys)}")
        for k in sorted(SHAPES[(name, keys)]):
            print(f"      {k:22s} = {SHAPES[(name, keys)][k]}")

    payload = {
        "impls": "see stdout",
        "calls": {f"{n}|{'/'.join(k)}": c for (n, k), c in RECORDS.items()},
        "shapes": {f"{n}|{'/'.join(k)}": v for (n, k), v in SHAPES.items()},
    }
    (OUT / "attn_path.json").write_text(json.dumps(payload, indent=2, default=str))
    print(f"\n  wrote {OUT / 'attn_path.json'}")
    print("ATTN_PATH_DONE", flush=True)


if __name__ == "__main__":
    main()
