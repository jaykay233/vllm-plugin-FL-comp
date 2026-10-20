#!/usr/bin/env python3
"""Recalibrate static-INT8 xmax with prefill ∪ decode activations.

Original ``calibrate_static_int8_l3.py`` only runs ``model(**ids)`` (prefill).
Decode-INT8 needs scales that also cover M=1 residual-stream activations
during generation.  This script:

1. Hooks every 2D linear (same filter as the original).
2. For each math_500 Level-3 prompt, applies the chat template and calls
   ``generate`` so both prefill and decode steps update xmax.
3. Writes ``minicpm5_2b_l3_static_int8_decode.pt`` (max over all steps).

Does not change default vLLM serve args; point the plugin at the new file
via ``VLLM_FL_ILUVATAR_STATIC_INT8_CALIB``.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "/workspace/MiniCPM5-2B"
DATA = "/workspace/evalscope-datasets/math_500/test.jsonl"
OUT = Path(
    "/workspace/vllm-plugin-FL-comp/experiments/calib/minicpm5_2b_l3_static_int8_decode.pt"
)
OLD = Path(
    "/workspace/vllm-plugin-FL-comp/vllm_fl/dispatch/backends/vendor/iluvatar/"
    "minicpm5_2b_l3_static_int8.pt"
)
MAX_LEN = 1024
MAX_NEW = int(os.getenv("CALIB_MAX_NEW_TOKENS", "256"))
ALPHA = 0.5
N_PROMPTS = int(os.getenv("CALIB_N_PROMPTS", "0"))  # 0 = all Level 3


def level3_problems():
    rows = []
    with open(DATA) as f:
        for line in f:
            rec = json.loads(line)
            if int(rec.get("level", -1)) == 3:
                # Match L3 eval phrasing so decode residual is on-distribution.
                rows.append(
                    rec["problem"].rstrip()
                    + "\nPlease reason step by step, and put your final answer "
                    "within \\boxed{}."
                )
    if N_PROMPTS > 0:
        rows = rows[:N_PROMPTS]
    return rows


def main() -> None:
    problems = level3_problems()
    print(
        f"level3 prompts: {len(problems)} max_new={MAX_NEW} max_len={MAX_LEN}",
        flush=True,
    )
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map=None,
        low_cpu_mem_usage=False,
    )
    model.to("cuda")
    model.eval()
    xmax: dict[str, torch.Tensor] = {}

    def make_hook(name: str):
        def hook(mod, inp, _out):
            x = inp[0].detach()
            m = x.abs().amax(dim=tuple(range(x.ndim - 1))).float().cpu()
            if name not in xmax:
                xmax[name] = m
            else:
                xmax[name] = torch.maximum(xmax[name], m)

        return hook

    handles = []
    for name, mod in model.named_modules():
        w = getattr(mod, "weight", None)
        if not isinstance(w, torch.Tensor) or w.ndim != 2:
            continue
        if any(s in name for s in ("embed", "lm_head", "rotary")):
            continue
        handles.append(mod.register_forward_hook(make_hook(name)))

    with torch.no_grad():
        for i, text in enumerate(problems):
            prompt = tok.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=False,
                add_generation_prompt=True,
            )
            ids = tok(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_LEN,
            )
            ids = {k: v.cuda() for k, v in ids.items()}
            model.generate(
                **ids,
                max_new_tokens=MAX_NEW,
                do_sample=False,
                use_cache=True,
            )
            if (i + 1) % 5 == 0 or i == 0 or i + 1 == len(problems):
                print(f"calib {i+1}/{len(problems)} keys={len(xmax)}", flush=True)

    for h in handles:
        h.remove()

    payload = {
        "alpha": ALPHA,
        "xmax": {k: v.half() for k, v in xmax.items()},
        "n_prompts": len(problems),
        "max_len": MAX_LEN,
        "max_new_tokens": MAX_NEW,
        "mode": "prefill_union_decode_chat",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, OUT)
    print(f"wrote {OUT} keys={len(xmax)}", flush=True)

    # Compare vs prefill-only calib if present (repo or site-packages copy).
    old_path = OLD if OLD.is_file() else Path(
        "/usr/local/lib/python3.12/site-packages/vllm_fl/dispatch/backends/"
        "vendor/iluvatar/minicpm5_2b_l3_static_int8.pt"
    )
    if old_path.is_file():
        old = torch.load(old_path, map_location="cpu", weights_only=False)["xmax"]
        ratios = []
        for k, v in xmax.items():
            if k not in old:
                continue
            o = old[k].float()
            n = v.float()
            if o.numel() != n.numel():
                continue
            r = (n / o.clamp_min(1e-8)).mean().item()
            ratios.append((r, k, n.max().item(), o.max().item()))
        ratios.sort(reverse=True)
        print(f"vs old calib matched={len(ratios)}", flush=True)
        if ratios:
            rs = [r for r, *_ in ratios]
            print(
                f"  mean_ratio={sum(rs)/len(rs):.3f} "
                f"p50={sorted(rs)[len(rs)//2]:.3f} "
                f"max_ratio={rs[0]:.3f} ({ratios[0][1]})",
                flush=True,
            )
            print("  top5 mean_ratio:", flush=True)
            for r, k, nm, om in ratios[:5]:
                print(f"    {r:.3f}  new_max={nm:.4g} old_max={om:.4g}  {k}", flush=True)


if __name__ == "__main__":
    main()
