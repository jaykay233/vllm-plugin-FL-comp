#!/usr/bin/env python3
"""Calibrate SmoothQuant-style static INT8 scales on math_500 Level 3.

Writes a tensor dict consumed by the Iluvatar static-INT8 linear patch.
Does not change vLLM serve args.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "/workspace/MiniCPM5-2B"
DATA = "/workspace/evalscope-datasets/math_500/test.jsonl"
OUT = Path("/workspace/vllm-plugin-FL-comp/experiments/calib/minicpm5_2b_l3_static_int8.pt")
MAX_LEN = 1024
ALPHA = 0.5


def level3_problems():
    rows = []
    with open(DATA) as f:
        for line in f:
            rec = json.loads(line)
            if int(rec.get("level", -1)) == 3:
                rows.append(rec["problem"])
    return rows


def main() -> None:
    problems = level3_problems()
    print(f"level3 prompts: {len(problems)}")
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
        if not hasattr(mod, "weight"):
            continue
        w = getattr(mod, "weight", None)
        if not isinstance(w, torch.Tensor) or w.ndim != 2:
            continue
        if any(s in name for s in ("embed", "lm_head", "rotary")):
            continue
        handles.append(mod.register_forward_hook(make_hook(name)))

    with torch.no_grad():
        for i, text in enumerate(problems):
            ids = tok(text, return_tensors="pt", truncation=True, max_length=MAX_LEN)
            ids = {k: v.cuda() for k, v in ids.items()}
            model(**ids)
            if (i + 1) % 15 == 0 or i == 0:
                print(f"calib {i+1}/{len(problems)}")

    for h in handles:
        h.remove()

    payload = {
        "alpha": ALPHA,
        "xmax": {k: v.half() for k, v in xmax.items()},
        "n_prompts": len(problems),
        "max_len": MAX_LEN,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, OUT)
    print(f"wrote {OUT} keys={len(xmax)}")


if __name__ == "__main__":
    main()
