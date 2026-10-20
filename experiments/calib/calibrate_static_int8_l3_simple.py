#!/usr/bin/env python3
"""Calibrate activation xmax for SIMPLE static W8A8 (no SmoothQuant / QuaRot).

Collects per-channel input xmax over prefill∪decode on math_500 Level-3,
same hooks as decode calib.  SIMPLE quant uses:
  * weight: per-channel or per-head absmax (from W, online)
  * act: static per-tensor a_scale = amax(xmax)/127

Writes ``minicpm5_2b_l3_static_int8_simple.pt`` (+ site copy).
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
    "/workspace/vllm-plugin-FL-comp/experiments/calib/"
    "minicpm5_2b_l3_static_int8_simple.pt"
)
MAX_LEN = 1024
MAX_NEW = int(os.getenv("CALIB_MAX_NEW_TOKENS", "128"))
N_PROMPTS = int(os.getenv("CALIB_N_PROMPTS", "20"))  # fast default; 0=all L3


def level3_problems():
    rows = []
    with open(DATA) as f:
        for line in f:
            rec = json.loads(line)
            if int(rec.get("level", -1)) == 3:
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
        f"simple-static calib prompts={len(problems)} max_new={MAX_NEW}",
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
        "alpha": 0.0,  # unused by SIMPLE; documents no SmoothQuant
        "xmax": {k: v.half() for k, v in xmax.items()},
        "n_prompts": len(problems),
        "max_len": MAX_LEN,
        "max_new_tokens": MAX_NEW,
        "mode": "prefill_union_decode_chat_simple_static",
        "note": "for VLLM_FL_ILUVATAR_STATIC_INT8_SIMPLE=1 (per-channel/head W, static A)",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, OUT)
    site = Path(
        "/usr/local/lib/python3.12/site-packages/vllm_fl/dispatch/backends/"
        "vendor/iluvatar"
    )
    site.mkdir(parents=True, exist_ok=True)
    torch.save(payload, site / OUT.name)
    print(f"wrote {OUT} and site copy keys={len(xmax)}", flush=True)


if __name__ == "__main__":
    main()
