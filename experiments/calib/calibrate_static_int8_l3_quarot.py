#!/usr/bin/env python3
"""Calibrate xmax under QuaRot-lite activation rotation (prefill ∪ decode).

Hooks record amax(|x @ H @ diag(s)|) so SmoothQuant matches the online
Hadamard used by ``VLLM_FL_ILUVATAR_STATIC_INT8_QUAROT=1``.  Seed/signs must
match ``static_int8._quarot_signs``.
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
    "minicpm5_2b_l3_static_int8_decode_quarot_a075.pt"
)
MAX_LEN = 1024
MAX_NEW = int(os.getenv("CALIB_MAX_NEW_TOKENS", "256"))
ALPHA = float(os.getenv("CALIB_ALPHA", "0.75"))
N_PROMPTS = int(os.getenv("CALIB_N_PROMPTS", "0"))

_HADAMARD_SIGNS: dict[int, torch.Tensor] = {}


def _is_pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _fwht_pow2(x: torch.Tensor) -> torch.Tensor:
    n = x.shape[-1]
    out = x.float()
    h = 1
    while h < n:
        out = out.reshape(*out.shape[:-1], n // (2 * h), 2, h)
        a = out[..., 0, :]
        b = out[..., 1, :]
        out = torch.stack((a + b, a - b), dim=-2).reshape(*x.shape[:-1], n)
        h *= 2
    return out * (n**-0.5)


def _fwht_last(x: torch.Tensor) -> torch.Tensor:
    k = x.shape[-1]
    if _is_pow2(k):
        return _fwht_pow2(x)
    block = 1 << (k.bit_length() - 1)
    while block > 1 and k % block != 0:
        block >>= 1
    if block < 2 or k % block != 0:
        return x
    y = x.reshape(*x.shape[:-1], k // block, block)
    return _fwht_pow2(y).reshape(*x.shape)


def _signs(k: int, device, dtype) -> torch.Tensor:
    signs = _HADAMARD_SIGNS.get(k)
    if signs is None:
        g = torch.Generator(device="cpu")
        g.manual_seed(0x51A207 ^ (k * 0x9E3779B9))
        s = torch.randint(0, 2, (k,), generator=g, dtype=torch.int8)
        signs = (s * 2 - 1).to(torch.float32)
        _HADAMARD_SIGNS[k] = signs
    return signs.to(device=device, dtype=dtype)


def hadamard_act(x: torch.Tensor) -> torch.Tensor:
    y = _fwht_last(x)
    return y * _signs(y.shape[-1], y.device, y.dtype)


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
        f"quarot calib prompts={len(problems)} alpha={ALPHA} max_new={MAX_NEW}",
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
            x = hadamard_act(inp[0].detach())
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
        "mode": "prefill_union_decode_chat_quarot",
        "note": "xmax after Hadamard+signs; pair with QUAROT=1",
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
