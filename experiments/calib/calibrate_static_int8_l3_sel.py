#!/usr/bin/env python3
"""Progressive SmoothQuant calib for selective W8A8 (gate_up + down only).

Pass 1: bf16 forward → xmax for gate/up (and reference down).
Pass 2: run with gate/up under simulated W8A8 → xmax for down only.

Writes a calib payload consumed when STATIC_INT8_ROLES selects MLP GEMMs.
Does not change vLLM serve args.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "/workspace/MiniCPM5-2B"
DATA = "/workspace/evalscope-datasets/math_500/test.jsonl"
OUT = Path(
    "/workspace/vllm-plugin-FL-comp/experiments/calib/"
    "minicpm5_2b_l3_static_int8_sel.pt"
)
# Also install next to the runtime module for the serve path.
OUT_RUNTIME = Path(
    "/workspace/vllm-plugin-FL-comp/vllm_fl/dispatch/backends/vendor/"
    "iluvatar/minicpm5_2b_l3_static_int8_sel.pt"
)
MAX_LEN = 1024
ALPHA = 0.5


def level3_problems() -> list[str]:
    rows = []
    with open(DATA) as f:
        for line in f:
            rec = json.loads(line)
            if int(rec.get("level", -1)) == 3:
                rows.append(rec["problem"])
    return rows


def _quant_weight(weight: torch.Tensor, xmax: torch.Tensor, alpha: float) -> dict:
    n, k = weight.shape
    xm = xmax.to(device=weight.device, dtype=torch.float32).reshape(-1)
    if xm.numel() != k:
        raise RuntimeError(f"xmax K mismatch: {xm.numel()} vs {k}")
    w = weight.detach().float()
    wmax = w.abs().amax(dim=0).clamp_min(1e-6)
    s = (xm.clamp_min(1e-6).pow(alpha)) / wmax.pow(1.0 - alpha)
    s = s.clamp(1e-3, 1e3)
    w_s = w * s
    w_scale = w_s.abs().amax(dim=1).clamp_min(1e-6) / 127.0
    w8 = (w_s / w_scale.unsqueeze(1)).round().clamp(-128, 127).to(torch.int8)
    a_scale = ((xm / s).amax() / 127.0).clamp_min(1e-8)
    q_scale = (s.reciprocal() / a_scale).contiguous()
    out_scale = (a_scale * w_scale).contiguous()
    return {
        "w8": w8.contiguous(),
        "q_scale": q_scale,
        "out_scale": out_scale,
    }


def _simulate_w8a8(x: torch.Tensor, meta: dict, bias: torch.Tensor | None) -> torch.Tensor:
    """Float simulation of static W8A8 (same scales as runtime)."""
    orig = x.shape
    k = orig[-1]
    x2 = x.reshape(-1, k).float()
    q = meta["q_scale"].to(device=x2.device, dtype=torch.float32)
    x8 = (x2 * q).round().clamp(-128, 127)
    w8 = meta["w8"].to(device=x2.device, dtype=torch.float32)
    acc = x8 @ w8.t()
    out = acc * meta["out_scale"].to(device=x2.device, dtype=torch.float32)
    if bias is not None:
        out = out + bias.float()
    return out.to(dtype=x.dtype).reshape(*orig[:-1], out.shape[-1])


class _W8A8Linear(nn.Module):
    def __init__(self, base: nn.Linear, meta: dict):
        super().__init__()
        self.meta = {k: v.cpu() for k, v in meta.items()}
        self.bias = base.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        meta = {k: v.to(device=x.device) for k, v in self.meta.items()}
        return _simulate_w8a8(x, meta, self.bias)


def _is_target(name: str) -> bool:
    return any(
        s in name
        for s in (
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
            "mlp.gate_up_proj",
        )
    )


def _collect_xmax(model: nn.Module, tok, problems: list[str], name_pred) -> dict[str, torch.Tensor]:
    xmax: dict[str, torch.Tensor] = {}
    handles = []

    def make_hook(name: str):
        def hook(mod, inp, _out):
            x = inp[0].detach()
            m = x.abs().amax(dim=tuple(range(x.ndim - 1))).float().cpu()
            if name not in xmax:
                xmax[name] = m
            else:
                xmax[name] = torch.maximum(xmax[name], m)

        return hook

    for name, mod in model.named_modules():
        if not name_pred(name):
            continue
        if not isinstance(getattr(mod, "weight", None), torch.Tensor):
            continue
        handles.append(mod.register_forward_hook(make_hook(name)))

    with torch.no_grad():
        for i, text in enumerate(problems):
            ids = tok(text, return_tensors="pt", truncation=True, max_length=MAX_LEN)
            ids = {k: v.cuda() for k, v in ids.items()}
            model(**ids)
            if (i + 1) % 15 == 0 or i == 0:
                print(f"  calib {i+1}/{len(problems)} keys={len(xmax)}")

    for h in handles:
        h.remove()
    return xmax


def main() -> None:
    problems = level3_problems()
    print(f"level3 prompts: {len(problems)} alpha={ALPHA}")
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

    print("pass1: bf16 xmax for gate/up/down")
    xmax_p1 = _collect_xmax(model, tok, problems, _is_target)

    # Replace gate/up with simulated W8A8 using pass1 scales.
    replaced = []
    for name, mod in list(model.named_modules()):
        if not isinstance(mod, nn.Linear):
            continue
        if not any(s in name for s in ("mlp.gate_proj", "mlp.up_proj", "mlp.gate_up_proj")):
            continue
        xm = xmax_p1.get(name)
        if xm is None:
            print(f"  skip replace {name}: no xmax")
            continue
        meta = _quant_weight(mod.weight, xm, ALPHA)
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child, _W8A8Linear(mod, meta))
        replaced.append(name)
    print(f"pass1 replaced {len(replaced)} gate/up modules with sim W8A8")

    print("pass2: xmax for down under quantized gate/up")
    xmax_down = _collect_xmax(
        model,
        tok,
        problems,
        lambda n: "mlp.down_proj" in n,
    )

    xmax: dict[str, torch.Tensor] = {}
    for k, v in xmax_p1.items():
        if "down_proj" in k:
            continue
        xmax[k] = v
    for k, v in xmax_down.items():
        xmax[k] = v

    payload = {
        "alpha": ALPHA,
        "xmax": {k: v.half() for k, v in xmax.items()},
        "n_prompts": len(problems),
        "max_len": MAX_LEN,
        "roles": ["gate_up", "down"],
        "progressive": True,
        "note": "gate/up xmax from bf16; down xmax with gate/up simulated W8A8",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, OUT)
    torch.save(payload, OUT_RUNTIME)
    print(f"wrote {OUT} and {OUT_RUNTIME} keys={len(xmax)}")
    for k in sorted(xmax)[:6]:
        print(f"  {k} shape={tuple(xmax[k].shape)}")


if __name__ == "__main__":
    main()
