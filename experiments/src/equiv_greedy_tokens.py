#!/usr/bin/env python3
"""Greedy-output equivalence check for the dense-KV prefill path.

Runs the same prompts twice in separate processes (caller toggles
VLLM_FL_METAX_PREFILL_DENSE_KV) and writes the generated token ids, so the two
arms can be diffed exactly. Prompts are longer than one chunk so the run also
exercises the mixed paged+dense split.

Run: OUT=/root/bench_results/equiv/on  python .../equiv_greedy_tokens.py
     OUT=/root/bench_results/equiv/off VLLM_FL_METAX_PREFILL_DENSE_KV=0 python ...
     diff -> must be empty
"""
from __future__ import annotations

import json
import os
from pathlib import Path

OUT = Path(os.environ.get("OUT", "/root/bench_results/equiv/on"))
OUT.mkdir(parents=True, exist_ok=True)
IN_LEN = int(os.environ.get("IN_LEN", "12288"))
CONC = int(os.environ.get("CONC", "4"))
NEW = int(os.environ.get("NEW", "16"))


def main() -> None:
    from vllm import LLM, SamplingParams

    llm = LLM(
        model="/workspace/MiniCPM5-2B",
        dtype="bfloat16",
        max_model_len=IN_LEN + 2048,
        gpu_memory_utilization=0.85,
        max_num_batched_tokens=8192,
        enable_prefix_caching=False,
    )
    tok = llm.get_tokenizer()
    unit = tok("Explain the theory of relativity in detail. ")["input_ids"]
    prompts = []
    for i in range(CONC):
        ids = (unit * (IN_LEN // len(unit) + 1))[:IN_LEN]
        ids[0] = 100 + i
        prompts.append({"prompt_token_ids": ids})

    sp = SamplingParams(max_tokens=NEW, temperature=0.0, ignore_eos=True,
                        logprobs=1)
    outs = llm.generate(prompts, sp)
    res = [{"i": i, "tokens": list(o.outputs[0].token_ids)} for i, o in enumerate(outs)]
    (OUT / "tokens.json").write_text(json.dumps(res))
    print("dense_kv =", os.environ.get("VLLM_FL_METAX_PREFILL_DENSE_KV", "unset"))
    for r in res:
        print(r["i"], r["tokens"][:8], "...")
    print("wrote", OUT / "tokens.json")


if __name__ == "__main__":
    main()
