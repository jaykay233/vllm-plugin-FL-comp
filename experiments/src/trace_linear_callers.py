#!/usr/bin/env python3
"""Which shapes actually reach FlagGems' `linear` during real inference?

The end-to-end A/B showed the GEMV fast path was reached only once, by the
lm_head. This run records every distinct (M, N, K) that enters
``flag_gems.linear`` -- including the generic fallback branch -- so it is
unambiguous which callers route through FlagGems and which bypass it.

Expected shapes if vLLM's per-layer projections went through FlagGems:
    N=2560 (qkv_proj), N=2048 (o_proj / down_proj), N=12288 (gate_up_proj),
    N=130560 (lm_head)

Run:  conda activate mx && python /root/src/trace_linear_callers.py
"""

from __future__ import annotations

import os
from pathlib import Path

TRACE = Path("/root/bench_results/linear_callers.txt")

os.environ["FLAG_GEMS_METAX_GEMV"] = "1"
os.environ["FLAG_GEMS_METAX_GEMV_TRACE"] = str(TRACE)

if TRACE.exists():
    TRACE.unlink()

from vllm import LLM, SamplingParams  # noqa: E402

EXPECTED = {
    2560: "qkv_proj (per layer)",
    2048: "o_proj / down_proj (per layer)",
    12288: "gate_up_proj (per layer)",
    130560: "lm_head",
}


def main() -> None:
    model = "/root/models/MiniCPM5-2B"
    llm = LLM(
        model=model,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=2048,
        gpu_memory_utilization=0.85,
        enforce_eager=False,
    )
    sp = SamplingParams(max_tokens=16, temperature=0.0)
    out = llm.generate(["The capital of France is"], sp)
    print("generated:", out[0].outputs[0].text[:60])
    print()

    lines = TRACE.read_text().splitlines() if TRACE.exists() else []
    print("=" * 84)
    print(f"shapes that reached flag_gems.linear  ({len(lines)} distinct)")
    print("=" * 84)
    seen_ns = set()
    for ln in lines:
        print(f"  {ln}")
        for part in ln.split():
            if part.startswith("N="):
                seen_ns.add(int(part[2:]))
    print()
    print("  interpretation:")
    for n in sorted(seen_ns):
        print(f"    N={n:<7d} -> {EXPECTED.get(n, 'unexpected')}")
    print()
    missing = [n for n in EXPECTED if n not in seen_ns]
    print(f"  expected shapes NOT seen : "
          f"{[(n, EXPECTED[n]) for n in missing]}")
    print()
    if 2560 not in seen_ns:
        print("  VERDICT: per-layer projections BYPASS flag_gems.linear.")
        print("           Only the lm_head (N=130560) is routed through it.")
    print("TRACE_CALLERS_DONE", flush=True)


if __name__ == "__main__":
    main()
