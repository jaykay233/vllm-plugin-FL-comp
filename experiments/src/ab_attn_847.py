#!/usr/bin/env python3
"""A/B the blocking D2H in the MetaX flash-attn path (flash_attn.py:847).

The probe (probe_d2h_sites.py) showed exactly one blocking D2H site in the
plugin: ``[0] + attn_metadata.prefill_seq_lens.tolist()`` followed by a
``torch.tensor(..., device=...)`` built from that Python list.  It fires once
per layer (42x) on every step that contains a prefill, and costs ~471 us/call
-- 19.8 ms per prefill -- while moving 4 bytes.

That 471 us cannot tell us how much is *incremental*: a blocking copy also
waits for prior GPU work, which the step would have paid for anyway.  The only
way to separate the two is to remove the D2H and measure.

Measured here is a TTFT proxy: ``max_tokens=1`` so wall time is prefill plus a
single decode step.  Prompts are unique per iteration so prefix caching cannot
turn the "prefill" into a cache hit.

Run:
    MODE=compile N=20 /opt/conda/envs/mx/bin/python /root/src/ab_attn_847.py
    MODE=eager   N=20 /opt/conda/envs/mx/bin/python /root/src/ab_attn_847.py
"""

from __future__ import annotations

import os
import statistics
import time

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_FL_PREFER", "flagos")
os.environ.setdefault("USE_FLAGGEMS", "1")

MODE = os.environ.get("MODE", "compile")
N = int(os.environ.get("N", "20"))
VARIANT = os.environ.get("VARIANT", "unknown")
MODEL = "/root/models/MiniCPM5-2B"
PREFILL_TOKENS = int(os.environ.get("PREFILL_TOKENS", "500"))


def main() -> None:
    import json

    from vllm import LLM, SamplingParams

    kwargs = dict(
        model=MODEL,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=2048,
        gpu_memory_utilization=0.85,
        enable_prefix_caching=False,
    )
    if MODE == "eager":
        kwargs["enforce_eager"] = True

    llm = LLM(**kwargs)

    # +1 token: TTFT (prefill) plus a single decode step.
    sp = SamplingParams(max_tokens=1, temperature=0.0, ignore_eos=True)
    base = "Explain the theory of relativity in detail. " * (PREFILL_TOKENS // 8)

    n_prompt = len(llm.get_tokenizer().encode(base))
    print(f"prompt tokens ~= {n_prompt}", flush=True)

    # Warmup: keeps compile / cudagraph capture out of the measurement.
    for i in range(2):
        llm.generate([f"{base} warmup {i}"], SamplingParams(max_tokens=1, temperature=0.0))

    # ---- correctness reference: greedy, fixed prompt, exact token ids ----
    # The D2H only feeds a cumsum of prefill sequence lengths, so removing it
    # must not change a single generated token.  Any divergence fails the A/B.
    ref_sp = SamplingParams(max_tokens=32, temperature=0.0, ignore_eos=True)
    ref_out = llm.generate([base], ref_sp)
    ref_tokens = list(ref_out[0].outputs[0].token_ids)
    ref_text = ref_out[0].outputs[0].text
    print(f"ref tokens[:12] = {ref_tokens[:12]}", flush=True)

    samples = []
    for i in range(N):
        # Unique prompts so every iteration is a real prefill.
        prompt = f"{base} measurement request number {i}."
        t0 = time.perf_counter()
        llm.generate([prompt], sp)
        samples.append((time.perf_counter() - t0) * 1e3)

    samples.sort()
    med = statistics.median(samples)
    p10 = samples[max(int(len(samples) * 0.1) - 1, 0)]
    p90 = samples[min(int(len(samples) * 0.9), len(samples) - 1)]

    print("=" * 84)
    print(f"TTFT proxy (prefill + 1 decode)   MODE={MODE}  VARIANT={VARIANT}  N={N}")
    print(f"  prompt ~{n_prompt} tok")
    print(f"  min={samples[0]:8.2f} ms   p10={p10:8.2f}   median={med:8.2f}   "
          f"p90={p90:8.2f}   max={samples[-1]:8.2f}")
    print(f"  sigma={statistics.pstdev(samples):7.2f} ms   "
          f"mean={statistics.fmean(samples):8.2f} ms")
    print("=" * 84)

    outdir = os.environ.get("OUTDIR", "/root/bench_results/ab_847")
    os.makedirs(outdir, exist_ok=True)
    rec = {
        "variant": VARIANT,
        "mode": MODE,
        "n": N,
        "prompt_tokens": n_prompt,
        "median_ms": med,
        "mean_ms": statistics.fmean(samples),
        "min_ms": samples[0],
        "p10_ms": p10,
        "p90_ms": p90,
        "max_ms": samples[-1],
        "ref_tokens": ref_tokens,
        "ref_text": ref_text,
    }
    path = os.path.join(outdir, f"{VARIANT}.json")
    with open(path, "w") as fh:
        json.dump(rec, fh, ensure_ascii=False, indent=2)
    print(f"wrote {path}")
    print("AB_847_DONE", flush=True)


if __name__ == "__main__":
    main()
