#!/usr/bin/env python3
"""mcTracer workload with begin/end marker kernels.

The Perfetto trace spans the whole process (model load + warmup + generation),
so the region of interest has to be identifiable. Two uniquely named triton
kernels bracket the measured generation so the analyzer can window on them:

    _mcprof_begin_marker   -> marks the first measured forward
    _mcprof_end_marker     -> marks the end of the measured generation

MODE=eager   -> enforce_eager=True  (plugin -> FlagGems path active)
MODE=compile -> vLLM compile + cudagraph (custom_ops=['none'], FlagGems bypassed)

Run:
  MODE=eager python /root/src/mctracer_marked.py
  mcTracer --odname <name> python /root/src/mctracer_marked.py
"""

from __future__ import annotations

import os
import time

MODE = os.environ.get("MODE", "eager")
MODEL = "/root/models/MiniCPM5-2B"
STEPS = int(os.environ.get("STEPS", "16"))

os.environ.setdefault("VLLM_FL_PREFER", "flagos")
os.environ.setdefault("USE_FLAGGEMS", "1")
# Run the engine in-process so the marker kernels and the model kernels land in
# the same trace file and share one clock base.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import triton  # noqa: E402
import triton.language as tl  # noqa: E402


@triton.jit
def _mcprof_begin_marker(ptr):
    tl.store(ptr, 1.0)


@triton.jit
def _mcprof_end_marker(ptr):
    tl.store(ptr, 2.0)


def main() -> None:
    import torch
    from vllm import LLM, SamplingParams

    kwargs = dict(
        model=MODEL,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=2048,
        gpu_memory_utilization=0.85,
        # Prefix caching would make the measured run reuse the warmup's prefix
        # and skip the real prefill -- so the trace would contain no prefill at
        # all. Disable it so the first forward is a genuine ~500-token prefill.
        enable_prefix_caching=False,
    )
    if MODE == "eager":
        kwargs["enforce_eager"] = True

    print(f"########## marked workload MODE={MODE} STEPS={STEPS} ##########", flush=True)
    print(f"device: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"whitelist={os.environ.get('VLLM_FL_FLAGOS_WHITELIST', '<none: all GEMS>')}",
          flush=True)

    t0 = time.perf_counter()
    llm = LLM(**kwargs)
    print(f"init={time.perf_counter() - t0:.1f}s", flush=True)

    sp = SamplingParams(max_tokens=STEPS, temperature=0.0, ignore_eos=True)
    # Unique prompt per phase so no cache can match it even if caching is on.
    prompt = ("Explain the theory of relativity in detail. " * 40)  # ~500 tokens
    prefill_probe = "Summarize the history of computing in one paragraph. " * 40

    # warmup: keeps compile/cudagraph capture out of the measured region
    for _ in range(3):
        llm.generate([prefill_probe], SamplingParams(max_tokens=4, temperature=0.0, ignore_eos=True))
    torch.cuda.synchronize()

    scratch = torch.zeros(1, device="cuda", dtype=torch.float32)
    torch.cuda.synchronize()

    # ---- measured region ----
    _mcprof_begin_marker[(1,)](scratch)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    out = llm.generate([prompt], sp)
    torch.cuda.synchronize()
    gen_s = time.perf_counter() - t0

    _mcprof_end_marker[(1,)](scratch)
    torch.cuda.synchronize()
    # ---- end measured region ----

    n_out = len(out[0].outputs[0].token_ids)
    print(f"gen={gen_s:.3f}s tokens={n_out} => {n_out / gen_s:.1f} tok/s", flush=True)
    print("MARKED_WORKLOAD_DONE", flush=True)


if __name__ == "__main__":
    main()
