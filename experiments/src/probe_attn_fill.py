#!/usr/bin/env python3
"""Ground truth: how often does the MetaX attention forward hit `output.fill_(0)`?

The Python stack probe caught `fill_` on the attention output exactly 84 times for
an entire run (2 per layer), yet the GPU executes `zero_persistent_kernel` 84.67
times per decode step (~223 us/step).  The two can only be reconciled if the fill
is recorded into the captured CUDA graph and replayed every step, or if the fill
runs on a path that Python-level patching cannot see.

This counts, in one process:

  * FlashAttentionImpl.forward calls, split by whether attn_metadata is None
  * FlagGems `_launch_zero` calls (the Python entry that launches the kernel)
  * the fill branch's share of them

Run:  python /root/src/probe_attn_fill.py
"""

from __future__ import annotations

import importlib
import os
import statistics
import time

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

from collections import Counter  # noqa: E402

COUNTS: Counter = Counter()


def patch() -> None:
    import vllm  # noqa: F401  triggers platform plugin + vllm_fl import

    mod = importlib.import_module(
        "vllm_fl.dispatch.backends.vendor.metax.impl.attention.flash_attn"
    )
    cls = mod.FlashAttentionImpl
    orig = cls.forward

    def forward(self, layer, query, key, value, kv_cache, attn_metadata, *a, **kw):
        COUNTS["attn_forward"] += 1
        if attn_metadata is None:
            COUNTS["attn_forward_meta_none"] += 1
        return orig(self, layer, query, key, value, kv_cache, attn_metadata, *a, **kw)

    cls.forward = forward
    print("[patch] FlashAttentionImpl.forward wrapped", flush=True)

    # FlagGems zero entry (Python-level; the compiled path bypasses it)
    try:
        zmod = importlib.import_module(
            "flag_gems.runtime.backend._metax.ops.zero"
        )
        z_orig = zmod._launch_zero

        def _launch_zero(t):
            COUNTS["gems_launch_zero"] += 1
            COUNTS[f"gems_zero_numel_{t.numel()}"] += 1
            return z_orig(t)

        zmod._launch_zero = _launch_zero
        print("[patch] flag_gems _launch_zero wrapped", flush=True)
    except Exception as e:  # pragma: no cover
        print(f"[patch] zero wrap failed: {e}", flush=True)

    # also the tensor-level entry used by the global dispatch table
    try:
        import flag_gems

        z2 = flag_gems.zero_

        def zero_wrap(*a, **kw):
            COUNTS["gems_zero_fn"] += 1
            return z2(*a, **kw)

        flag_gems.zero_ = zero_wrap
    except Exception as e:  # pragma: no cover
        print(f"[patch] flag_gems.zero_ wrap failed: {e}", flush=True)


def main() -> None:
    patch()

    import torch
    from torch.profiler import ProfilerActivity, profile
    from vllm import LLM, SamplingParams

    llm = LLM(
        model="/root/models/MiniCPM5-2B",
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=2048,
        gpu_memory_utilization=0.30,
        compilation_config={"mode": "VLLM_COMPILE"},
    )
    sp = SamplingParams(max_tokens=64, temperature=0.0, ignore_eos=True)
    prompt = "The theory of general relativity describes gravity as"

    llm.generate([prompt], sp, use_tqdm=False)
    print(f"[phase] after warmup: {dict(COUNTS)}", flush=True)
    base = dict(COUNTS)

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        out = llm.generate([prompt], sp, use_tqdm=False)
        torch.cuda.synchronize()
    n = len(out[0].outputs[0].token_ids)

    delta = {k: v - base.get(k, 0) for k, v in COUNTS.items()}
    print(f"[phase] profiled generate ({n} steps): {delta}", flush=True)

    CM = torch.autograd.DeviceType.CUDA
    zk = Counter()
    zt = 0.0
    for ev in prof.events():
        if ev.device_type == CM and ev.name and "zero_persistent" in ev.name:
            zk[ev.name] += ev.count
            zt += ev.device_time
    print(f"[gpu] zero_persistent_kernel {sum(zk.values())} launches, "
          f"{zt / max(n - 1, 1):.1f} us/step", flush=True)

    print("\n--- per-token-normalised counters (steps = "
          f"{n - 1}) ---", flush=True)
    for k, v in sorted(COUNTS.items()):
        print(f"  {k:34s} total={v:8d}  per_step={v / max(n - 1, 1):7.2f}",
              flush=True)
    print("ATTN_FILL_DONE", flush=True)


if __name__ == "__main__":
    main()
