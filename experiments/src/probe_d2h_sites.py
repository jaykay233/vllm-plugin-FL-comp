#!/usr/bin/env python3
"""Attribute the blocking D2H copies that mcTracer cannot see.

Background
----------
mcTracer only instruments MACA's *async* copy API.  In the marked traces every
host-side memcpy was ``mcMemcpyAsync`` and there was not a single synchronous
``mcMemcpy`` -- yet the prefill step should contain 42 ``.tolist() -> torch.tensor(device=)``
round trips in ``flash_attn.py:847``.  The conclusion is that blocking D2H
(``Tensor.tolist`` / ``Tensor.cpu``) is a profiling blind spot, so the
"0.42% of gpu/step" transfer budget covers only the async path.

This script measures the blind spot two ways:

1. ``torch.profiler`` key averages for ``Memcpy DtoH`` / ``Memcpy HtoD``,
   split by pageable vs pinned.
2. A method-level probe: ``Tensor.tolist`` / ``Tensor.cpu`` are wrapped to
   count calls on CUDA tensors and bucket them by the calling frame, so each
   count is attributed to a source file:line.

Usage:
    /opt/conda/envs/mx/bin/python /root/src/probe_d2h_sites.py
"""

from __future__ import annotations

import os
import traceback
from collections import Counter, defaultdict

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_FL_PREFER", "flagos")
os.environ.setdefault("USE_FLAGGEMS", "1")

MODEL = "/root/models/MiniCPM5-2B"
STEPS = int(os.environ.get("STEPS", "16"))
PREFILL_TOKENS = int(os.environ.get("PREFILL_TOKENS", "500"))

# Collect only frames that live in the plugin / vllm, so the bucket key points
# at real code rather than at torch internals.
_KEEP = ("vllm_fl", "vllm/", "flag_gems")


def _bucket() -> str:
    for frame in reversed(traceback.extract_stack()[:-2]):
        if any(k in frame.filename for k in _KEEP):
            return f"{frame.filename.split('/')[-1]}:{frame.lineno} {frame.name}"
    return "<non-plugin>"


def main() -> None:
    import time

    import torch
    from vllm import LLM, SamplingParams

    counts: Counter[str] = Counter()
    bytes_by_site: defaultdict[str, int] = defaultdict(int)
    secs_by_site: defaultdict[str, float] = defaultdict(float)
    durs_by_site: defaultdict[str, list[float]] = defaultdict(list)

    orig_tolist = torch.Tensor.tolist
    orig_cpu = torch.Tensor.cpu

    def traced_tolist(self, *a, **k):
        if self.is_cuda:
            site = _bucket()
            t0 = time.perf_counter()
            out = orig_tolist(self, *a, **k)
            # time.perf_counter() only returns after the blocking D2H has
            # landed, so this delta *is* the sync cost we are chasing.  It
            # includes waiting for prior GPU work on the stream, which is why
            # the per-call distribution matters more than the mean.
            dt = time.perf_counter() - t0
            secs_by_site[site] += dt
            durs_by_site[site].append(dt * 1e6)
            counts[f"tolist {site}"] += 1
            bytes_by_site[site] += self.numel() * self.element_size()
            return out
        return orig_tolist(self, *a, **k)

    def traced_cpu(self, *a, **k):
        if self.is_cuda and not a and not k:
            site = _bucket()
            t0 = time.perf_counter()
            out = orig_cpu(self, *a, **k)
            dt = time.perf_counter() - t0
            secs_by_site[site] += dt
            durs_by_site[site].append(dt * 1e6)
            counts[f"cpu {site}"] += 1
            bytes_by_site[site] += self.numel() * self.element_size()
            return out
        return orig_cpu(self, *a, **k)

    torch.Tensor.tolist = traced_tolist
    torch.Tensor.cpu = traced_cpu

    llm = LLM(
        model=MODEL,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=2048,
        gpu_memory_utilization=0.85,
        enable_prefix_caching=False,
    )
    sp = SamplingParams(max_tokens=STEPS, temperature=0.0, ignore_eos=True)
    prompt = "Explain the theory of relativity in detail. " * (PREFILL_TOKENS // 8)

    # Warmup so compile / cudagraph capture is not inside the profile.
    for _ in range(2):
        llm.generate([prompt + " warmup"], SamplingParams(max_tokens=4, temperature=0.0))
    torch.cuda.synchronize()
    counts.clear()
    bytes_by_site.clear()
    secs_by_site.clear()
    durs_by_site.clear()

    from torch.profiler import ProfilerActivity, profile

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
        with_stack=False,
    ) as prof:
        t_gen = time.perf_counter()
        out = llm.generate([prompt], sp)
        gen_s = time.perf_counter() - t_gen

    torch.cuda.synchronize()
    torch.Tensor.tolist = orig_tolist
    torch.Tensor.cpu = orig_cpu

    n_out = len(out[0].outputs[0].token_ids)
    steps = 1 + (n_out - 1)  # 1 prefill + (n_out-1) decode steps

    print("=" * 92)
    print(f"generated {n_out} tokens in {gen_s:.3f} s  "
          f"=> 1 prefill + {n_out - 1} decode steps  ({n_out / gen_s:.1f} tok/s)")
    print("=" * 92)

    print("\n--- blocking D2H sites on CUDA tensors (method probe) ---")
    if not counts:
        print("  (none) -- no blocking .tolist()/.cpu() on a CUDA tensor")
    tot_calls = 0
    tot_secs = 0.0
    for site, cnt in counts.most_common(20):
        key = site.split(" ", 1)[1]
        tot_calls += cnt
        tot_secs += secs_by_site[key]
        ds = sorted(durs_by_site[key])
        print(f"  {cnt:6d} calls  {bytes_by_site[key] / 1e6:9.5f} MB  "
              f"{secs_by_site[key] * 1e3:8.3f} ms total  "
              f"min/p50/p90/max = {ds[0]:.1f}/{ds[len(ds) // 2]:.1f}/"
              f"{ds[min(int(len(ds) * 0.9), len(ds) - 1)]:.1f}/{ds[-1]:.1f} µs  {site}")
    print(f"  TOTAL: {tot_calls} calls, {tot_secs * 1e3:.3f} ms blocked in D2H "
          f"({tot_secs * 1e6 / max(tot_calls, 1):.2f} µs/call)")
    for site, ds in durs_by_site.items():
        print(f"\n  per-call µs ({site}), n={len(ds)}:")
        print("   ", " ".join(f"{d:.0f}" for d in ds))

    print("\n--- torch.profiler key averages (Memcpy only) ---")
    for row in prof.key_averages():
        if "Memcpy" not in row.key and "memcpy" not in row.key:
            continue
        dev = getattr(row, "device_time_total", None)
        if dev is None:
            dev = getattr(row, "cuda_time_total", 0.0)
        print(f"  {row.key:46s} calls={row.count:6d} "
              f"device_total={dev / 1e3:9.2f} ms "
              f"avg={dev / max(row.count, 1) / 1e3:7.3f} µs")

    print("\nPROBE_D2H_DONE", flush=True)


if __name__ == "__main__":
    main()
