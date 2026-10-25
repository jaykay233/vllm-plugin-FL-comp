#!/usr/bin/env python3
"""Where does the 4k official bench's wall time actually go: prefill or decode?

The pure-decode trace says a decode step at ctx=4096 costs 17.84 ms. The bench
reports TPOT 23.63 ms, and 4096 decode steps x 17.84 ms = 73.1 s against a
104.5 s total. So ~31 s is unaccounted for. Either prefill is not hidden behind
decode (my earlier assumption), or in-flight decode steps are slowed by prefill
sharing the GPU.

The engine's own counters settle it. Poll /metrics once a second while the
official 4k bench runs and classify each interval by whether prompt tokens moved:

  prompt tokens increased  -> prefill work happened in that second
  only generation moved    -> pure decode

Then report seconds and token rates per class. This is the cheapest possible way
to decide whether remaining effort belongs on prefill or decode.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
import urllib.request

OUT = "/workspace/vllm-plugin-FL-comp/experiments/bench_results/prefill_decode_split_20261025"
MODEL = "/workspace/MiniCPM5-2B"
PORT = 9037
POLL = 1.0


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True)


def metrics():
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/metrics", timeout=3) as r:
            txt = r.read().decode()
    except Exception:
        return None

    def val(name):
        m = re.search(rf"^{name}\{{[^}}]*\}}\s+([0-9.eE+]+)$", txt, re.M)
        if not m:
            m = re.search(rf"^{name}\s+([0-9.eE+]+)$", txt, re.M)
        return float(m.group(1)) if m else None

    return {
        "running": val("vllm:num_requests_running"),
        "waiting": val("vllm:num_requests_waiting"),
        "prompt": val("vllm:prompt_tokens_total"),
        "gen": val("vllm:generation_tokens_total"),
    }


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    sh("pkill -9 -f 'vllm serve' ; pkill -9 -f 'VLLM::EngineCore' ; pkill -9 -f 'vllm bench serve'")
    time.sleep(4)
    env = os.environ.copy()
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env.update({"VLLM_PLUGINS": "fl"})
    cmd = (
        f"/opt/conda/bin/vllm serve {MODEL} --port {PORT} --served-model-name minicpm "
        "--gpu-memory-utilization 0.85 --max-model-len 131072 "
        "--no-enable-prefix-caching"
    )
    logf = open(os.path.join(OUT, "server.log"), "w")
    subprocess.Popen(cmd, shell=True, env=env, stdout=logf, stderr=subprocess.STDOUT,
                     start_new_session=True)
    t0 = time.time()
    while time.time() - t0 < 600:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2) as r:
                if r.status == 200:
                    break
        except Exception:
            pass
        time.sleep(5)
    else:
        print("NOT HEALTHY", flush=True)
        return 1
    print(f"HEALTHY after {time.time() - t0:.0f}s", flush=True)

    bench = (
        f"env -u CUDA_VISIBLE_DEVICES VLLM_PLUGINS= /opt/conda/bin/vllm bench serve "
        f"--backend vllm --model minicpm --tokenizer {MODEL} "
        f"--endpoint /v1/completions --host localhost --port {PORT} "
        "--dataset-name random --ignore-eos --random-input-len 4096 "
        "--random-output-len 1024 --max-concurrency 64 --num-prompts 256"
    )
    bf = open(os.path.join(OUT, "bench.log"), "w")
    bp = subprocess.Popen(bench, shell=True, stdout=bf, stderr=subprocess.STDOUT)

    samples = []
    tb = time.time()
    while bp.poll() is None and time.time() - tb < 600:
        m = metrics()
        if m and m["prompt"] is not None:
            samples.append({"t": time.time() - tb, **m})
            if len(samples) % 15 == 0:
                print(f"  t={samples[-1]['t']:5.0f}s running={m['running']:.0f} "
                      f"prompt={m['prompt']:.0f} gen={m['gen']:.0f}", flush=True)
        time.sleep(POLL)
    wall = time.time() - tb

    with open(os.path.join(OUT, "metrics.json"), "w") as f:
        json.dump({"wall_s": wall, "samples": samples}, f, indent=1)
    print(f"\nbench finished, wall {wall:.1f}s, {len(samples)} samples", flush=True)

    # classify each inter-sample interval
    prefill_s = decode_s = 0.0
    prefill_tok = gen_during_prefill = gen_during_decode = 0.0
    for a, b in zip(samples, samples[1:]):
        dt = b["t"] - a["t"]
        if dt <= 0:
            continue
        dp = (b["prompt"] or 0) - (a["prompt"] or 0)
        dg = (b["gen"] or 0) - (a["gen"] or 0)
        if dp > 0:
            prefill_s += dt
            prefill_tok += dp
            gen_during_prefill += dg
        else:
            decode_s += dt
            gen_during_decode += dg

    print(f"\n  intervals with prefill   {prefill_s:6.1f} s  ({prefill_s / wall * 100:4.1f}% of bench)")
    print(f"    prompt tokens there    {prefill_tok:10.0f}  -> {prefill_tok / max(prefill_s, 1e-9):8.0f} tok/s")
    print(f"    gen tokens there       {gen_during_prefill:10.0f}  -> {gen_during_prefill / max(prefill_s, 1e-9):8.1f} tok/s")
    print(f"  pure-decode intervals    {decode_s:6.1f} s  ({decode_s / wall * 100:4.1f}%)")
    print(f"    gen tokens there       {gen_during_decode:10.0f}  -> {gen_during_decode / max(decode_s, 1e-9):8.1f} tok/s")
    tot = prefill_tok + gen_during_prefill + gen_during_decode
    print(f"  total tokens counted     {tot:10.0f}  (expect {256 * 4096 + 256 * 1024:.0f})")
    print(f"\n  decode token rate during prefill intervals "
          f"{gen_during_prefill / max(prefill_s, 1e-9) / max(gen_during_decode / max(decode_s, 1e-9), 1e-9) * 100:.0f}%"
          f" of the pure-decode rate")
    sh("pkill -9 -f 'vllm serve' ; pkill -9 -f 'VLLM::EngineCore'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
