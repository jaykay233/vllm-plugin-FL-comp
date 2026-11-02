#!/usr/bin/env python3
"""Measure the served decode step time directly, via engine counters.

The torch-profiler HTTP endpoints are not exposed by this vLLM build (404), so
measure from the engine's own counters instead. Two facts make this exact:

  * every request in the workload has the same prompt length, so after the
    first prefill the prompt token counter never moves again;
  * generation_tokens_total increments by one per running request per step.

So sampling generation_tokens_total and num_requests_running at high frequency
over a pure-decode window gives (generation delta)/(running*steps). And because
num_requests_running pins at the concurrency for the whole window, the step
count is generation_delta / running, letting us convert wall time to ms/step
without assuming anything.

Run: python experiments/src/measure_served_decode_step.py
"""
from __future__ import annotations

import os
import re
import statistics as st
import subprocess
import time
import urllib.request

PORT = 9039
MODEL = "/workspace/MiniCPM5-2B"
OUT = "/root/bench_results/trace/served_decode_step"
POLL = 0.05


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True)


def metrics():
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/metrics", timeout=3) as r:
            txt = r.read().decode()
    except Exception:
        return None

    def val(name):
        m = re.search(rf"^{re.escape(name)}\{{[^}}]*\}}\s+([0-9.eE+]+)$", txt, re.M)
        if not m:
            m = re.search(rf"^{re.escape(name)}\s+([0-9.eE+]+)$", txt, re.M)
        return float(m.group(1)) if m else None

    return {
        "gen": val("vllm:generation_tokens_total"),
        "prompt": val("vllm:prompt_tokens_total"),
        "running": val("vllm:num_requests_running"),
    }


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    sh("pkill -9 -f 'vllm serve /workspace/MiniCPM5-2B'; pkill -9 -f 'VLLM::EngineCore'")
    for _ in range(60):
        used = sh("mx-smi 2>/dev/null | awk '/MiB *\\| Available/{split($0,a,\"/\");"
                  " gsub(/[^0-9]/,\"\",a[1]); print a[1]; exit}'").stdout.strip()
        if used.isdigit() and int(used) < 2000:
            break
        time.sleep(1)

    env = os.environ.copy()
    env.update({"VLLM_PLUGINS": "fl"})
    logf = open(os.path.join(OUT, "server.log"), "w")
    subprocess.Popen(
        f"/opt/conda/bin/vllm serve {MODEL} --port {PORT} --served-model-name minicpm "
        "--gpu-memory-utilization 0.85 --max-model-len 131072",
        shell=True, env=env, stdout=logf, stderr=subprocess.STDOUT,
        start_new_session=True)
    t0 = time.time()
    while time.time() - t0 < 900:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=3) as r:
                if r.status == 200:
                    break
        except Exception:
            pass
        time.sleep(5)
    else:
        print("NOT HEALTHY", flush=True)
        return 1
    print(f"HEALTHY after {time.time() - t0:.0f}s", flush=True)

    # Long outputs so the sampled window is almost entirely decode.
    bench = (f"env CUDA_VISIBLE_DEVICES= VLLM_PLUGINS= /opt/conda/bin/vllm bench serve "
             f"--backend vllm --model minicpm --tokenizer {MODEL} "
             f"--endpoint /v1/completions --host localhost --port {PORT} "
             "--dataset-name random --ignore-eos "
             "--random-input-len 4096 --random-output-len 512 --max-concurrency 64 "
             "--num-prompts 64 --disable-tqdm")
    bp = subprocess.Popen(bench, shell=True, stdout=open(os.path.join(OUT, "bench.log"), "w"),
                          stderr=subprocess.STDOUT)

    samples = []
    while bp.poll() is None:
        m = metrics()
        if m and m["gen"] is not None:
            samples.append({"t": time.time(), **m})
        time.sleep(POLL)
    bp.wait()

    with open(os.path.join(OUT, "metrics.json"), "w") as f:
        import json
        json.dump(samples, f)

    # Keep only intervals where the batch is full and no prefill happened.
    full = [s["running"] for s in samples if s["running"]]
    conc = int(round(max(full))) if full else 0
    steady = []
    for a, b in zip(samples, samples[1:]):
        if a["running"] is None or b["running"] is None:
            continue
        if abs(a["running"] - conc) > 0.01 or abs(b["running"] - conc) > 0.01:
            continue
        if (b["prompt"] or 0) != (a["prompt"] or 0):
            continue
        dg = (b["gen"] or 0) - (a["gen"] or 0)
        dt = b["t"] - a["t"]
        if dg > 0 and dt > 0:
            steady.append((dt, dg))

    if not steady:
        print("no steady decode intervals found", flush=True)
        return 1

    dt = sum(x[0] for x in steady)
    dg = sum(x[1] for x in steady)
    steps = dg / conc
    ms_per_step = dt / steps * 1e3
    tok_per_s = dg / dt
    step_times = [(d / (g / conc)) * 1e3 for d, g in steady if g > 0]

    print(f"\nsteady decode window: {dt:.1f} s, {len(steady)} samples, "
          f"concurrency {conc}", flush=True)
    print(f"  generation tokens   {dg:.0f}", flush=True)
    print(f"  decode steps        {steps:.0f}", flush=True)
    print(f"  ms/step             {ms_per_step:.2f}", flush=True)
    print(f"  median ms/step      {st.median(step_times):.2f}", flush=True)
    print(f"  output tok/s        {tok_per_s:.0f}", flush=True)
    print(f"\n  in-process harness (MBT 8192):  16.11 ms/step, 3972 tok/s", flush=True)
    print(f"  ratio                         {ms_per_step / 16.11:.2f}x", flush=True)

    sh("pkill -9 -f 'vllm serve /workspace/MiniCPM5-2B'; pkill -9 -f 'VLLM::EngineCore'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
