#!/usr/bin/env python3
"""Profile the *real* served decode loop and measure its step time.

§33 concluded that a service decode step costs ~22.6 ms against the in-process
harness's 16.1 ms, but both of those numbers are indirect: the service figure
comes from client-side TPOT and from /metrics token rates that assume a full
batch of 64. This uses vLLM's own torch-profiler endpoint so the number is
measured inside the engine, on the same code path the benchmark exercises.

Workload: 64 concurrent streams with a 4096-token prompt and a long output, so
almost all of the profiled window is pure decode after the first prefill.

Run: python experiments/src/profile_served_decode.py
"""
from __future__ import annotations

import json
import os
import statistics as st
import subprocess
import time
import urllib.request

PORT = 9039
MODEL = "/workspace/MiniCPM5-2B"
PROF_DIR = "/root/bench_results/trace/served_decode"
OUT = "/root/bench_results/trace/served_decode"


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True)


def post(path, timeout=30):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode()[:200]
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)


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
    env.update({"VLLM_PLUGINS": "fl", "VLLM_TORCH_PROFILER_DIR": PROF_DIR,
                "VLLM_RPC_TIMEOUT": "600000"})
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

    # Warm up the exact shape first so graph capture is not in the window.
    bench = (f"env CUDA_VISIBLE_DEVICES= VLLM_PLUGINS= /opt/conda/bin/vllm bench serve "
             f"--backend vllm --model minicpm --tokenizer {MODEL} "
             f"--endpoint /v1/completions --host localhost --port {PORT} "
             "--dataset-name random --ignore-eos "
             "--random-input-len 4096 --random-output-len 64 --max-concurrency 64 "
             "--num-prompts 64 --disable-tqdm")
    print("warmup:", sh(bench).returncode, flush=True)

    print("start_profile:", post("/start_profile"), flush=True)
    r = sh(bench.replace("--random-output-len 64", "--random-output-len 256")
             .replace("--num-prompts 64", "--num-prompts 64"))
    print("bench rc:", r.returncode, flush=True)
    time.sleep(3)
    print("stop_profile:", post("/stop_profile", timeout=120), flush=True)
    time.sleep(10)

    files = [os.path.join(dp, f) for dp, _, fs in os.walk(PROF_DIR) for f in fs
             if f.endswith(".json")]
    print("traces:", files, flush=True)
    if not files:
        return 1
    path = max(files, key=os.path.getsize)
    ev = json.load(open(path))["traceEvents"]
    k = [e for e in ev if e.get("ph") == "X"
         and e.get("cat") in ("kernel", "gpu_memcpy", "Kernel")]
    k.sort(key=lambda e: e["ts"])
    att = [e for e in k if "flash_fwd" in e["name"]]
    print(f"kernels={len(k)}  attention kernels={len(att)}", flush=True)
    # Decode steps = attention kernel groups; use the biggest-gap split on
    # attention kernels only, which fire once per layer per step.
    gaps = sorted(((b["ts"] - a["ts"]) for a, b in zip(att, att[1:])), reverse=True)
    print("largest attention-to-attention gaps (ms):",
          [round(g / 1e3, 2) for g in gaps[:10]], flush=True)
    tot = sum(e.get("dur", 0) for e in k)
    span = (k[-1]["ts"] + k[-1].get("dur", 0) - k[0]["ts"])
    print(f"total kernel {tot / 1e3:.1f} ms over span {span / 1e3:.1f} ms "
          f"({100 * tot / span:.1f}% busy)", flush=True)
    sh("pkill -9 -f 'vllm serve /workspace/MiniCPM5-2B'; pkill -9 -f 'VLLM::EngineCore'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
