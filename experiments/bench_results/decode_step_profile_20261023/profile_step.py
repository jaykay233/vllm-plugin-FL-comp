#!/usr/bin/env python3
"""Capture a torch trace of steady-state decode via vLLM's profiler endpoints.

Answers "decode step is 25.3 ms: GEMM 4.85 + attention ~9-10 + lm_head 0.48 =
~15 ms. Where is the other ~10 ms?"

Trace lands in VLLM_TORCH_PROFILER_DIR; analyse with analyze_step_trace.py.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request

OUT = "/workspace/vllm-plugin-FL-comp/experiments/bench_results/decode_step_profile_20261023"
MODEL = "/workspace/MiniCPM5-2B"
PORT = 9032


def sh(cmd, **kw):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, **kw)


def health(timeout=300):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        rc = sh("pgrep -f 'vllm serve /workspace/MiniCPM5-2B' >/dev/null")
        if rc.returncode != 0:
            print("server died", flush=True)
            return False
        time.sleep(5)
    return False


def post(path):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", method="POST", data=b"")
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status
    except Exception as e:  # noqa: BLE001
        return f"ERR {e}"


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    trace_dir = os.path.join(OUT, "trace")
    os.makedirs(trace_dir, exist_ok=True)
    for f in os.listdir(trace_dir):
        os.remove(os.path.join(trace_dir, f))

    sh("pkill -9 -f 'vllm serve' ; pkill -9 -f 'VLLM::EngineCore'")
    time.sleep(3)

    env = os.environ.copy()
    env.pop("CUDA_VISIBLE_DEVICES", None)
    for k in ("VLLM_METAX_LM_HEAD_GEMM", "VLLM_METAX_MLP_SILU_FUSION"):
        env.pop(k, None)
    env.update(
        {
            "VLLM_PLUGINS": "fl",
            "VLLM_TORCH_PROFILER_DIR": trace_dir,
            "VLLM_FL_METAX_MAX_NUM_BATCHED_TOKENS": "8192",
            "VLLM_FL_METAX_CUDAGRAPH_SPARE": "72",
            "FLAG_GEMS_METAX_ALIGN32_GEOMETRIC_CAP": "8192",
            "VLLM_FL_FLAGOS_WHITELIST": "silu_and_mul,rms_norm,rotary_embedding",
        }
    )
    cmd = (
        f"/opt/conda/bin/vllm serve {MODEL} --port {PORT} --served-model-name minicpm "
        "--gpu-memory-utilization 0.85 --max-model-len 131072 "
        "--max-num-batched-tokens 8192 --no-enable-prefix-caching "
        # /start_profile is only mounted when profiler_config.profiler is set,
        # and the torch profiler also needs its output dir in the same config.
        f"--profiler-config '{{\"profiler\": \"torch\", \"torch_profiler_dir\": \"{trace_dir}\"}}'"
    )
    logf = open(os.path.join(OUT, "server.log"), "w")
    subprocess.Popen(
        cmd, shell=True, env=env, stdout=logf, stderr=subprocess.STDOUT, start_new_session=True
    )
    if not health():
        print("NOT HEALTHY", flush=True)
        return 1
    print("HEALTHY", flush=True)

    # Admit requests with short output so we reach steady decode fast, then keep
    # them decoding while the profiler runs.
    bench = (
        f"env -u CUDA_VISIBLE_DEVICES VLLM_PLUGINS= /opt/conda/bin/vllm bench serve "
        f"--backend vllm --model minicpm --tokenizer {MODEL} "
        f"--endpoint /v1/completions --host localhost --port {PORT} "
        "--dataset-name random --ignore-eos --random-input-len 512 "
        "--random-output-len 2048 --max-concurrency 64 --num-prompts 128"
    )
    bf = open(os.path.join(OUT, "bench.log"), "w")
    bp = subprocess.Popen(bench, shell=True, stdout=bf, stderr=subprocess.STDOUT)

    time.sleep(35)  # past prefill, into steady decode
    print("start_profile", post("/start_profile"), flush=True)
    time.sleep(4)
    print("stop_profile", post("/stop_profile"), flush=True)
    time.sleep(2)
    bp.terminate()
    sh("pkill -9 -f 'vllm bench serve'")
    time.sleep(1)
    sh("pkill -9 -f 'vllm serve' ; pkill -9 -f 'VLLM::EngineCore'")

    files = [f for f in os.listdir(trace_dir) if f.endswith(".json") or f.endswith(".pt.trace.json")]
    print("traces:", files, flush=True)
    (pathlib := __import__("pathlib")).Path(os.path.join(OUT, "trace_files.json")).write_text(
        json.dumps(files, indent=2)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
