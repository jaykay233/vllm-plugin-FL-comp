#!/usr/bin/env python3
"""Capture a non-empty decode trace on the current stack, official 4k shape.

The previous attempt (decode_step_profile_4k_20261023) produced a trace with a
single mcDeviceSynchronize in it, because the harness slept 50 s before starting
the profiler while its bench -- 64 prompts x 400 output tokens at concurrency 64
-- now finishes in about 10 s.  On the current stack the profiler window landed
after every request had drained.

This uses the official case instead: 256 x (4096 in + 1024 out) at concurrency
64, which runs ~108 s, so a window starting at 20 s sits squarely in steady-state
decode with ~85 s of margin.

The question being answered is how much of a decode step runs *outside* the CUDA
graph.  vLLM's sampler cannot be captured (it consumes values produced by the
step), so the tail -- lm_head, softmax, top-p kernels, RNG -- is launched
individually and sits on the critical path.  Section 18.5 estimated ~0.4 ms for
"mask/softmax/random" but that number came from a trace predating sections 19/22;
section 22.7 is explicit that such figures must not be reused.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request

OUT = "/workspace/vllm-plugin-FL-comp/experiments/bench_results/decode_trace_current_20261025"
MODEL = "/workspace/MiniCPM5-2B"
PORT = 9033
# Admit exactly one wave and let it decode with nothing queued behind it.  The
# official shape mixes prefill into almost every step (chunked prefill keeps
# feeding wave 2 while wave 1 decodes), which is why the first attempt's window
# was 100% prefill at 264 ms/step.  With num-prompts == concurrency there is no
# second wave: ~8 s of prefill, then ~19 s of pure decode.
NUM_PROMPTS = 64
BENCH_LEAD = 13   # past the single wave's prefill, inside pure decode
PROFILE_WINDOW = 4


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True)


def health(timeout=300):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        if sh("pgrep -f 'vllm serve /workspace/MiniCPM5-2B' >/dev/null").returncode != 0:
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

    # No overrides: this must describe the shipped defaults, otherwise the trace
    # answers a question nobody is asking (22.7).
    env = os.environ.copy()
    env.pop("CUDA_VISIBLE_DEVICES", None)
    for k in (
        "VLLM_METAX_LM_HEAD_GEMM",
        "VLLM_METAX_MLP_SILU_FUSION",
        "VLLM_FL_FLAGOS_WHITELIST",
        "FLAG_GEMS_METAX_ALIGN32_GEOMETRIC_CAP",
        "VLLM_FL_METAX_MAX_NUM_BATCHED_TOKENS",
    ):
        env.pop(k, None)
    env.update(
        {
            "VLLM_PLUGINS": "fl",
            "VLLM_TORCH_PROFILER_DIR": trace_dir,
        }
    )
    cmd = (
        f"/opt/conda/bin/vllm serve {MODEL} --port {PORT} --served-model-name minicpm "
        "--gpu-memory-utilization 0.85 --max-model-len 131072 "
        "--no-enable-prefix-caching "
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

    # One wave, no queue: pure decode once prefill finishes.
    bench = (
        f"env -u CUDA_VISIBLE_DEVICES VLLM_PLUGINS= /opt/conda/bin/vllm bench serve "
        f"--backend vllm --model minicpm --tokenizer {MODEL} "
        f"--endpoint /v1/completions --host localhost --port {PORT} "
        "--dataset-name random --ignore-eos --random-input-len 4096 "
        f"--random-output-len 1024 --max-concurrency 64 --num-prompts {NUM_PROMPTS}"
    )
    bf = open(os.path.join(OUT, "bench.log"), "w")
    bp = subprocess.Popen(bench, shell=True, stdout=bf, stderr=subprocess.STDOUT)

    time.sleep(BENCH_LEAD)
    print("start_profile", post("/start_profile"), flush=True)
    time.sleep(PROFILE_WINDOW)
    print("stop_profile", post("/stop_profile"), flush=True)
    time.sleep(2)
    bp.terminate()
    sh("pkill -9 -f 'vllm bench serve'")
    time.sleep(1)
    sh("pkill -9 -f 'vllm serve' ; pkill -9 -f 'VLLM::EngineCore'")

    files = [f for f in os.listdir(trace_dir) if f.endswith(".json") or f.endswith(".gz")]
    sizes = {f: os.path.getsize(os.path.join(trace_dir, f)) for f in files}
    print("traces:", json.dumps(sizes, indent=2), flush=True)
    open(os.path.join(OUT, "trace_files.json"), "w").write(json.dumps(sizes, indent=2))

    # Fail loudly if the window missed again.
    worker = [f for f in files if f.startswith("rank0")]
    if not worker or sizes[worker[0]] < 100_000:
        print("WARN: worker trace looks empty -- window missed the bench", flush=True)
        return 2
    print("OK", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
