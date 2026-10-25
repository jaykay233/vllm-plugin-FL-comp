#!/usr/bin/env python3
"""Capture a trace of pure decode by watching the server's own metrics.

Two earlier attempts landed entirely in prefill.  The reason is that while the
torch profiler is active this stack stops replaying CUDA graphs (the trace shows
0 mcGraphLaunch per step) and the step degrades to eager, so wall time per step
is ~264 ms instead of ~18-24 ms.  Any fixed sleep calibrated against normal
serving therefore fires far too early: prefill alone, which normally finishes in
2 s, is still running tens of seconds later under the profiler.

So don't guess a time.  Poll /metrics until the engine reports a steady decode
regime, then start profiling:

  * vllm:num_requests_running > 0            -- requests are in flight
  * prompt tokens/step == 0 for N polls      -- nothing left to prefill
  * generation tokens/step > 0               -- and it is producing output

Prefill that co-runs with decode is exactly what made the earlier windows
useless, so requiring zero prompt tokens is the whole point.

Output length is deliberately large (4096) so decode dominates even if eager
execution makes everything slower; there is no queue behind the single wave, so
all of it is decode once prefill drains.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
import urllib.request

OUT = "/workspace/vllm-plugin-FL-comp/experiments/bench_results/decode_trace_current_20261025"
MODEL = "/workspace/MiniCPM5-2B"
PORT = 9034
OUTPUT_LEN = 4096
CONC = 64
POLL = 1.0
NEED_QUIET = 4          # consecutive polls with no prompt tokens
PROFILE_WINDOW = 4
MAX_WAIT = 240


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True)


def health(timeout=600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        if sh("pgrep -f 'vllm serve /workspace/MiniCPM5-2B' >/dev/null").returncode != 0:
            print(f"server process gone at t={time.time() - t0:.0f}s", flush=True)
            return False
        if int(time.time() - t0) % 30 < 5:
            print(f"  waiting for /health t={time.time() - t0:.0f}s", flush=True)
        time.sleep(5)
    print(f"health timeout after {timeout}s", flush=True)
    return False


def host_rss_gb():
    try:
        with open("/sys/fs/cgroup/memory/memory.stat") as f:
            for ln in f:
                if ln.startswith("rss "):
                    return int(ln.split()[1]) / 2**30
    except OSError:
        pass
    return 0.0


def post(path):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", method="POST", data=b"")
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status
    except Exception as e:  # noqa: BLE001
        return f"ERR {e}"


def metrics():
    """-> (running, prompt_delta, gen_delta) from /metrics."""
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
    return (
        val("vllm:num_requests_running"),
        val("vllm:prompt_tokens_total"),
        val("vllm:generation_tokens_total"),
    )


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    trace_dir = os.path.join(OUT, "trace")
    os.makedirs(trace_dir, exist_ok=True)
    for f in os.listdir(trace_dir):
        os.remove(os.path.join(trace_dir, f))

    sh("pkill -9 -f 'vllm serve' ; pkill -9 -f 'VLLM::EngineCore' ; pkill -9 -f 'vllm bench serve'")
    time.sleep(3)
    # A failed attempt leaves an orphaned EngineCore holding ~21 GB.  The cgroup
    # ceiling is 64 GiB and the server's own peak already reaches it, so an
    # orphan guarantees an OOM kill of the next attempt.  Refuse to start until
    # the host is actually clean.
    rss = host_rss_gb()
    if rss > 4:
        print(f"refusing to start: host rss still {rss:.1f} GB", flush=True)
        return 3
    print(f"host rss before start: {rss:.2f} GB", flush=True)

    env = os.environ.copy()
    env.pop("CUDA_VISIBLE_DEVICES", None)
    for k in (
        "VLLM_METAX_LM_HEAD_GEMM", "VLLM_METAX_MLP_SILU_FUSION",
        "VLLM_FL_FLAGOS_WHITELIST", "FLAG_GEMS_METAX_ALIGN32_GEOMETRIC_CAP",
        "VLLM_FL_METAX_MAX_NUM_BATCHED_TOKENS",
    ):
        env.pop(k, None)
    env.update({"VLLM_PLUGINS": "fl", "VLLM_TORCH_PROFILER_DIR": trace_dir})
    cmd = (
        f"/opt/conda/bin/vllm serve {MODEL} --port {PORT} --served-model-name minicpm "
        # 131072 makes the host-side peak hit the 64 GiB cgroup ceiling and the
        # server gets OOM-killed mid-startup (observed: max_usage == limit).
        # Decode at seq ~4096 only needs 8k of context, so 16384 is ample and
        # keeps the process comfortably inside the cgroup.
        "--gpu-memory-utilization 0.85 --max-model-len 16384 "
        "--no-enable-prefix-caching "
        f"--profiler-config '{{\"profiler\": \"torch\", \"torch_profiler_dir\": \"{trace_dir}\"}}'"
    )
    logf = open(os.path.join(OUT, "server.log"), "w")
    subprocess.Popen(cmd, shell=True, env=env, stdout=logf, stderr=subprocess.STDOUT,
                     start_new_session=True)
    if not health():
        print("NOT HEALTHY", flush=True)
        return 1
    print("HEALTHY", flush=True)

    bench = (
        f"env -u CUDA_VISIBLE_DEVICES VLLM_PLUGINS= /opt/conda/bin/vllm bench serve "
        f"--backend vllm --model minicpm --tokenizer {MODEL} "
        f"--endpoint /v1/completions --host localhost --port {PORT} "
        "--dataset-name random --ignore-eos --random-input-len 4096 "
        f"--random-output-len {OUTPUT_LEN} --max-concurrency {CONC} --num-prompts {CONC}"
    )
    bf = open(os.path.join(OUT, "bench.log"), "w")
    bp = subprocess.Popen(bench, shell=True, stdout=bf, stderr=subprocess.STDOUT)

    # wait for steady decode, decided by the engine's own numbers
    t0 = time.time()
    quiet = 0
    prev = None
    started = False
    while time.time() - t0 < MAX_WAIT:
        m = metrics()
        if m and m[0] is not None and prev is not None:
            running, ptok, gtok = m
            dp = (ptok or 0) - (prev[1] or 0)
            dg = (gtok or 0) - (prev[2] or 0)
            el = time.time() - t0
            print(f"  t={el:5.1f}s running={running:.0f} prompt+={dp:.0f} gen+={dg:.0f}",
                  flush=True)
            if dp <= 0 and dg > 0 and running > 0:
                quiet += 1
            else:
                quiet = 0
            if quiet >= NEED_QUIET:
                print(f"steady decode after {el:.1f}s", flush=True)
                started = True
                break
        prev = m if m else prev
        time.sleep(POLL)

    if not started:
        print("WARN: never reached steady decode, profiling anyway", flush=True)

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
    worker = [f for f in files if f.startswith("rank0")]
    if not worker or sizes[worker[0]] < 100_000:
        print("WARN: worker trace looks empty", flush=True)
        return 2
    print("OK", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
