#!/usr/bin/env python3
"""Fully-detached vllm serve launcher for host-gap measurement (§5.2).

P2 env + VLLM_ITER_STAGE_PROFILE + --enable-logging-iteration-details.
Double-fork + setsid; records wait() status to server.exit.
"""
from __future__ import annotations

import os
import signal
import sys
import time

OUT = "/workspace/vllm-plugin-FL-comp/experiments/bench_results/hostgap_measure_20261023"
LOG = os.path.join(OUT, "server.log")
PIDF = os.path.join(OUT, "server.pid")
EXITF = os.path.join(OUT, "server.exit")
ERRF = os.path.join(OUT, "launch_serve.err")
STAGEF = os.path.join(OUT, "stage_prof.txt")

os.makedirs(OUT, exist_ok=True)

# --- first fork ---
if os.fork() > 0:
    for _ in range(50):
        if os.path.exists(PIDF):
            break
        time.sleep(0.05)
    sys.exit(0)

os.setsid()
# --- second fork ---
if os.fork() > 0:
    sys.exit(0)

signal.signal(signal.SIGHUP, signal.SIG_IGN)
os.chdir("/workspace/vllm-plugin-FL-comp")
os.umask(0)

sys.stdout.flush()
sys.stderr.flush()
logf = open(LOG, "w", buffering=1)
os.dup2(logf.fileno(), 1)
os.dup2(logf.fileno(), 2)
devnull = open(os.devnull, "r")
os.dup2(devnull.fileno(), 0)

env = os.environ.copy()
env["VLLM_PLUGINS"] = "fl"
env["PYTHONFAULTHANDLER"] = "1"
env["PYTHONUNBUFFERED"] = "1"
env["PATH"] = "/opt/conda/bin:" + env.get("PATH", "")
_cvd = env.get("CUDA_VISIBLE_DEVICES")
if _cvd is None or str(_cvd).strip() == "":
    env.pop("CUDA_VISIBLE_DEVICES", None)

env["FLAG_GEMS_METAX_MM_GEMV"] = "1"
env["VLLM_METAX_MLP_SILU_FUSION"] = "1"
env["VLLM_FL_METAX_MAX_NUM_BATCHED_TOKENS"] = "8192"
env["VLLM_FL_METAX_CUDAGRAPH_SPARE"] = "72"
env["FLAG_GEMS_METAX_ALIGN32_GEOMETRIC_CAP"] = "8192"
env["VLLM_FL_FLAGOS_WHITELIST"] = "silu_and_mul,rms_norm,rotary_embedding"
env["VLLM_METAX_LM_HEAD_GEMM"] = "0"
env["VLLM_GC_DEBUG"] = "0"
# §5.2 host-gap stage breakdown
env["VLLM_ITER_STAGE_PROFILE"] = "1"
env["VLLM_ITER_STAGE_WINDOW"] = "10"
env["VLLM_ITER_STAGE_SLOW"] = "0.5"
env["VLLM_ITER_STAGE_FILE"] = STAGEF

for k in (
    "VLLM_FL_FLAGOS_BLACKLIST",
    "VLLM_FL_CUDAGRAPH_ONLY",
    "MACA_GRAPH_LAUNCH_MODE",
):
    env.pop(k, None)

cmd = [
    "/opt/conda/bin/vllm",
    "serve",
    "/workspace/MiniCPM5-2B",
    "--port",
    "9031",
    "--served-model-name",
    "minicpm",
    "--gpu-memory-utilization",
    "0.85",
    "--max-model-len",
    "131072",
    "--max-num-batched-tokens",
    "8192",
    "--no-enable-prefix-caching",
    "--enable-logging-iteration-details",
]

pid = os.spawnvpe(os.P_NOWAIT, cmd[0], cmd, env)
with open(PIDF, "w") as f:
    f.write(str(pid))

try:
    _, status = os.waitpid(pid, 0)
except ChildProcessError:
    status = -1

if os.WIFSIGNALED(status):
    sig = os.WTERMSIG(status)
    rc_txt = f"rc=-{sig} signal={sig} ({signal.Signals(sig).name}) time={time.strftime('%H:%M:%S')}\n"
elif os.WIFEXITED(status):
    rc_txt = f"rc={os.WEXITSTATUS(status)} time={time.strftime('%H:%M:%S')}\n"
else:
    rc_txt = f"rc=unknown status={status} time={time.strftime('%H:%M:%S')}\n"

with open(EXITF, "w") as f:
    f.write(rc_txt)
with open(ERRF, "a") as f:
    f.write(rc_txt)
