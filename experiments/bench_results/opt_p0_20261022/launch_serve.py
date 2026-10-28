#!/usr/bin/env python3
"""Fully-detached vllm serve launcher with exit-code logging.

Double-fork + setsid so agent shell / job-control cannot signal the server.
Records wait() status to server.exit (rc=-N means signal N).
"""
from __future__ import annotations

import os
import signal
import sys
import time

OUT = "/workspace/vllm-plugin-FL-comp/experiments/bench_results/opt_p0_20261022"
LOG = os.path.join(OUT, "server.log")
PIDF = os.path.join(OUT, "server.pid")
EXITF = os.path.join(OUT, "server.exit")
ERRF = os.path.join(OUT, "launch_serve.err")

os.makedirs(OUT, exist_ok=True)

# --- first fork ---
if os.fork() > 0:
    # parent: wait briefly for pid file then exit
    for _ in range(50):
        if os.path.exists(PIDF):
            break
        time.sleep(0.05)
    sys.exit(0)

os.setsid()
# --- second fork ---
if os.fork() > 0:
    sys.exit(0)

# daemon body
signal.signal(signal.SIGHUP, signal.SIG_IGN)
os.chdir("/workspace/vllm-plugin-FL-comp")
os.umask(0)

# close stdio, reopen log
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
# P0: GEMV (FlagGems default on), MLP silu fusion, MBT default 4096
env["FLAG_GEMS_METAX_MM_GEMV"] = "1"
env["VLLM_METAX_MLP_SILU_FUSION"] = "1"
env["VLLM_FL_METAX_MAX_NUM_BATCHED_TOKENS"] = "4096"
for k in (
    "VLLM_FL_FLAGOS_WHITELIST",
    "VLLM_FL_FLAGOS_BLACKLIST",
    "VLLM_FL_CUDAGRAPH_ONLY",
    "VLLM_GC_DEBUG",
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
    "4096",
]

pid = os.spawnvpe(os.P_NOWAIT, cmd[0], cmd, env)
with open(PIDF, "w") as f:
    f.write(str(pid))

# reap child
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
# keep err copy
with open(ERRF, "a") as f:
    f.write(rc_txt)
