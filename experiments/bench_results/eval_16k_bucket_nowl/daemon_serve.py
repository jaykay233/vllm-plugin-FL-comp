#!/usr/bin/env python3
import os, subprocess, sys, time
OUT = "/workspace/vllm-plugin-FL-comp/experiments/bench_results/eval_16k_bucket_nowl"
LOG = os.path.join(OUT, "server.log")
PID = os.path.join(OUT, "server.pid")
STAT = os.path.join(OUT, "server.exit")
os.makedirs(OUT, exist_ok=True)
env = os.environ.copy()
env["VLLM_PLUGINS"] = "fl"
env["PYTHONFAULTHANDLER"] = "1"
env["PYTHONUNBUFFERED"] = "1"
env["PATH"] = "/opt/conda/bin:" + env.get("PATH", "")
for k in ("VLLM_FL_FLAGOS_WHITELIST","VLLM_FL_FLAGOS_BLACKLIST","VLLM_FL_CUDAGRAPH_ONLY","VLLM_GC_DEBUG"):
    env.pop(k, None)
logf = open(LOG, "w", buffering=1)
cmd = [
    "/opt/conda/bin/vllm", "serve", "/workspace/MiniCPM5-2B",
    "--port", "9031", "--served-model-name", "minicpm",
    "--gpu-memory-utilization", "0.85", "--max-model-len", "131072",
]
# Double-fork style: this process becomes session leader and *waits*,
# so we record how serve exits. Launch via start_new_session from caller.
p = subprocess.Popen(
    cmd, stdout=logf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
    env=env, cwd="/workspace/vllm-plugin-FL-comp",
)
open(PID, "w").write(str(p.pid))
print(f"started pid={p.pid}", flush=True)
rc = p.wait()
open(STAT, "w").write(f"rc={rc} time={time.strftime('%H:%M:%S')}\n")
print(f"serve exited rc={rc}", flush=True)
sys.exit(rc if isinstance(rc, int) and rc >= 0 else 1)
