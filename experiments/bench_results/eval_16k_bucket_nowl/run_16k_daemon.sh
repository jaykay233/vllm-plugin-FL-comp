#!/bin/bash
# MetaX 16k throughput baseline.
# CRITICAL: only the serve process may load VLLM_PLUGINS=fl.
# The bench client MUST run with VLLM_PLUGINS= (empty); otherwise
# importing fl/MetaX in a second process SIGKILLs the live serve.
set -u
source /opt/conda/etc/profile.d/conda.sh
conda activate base

OUT=/workspace/vllm-plugin-FL-comp/experiments/bench_results/eval_16k_bucket_nowl
PY=/opt/conda/bin/python
MODEL=/workspace/MiniCPM5-2B
PORT=9031
SERVED=minicpm

export PATH=/opt/conda/bin:$PATH
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
# Do NOT export VLLM_PLUGINS here — serve launcher sets it itself.
unset VLLM_FL_FLAGOS_WHITELIST VLLM_FL_FLAGOS_BLACKLIST VLLM_FL_CUDAGRAPH_ONLY VLLM_GC_DEBUG
unset VLLM_PLUGINS

ts() { date '+%H:%M:%S'; }
say() { echo "[$(ts)] $*"; }

mkdir -p "$OUT"
: > "$OUT/bench.log"
rm -f "$OUT/server.exit"

# Reuse live serve if healthy; otherwise (re)launch.
# CRITICAL: do NOT import torch.cuda / MetaX in this process while serve is up —
# a second CUDA init SIGKILLs the APIServer (same failure mode as VLLM_PLUGINS=fl bench).
NEED_SERVE=1
if [ -f "$OUT/server.pid" ] && [ -d "/proc/$(cat "$OUT/server.pid")" ]; then
  if curl -sf -m 2 "http://127.0.0.1:${PORT}/health" >/dev/null; then
    say "reuse live serve pid=$(cat "$OUT/server.pid")"
    NEED_SERVE=0
  fi
fi

if [ "$NEED_SERVE" = 1 ]; then
  # clean leftovers only when we are about to relaunch
  pkill -9 -f 'VLLM::EngineCore|vllm serve.*9031' 2>/dev/null || true
  sleep 2
  say "GPU health (before serve only)"
  $PY - <<'PY' || exit 1
import torch
s = torch.cuda.Stream()
with torch.cuda.stream(s):
    x = torch.randn(256, 256, device="cuda")
    y = x @ x
s.synchronize()
print("OK", torch.cuda.get_device_name(0))
PY
  rm -f "$OUT/server.exit" "$OUT/server.pid"
  say "launch detached serve (double-fork)"
  $PY "$OUT/launch_serve.py"
  sleep 2
  SRV=$(cat "$OUT/server.pid")
  say "server pid=$SRV"

  READY=0
  for i in $(seq 1 360); do
    if grep -qa "Application startup complete" "$OUT/server.log" 2>/dev/null; then
      READY=1
      break
    fi
    if [ -f "$OUT/server.exit" ]; then
      say "serve exited early: $(cat "$OUT/server.exit")"
      tr '\r' '\n' < "$OUT/server.log" | tail -40
      exit 1
    fi
    if [ $((i % 3)) -eq 0 ]; then
      say "wait ${i}0s bytes=$(wc -c <"$OUT/server.log") parent=$([ -d /proc/$SRV ] && echo up || echo down) ec=$(pgrep -c -f 'VLLM::EngineCore' 2>/dev/null || echo 0)"
    fi
    sleep 10
  done
  [ "$READY" = 1 ] || { say "NOT READY"; exit 1; }
fi

say "READY"
tr '\r' '\n' < "$OUT/server.log" | grep -aE "Loading weights took|GPU KV cache size|Maximum concurrency" | head -8

for i in $(seq 1 30); do
  if curl -sf -m 2 "http://127.0.0.1:${PORT}/health" >/dev/null; then
    say "health OK"
    break
  fi
  sleep 2
done

say "PART B | 16k [[16384,1024,64,128]] client plugins empty"
cd /workspace
# Explicit empty plugins — do not inherit fl from anywhere.
env -u VLLM_PLUGINS VLLM_PLUGINS= \
  BENCH_RUNS=4 BENCH_SKIP_FIRST=1 \
  $PY -u /workspace/vllm-plugin-FL-comp/benchmarks/benchmark_throughput_serve.py \
  --model "$MODEL" --served-model-name "$SERVED" --port "$PORT" \
  --test-cases '[[16384,1024,64,128]]' \
  2>&1 | tee "$OUT/bench.log"

say "DONE"
echo "======== SUMMARY ========"
grep -aE "Successful requests|Failed requests|Total token throughput|^  Prefill=|Output token throughput" "$OUT/bench.log" || true
[ -f "$OUT/server.exit" ] && say "WARNING server exit: $(cat "$OUT/server.exit")"
curl -sf -m 2 "http://127.0.0.1:${PORT}/health" >/dev/null && say "serve still healthy after bench" || say "serve DEAD after bench"
