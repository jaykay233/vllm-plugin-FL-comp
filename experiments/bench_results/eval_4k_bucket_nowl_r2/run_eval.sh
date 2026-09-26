#!/bin/bash
# Official-case eval WITHOUT flagos whitelist + faulthandler for native crashes.
# Relies on FlagGems align32_geometric M-bucketing.
# Flow: GPU health -> serve -> 4k throughput -> evalscope math_500 L3
set -u
source /opt/conda/etc/profile.d/conda.sh
conda activate mx

OUT=/workspace/vllm-plugin-FL-comp/experiments/bench_results/eval_4k_bucket_nowl_r2
mkdir -p "$OUT"
PY=/opt/conda/envs/mx/bin/python
MODEL=/workspace/MiniCPM5-2B
PORT=9031
SERVED=minicpm

export PATH=/opt/conda/envs/mx/bin:$PATH
export VLLM_PLUGINS=fl
export PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1
unset VLLM_FL_FLAGOS_WHITELIST
unset VLLM_FL_FLAGOS_BLACKLIST
unset VLLM_FL_CUDAGRAPH_ONLY
unset FL_METAX_ATTN_NUM_SPLITS
unset VLLM_GC_DEBUG
unset VLLM_ITER_STAGE_PROFILE

ts() { date '+%H:%M:%S'; }
say() { echo "[$(ts)] $*"; }

# whitelist must be empty
cat > "$OUT/check_wl.py" <<'PY'
from vllm_fl.dispatch.config import get_flagos_whitelist
w = get_flagos_whitelist()
open("/workspace/vllm-plugin-FL-comp/experiments/bench_results/eval_4k_bucket_nowl_r2/wl_check.txt", "w").write(repr(w))
PY
$PY "$OUT/check_wl.py" >/dev/null 2>&1 || true
WL=$(cat "$OUT/wl_check.txt" 2>/dev/null || echo MISSING)
say "flagos_whitelist = $WL"
if [ "$WL" != "[]" ] && [ "$WL" != "None" ]; then
  say "ABORT: whitelist still active"
  exit 1
fi

say "GPU health: Stream + matmul"
$PY - <<'PY' || { say "ABORT: GPU unhealthy"; exit 1; }
import torch, sys
assert torch.cuda.is_available()
s = torch.cuda.Stream()
with torch.cuda.stream(s):
    x = torch.randn(512, 512, device="cuda")
    y = x @ x
s.synchronize()
print("OK", torch.cuda.get_device_name(0), float(y.mean()))
PY

say "cleanup old server on $PORT"
$PY - <<'PY'
import os, signal, time
me = os.getpid()
killed = []
for pid in os.listdir("/proc"):
    if not pid.isdigit() or int(pid) == me:
        continue
    try:
        cl = open(f"/proc/{pid}/cmdline", "rb").read().decode("utf8", "replace")
    except Exception:
        continue
    try:
        cwd = os.readlink(f"/proc/{pid}/cwd")
    except Exception:
        cwd = ""
    is_server = "vllm" in cl and "serve" in cl and "9031" in cl
    is_orphan = cl.startswith("VLLM::EngineCore") and cwd.startswith("/workspace")
    if is_server or is_orphan:
        try:
            os.kill(int(pid), signal.SIGKILL)
            killed.append(pid)
        except Exception:
            pass
print("  killed", killed or "none")
time.sleep(3)
PY

: > "$OUT/server.log"
: > "$OUT/bench.log"
: > "$OUT/evalscope.log"

say "PART A | start vllm serve - no whitelist, faulthandler on"
cd /workspace/vllm-plugin-FL-comp
setsid vllm serve "$MODEL" --port "$PORT" --served-model-name "$SERVED" \
  --gpu-memory-utilization 0.85 --max-model-len 131072 \
  > "$OUT/server.log" 2>&1 < /dev/null &
echo $! > "$OUT/server.pid"
say "server pid=$(cat "$OUT/server.pid")"

READY=0
for i in $(seq 1 360); do
  if grep -qa "Application startup complete" "$OUT/server.log"; then
    READY=1
    break
  fi
  if ! kill -0 "$(cat "$OUT/server.pid")" 2>/dev/null; then
    say "server died; tail:"
    tr '\r' '\n' < "$OUT/server.log" | tail -60
    exit 1
  fi
  # early detect EngineCore death during startup
  if grep -qa "engine core exited unexpectedly" "$OUT/server.log"; then
    say "EngineCore died during startup; tail:"
    tr '\r' '\n' < "$OUT/server.log" | tail -60
    exit 1
  fi
  if [ $((i % 6)) -eq 0 ]; then
    say "waiting ${i}0s bytes=$(wc -c < "$OUT/server.log")"
  fi
  sleep 10
done
if [ "$READY" != 1 ]; then
  say "NOT READY; tail:"
  tr '\r' '\n' < "$OUT/server.log" | tail -40
  exit 1
fi
say "PART A | READY"
if tr '\0' '\n' < "/proc/$(cat "$OUT/server.pid")/environ" 2>/dev/null | grep -q VLLM_FL_FLAGOS_WHITELIST; then
  say "WARN: WHITELIST env still on server"
else
  say "confirmed no WHITELIST env on server"
fi
tr '\r' '\n' < "$OUT/server.log" | grep -aE "flagos_whitelist|Using platform config|Loading weights took|GPU KV cache size|Maximum concurrency|Asynchronous scheduling|faulthandler" | head -20

# watchdog: if EngineCore dies mid-bench, abort early
watch_engine() {
  while kill -0 "$1" 2>/dev/null; do
    if grep -qa "engine core exited unexpectedly" "$OUT/server.log"; then
      say "WATCHDOG: EngineCore died — killing bench"
      pkill -9 -f "benchmark_throughput_serve|vllm bench serve" 2>/dev/null || true
      return 1
    fi
    sleep 5
  done
  return 0
}

say "PART B | 4k throughput"
cd /workspace
watch_engine "$(cat "$OUT/server.pid")" &
WATCH_PID=$!
$PY -u /workspace/vllm-plugin-FL-comp/benchmarks/benchmark_throughput_serve.py \
  --model "$MODEL" --served-model-name "$SERVED" --port "$PORT" \
  --test-cases '[[4096,1024,64,256]]' \
  2>&1 | tee "$OUT/bench.log"
BENCH_RC=${PIPESTATUS[0]}
kill "$WATCH_PID" 2>/dev/null || true
wait "$WATCH_PID" 2>/dev/null || true

if grep -qa "engine core exited unexpectedly" "$OUT/server.log"; then
  say "PART B | ABORT due to EngineCore crash"
  tr '\r' '\n' < "$OUT/server.log" | grep -aE "Fatal|faulthandler|Segmentation|Fatal Python|exited unexpectedly" | tail -40
  exit 2
fi
say "PART B | done rc=$BENCH_RC"
grep -aE "Successful requests|Failed requests|Total token throughput|^  Prefill=" "$OUT/bench.log" || true

# refuse evalscope if any failed requests in kept runs
if grep -a "Failed requests:" "$OUT/bench.log" | grep -qv "Failed requests:[[:space:]]*0"; then
  say "PART C | SKIP evalscope: bench had failed requests"
  exit 3
fi

say "PART C | evalscope math_500 Level 3"
rm -rf /workspace/evalscope-datasets/level3
evalscope eval \
  --model "$SERVED" \
  --api-url "http://127.0.0.1:$PORT/v1/chat/completions" \
  --api-key EMPTY \
  --eval-type openai_api \
  --datasets math_500 \
  --dataset-args '{"math_500": {"dataset_id": "/workspace/evalscope-datasets/math_500", "subset_list": ["Level 3"]}}' \
  --eval-batch-size 8 \
  --timeout 3600 \
  --generation-config '{"temperature": 1.0, "top_p": 0.95, "max_tokens": 32768}' \
  --work-dir /workspace/evalscope-datasets/level3 \
  --ignore-errors \
  2>&1 | tee "$OUT/evalscope.log"
say "PART C | done"

echo
echo "================ FINAL ================"
grep -aE "Successful requests|Failed requests|Total token throughput|^  Prefill=" "$OUT/bench.log" || true
grep -aiE "accuracy|acc|score|Level 3|math_500" "$OUT/evalscope.log" | tail -20 || true
echo "logs: $OUT"
say "ALL DONE"
