#!/bin/bash
# A2 control arm: vllm_c first, measured *after* B in the same session.
#
# B (flagos-first) came in 1.54% below A, but A itself sits 1.6% below the
# 12186-12199 baseline that section 21 recorded minutes earlier.  Those two
# numbers are the same size, so before believing the A/B we have to know
# whether the machine drifted between the two arms.  B is followed here by a
# fresh A; if A2 lands back near 11993 the arm-to-arm delta stands, and the
# 12186 -> 11993 shift is a separate (real) baseline move to chase.
#
# Usage: run_arm_norm.sh <label> <priority-json-list>
set -u
export PATH=/opt/conda/bin:$PATH
export VLLM_PLUGINS=fl
PY=/opt/conda/bin/python
LABEL="${1:?label}"
PRIO="${2:?priority list}"
OUT=/workspace/vllm-plugin-FL-comp/experiments/bench_results/norm_provider_ab_20261023
PORT=9031
KC="{\"ir_op_priority\": {\"fused_add_rms_norm\": $PRIO}}"
mkdir -p "$OUT"
say(){ echo "[$(date +%H:%M:%S)] $*"; }

server_pid(){ pgrep -f "vllm serve /workspace/MiniCPM5-2B" | head -1; }

stop_server(){
  local pid; pid=$(server_pid)
  [ -n "${pid:-}" ] && { kill -TERM "$pid"; say "stop server $pid"; }
  for _ in $(seq 1 20); do [ -z "$(server_pid)" ] && break; sleep 3; done
  pid=$(server_pid); [ -n "${pid:-}" ] && kill -9 "$pid"
  pkill -9 -f 'VLLM::EngineCore' 2>/dev/null
  sleep 3
}

stop_server
say "gpu quiet: $(mx-smi | rg -o 'GPU-Util' -A0 | head -1)"

say "start [$LABEL] $KC"
cd /workspace/vllm-plugin-FL || exit 1
setsid /opt/conda/bin/vllm serve /workspace/MiniCPM5-2B \
  --port "$PORT" --served-model-name minicpm \
  --gpu-memory-utilization 0.85 --max-model-len 131072 \
  --kernel-config "$KC" \
  > "$OUT/server_${LABEL}.log" 2>&1 < /dev/null &
disown 2>/dev/null
cd - >/dev/null

t0=$SECONDS
until [ "$(curl -s -m 3 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" 2>/dev/null)" = "200" ]; do
  [ $((SECONDS-t0)) -gt 900 ] && { say "TIMEOUT"; exit 1; }
  [ -z "$(server_pid)" ] && { say "SERVER DIED"; exit 1; }
  sleep 5
done
say "HEALTHY [$LABEL] after $((SECONDS-t0))s"

say "priority in effect: $(rg -ao 'Final IR op priority after setting platform defaults: .*fused_add_rms_norm=\[[^]]*\]' "$OUT/server_${LABEL}.log" | tail -1 | sed 's/.*IrOpPriorityConfig/IrOpPriorityConfig/')"

say "bench [$LABEL] RUNS=2"
( cd /workspace && env VLLM_PLUGINS= RUNS=2 SKIP_FIRST=0 PYTHONPATH=/workspace/vllm-plugin-FL-comp/experiments/src \
    $PY /workspace/vllm-plugin-FL-comp/experiments/src/runs2_bench.py \
    --model /workspace/MiniCPM5-2B --served-model-name minicpm --port "$PORT" \
    --test-cases '[[4096,1024,64,256]]' > "$OUT/bench_${LABEL}.log" 2>&1 )

rg -o 'Total token throughput \(tok/s\): +[0-9.]+|Mean TPOT \(ms\): +[0-9.]+|Mean TTFT \(ms\): +[0-9.]+' "$OUT/bench_${LABEL}.log"
stop_server
say "DONE [$LABEL]"
