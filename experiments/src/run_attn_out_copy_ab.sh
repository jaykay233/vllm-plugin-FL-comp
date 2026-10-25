#!/bin/bash
# A/B the decode attention-output writeback at the official 4k shape.
#
# A: VLLM_METAX_FAST_ATTN_OUT_WRITEBACK=0 -> `output[:n] = attn_out` (memcpy)
# B: default                                -> `add(x, 0, out=)` (kernel)
#
# Both arms run in one session, 2 bench runs each, because the session-to-session
# band is +-0.94% (FINDINGS 23.5) and the predicted effect is ~7%, so a
# cross-session comparison would still work but same-session is cleaner.
set -u
export PATH=/opt/conda/bin:$PATH
export VLLM_PLUGINS=fl
PY=/opt/conda/bin/python
OUT=/workspace/vllm-plugin-FL-comp/experiments/bench_results/attn_out_copy_20261025
PORT=9036
mkdir -p "$OUT"
say(){ echo "[$(date +%H:%M:%S)] $*"; }

server_pid(){ pgrep -f "vllm serve /workspace/MiniCPM5-2B" | head -1; }

stop_server(){
  local pid; pid=$(server_pid)
  [ -n "${pid:-}" ] && { kill -TERM "$pid"; say "stop server $pid"; }
  for _ in $(seq 1 20); do [ -z "$(server_pid)" ] && break; sleep 3; done
  pid=$(server_pid); [ -n "${pid:-}" ] && kill -9 "$pid"
  pkill -9 -f 'VLLM::EngineCore' 2>/dev/null
  sleep 8
}

run_arm(){
  local label="$1" knob="$2"
  stop_server
  say "start [$label] FAST_ATTN_OUT_WRITEBACK=$knob"
  cd /workspace/vllm-plugin-FL || exit 1
  setsid env VLLM_METAX_FAST_ATTN_OUT_WRITEBACK="$knob" \
    /opt/conda/bin/vllm serve /workspace/MiniCPM5-2B \
    --port "$PORT" --served-model-name minicpm \
    --gpu-memory-utilization 0.85 --max-model-len 131072 \
    --no-enable-prefix-caching \
    > "$OUT/server_${label}.log" 2>&1 < /dev/null &
  disown 2>/dev/null
  cd - >/dev/null

  local t0=$SECONDS
  until [ "$(curl -s -m 3 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" 2>/dev/null)" = "200" ]; do
    [ $((SECONDS-t0)) -gt 900 ] && { say "TIMEOUT"; return 1; }
    [ -z "$(server_pid)" ] && { say "SERVER DIED"; tail -20 "$OUT/server_${label}.log"; return 1; }
    sleep 5
  done
  say "HEALTHY after $((SECONDS-t0))s"

  ( cd /workspace && env VLLM_PLUGINS= RUNS=2 SKIP_FIRST=0 \
      PYTHONPATH=/workspace/vllm-plugin-FL-comp/experiments/src \
      $PY /workspace/vllm-plugin-FL-comp/experiments/src/runs2_bench.py \
      --model /workspace/MiniCPM5-2B --served-model-name minicpm --port "$PORT" \
      --test-cases '[[4096,1024,64,256]]' > "$OUT/bench_${label}.log" 2>&1 )

  say "results [$label]"
  rg -o 'Total token throughput \(tok/s\): +[0-9.]+|Output token throughput \(tok/s\): +[0-9.]+|Mean (TTFT|TPOT) \(ms\): +[0-9.]+|Successful requests: +[0-9.]+|Failed requests: +[0-9.]+' "$OUT/bench_${label}.log"
  stop_server
}

# interleaved A B A B so a slow drift in the session cannot favour one arm
run_arm A1 0
run_arm B1 1
run_arm A2 0
run_arm B2 1

say "DONE"
rg -n 'Total token throughput' "$OUT"/bench_*.log
