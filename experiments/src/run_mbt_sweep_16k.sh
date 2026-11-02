#!/bin/bash
# Does MBT 16384 also help at the 16k shape, or does it regress?
#
# The 4k sweep (mbt_sweep_4k_20261025) gave, on the post-fix stack:
#   MBT  8192 -> 12542 tok/s  (anchor A 12542, anchor D 12541, 0.01% apart)
#   MBT 16384 -> 12657 tok/s  (+0.92%)
#   MBT 32768 -> 12654 tok/s  (saturated, and TTFT 3.7 s vs 2.5 s)
#
# 16384 wins at 4k. But 16k has 16384-token prompts, so MBT 16384 holds exactly
# one prompt per chunk and larger MBT changes chunk *count*, not chunk *size*.
# The risk is worse: a max_num_batched_tokens above the prompt length can make
# the scheduler admit more sequences than it would otherwise, and 16k is the
# shape where KV pressure actually bites.
#
# So A/B 8192 vs 16384 at 16k, with a 16384 anchor at the end.
# 2 runs per arm, official [[16384,1024,64,256]], no prefix caching.
set -u
export PATH=/opt/conda/bin:$PATH
export VLLM_PLUGINS=fl
PY=/opt/conda/bin/python
OUT=/workspace/vllm-plugin-FL-comp/experiments/bench_results/mbt_sweep_16k_20261025
PORT=9039
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
  local label="$1" mbt="$2"
  stop_server
  say "start [$label] MBT=$mbt"
  cd /workspace/vllm-plugin-FL || exit 1
  setsid env VLLM_FL_METAX_MAX_NUM_BATCHED_TOKENS="$mbt" \
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
  say "MBT in effect: $(rg -ao 'max_num_batched_tokens[^,)]{0,14}' "$OUT/server_${label}.log" | head -1)"
  say "kv: $(rg -ao 'GPU KV cache size: [0-9,]+ tokens' "$OUT/server_${label}.log" | head -1)"

  ( cd /workspace && env VLLM_PLUGINS= RUNS=2 SKIP_FIRST=0 \
      $PY /workspace/vllm-plugin-FL-comp/experiments/src/runs2_bench.py \
      --model /workspace/MiniCPM5-2B --served-model-name minicpm --port "$PORT" \
      --test-cases '[[16384,1024,64,256]]' > "$OUT/bench_${label}.log" 2>&1 )

  say "results [$label] MBT=$mbt"
  rg -o 'Total token throughput \(tok/s\): +[0-9.]+|Output token throughput \(tok/s\): +[0-9.]+|Mean (TTFT|TPOT) \(ms\): +[0-9.]+' "$OUT/bench_${label}.log"
  stop_server
}

run_arm A 8192
run_arm B 16384
run_arm C 16384

say "DONE"
for f in "$OUT"/bench_*.log; do echo "$f: $(rg -o 'Total token throughput \(tok/s\): +[0-9.]+' "$f" | tr -s ' ' | tr '\n' ' ')"; done
