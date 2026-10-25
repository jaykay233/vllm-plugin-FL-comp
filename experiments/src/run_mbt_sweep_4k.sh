#!/bin/bash
# Sweep max_num_batched_tokens at the official 4k shape on the post-fix stack.
#
# Why: measuring the prefill/decode split with /metrics showed 28.3% of bench
# wall has prefill work in flight, and while it does, decode collapses to 15% of
# its normal rate (392 vs 2642 tok/s). Prefill itself runs at 28.1k tok/s. 4k
# produces 80% of its tokens as prompt tokens, so prefill efficiency dominates
# total throughput.
#
# MBT 4096 -> 8192 previously gave +4.3% (FINDINGS 14) purely by making prefill
# chunks bigger, but nothing above 8192 was ever tried. With 4096-token prompts,
# MBT 8192 packs 2 prompts per chunk, 16384 packs 4, 32768 packs 8. Bigger chunks
# mean fewer, larger prefill GEMMs.
#
# Arms: 8192 16384 32768 8192 (the repeated anchor at the end detects drift).
# 2 runs per arm, official [[4096,1024,64,256]], no prefix caching.
set -u
export PATH=/opt/conda/bin:$PATH
export VLLM_PLUGINS=fl
PY=/opt/conda/bin/python
OUT=/workspace/vllm-plugin-FL-comp/experiments/bench_results/mbt_sweep_4k_20261025
PORT=9038
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

  # confirm what the engine actually resolved
  say "MBT in effect: $(rg -ao 'max_num_batched_tokens[^,)]{0,14}' "$OUT/server_${label}.log" | head -1)"
  say "kv: $(rg -ao 'GPU KV cache size: [0-9,]+ tokens' "$OUT/server_${label}.log" | head -1)"

  ( cd /workspace && env VLLM_PLUGINS= RUNS=2 SKIP_FIRST=0 \
      PYTHONPATH=/workspace/vllm-plugin-FL-comp/experiments/src \
      $PY /workspace/vllm-plugin-FL-comp/experiments/src/runs2_bench.py \
      --model /workspace/MiniCPM5-2B --served-model-name minicpm --port "$PORT" \
      --test-cases '[[4096,1024,64,256]]' > "$OUT/bench_${label}.log" 2>&1 )

  say "results [$label] MBT=$mbt"
  rg -o 'Total token throughput \(tok/s\): +[0-9.]+|Output token throughput \(tok/s\): +[0-9.]+|Mean (TTFT|TPOT) \(ms\): +[0-9.]+' "$OUT/bench_${label}.log"
  stop_server
}

run_arm A 8192
run_arm B 16384
run_arm C 32768
run_arm D 8192

say "DONE"
for f in "$OUT"/bench_*.log; do echo "$f: $(rg -o 'Total token throughput \(tok/s\): +[0-9.]+' "$f" | tr -s ' ' | tr '\n' ' ')"; done
