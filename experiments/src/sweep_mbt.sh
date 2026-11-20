# 4k sweep over max_num_batched_tokens (prefill chunk size).
# Each arm gets its own compile cache: the compiled shape ranges depend on MBT.
set -u
OUT=${OUT:-/root/bench_results/sweep_mbt}
mkdir -p "$OUT"
PORT=9201
MODEL=/workspace/MiniCPM5-2B
FRAMEWORK=/workspace/vllm-plugin-FL-comp
CASE="${CASE:-[[4096,1024,64,256]]}"
export VLLM_PLUGINS=fl
export PATH=/usr/local/corex/bin:$PATH
export PYTHONUNBUFFERED=1

teardown() {
  for p in $(pgrep -f "vllm serve $MODEL --port $PORT"); do kill -9 $p 2>/dev/null || true; done
  pkill -9 -f "VLLM::EngineCor[e]" 2>/dev/null || true
  sleep 10
}

run_arm() {
  local mbt="$1"
  local d="$OUT/mbt$mbt"
  mkdir -p "$d"
  teardown
  export VLLM_FL_ILUVATAR_MAX_NUM_BATCHED_TOKENS="$mbt"
  export VLLM_CACHE_ROOT="$d/cache"
  echo "=== mbt=$mbt start $(date +%H:%M:%S)" | tee -a "$OUT/run.log"
  cd "$FRAMEWORK" || return 1
  setsid nohup vllm serve "$MODEL" --port "$PORT" \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --served-model-name minicpm --gpu-memory-utilization 0.85 \
    --max-model-len 131072 > "$d/server.log" 2>&1 < /dev/null &
  for _ in $(seq 1 240); do
    curl -sf -m 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && break
    sleep 5
  done
  grep -ao "max_num_batched_tokens -> [0-9]*" "$d/server.log" | tail -1 | tee -a "$OUT/run.log"
  BENCH_RUNS=2 BENCH_SKIP_FIRST=0 python3 benchmarks/benchmark_throughput_serve.py \
    --model "$MODEL" --served-model-name minicpm --port "$PORT" \
    --test-cases "$CASE" > "$d/bench.log" 2>&1
  tr '\r' '\n' < "$d/bench.log" \
    | grep -aE "Benchmark duration|Total token throughput|Mean TPOT|Mean TTFT|Total tok/s" \
    | sed "s/^/  mbt=$mbt /" | tee -a "$OUT/run.log"
  echo "=== mbt=$mbt done $(date +%H:%M:%S)" | tee -a "$OUT/run.log"
}

: >> "$OUT/run.log"
echo "host $(hostname) start $(date -Is) case=$CASE" | tee -a "$OUT/run.log"
for m in ${MBTS:-2048 4096 8192 16384}; do run_arm "$m"; done
teardown
echo SWEEP_DONE | tee -a "$OUT/run.log"
