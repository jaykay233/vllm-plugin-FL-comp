#!/bin/bash
# Usage: run_profile.sh <split 0|1> <port>
# Profiles a window of engine iterations while the first prefill wave of the
# official 4k shape is being chunked alongside decodes, i.e. the mixed steps
# where split and no-split diverge.
set -u
SPLIT=$1
PORT=$2
OUT=/root/bench_results/engine_profile/split${SPLIT}
rm -rf "$OUT"
mkdir -p "$OUT/trace"

export VLLM_PLUGINS=fl
export VLLM_FL_ILUVATAR_IXFORMER_ATTN=1
export VLLM_FL_ILUVATAR_IXFORMER_SPLIT=$SPLIT
export VLLM_FL_ILUVATAR_IXFORMER_PREFILL_MIN_Q=256
unset VLLM_FL_CUDAGRAPH_ONLY

PROF_CFG="{\"profiler\":\"torch\",\"torch_profiler_dir\":\"$OUT/trace\",\"torch_profiler_with_stack\":false,\"torch_profiler_use_gzip\":false,\"ignore_frontend\":true,\"max_iterations\":120}"

cd /workspace/vllm-plugin-FL-comp || exit 1
nohup vllm serve /workspace/MiniCPM5-2B --port "$PORT" \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --served-model-name minicpm \
  --gpu-memory-utilization 0.85 \
  --max-model-len 131072 \
  --profiler-config "$PROF_CFG" \
  > "$OUT/server.log" 2>&1 &
SRV=$!
echo "$SRV" > "$OUT/server.pid"
echo "server pid=$SRV" > "$OUT/driver.log"

for i in $(seq 1 180); do
  curl -s "localhost:$PORT/v1/models" >/dev/null 2>&1 && break
  sleep 5
done
echo "ready at $(date +%T)" >> "$OUT/driver.log"

nohup vllm bench serve --backend vllm --model minicpm \
  --tokenizer /workspace/MiniCPM5-2B --endpoint /v1/completions \
  --host localhost --port "$PORT" --dataset-name random --ignore-eos \
  --random-input-len 4096 --random-output-len 1024 \
  --max-concurrency 64 --num-prompts 128 \
  > "$OUT/bench.log" 2>&1 &
BENCH=$!

# Steady state: wait for a pure-decode stats line (first wave fully prefilled),
# then for prompt throughput to come back (second wave trickling in, one new
# request per finished one, ~63 decodes + 1 chunk per mixed step).
seen_decode=0
for i in $(seq 1 600); do
  line=$(rg "Engine 000: Avg prompt throughput" "$OUT/server.log" | tail -1)
  p=$(echo "$line" | sed -E 's/.*Avg prompt throughput: ([0-9.]+).*/\1/')
  g=$(echo "$line" | sed -E 's/.*Avg generation throughput: ([0-9.]+).*/\1/')
  if [ -n "$line" ] && [ "${p%.*}" = "0" ] && [ "${g%.*}" != "0" ]; then
    seen_decode=1
  elif [ "$seen_decode" = "1" ] && [ -n "$p" ] && [ "${p%.*}" != "0" ]; then
    break
  fi
  sleep 2
done
echo "second wave seen at $(date +%T): $line" >> "$OUT/driver.log"
echo "start_profile at $(date +%T)" >> "$OUT/driver.log"
curl -s -X POST "localhost:$PORT/start_profile" >> "$OUT/driver.log" 2>&1
sleep 40
echo "stop_profile at $(date +%T)" >> "$OUT/driver.log"
curl -s -X POST "localhost:$PORT/stop_profile" >> "$OUT/driver.log" 2>&1
echo "stopped at $(date +%T)" >> "$OUT/driver.log"

kill "$BENCH" 2>/dev/null
pkill -9 -P "$SRV" 2>/dev/null
kill -9 "$SRV" 2>/dev/null
sleep 5
pkill -9 -f "EngineCor" 2>/dev/null
pkill -9 -f "vllm serve /workspace/MiniCPM5-2B --port $PORT" 2>/dev/null
echo "done at $(date +%T)" >> "$OUT/driver.log"
