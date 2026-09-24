#!/bin/bash
# 沐曦官方口径评测：性能 (benchmark_throughput_serve) + 正确性 (evalscope math_500 Level 3)
# 用法: bash /root/src/run_eval_all.sh
set -u

export PATH=/opt/conda/envs/mx/bin:$PATH
export VLLM_PLUGINS=fl
unset VLLM_FL_CUDAGRAPH_ONLY   # 走插件默认（metax => CUDAGraph-only）

PORT=9031
MODEL=/workspace/MiniCPM5-2B
SERVED=minicpm
OUT=/root/bench_results/eval_official
SERVER_LOG=$OUT/server.log
BENCH_LOG=$OUT/bench.log
EVAL_LOG=$OUT/evalscope.log
PY=/opt/conda/envs/mx/bin/python

ts() { date '+%H:%M:%S'; }
say() { echo "[$(ts)] $*"; }

mkdir -p "$OUT"

# ---------------------------------------------------------------- PART A: serve
say "PART A | 启动 vllm serve（不传 --compilation-config，即评测口径）"
cd /workspace/vllm-plugin-FL || exit 1
say "HEAD = $(git rev-parse --short HEAD) $(git log -1 --format=%s)"
nohup vllm serve $MODEL --port $PORT --served-model-name $SERVED \
  --gpu-memory-utilization 0.85 --max-model-len 131072 \
  > "$SERVER_LOG" 2>&1 &
say "server pid=$!"

READY=0
for _ in $(seq 1 360); do
  grep -qa "Application startup complete" "$SERVER_LOG" 2>/dev/null && { READY=1; break; }
  sleep 10
done
if [ "$READY" != 1 ]; then
  say "PART A | 失败，尾部日志："; tr '\r' '\n' < "$SERVER_LOG" | tail -15; exit 1
fi
say "PART A | READY"
echo "--- 启动证据 ---"
tr '\r' '\n' < "$SERVER_LOG" | grep -aE "CUDAGraph-only|Loading weights took|Dynamo bytecode|took .* s in total|GPU KV cache size|Maximum concurrency" | head -8
tr '\r' '\n' < "$SERVER_LOG" | grep -aoE "\(decode, FULL\): +100%[^]]*\]" | tail -1
curl -s -m 20 "http://localhost:$PORT/v1/models" | head -c 160; echo

# ---------------------------------------------------------------- PART B: 性能
say "PART B | 性能基准 (4k/16k, RUNS=4, SKIP_FIRST=1, 官方 2 个 case)"
cd /workspace || exit 1
$PY /workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py \
  --model "$MODEL" --served-model-name "$SERVED" --port "$PORT" \
  --test-cases '[[4096,1024,64,256],[16384,1024,64,128]]' \
  2>&1 | tee "$BENCH_LOG"
say "PART B | 完成"

# ------------------------------------------------------------ PART C: 正确性
say "PART C | 正确性评测 evalscope math_500 Level 3"
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
  2>&1 | tee "$EVAL_LOG"
say "PART C | 完成"

echo; echo "================ 产物 ================"
ls -la /workspace/benchmark_results/ 2>/dev/null | tail -5
ls -la /workspace/evalscope-datasets/level3/ 2>/dev/null | tail -8
echo "server log : $SERVER_LOG"
echo "bench log  : $BENCH_LOG"
echo "eval log   : $EVAL_LOG"
