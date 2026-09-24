#!/bin/bash
# 沐曦官方评测脚本编排：等 server 就绪 -> 跑 benchmark -> 汇总
# 用法: bash /root/src/run_official_eval.sh
set -u

export PATH=/opt/conda/envs/mx/bin:$PATH
export VLLM_PLUGINS=fl

PORT=9031
MODEL=/workspace/MiniCPM5-2B
SERVED=minicpm
SERVER_LOG=/root/bench_results/eval_current/server.log
BENCH_LOG=/root/bench_results/eval_current/bench.log
CASES='[[4096,1024,64,256],[16384,1024,64,128]]'

echo "[$(date +%H:%M:%S)] [wait] 等待 server 就绪（最多 40 分钟）..."
READY=0
for _ in $(seq 1 240); do
  if grep -qa "Application startup complete" "$SERVER_LOG" 2>/dev/null; then READY=1; break; fi
  sleep 10
done

if [ "$READY" != "1" ]; then
  echo "[$(date +%H:%M:%S)] [wait] 超时或失败，最后日志："
  tail -c 800 "$SERVER_LOG" | tr '\r' '\n' | tail -12
  exit 1
fi

echo "[$(date +%H:%M:%S)] [wait] server READY"
echo "--- 启动摘要 ---"
grep -aoE "torch.compile and initial profiling/warmup run together took[^\r]{0,20}|Capturing CUDA graphs[^\r]{0,30}\| [0-9]+/[0-9]+ \[[0-9:]+<[0-9:]+, *[0-9.]+s/it|GPU KV cache size[^\r]{0,30}|Maximum concurrency[^\r]{0,30}" \
  "$SERVER_LOG" | tail -4

# 冒烟：确认接口可用
curl -s -m 20 "http://localhost:${PORT}/v1/models" | head -c 200; echo

echo "[$(date +%H:%M:%S)] [bench] 开始（RUNS=4, SKIP_FIRST=1, cases=$CASES）"
cd /workspace || exit 1
/opt/conda/envs/mx/bin/python \
  /workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py \
  --model "$MODEL" \
  --served-model-name "$SERVED" \
  --port "$PORT" \
  --test-cases "$CASES" \
  2>&1 | tee "$BENCH_LOG"

echo "[$(date +%H:%M:%S)] [bench] 结束"
ls -la /workspace/benchmark_results/
