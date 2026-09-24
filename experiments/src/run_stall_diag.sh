#!/bin/bash
# 尾部停顿诊断：1s 粒度采样 GPU/CPU + 跑官方 4k case ×4
set -u
export PATH=/opt/conda/envs/mx/bin:$PATH
PY=/opt/conda/envs/mx/bin/python3
D=/root/bench_results/tail_stall
SERVE=524892
ENG=525406

echo "[$(date +%H:%M:%S)] 启动采样器"
setsid $PY /root/src/stall_sampler.py "$D/samples.csv" "apiserver:$SERVE" "engine:$ENG" \
  > "$D/sampler.out" 2>&1 < /dev/null &
echo $! > "$D/sampler.pid"
sleep 3
echo "  采样器 pid=$(cat $D/sampler.pid)  样本行数=$(wc -l < $D/samples.csv 2>/dev/null)"

echo "[$(date +%H:%M:%S)] 开始 4k 基准 ×4"
cd /workspace || exit 1
$PY /workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py \
  --model /workspace/MiniCPM5-2B --served-model-name minicpm --port 9031 \
  --test-cases '[[4096,1024,64,256]]' > "$D/bench.log" 2>&1

echo "[$(date +%H:%M:%S)] 基准结束，停采样器"
kill "$(cat $D/sampler.pid)" 2>/dev/null
sleep 2
echo "  采样行数=$(wc -l < $D/samples.csv)"
echo "[$(date +%H:%M:%S)] DONE"
