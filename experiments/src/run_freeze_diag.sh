#!/bin/bash
# 冻结诊断：启动 catcher + sampler，然后跑 4k 基准
set -u
export PATH=/opt/conda/envs/mx/bin:$PATH
PY=/opt/conda/envs/mx/bin/python
D=/root/bench_results/tail_stall
ENG=$(tr '\r' '\n' < "$D/server.log" | grep -aoE "EngineCore pid=[0-9]+" | head -1 | grep -oE "[0-9]+")
echo "[$(date +%H:%M:%S)] engine=$ENG"
[ -z "$ENG" ] && { echo "engine 未找到"; exit 1; }

setsid $PY /root/src/freeze_catcher.py "$ENG" "$D/freeze.txt" \
  > "$D/freeze_catcher.out" 2>&1 < /dev/null &
echo $! > "$D/catcher.pid"
setsid $PY /root/src/stall_sampler.py "$D/samples2.csv" "engine:$ENG" \
  > /dev/null 2>&1 < /dev/null &
echo $! > "$D/sampler2.pid"
sleep 4
echo "[$(date +%H:%M:%S)] catcher=$(cat $D/catcher.pid) sampler=$(cat $D/sampler2.pid)"
head -3 "$D/freeze.txt" 2>/dev/null | sed 's/^/  /'

echo "[$(date +%H:%M:%S)] 开始 4k 基准 ×4"
cd /workspace || exit 1
$PY /workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py \
  --model /workspace/MiniCPM5-2B --served-model-name minicpm --port 9031 \
  --test-cases '[[4096,1024,64,256]]' > "$D/bench2.log" 2>&1

echo "[$(date +%H:%M:%S)] 基准结束"
kill "$(cat $D/catcher.pid)" 2>/dev/null
kill "$(cat $D/sampler2.pid)" 2>/dev/null
sleep 2
echo "[$(date +%H:%M:%S)] DONE  捕获=$(grep -c '捕获冻结' $D/freeze.txt 2>/dev/null)"
