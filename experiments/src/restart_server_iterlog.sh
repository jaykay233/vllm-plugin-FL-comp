#!/bin/bash
# 重启 server 并开启逐迭代日志，用于在冻结发生时抓到引擎当步在做什么。
# 用脚本文件承载匹配串，避免 pgrep 匹配到自身命令行（此前踩过两次）。
set -u
export PATH=/opt/conda/envs/mx/bin:$PATH
export VLLM_PLUGINS=fl
unset VLLM_FL_CUDAGRAPH_ONLY FL_METAX_ATTN_NUM_SPLITS

D=/root/bench_results/iterlog
mkdir -p "$D"
LOG=$D/server.log

echo "[$(date +%H:%M:%S)] 停止旧 server"
/opt/conda/envs/mx/bin/python - <<'PY'
import os, signal, time
me = os.getpid()
targets = []
for pid in os.listdir('/proc'):
    if not pid.isdigit() or int(pid) == me:
        continue
    try:
        cl = open(f'/proc/{pid}/cmdline', 'rb').read().decode('utf8', 'replace')
    except Exception:
        continue
    if 'vllm' in cl and 'serve' in cl and 'MiniCPM5-2B' in cl:
        targets.append(int(pid))
for p in targets:
    print(f"  TERM {p}")
    try: os.kill(p, signal.SIGTERM)
    except Exception as e: print('   ', e)
time.sleep(10)
for p in targets:
    try:
        os.kill(p, signal.SIGKILL); print(f"  KILL {p}")
    except Exception: pass
PY
sleep 3

echo "[$(date +%H:%M:%S)] 启动 server（--enable-logging-iteration-details）"
cd /workspace/vllm-plugin-FL || exit 1
setsid vllm serve /workspace/MiniCPM5-2B --port 9031 --served-model-name minicpm \
  --gpu-memory-utilization 0.85 --max-model-len 131072 \
  --enable-logging-iteration-details \
  > "$LOG" 2>&1 < /dev/null &
echo $! > "$D/server.pid"

for _ in $(seq 1 120); do
  grep -qa "Application startup complete" "$LOG" && break
  sleep 5
done
if grep -qa "Application startup complete" "$LOG"; then
  echo "[$(date +%H:%M:%S)] READY"
  tr '\r' '\n' < "$LOG" | grep -aE "CUDAGraph-only|iteration|Loading weights took|startup complete" | tail -3
else
  echo "[$(date +%H:%M:%S)] 未就绪，尾部："; tr '\r' '\n' < "$LOG" | tail -8
fi
