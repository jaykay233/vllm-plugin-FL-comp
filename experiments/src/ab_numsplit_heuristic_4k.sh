#!/bin/bash
# A 臂：FL_METAX_ATTN_NUM_SPLITS=0（纯 heuristic = 上游行为）
# 对比 B 臂（默认自适应，已测）= /workspace/benchmark_results/summary_20260923_221038.csv 4k: 4824.45
set -u
export PATH=/opt/conda/envs/mx/bin:$PATH
export VLLM_PLUGINS=fl
PY=/opt/conda/envs/mx/bin/python
OUT=/root/bench_results/numsplit_ab
mkdir -p "$OUT"
say(){ echo "[$(date +%H:%M:%S)] $*"; }

say "停掉当前 server"
P=$(pgrep -f "mx/bin/vll[m] serve" | head -1)
[ -n "${P:-}" ] && kill -TERM "$P" && say "stop $P"
for _ in $(seq 1 25); do pgrep -f "mx/bin/vll[m] serve" >/dev/null || break; sleep 3; done
P=$(pgrep -f "mx/bin/vll[m] serve" | head -1); [ -n "${P:-}" ] && kill -9 "$P"
sleep 4
say "GPU: $(mx-smi 2>/dev/null | sed -n '15p' | tr -s ' ')"

LOG=$OUT/server_heuristic.log
say "启动 server（FL_METAX_ATTN_NUM_SPLITS=0）"
( cd /workspace/vllm-plugin-FL && FL_METAX_ATTN_NUM_SPLITS=0 nohup vllm serve /workspace/MiniCPM5-2B \
    --port 9031 --served-model-name minicpm \
    --gpu-memory-utilization 0.85 --max-model-len 131072 > "$LOG" 2>&1 & )
say "server launched, 等待就绪（冷编译可能 ~20min）"
for _ in $(seq 1 300); do
  grep -qa "Application startup complete" "$LOG" && { say "READY"; break; }
  sleep 10
done
grep -qa "Application startup complete" "$LOG" || { say "TIMEOUT"; tr '\r' '\n' < "$LOG" | tail -12; exit 1; }
say "--- num_splits 观测 ---"
tr '\r' '\n' < "$LOG" | grep -a "FL decode attention" | sort -u | head -20
say "--- 编译/捕获 ---"
tr '\r' '\n' < "$LOG" | grep -aoE "took [0-9.]+ s in total|\([a-z -]+, [A-Z_]+\): +100%[^]]*\]" | tail -3

say "bench 4k x4"
( cd /workspace && $PY /workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py \
    --model /workspace/MiniCPM5-2B --served-model-name minicpm --port 9031 \
    --test-cases '[[4096,1024,64,256]]' > "$OUT/bench_heuristic.log" 2>&1 )
$PY - <<'PY'
import re
t=open('/root/bench_results/numsplit_ab/bench_heuristic.log',errors='ignore').read()
b=re.split(r'Running: (\S+) \| Run (\d+)/(\d+)', t)
g=lambda s,p:(float(re.search(p,s).group(1)) if re.search(p,s) else None)
rows=[];i=1
while i+3<len(b):
    body=b[i+3]
    rows.append((int(b[i+1]),g(body,r'Total token throughput \(tok/s\): +([0-9.]+)'),
                 g(body,r'Mean TTFT \(ms\): +([0-9.]+)')))
    i+=4
for r in rows: print(f"  heuristic run{r[0]}: total={r[1]:.2f}  TTFT={r[2]:.2f}")
v=[r for r in rows if r[0]!=1]
if v:
    mt=sum(r[1] for r in v)/len(v); mf=sum(r[2] for r in v)/len(v)
    print(f"  >> heuristic 有效均值 total={mt:.2f}  TTFT={mf:.2f}")
    print(f"  >> B 臂(自适应) 参考值 total=4824.45  TTFT=2931.38")
    print(f"  >> 差异: total {mt-4824.45:+.2f} ({(mt/4824.45-1):+.2%})")
PY
say "DONE"
