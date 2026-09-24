#!/bin/bash
# 隔离测试：num_splits 自适应规则是否拖累了 4k（decode 重）案例
# 假设 server 已在 9031 就绪。依次跑：
#   A) FL_METAX_ATTN_NUM_SPLITS=0（纯 heuristic，上游行为）  需要重启 server
#   B) 默认（自适应，batch<16 强制 16 路）                   需要重启 server
# 结论：比较两者的 Total tok/s
set -u
export PATH=/opt/conda/envs/mx/bin:$PATH
export VLLM_PLUGINS=fl
PY=/opt/conda/envs/mx/bin/python
OUT=/root/bench_results/numsplit_ab
mkdir -p "$OUT"
say(){ echo "[$(date +%H:%M:%S)] $*"; }

stop_server(){
  local pid; pid=$(pgrep -f "mx/bin/vll[m] serve" | head -1)
  [ -n "${pid:-}" ] && { kill -TERM "$pid"; say "stop server $pid"; }
  for _ in $(seq 1 20); do pgrep -f "mx/bin/vll[m] serve" >/dev/null || break; sleep 3; done
  local p2; p2=$(pgrep -f "mx/bin/vll[m] serve" | head -1); [ -n "${p2:-}" ] && kill -9 "$p2"
  sleep 3
}

start_server(){ # $1=label  $2=extra env (如 "FL_METAX_ATTN_NUM_SPLITS=0")
  local label=$1 extra=${2:-}
  local log=$OUT/server_$label.log
  say "start server [$label] env='$extra'"
  ( cd /workspace/vllm-plugin-FL && env $extra nohup vllm serve /workspace/MiniCPM5-2B \
      --port 9031 --served-model-name minicpm \
      --gpu-memory-utilization 0.85 --max-model-len 131072 > "$log" 2>&1 & )
  for _ in $(seq 1 120); do
    grep -qa "Application startup complete" "$log" && { say "READY [$label]"; return 0; }
    sleep 10
  done
  say "TIMEOUT [$label]"; return 1
}

bench(){ # $1=label
  local label=$1
  say "bench [$label] 4k x4"
  ( cd /workspace && $PY /workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py \
      --model /workspace/MiniCPM5-2B --served-model-name minicpm --port 9031 \
      --test-cases '[[4096,1024,64,256]]' > "$OUT/bench_$label.log" 2>&1 )
  $PY - "$OUT/bench_$label.log" "$label" <<'PY'
import re,sys
t=open(sys.argv[1],errors='ignore').read()
b=re.split(r'Running: (\S+) \| Run (\d+)/(\d+)', t)
g=lambda s,p:(float(re.search(p,s).group(1)) if re.search(p,s) else None)
rows=[]; i=1
while i+3<len(b):
    body=b[i+3]
    rows.append((int(b[i+1]), g(body,r'Total token throughput \(tok/s\): +([0-9.]+)'),
                 g(body,r'Mean TTFT \(ms\): +([0-9.]+)')))
    i+=4
valid=[r for r in rows if r[0]!=1]
print(f"  [{sys.argv[2]}] runs: "+", ".join(f"run{r[0]}={r[1]:.0f}" for r in rows))
if valid:
    mt=sum(r[1] for r in valid)/len(valid); mf=sum(r[2] for r in valid)/len(valid)
    print(f"  [{sys.argv[2]}] 有效均值 total={mt:.2f}  TTFT={mf:.2f}")
PY
}

stop_server
start_server heuristic "FL_METAX_ATTN_NUM_SPLITS=0"
bench heuristic
stop_server
start_server adaptive ""
bench adaptive
stop_server
say "DONE  结果见 $OUT"
