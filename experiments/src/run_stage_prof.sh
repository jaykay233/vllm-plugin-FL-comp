#!/bin/bash
# 带 EngineCore 阶段埋点重启 server，跑 4k 基准，用于定位：
#   1) 迭代之间那 ~23ms/次 的未计时开销去了哪里
#   2) 偶发的 ~20s 停顿落在哪个阶段
set -u
export PATH=/opt/conda/envs/mx/bin:$PATH
export VLLM_PLUGINS=fl
export VLLM_ITER_STAGE_PROFILE=1
export VLLM_ITER_STAGE_SLOW=0.5
# VLLM_GC_DEBUG 已关闭：它为 GC 回调挂上 gc.get_objects()，而 vLLM 自己在
# gc_utils.py 里就注明这段代码「occasionally it would run into signals which
# kills the engine」。上一轮开启后 EngineCore 在 08:32:59 异常退出（无 Python
# 栈，信号级），导致该轮 208/256 请求失败。GC 结论已拿到（max 8.52ms，已排除），
# 不再需要这个开关。
export VLLM_GC_DEBUG=0
unset VLLM_FL_CUDAGRAPH_ONLY FL_METAX_ATTN_NUM_SPLITS

D=/root/bench_results/stage_prof
mkdir -p "$D"
LOG=$D/server.log
# 停顿栈由 EngineCore 内的看门狗写入，必须每轮清空，否则会混入旧运行的内容
rm -f "$D/stall_stacks.txt"

echo "[$(date +%H:%M:%S)] 停止旧 server（只杀占用 9031 端口的，避免误伤其他 session）"
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
    # 必须同时是 vllm serve 且显式使用 --port 9031，绝不按模型名匹配
    if 'vllm' in cl and 'serve' in cl and '--port' in cl and '9031' in cl:
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

echo "[$(date +%H:%M:%S)] 启动 server（VLLM_ITER_STAGE_PROFILE=1）"
cd /workspace/vllm-plugin-FL || exit 1
setsid vllm serve /workspace/MiniCPM5-2B --port 9031 --served-model-name minicpm \
  --gpu-memory-utilization 0.85 --max-model-len 131072 \
  --enable-logging-iteration-details \
  > "$LOG" 2>&1 < /dev/null &
echo $! > "$D/server.pid"

for _ in $(seq 1 150); do
  grep -qa "Application startup complete" "$LOG" && break
  sleep 5
done
if ! grep -qa "Application startup complete" "$LOG"; then
  echo "[$(date +%H:%M:%S)] 未就绪，尾部："; tr '\r' '\n' < "$LOG" | tail -8; exit 1
fi
echo "[$(date +%H:%M:%S)] READY"

echo "[$(date +%H:%M:%S)] 开始 4k 基准（官方脚本内部 4 轮）"
cd /workspace
# -u: 无缓冲。重定向到文件时 Python 会缓冲 stdout，一旦进程被信号带走
# （本次就被环境重启带走一次），缓冲里的结果会全部丢失，bench.log 变成 0 字节。
/opt/conda/envs/mx/bin/python -u /workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py \
  --model /workspace/MiniCPM5-2B --served-model-name minicpm --port 9031 \
  --test-cases '[[4096,1024,64,256]]' > "$D/bench.log" 2>&1
echo "[$(date +%H:%M:%S)] 基准结束"
echo "=== 官方口径结果（脚本自动跑 4 轮、丢首轮、取后 3 轮平均）==="
grep -a "^  Prefill=" "$D/bench.log" | sed 's/^/  /'
echo
echo "=== 各内部轮 Total tok/s ==="
grep -a "Total token throughput" "$D/bench.log" | sed 's/^/  /'
echo
echo "=== 停顿时刻的 Python 栈（看门狗抓取）==="
if [ -s "$D/stall_stacks.txt" ]; then
  echo "  文件行数: $(wc -l < "$D/stall_stacks.txt")"
  echo "  --- 每个停顿采样的前 2 条 Traceback ---"
  grep -a -A 6 "^----- step running" "$D/stall_stacks.txt" | head -80 | sed 's/^/  /'
else
  echo "  无（本轮未复现停顿）"
fi
echo
echo "=== 阶段埋点窗口汇总 ==="
grep -a "stage-prof] win" "$LOG" | sed 's/.*stage-prof] //' | head -40
echo
echo "=== 慢循环告警 ==="
grep -a "stage-prof] slow loop" "$LOG" | sed 's/.*stage-prof] //' | head -20
echo
echo "=== GC 停顿（按耗时降序，前 20）==="
grep -a "GC took" "$LOG" | sed 's/.*] //' \
  | awk -F'[ .]' '{print $3, $0}' | sort -rn | head -20 | cut -d' ' -f2-
echo
echo "=== GC 停顿统计 ==="
/opt/conda/envs/mx/bin/python - <<'PY'
import re
L="/root/bench_results/stage_prof/server.log"
rows=[]
for line in open(L, errors="ignore"):
    m=re.search(r"GC took ([\d.]+)ms to complete\. Collected (\S+) objects \(out of (\d+)\) in GC generation (\d+)", line)
    if m:
        rows.append((float(m.group(1)), m.group(2), int(m.group(3)), int(m.group(4))))
print(f"  共 {len(rows)} 次 GC")
if rows:
    ms=[r[0] for r in rows]
    ms_s=sorted(ms)
    n=len(ms_s)
    print(f"  median {ms_s[n//2]:.1f}ms   p90 {ms_s[int(n*0.9)]:.1f}ms   max {ms_s[-1]:.1f}ms")
    print(f"  合计 GC 耗时 {sum(ms)/1000:.1f}s")
    from collections import Counter
    print(f"  按代: {dict(Counter(r[3] for r in rows))}")
    print(f"\n  最慢 10 次:")
    for m_,col,out,gen in sorted(rows, reverse=True)[:10]:
        print(f"    {m_:>10.1f}ms  gen{gen}  collected {col}/{out}")
PY
echo "DONE $(date +%H:%M:%S)"
