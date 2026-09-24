#!/bin/bash
# 官方口径评测（性能 + 正确性），叠加 §4.10 验证过的 FlagOS 白名单。
# 相对 run_eval_all.sh 的差异只有一处：多了 VLLM_FL_FLAGOS_WHITELIST，
# 其余（不传 --compilation-config、gpu-mem 0.85、max-model-len 131072）保持评测口径。
# 目的：确认白名单在官方 case（4k 与 16k）下既提速又不损正确性。
set -u

export PATH=/opt/conda/envs/mx/bin:$PATH
export VLLM_PLUGINS=fl
unset VLLM_FL_CUDAGRAPH_ONLY
# §4.10：`mm` 未列入 => 回落原生实现，同时消除 autotune 长停顿（A/B 已验证）。
export VLLM_FL_FLAGOS_WHITELIST=silu_and_mul,rms_norm,rotary_embedding

PORT=9031
MODEL=/workspace/MiniCPM5-2B
SERVED=minicpm
OUT=${EVAL_OUT:-/root/bench_results/eval_official_wl}
SERVER_LOG=$OUT/server.log
BENCH_LOG=$OUT/bench.log
EVAL_LOG=$OUT/evalscope.log
PY=/opt/conda/envs/mx/bin/python

ts() { date '+%H:%M:%S'; }
say() { echo "[$(ts)] $*"; }

# 退出时收掉自己启动的 server 及其子进程（EngineCore）。
# 踩过的坑：`vllm serve` 的 EngineCore 是**子进程**，杀掉父进程后它会 re-parent
# 到 init 并继续占着显存，导致下一次启动报
# `Free memory ... is less than desired`。孤儿不会自己消失，必须显式收。
SERVER_TREE=""
cleanup_server() {
  [ -z "${SERVER_TREE:-}" ] && return 0
  say "TEARDOWN | 停止本轮 server（含 EngineCore 子进程）"
  /opt/conda/envs/mx/bin/python - "$SERVER_TREE" <<'PYEOF'
import os, signal, sys, time
roots = [int(x) for x in sys.argv[1].split() if x.isdigit()]
# 收集整棵子树（父/子关系可能在 re-parent 后丢失，故同时按 cmdline 兜底）
pids = set(roots)
changed = True
while changed:
    changed = False
    for pid in os.listdir('/proc'):
        if not pid.isdigit():
            continue
        try:
            ppid = int(open(f'/proc/{pid}/stat').read().split()[3])
        except Exception:
            continue
        if ppid in pids and int(pid) not in pids:
            pids.add(int(pid)); changed = True
for pid in pids:
    try:
        os.kill(pid, signal.SIGKILL)
    except Exception:
        pass
time.sleep(1)
left = [p for p in pids if os.path.isdir(f'/proc/{p}')]
print(f"  killed={sorted(pids)} still_alive={left or 'none'}")
PYEOF
}
trap cleanup_server EXIT

# 单实例锁。踩过的坑：两个实例并存时，先启动那个的 server 崩了、就绪轮询仍在
# grep 同一个 server.log；后启动的 server 把 "Application startup complete"
# 写进同一文件，于是**僵死的旧实例被唤醒**，两个 benchmark client 打了同一个
# server，两边结果互相污染且 bench.log 交错。宁可拒绝启动也不要出一份脏数据。
LOCK=/tmp/run_eval_whitelist.lock
exec 9>"$LOCK"
if ! flock -n 9; then
  say "已有另一个评测实例在跑（$LOCK 被占用），本实例退出。"
  say "如需强制重跑，先确认旧实例已停：pgrep -af run_eval_whitelist"
  exit 1
fi

mkdir -p "$OUT"

# 启动前必须确认 9031 端口没有别的 server 占着，否则新 server 起不来而
# benchmark 会连到旧 server 上，结果看着正常但其实是另一个配置。
#
# 归属判据（保守，宁漏勿误杀）：
#   1) 占用 9031 端口的进程 —— 本脚本自己用的端口，可认定是上一轮的我方 server；
#   2) ppid==1 的 `VLLM::EngineCore` 孤儿且 cwd 在 /workspace 下 —— 杀父进程不会
#      带走 EngineCore（它会 re-parent 到 init），它会**继续占着显存**导致新 server
#      `Free memory ... is less than desired` 而启动失败（已踩过一次）。
#      用 cwd 限定归属，避免误杀别的 session 的孤儿。
# 不匹配上面任何一条的进程一律不动，并打印出来供人工确认。
say "PART 0 | 清理 9031 端口上的旧 server 及其孤儿 EngineCore"
$PY - <<'PYEOF'
import os, signal

def read(path, binary=True):
    try:
        with open(path, 'rb') as f:
            data = f.read()
        return data if binary else data.decode(errors='ignore')
    except Exception:
        return None

me = os.getpid()
killed, skipped = [], []
for pid in os.listdir('/proc'):
    if not pid.isdigit() or int(pid) == me:
        continue
    cl = read(f'/proc/{pid}/cmdline')
    if not cl:
        continue
    cl = cl.decode(errors='ignore')
    is_server = '9031' in cl and 'vllm' in cl
    # cwd 是否在 /workspace 下（我们自己的工作目录）
    try:
        cwd = os.readlink(f'/proc/{pid}/cwd')
        in_ws = cwd.startswith('/workspace')
    except Exception:
        cwd, in_ws = '?', False
    if is_server or (cl.startswith('VLLM::EngineCore') and in_ws):
        try:
            os.kill(int(pid), signal.SIGKILL)
            killed.append((pid, cwd, cl[:40]))
        except Exception:
            pass

# 再报一遍仍活着的 vLLM 类进程，便于人工确认没有漏/误
alive = []
for pid in os.listdir('/proc'):
    if not pid.isdigit():
        continue
    cl = read(f'/proc/{pid}/cmdline')
    if cl and (b'VLLM::EngineCore' in cl or (b'vllm' in cl and b'serve' in cl)):
        try:
            cwd = os.readlink(f'/proc/{pid}/cwd')
        except Exception:
            cwd = '?'
        alive.append((pid, cwd))
print("  killed:", [k[0] for k in killed] or "none")
for pid, cwd, cl in killed:
    print(f"    - {pid} cwd={cwd} {cl}")
print("  仍存活的 vLLM 进程（未动）:", alive or "none")
PYEOF
sleep 3

# ---------------------------------------------------------------- PART A: serve
say "PART A | 启动 vllm serve（评测口径 + 白名单）"
cd /workspace/vllm-plugin-FL || exit 1
say "HEAD = $(git rev-parse --short HEAD) $(git log -1 --format=%s)"

# 就绪检测只看「本轮启动之后」写入的内容。
# 踩过的坑：直接 grep 整个 server.log 时，若同时存在另一个实例（或上一轮的
# 残留日志），会匹配到**别人**写的 "Application startup complete"，
# 于是自己的 server 明明没起来也当成 READY，后续 benchmark 直接连到别人的 server。
# 记录启动前的字节偏移，之后只从该偏移开始看。
: > "$SERVER_LOG"
OFFSET=0

nohup vllm serve $MODEL --port $PORT --served-model-name $SERVED \
  --gpu-memory-utilization 0.85 --max-model-len 131072 \
  > "$SERVER_LOG" 2>&1 &
SRV_LAUNCHED=$!
SERVER_TREE="$SRV_LAUNCHED"
say "server pid=$SRV_LAUNCHED"

READY=0
for _ in $(seq 1 360); do
  # tail -c +$((OFFSET+1)) 等价于「只取第 OFFSET+1 字节起的内容」
  if tail -c +$((OFFSET + 1)) "$SERVER_LOG" 2>/dev/null | grep -qa "Application startup complete"; then
    READY=1; break
  fi
  # 自己的 server 已经死了就别再干等（否则要空转满 60 分钟）
  if ! kill -0 "$SRV_LAUNCHED" 2>/dev/null; then
    say "PART A | 失败：server 进程已退出。根因如下："
    tail -c +$((OFFSET + 1)) "$SERVER_LOG" | tr '\r' '\n' | grep -aE "Error|error|ValueError|Traceback|RuntimeError" | tail -8
    exit 1
  fi
  sleep 10
done
if [ "$READY" != 1 ]; then
  say "PART A | 失败，尾部日志："; tail -c +$((OFFSET + 1)) "$SERVER_LOG" | tr '\r' '\n' | tail -15; exit 1
fi
say "PART A | READY"

# 核实白名单真的生效：只看 server 进程环境，不看启动命令（免得脚本写对了却没传下去）
SRV_PID=$(ps -eo pid,cmd | grep "[v]llm serve" | awk '{print $1}' | head -1)
if tr '\0' '\n' < "/proc/$SRV_PID/environ" 2>/dev/null | grep -q "VLLM_FL_FLAGOS_WHITELIST=silu_and_mul,rms_norm,rotary_embedding"; then
  say "PART A | 白名单已在 server 环境中确认"
else
  say "PART A | 警告：白名单未在 server 环境中找到，本结果不可用于对照"
fi
tr '\r' '\n' < "$SERVER_LOG" | grep -aE "Loading weights took|GPU KV cache size|Maximum concurrency" | head -5

# ---------------------------------------------------------------- PART B: 性能
say "PART B | 性能基准 (4k/16k, RUNS=4, SKIP_FIRST=1, 官方 2 个 case)"
cd /workspace || exit 1
# -u：重定向到文件时 Python 会缓冲 stdout，进程被信号带走会丢结果
$PY -u /workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py \
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

echo
echo "================ 官方口径汇总 ================"
echo "--- 性能（脚本自动丢首轮、取后 3 轮平均）---"
grep -a "^  Prefill=" "$BENCH_LOG" 2>/dev/null || echo "  未取到 Summary 行"
echo "--- 各内部轮 ---"
grep -a "Total token throughput" "$BENCH_LOG" 2>/dev/null
echo "--- 正确性 ---"
grep -aiE "accuracy|acc|score|level 3|math_500" "$EVAL_LOG" 2>/dev/null | tail -12
echo
echo "server log : $SERVER_LOG"
echo "bench log  : $BENCH_LOG"
echo "eval log   : $EVAL_LOG"
