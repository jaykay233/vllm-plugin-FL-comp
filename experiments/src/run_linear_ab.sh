#!/bin/bash
# linear 白名单 e2e A/B/C —— 回答「能否把 linear 收回 FlagGems 而不掉性能」。
#
#   A  白名单 = silu_and_mul,rms_norm,rotary_embedding          -> linear 走原生（当前基线）
#   B  白名单 + linear，linear.py 带 M>1->mm 补丁                -> linear 走 metax mm
#   C  白名单 + linear，linear.py 回落到通用 linear_kernel       -> 对照组：分离补丁的贡献
#
# B vs A：能否把 linear 收回 FlagGems（合规友好方向）
# B vs C：补丁本身值多少
#
# 判决依据是官方口径 benchmark（4k + 16k 两个 case），并且用
# FLAG_GEMS_METAX_GEMV_TRACE 直接证明 linear 到底有没有走到 FlagGems 里。
set -u

export PATH=/opt/conda/envs/mx/bin:$PATH
export VLLM_PLUGINS=fl
unset VLLM_FL_CUDAGRAPH_ONLY

PORT=9031
MODEL=/workspace/MiniCPM5-2B
SERVED=minicpm
ROOT_OUT=/root/bench_results/linear_ab
PY=/opt/conda/envs/mx/bin/python
WL_BASE=silu_and_mul,rms_norm,rotary_embedding

ts() { date '+%H:%M:%S'; }
say() { echo "[$(ts)] $*"; }

LINEAR_SRC=/opt/conda/envs/mx/lib/python3.12/site-packages/flag_gems/runtime/backend/_metax/ops/linear.py
LINEAR_PATCHED=/tmp/linear.py.patched      # 带 M>1->mm 补丁
LINEAR_ORIG=/tmp/linear.py.bak             # 原始（M>1 回落通用）

SERVER_TREE=""
cleanup_server() {
  [ -z "${SERVER_TREE:-}" ] && return 0
  say "TEARDOWN | 停止 server（含 EngineCore 子进程）"
  $PY - "$SERVER_TREE" <<'PYEOF'
import os, signal, sys, time
roots = [int(x) for x in sys.argv[1].split() if x.isdigit()]
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
print(f"  killed={sorted(pids)} still_alive={[p for p in pids if os.path.isdir(f'/proc/{p}')] or 'none'}")
PYEOF
  SERVER_TREE=""
}
trap cleanup_server EXIT

LOCK=/tmp/run_linear_ab.lock
exec 9>"$LOCK"
if ! flock -n 9; then
  say "已有另一个实例在跑（$LOCK），退出。"; exit 1
fi

mkdir -p "$ROOT_OUT"

# ---- 端口/孤儿清理（同 run_eval_whitelist.sh 的保守判据）
say "PART 0 | 清理 9031 上的旧 server 与 /workspace 下的孤儿 EngineCore"
$PY - <<'PYEOF'
import os, signal

def cmdline(pid):
    try:
        with open(f'/proc/{pid}/cmdline', 'rb') as f:
            return f.read().decode(errors='ignore')
    except Exception:
        return None

me = os.getpid()
killed = []
for pid in os.listdir('/proc'):
    if not pid.isdigit() or int(pid) == me:
        continue
    cl = cmdline(pid)
    if not cl:
        continue
    try:
        cwd = os.readlink(f'/proc/{pid}/cwd')
    except Exception:
        cwd = '?'
    in_ws = cwd.startswith('/workspace')
    if ('9031' in cl and 'vllm' in cl) or (cl.startswith('VLLM::EngineCore') and in_ws):
        try:
            os.kill(int(pid), signal.SIGKILL); killed.append(pid)
        except Exception:
            pass
print("  killed:", killed or "none")
PYEOF
sleep 3

run_arm() {
  local ARM=$1 WL=$2 LINMODE=$3 DESC=$4
  local OUT="$ROOT_OUT/$ARM"
  mkdir -p "$OUT"
  local SERVER_LOG="$OUT/server.log" BENCH_LOG="$OUT/bench.log"

  say "================ ARM $ARM | $DESC ================"
  # 按 arm 切换 linear.py（补丁 / 原始）
  case "$LINMODE" in
    patched) cp "$LINEAR_PATCHED" "$LINEAR_SRC" ;;
    orig)    cp "$LINEAR_ORIG"    "$LINEAR_SRC" ;;
  esac
  say "ARM $ARM | linear.py = $LINMODE"

  export VLLM_FL_FLAGOS_WHITELIST="$WL"
  export FLAG_GEMS_METAX_GEMV_TRACE="$OUT/gemv_trace.txt"
  : > "$FLAG_GEMS_METAX_GEMV_TRACE"

  cd /workspace/vllm-plugin-FL || return 1
  : > "$SERVER_LOG"
  nohup vllm serve $MODEL --port $PORT --served-model-name $SERVED \
    --gpu-memory-utilization 0.85 --max-model-len 131072 \
    > "$SERVER_LOG" 2>&1 &
  SERVER_TREE=$!
  local SRV=$SERVER_TREE
  say "ARM $ARM | server pid=$SRV"

  local READY=0
  for _ in $(seq 1 360); do
    if tail -c +1 "$SERVER_LOG" 2>/dev/null | grep -qa "Application startup complete"; then
      READY=1; break
    fi
    if ! kill -0 "$SRV" 2>/dev/null; then
      say "ARM $ARM | server 进程已退出，根因："
      tr '\r' '\n' < "$SERVER_LOG" | grep -aE "Error|ValueError|RuntimeError|Traceback" | tail -8
      return 1
    fi
    sleep 10
  done
  [ "$READY" = 1 ] || { say "ARM $ARM | 就绪超时"; tail -15 "$SERVER_LOG"; return 1; }
  say "ARM $ARM | READY"

  # 核实白名单确实进了 server 环境
  local SRV_PID
  SRV_PID=$(ps -eo pid,cmd | grep "[v]llm serve" | awk '{print $1}' | head -1)
  if tr '\0' '\n' < "/proc/$SRV_PID/environ" 2>/dev/null | grep -qF "VLLM_FL_FLAGOS_WHITELIST=$WL"; then
    say "ARM $ARM | 白名单已在 server 环境中确认"
  else
    say "ARM $ARM | 警告：白名单未在 server 环境确认，本臂结果不可用"
  fi
  tr '\r' '\n' < "$SERVER_LOG" | grep -aE "Loading weights took|GPU KV cache size|Maximum concurrency" | head -4

  say "ARM $ARM | benchmark（官方 4k + 16k）"
  cd /workspace || return 1
  $PY -u /workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py \
    --model "$MODEL" --served-model-name "$SERVED" --port "$PORT" \
    --test-cases '[[4096,1024,64,256],[16384,1024,64,128]]' \
    2>&1 | tee "$BENCH_LOG"

  say "ARM $ARM | 本轮结果"
  grep -a "^  Prefill=" "$BENCH_LOG" 2>/dev/null | sed "s/^/  [$ARM] /"
  grep -a "Total token throughput" "$BENCH_LOG" 2>/dev/null | sed "s/^/  [$ARM] /"

  say "ARM $ARM | linear 是否走到 FlagGems（trace 的 via= 计数）"
  if [ -s "$FLAG_GEMS_METAX_GEMV_TRACE" ]; then
    sort "$FLAG_GEMS_METAX_GEMV_TRACE" | uniq -c | sort -rn | head -8 | sed "s/^/  [$ARM] /"
  else
    say "ARM $ARM | trace 为空 -> linear 未进 FlagGems（本臂 linear 走原生）"
  fi

  cleanup_server
  sleep 5
}

# 保存带补丁版本，之后按 arm 切换
cp "$LINEAR_SRC" "$LINEAR_PATCHED" 2>/dev/null

run_arm A "$WL_BASE"                 patched "linear 走原生（基线；补丁在但未触发）"
run_arm B "$WL_BASE,linear"          patched "linear 进 FlagGems + M>1->mm 补丁"
run_arm C "$WL_BASE,linear"          orig    "linear 进 FlagGems，无补丁（回落通用 kernel）"

# 收尾：恢复成带补丁版本
cp "$LINEAR_PATCHED" "$LINEAR_SRC"

echo
echo "================ linear A/B/C 汇总 ================"
for A in A B C; do
  echo "--- ARM $A ---"
  grep -a "^  Prefill=" "$ROOT_OUT/$A/bench.log" 2>/dev/null || echo "  (无结果)"
done
