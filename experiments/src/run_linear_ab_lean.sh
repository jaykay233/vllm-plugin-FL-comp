#!/bin/bash
# linear 白名单 e2e A/B/C —— 精简版：只跑 4k，每臂「1 次预热 + 1 次正式」。
#
#   A  白名单 = silu_and_mul,rms_norm,rotary_embedding   -> linear 走原生（基线）
#   B  白名单 + linear，linear.py 带 M>1->mm 补丁         -> linear 走 metax mm
#   C  白名单 + linear，linear.py 回落通用 linear_kernel  -> 对照组（分离补丁贡献）
#
# 为什么不直接跑官方 benchmark_throughput_serve.py：
#   它的 RUNS=4 / SKIP_FIRST=1 是模块级常量，改它就得动官方脚本；竞赛要求可复现，
#   不该动。这里直接调用它底层那条 `vllm bench serve`，自己做「预热 1 次 + 计 1 次」，
#   既拿到无偏的第一手数，又把 8 次 bench 压到 2 次。
#
# 注意：这是**决策用**的精简探测。最终对外数字仍应回到官方 4 轮口径复现。
set -u

export PATH=/opt/conda/envs/mx/bin:$PATH
export VLLM_PLUGINS=fl
unset VLLM_FL_CUDAGRAPH_ONLY

PORT=9031
MODEL=/workspace/MiniCPM5-2B
SERVED=minicpm
ROOT_OUT=/root/bench_results/linear_ab_lean
PY=/opt/conda/envs/mx/bin/python
WL_BASE=silu_and_mul,rms_norm,rotary_embedding
# 4k：input 4096 / output 1024 / concurrency 64 / prompts 256（与官方一致）
CASE_ARGS="--random-input-len 4096 --random-output-len 1024 --max-concurrency 64 --num-prompts 256"
CASE_NAME=4k

ts() { date '+%H:%M:%S'; }
say() { echo "[$(ts)] $*"; }

LINEAR_SRC=/opt/conda/envs/mx/lib/python3.12/site-packages/flag_gems/runtime/backend/_metax/ops/linear.py
LINEAR_PATCHED=/tmp/linear.py.patched
LINEAR_ORIG=/tmp/linear.py.bak

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

LOCK=/tmp/run_linear_ab_lean.lock
exec 9>"$LOCK"
if ! flock -n 9; then say "已有另一个实例在跑，退出。"; exit 1; fi
mkdir -p "$ROOT_OUT"

say "PART 0 | 清理 9031 上的旧 server 与 /workspace 下的孤儿 EngineCore"
$PY - <<'PYEOF'
import os, signal

def cmdline(pid):
    try:
        return open(f'/proc/{pid}/cmdline', 'rb').read().decode(errors='ignore')
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
    if ('9031' in cl and 'vllm' in cl) or (cl.startswith('VLLM::EngineCore') and cwd.startswith('/workspace')):
        try:
            os.kill(int(pid), signal.SIGKILL); killed.append(pid)
        except Exception:
            pass
print("  killed:", killed or "none")
PYEOF
sleep 3

bench_once() {
  # $1 = 输出文件
  cd /workspace || return 1
  vllm bench serve --backend vllm --model "$SERVED" --tokenizer "$MODEL" \
    --endpoint /v1/completions --host localhost --port "$PORT" \
    --dataset-name random --ignore-eos $CASE_ARGS > "$1" 2>&1
  return $?
}

run_arm() {
  local ARM=$1 WL=$2 LINMODE=$3 DESC=$4
  local OUT="$ROOT_OUT/$ARM"
  mkdir -p "$OUT"
  local SERVER_LOG="$OUT/server.log"

  say "================ ARM $ARM | $DESC ================"
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

  local READY=0
  for _ in $(seq 1 60); do
    if grep -qa "Application startup complete" "$SERVER_LOG" 2>/dev/null; then READY=1; break; fi
    if ! kill -0 "$SRV" 2>/dev/null; then
      say "ARM $ARM | server 退出，根因："
      tr '\r' '\n' < "$SERVER_LOG" | grep -aE "Error|ValueError|RuntimeError|Traceback" | tail -8
      return 1
    fi
    sleep 5
  done
  [ "$READY" = 1 ] || { say "ARM $ARM | 就绪超时"; tail -15 "$SERVER_LOG"; return 1; }
  say "ARM $ARM | READY"

  local SRV_PID
  SRV_PID=$(ps -eo pid,cmd | grep "[v]llm serve" | awk '{print $1}' | head -1)
  if tr '\0' '\n' < "/proc/$SRV_PID/environ" 2>/dev/null | grep -qF "VLLM_FL_FLAGOS_WHITELIST=$WL"; then
    say "ARM $ARM | 白名单已在 server 环境确认"
  else
    say "ARM $ARM | 警告：白名单未确认，本臂结果不可用"
  fi

  say "ARM $ARM | 预热 1 次（丢弃：吸收 prefix cache / graph 预热）"
  bench_once "$OUT/warmup.txt"
  say "ARM $ARM | 预热完成：$(grep -a 'Total token throughput' "$OUT/warmup.txt" | tail -1)"

  say "ARM $ARM | 正式 1 次"
  bench_once "$OUT/bench.log"
  grep -aE "Total token throughput|Mean TTFT|Mean TPOT" "$OUT/bench.log" | sed "s/^/  [$ARM] /"

  say "ARM $ARM | linear 是否进 FlagGems"
  if [ -s "$FLAG_GEMS_METAX_GEMV_TRACE" ]; then
    sort "$FLAG_GEMS_METAX_GEMV_TRACE" | uniq -c | sort -rn | head -6 | sed "s/^/  [$ARM] /"
  else
    say "ARM $ARM | trace 空 -> linear 未进 FlagGems"
  fi

  cleanup_server
  sleep 5
}

cp "$LINEAR_SRC" "$LINEAR_PATCHED" 2>/dev/null

run_arm A "$WL_BASE"        patched "linear 走原生（基线）"
run_arm B "$WL_BASE,linear" patched "linear 进 FlagGems + M>1->mm 补丁"
run_arm C "$WL_BASE,linear" orig    "linear 进 FlagGems，无补丁（通用 kernel）"

cp "$LINEAR_PATCHED" "$LINEAR_SRC"

echo
echo "================ linear A/B/C 精简汇总（$CASE_NAME）================"
for A in A B C; do
  echo "--- ARM $A ---"
  grep -aE "Total token throughput|Mean TTFT|Mean TPOT" "$ROOT_OUT/$A/bench.log" 2>/dev/null | sed 's/^/  /' || echo "  (无)"
done
