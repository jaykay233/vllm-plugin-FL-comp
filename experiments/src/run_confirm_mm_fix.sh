#!/bin/bash
# 端到端确认：align32 修复后，白名单含 linear/mm 是否可用（不风暴）且不劣于生产白名单。
#
#   ARM E  白名单 = silu_and_mul,rms_norm,rotary_embedding            （生产参考）
#   ARM F  白名单 = silu_and_mul,rms_norm,rotary_embedding,linear,mm   （含修复后的 linear/mm）
#
# 每臂跑 4k 与 16k，各 1 次预热 + 1 次计量（诊断用，非官方 4 轮制）。
# 看门狗：连续 3 次 gen < 100 tok/s 判 STORM 早停。DB 前后快照。
set -u

export PATH=/opt/conda/envs/mx/bin:$PATH
export VLLM_PLUGINS=fl
unset VLLM_FL_CUDAGRAPH_ONLY

PORT=9031
MODEL=/workspace/MiniCPM5-2B
SERVED=minicpm
OUT=/root/bench_results/confirm_mm_fix
PY=/opt/conda/envs/mx/bin/python
DB=/root/.flaggems/config_cache/TunedConfig_metax_triton_3_0.db
LOCK=/tmp/run_confirm_mm_fix.lock

exec 9>"$LOCK"
flock -n 9 || { echo "已有实例在跑，退出"; exit 1; }

mkdir -p "$OUT"
say() { echo "[$(date '+%H:%M:%S')] $*"; }

db_snap() {
  $PY /root/src/db_fam_snap.py "$DB" "$1"
}

# 递归终止某个进程及其所有后代（只针对我们自己启动的 PID）
kill_tree() {
  [ -n "${1:-}" ] || return 0
  $PY /root/src/kill_tree.py "$1" 2>/dev/null
}

SERVER_PID=""
BENCH_PID=""
trap 'kill_tree "$BENCH_PID"; kill_tree "$SERVER_PID"; say "已清理"' EXIT

bench_case() {
  local ARM=$1 IN=$2 OTOK=$3 CONC=$4 N=$5 TAG=$6
  local LOG="$OUT/$ARM/$TAG.log"
  local SLOG="$OUT/$ARM/server.log"
  local MARK
  MARK=$(stat -c%s "$SLOG" 2>/dev/null || echo 0)

  cd /workspace || return 1
  nohup vllm bench serve --backend vllm --model "$SERVED" --tokenizer "$MODEL" \
    --endpoint /v1/completions --host localhost --port "$PORT" \
    --dataset-name random --ignore-eos \
    --random-input-len "$IN" --random-output-len "$OTOK" \
    --max-concurrency "$CONC" --num-prompts "$N" \
    > "$LOG" 2>&1 &
  BENCH_PID=$!

  # 看门狗：本目录跑的是服务端日志里 10s 一条的吞吐行。
  # 注意 bench 客户端先要 ~50s tokenize 提示词，这期间 server 空闲、
  # 打的是 gen=0.0/Running=0 —— 那**不是**停顿。
  # 真正的引擎停顿是 gen=0 但 Running>0（请求挂着不出 token）。
  # 因此只在 Running>0 时才累计低值，避免把客户端预热误判成 STORM。
  local VERDICT="TIMEOUT" LOW=0 i
  for i in $(seq 1 80); do
    sleep 15
    grep -qa "Total token throughput" "$LOG" 2>/dev/null && { VERDICT="DONE"; break; }
    kill -0 "$BENCH_PID" 2>/dev/null || { VERDICT="EXIT"; break; }
    local gt rn
    gt=$(tail -c +$((MARK+1)) "$SLOG" 2>/dev/null | tr '\r' '\n' \
         | grep -ao "Avg generation throughput: [0-9.]*" | tail -1 | grep -ao "[0-9.]*$")
    rn=$(tail -c +$((MARK+1)) "$SLOG" 2>/dev/null | tr '\r' '\n' \
         | grep -ao "Running: [0-9]*" | tail -1 | grep -ao "[0-9]*$")
    if [ -n "$gt" ] && [ -n "$rn" ] && [ "$rn" -gt 0 ] 2>/dev/null; then
      awk "BEGIN{exit !($gt < 100)}" && LOW=$((LOW+1)) || LOW=0
    fi
    [ "$LOW" -ge 3 ] && { VERDICT="STORM"; break; }
  done
  kill_tree "$BENCH_PID"; BENCH_PID=""
  echo "    $TAG -> $VERDICT"
  grep -aE "Total token throughput|Mean TTFT|Mean TPOT" "$LOG" 2>/dev/null | sed 's/^/      /'
}

run_arm() {
  local ARM=$1 WL=$2 DESC=$3
  mkdir -p "$OUT/$ARM"
  say "================ ARM $ARM | $DESC ================"
  export VLLM_FL_FLAGOS_WHITELIST="$WL"
  say "ARM $ARM | whitelist=$WL"
  db_snap "$OUT/$ARM/db_before.txt"

  local SLOG="$OUT/$ARM/server.log"
  : > "$SLOG"
  cd /workspace/vllm-plugin-FL || return 1
  nohup vllm serve "$MODEL" --port "$PORT" --served-model-name "$SERVED" \
    --gpu-memory-utilization 0.85 --max-model-len 131072 > "$SLOG" 2>&1 &
  SERVER_PID=$!

  local READY=0 OFFSET=0
  for _ in $(seq 1 120); do
    kill -0 "$SERVER_PID" 2>/dev/null || { say "ARM $ARM | 启动即退出"; tail -12 "$SLOG"; return 1; }
    [ -s "$SLOG" ] && break
    sleep 5
  done
  OFFSET=$(stat -c%s "$SLOG" 2>/dev/null || echo 0)
  for _ in $(seq 1 240); do
    tail -c +$((OFFSET+1)) "$SLOG" 2>/dev/null | grep -qa "Application startup complete" && { READY=1; break; }
    kill -0 "$SERVER_PID" 2>/dev/null || { say "ARM $ARM | server 中途退出"; tail -12 "$SLOG"; return 1; }
    sleep 5
  done
  [ "$READY" = 1 ] || { say "ARM $ARM | 就绪超时"; return 1; }
  say "ARM $ARM | READY"

  bench_case "$ARM" 4096  1024 64 256 "4k_warmup"
  bench_case "$ARM" 4096  1024 64 256 "4k"
  bench_case "$ARM" 16384 1024 64 128 "16k_warmup"
  bench_case "$ARM" 16384 1024 64 128 "16k"

  db_snap "$OUT/$ARM/db_after.txt"
  kill_tree "$SERVER_PID"; SERVER_PID=""
  sleep 3
}

run_arm E "silu_and_mul,rms_norm,rotary_embedding" "生产白名单（参考）"
run_arm F "silu_and_mul,rms_norm,rotary_embedding,linear,mm" "含修复后的 linear+mm"

echo
echo "================ 端到端确认汇总 ================"
for A in E F; do
  echo "--- ARM $A ---"
  for C in 4k 16k; do
    printf "  %-4s " "$C"
    grep -aE "Total token throughput|Mean TTFT|Mean TPOT" "$OUT/$A/$C.log" 2>/dev/null | tr '\n' ' '
    echo
  done
  echo "  DB before|after:"
  paste "$OUT/$A/db_before.txt" "$OUT/$A/db_after.txt" 2>/dev/null | sed 's/^/    /'
done
