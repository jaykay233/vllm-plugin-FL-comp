#!/bin/bash
# 精简确认：白名单含修复后的 linear/mm 是否风暴、是否劣于生产白名单。
#
#   ARM E  白名单 = silu_and_mul,rms_norm,rotary_embedding
#   ARM F  白名单 = silu_and_mul,rms_norm,rotary_embedding,linear,mm
#
# 只跑 4k / 各 1 次（诊断用，非官方 4 轮制）。
# 看门狗只在 Running>0 时累计低值 —— 否则会把客户端 tokenize 窗口(server 空闲)
# 误判成引擎停顿。
set -u

export PATH=/opt/conda/envs/mx/bin:$PATH
export VLLM_PLUGINS=fl
unset VLLM_FL_CUDAGRAPH_ONLY

PORT=9031
MODEL=/workspace/MiniCPM5-2B
SERVED=minicpm
OUT=/root/bench_results/confirm_lean
PY=/opt/conda/envs/mx/bin/python
DB=/root/.flaggems/config_cache/TunedConfig_metax_triton_3_0.db

mkdir -p "$OUT"
say() { echo "[$(date '+%H:%M:%S')] $*"; }

SERVER_PID=""
BENCH_PID=""
kill_tree() { [ -n "${1:-}" ] && $PY /root/src/kill_tree.py "$1" 2>/dev/null; }
trap 'kill_tree "$BENCH_PID"; kill_tree "$SERVER_PID"; say "已清理"' EXIT

bench_4k() {
  local ARM=$1 TAG=$2
  local LOG="$OUT/$ARM/$TAG.log" SLOG="$OUT/$ARM/server.log"
  local MARK; MARK=$(stat -c%s "$SLOG" 2>/dev/null || echo 0)

  cd /workspace || return 1
  nohup vllm bench serve --backend vllm --model "$SERVED" --tokenizer "$MODEL" \
    --endpoint /v1/completions --host localhost --port "$PORT" \
    --dataset-name random --ignore-eos \
    --random-input-len 4096 --random-output-len 1024 \
    --max-concurrency 64 --num-prompts 256 > "$LOG" 2>&1 &
  BENCH_PID=$!

  local VERDICT="TIMEOUT" LOW=0 i
  for i in $(seq 1 60); do
    sleep 15
    grep -qa "Total token throughput" "$LOG" 2>/dev/null && { VERDICT="DONE"; break; }
    kill -0 "$BENCH_PID" 2>/dev/null || { VERDICT="EXIT"; break; }
    local gt rn
    gt=$(tail -c +$((MARK+1)) "$SLOG" 2>/dev/null | tr '\r' '\n' | grep -ao "Avg generation throughput: [0-9.]*" | tail -1 | grep -ao "[0-9.]*$")
    rn=$(tail -c +$((MARK+1)) "$SLOG" 2>/dev/null | tr '\r' '\n' | grep -ao "Running: [0-9]*" | tail -1 | grep -ao "[0-9]*$")
    if [ -n "$gt" ] && [ -n "$rn" ] && [ "$rn" -gt 0 ] 2>/dev/null; then
      awk "BEGIN{exit !($gt < 100)}" && LOW=$((LOW+1)) || LOW=0
    fi
    [ "$LOW" -ge 3 ] && { VERDICT="STORM(gen<100且Running>0×3)"; break; }
  done
  kill_tree "$BENCH_PID"; BENCH_PID=""
  echo "    4k -> $VERDICT"
  grep -aE "Total token throughput|Mean TTFT|Mean TPOT" "$LOG" 2>/dev/null | sed 's/^/      /'
}

run_arm() {
  local ARM=$1 WL=$2 DESC=$3
  mkdir -p "$OUT/$ARM"
  say "======== ARM $ARM | $DESC ========"
  export VLLM_FL_FLAGOS_WHITELIST="$WL"
  $PY /root/src/db_fam_snap.py "$DB" "$OUT/$ARM/db_before.txt"

  local SLOG="$OUT/$ARM/server.log"; : > "$SLOG"
  cd /workspace/vllm-plugin-FL || return 1
  nohup vllm serve "$MODEL" --port "$PORT" --served-model-name "$SERVED" \
    --gpu-memory-utilization 0.85 --max-model-len 131072 > "$SLOG" 2>&1 &
  SERVER_PID=$!

  local READY=0 OFFSET=0
  for _ in $(seq 1 120); do
    kill -0 "$SERVER_PID" 2>/dev/null || { say "ARM $ARM 启动即退出"; tail -12 "$SLOG"; return 1; }
    [ -s "$SLOG" ] && break; sleep 5
  done
  OFFSET=$(stat -c%s "$SLOG" 2>/dev/null || echo 0)
  for _ in $(seq 1 180); do
    tail -c +$((OFFSET+1)) "$SLOG" 2>/dev/null | grep -qa "Application startup complete" && { READY=1; break; }
    kill -0 "$SERVER_PID" 2>/dev/null || { say "ARM $ARM server 中途退出"; tail -12 "$SLOG"; return 1; }
    sleep 5
  done
  [ "$READY" = 1 ] || { say "ARM $ARM 就绪超时"; return 1; }
  say "ARM $ARM READY"

  bench_4k "$ARM" "4k"
  $PY /root/src/db_fam_snap.py "$DB" "$OUT/$ARM/db_after.txt"
  kill_tree "$SERVER_PID"; SERVER_PID=""
  sleep 3
}

run_arm E "silu_and_mul,rms_norm,rotary_embedding" "生产白名单"
run_arm F "silu_and_mul,rms_norm,rotary_embedding,linear,mm" "含修复后 linear+mm"

echo
echo "======== 精简确认汇总 ========"
for A in E F; do
  echo "--- ARM $A ---"
  grep -aE "Total token throughput|Mean TTFT|Mean TPOT" "$OUT/$A/4k.log" 2>/dev/null | sed 's/^/  /'
  echo "  DB before -> after:"
  paste "$OUT/$A/db_before.txt" "$OUT/$A/db_after.txt" 2>/dev/null | sed 's/^/    /'
done
