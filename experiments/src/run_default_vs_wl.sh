#!/bin/bash
# 判断白名单是否还能去掉 —— 这决定合规上「是否还在切换算子」。
#
#   ARM G  flagos_whitelist: []            -> 零算子选择，全部走 FlagGems（prefer: flagos 默认）
#   ARM H  flagos_whitelist: 3 个融合算子  -> 出厂默认（现在的提交内容）
#
# 两臂都带 mm.py 的 align32 修复。各跑 4k 一次（诊断用）。
# 看门狗只在 Running>0 时累计低值 —— 否则会把客户端 tokenize 窗口误判成停顿。
set -u

export PATH=/opt/conda/envs/mx/bin:$PATH
export VLLM_PLUGINS=fl
unset VLLM_FL_CUDAGRAPH_ONLY VLLM_FL_FLAGOS_WHITELIST VLLM_FL_FLAGOS_BLACKLIST

PORT=9031
MODEL=/workspace/MiniCPM5-2B
SERVED=minicpm
OUT=/root/bench_results/default_vs_wl
PY=/opt/conda/envs/mx/bin/python
DB=/root/.flaggems/config_cache/TunedConfig_metax_triton_3_0.db
YAML=/workspace/vllm-plugin-FL/vllm_fl/dispatch/config/metax.yaml

mkdir -p "$OUT"
say() { echo "[$(date '+%H:%M:%S')] $*"; }

SERVER_PID=""
BENCH_PID=""
kill_tree() { [ -n "${1:-}" ] && $PY /root/src/kill_tree.py "$1" 2>/dev/null; }
trap 'kill_tree "$BENCH_PID"; kill_tree "$SERVER_PID"; $PY /root/src/set_wl_mode.py whitelist >/dev/null 2>&1; say "已清理并恢复出厂白名单"' EXIT

bench_4k() {
  local ARM=$1
  local LOG="$OUT/$ARM/4k.log" SLOG="$OUT/$ARM/server.log"
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
  local ARM=$1 MODE=$2 DESC=$3
  mkdir -p "$OUT/$ARM"
  say "======== ARM $ARM | $DESC ========"
  $PY /root/src/set_wl_mode.py "$MODE" | sed 's/^/    /'
  cd /root/src && $PY wl_probe.py "$ARM" 2>&1 | grep -aE "^\[" | sed 's/^/    /'
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

  bench_4k "$ARM"
  $PY /root/src/db_fam_snap.py "$DB" "$OUT/$ARM/db_after.txt"
  kill_tree "$SERVER_PID"; SERVER_PID=""
  sleep 3
}

run_arm G "off"       "零算子选择：全走 FlagGems（prefer: flagos）"
run_arm H "whitelist" "出厂白名单：只有 3 个融合算子走 FlagGems"

echo
echo "======== 默认 vs 白名单 汇总 ========"
for A in G H; do
  echo "--- ARM $A ---"
  grep -aE "Total token throughput|Mean TTFT|Mean TPOT" "$OUT/$A/4k.log" 2>/dev/null | sed 's/^/  /'
  echo "  DB before -> after:"
  paste "$OUT/$A/db_before.txt" "$OUT/$A/db_after.txt" 2>/dev/null | sed 's/^/    /'
done
echo
echo "  metax.yaml 当前: $(grep -A1 'flagos_whitelist' $YAML | tr '\n' ' ')"
