#!/bin/bash
# 验证 strategy=align32 修复：同一配置下「无修复」应风暴、「有修复」应健康。
#
# 白名单 = silu_and_mul,rms_norm,rotary_embedding,linear,mm
#   - 含 mm 是关键：修复前 mm 的 tuner 用恒等策略、按原始 M 建 key
#     => 每步 autotune => 风暴（已验证于 ARM B / 默认 prefer:flagos）
#
#   D0  strategy 回退（原始 mm.py）  -> 预期 STORM
#   D   strategy=align32（修复版）   -> 预期健康
#
# 看门狗：连续 3 次 gen < 100 tok/s 判 STORM 并早停；bench 硬上限 10 分钟。
set -u

export PATH=/opt/conda/envs/mx/bin:$PATH
export VLLM_PLUGINS=fl
unset VLLM_FL_CUDAGRAPH_ONLY

PORT=9031
MODEL=/workspace/MiniCPM5-2B
SERVED=minicpm
ROOT_OUT=/root/bench_results/mm_strategy_ab
PY=/opt/conda/envs/mx/bin/python
DB=/root/.flaggems/config_cache/TunedConfig_metax_triton_3_0.db

MM_SRC=/opt/conda/envs/mx/lib/python3.12/site-packages/flag_gems/runtime/backend/_metax/ops/mm.py
MM_FIXED=/tmp/mm.py.fixed
TS=$(cat /tmp/mm_strategy_ts)
MM_ORIG=/tmp/mm.py.nostrategy.$TS.bak

ts() { date '+%H:%M:%S'; }
say() { echo "[$(ts)] $*"; }
mkdir -p "$ROOT_OUT"

db_snap() {
  $PY - "$DB" "$1" <<'PYEOF'
import sqlite3, sys
from collections import Counter
db, out = sys.argv[1], sys.argv[2]
c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
tabs = [r[0] for r in c.execute("select name from sqlite_master where type='table'")]

def fam(t):
    for k in ("mm_kernel_nt", "mm_kernel_splitk", "mm_kernel_nn", "mm_kernel_general"):
        if t.startswith(k):
            return k
    return None

cnt, rows = Counter(), Counter()
for t in tabs:
    f = fam(t)
    if not f:
        continue
    cnt[f] += 1
    try:
        rows[f] += c.execute(f'select count(*) from "{t}"').fetchone()[0]
    except Exception:
        pass
with open(out, "w") as fh:
    for f in sorted(set(list(cnt) + list(rows))):
        fh.write(f"{f}\ttables={cnt[f]}\trows={rows[f]}\n")
PYEOF
}

kill_all() {
  for p in $(pgrep -f "vllm bench serve" 2>/dev/null); do kill -9 $p 2>/dev/null; done
  for p in $(pgrep -f "VLLM::EngineCore" 2>/dev/null); do kill -9 $p 2>/dev/null; done
  for p in $(pgrep -f "bin/vllm serve" 2>/dev/null); do kill -9 $p 2>/dev/null; done
  sleep 2
}
trap kill_all EXIT

run_arm() {
  local ARM=$1 MODE=$2 DESC=$3
  local OUT="$ROOT_OUT/$ARM"
  mkdir -p "$OUT"
  say "================ ARM $ARM | $DESC ================"

  if [ "$MODE" = "fixed" ]; then cp "$MM_FIXED" "$MM_SRC"; else cp "$MM_ORIG" "$MM_SRC"; fi
  say "ARM $ARM | mm.py 策略 = $MODE"

  export VLLM_FL_FLAGOS_WHITELIST=silu_and_mul,rms_norm,rotary_embedding,linear,mm

  kill_all
  db_snap "$OUT/db_before.txt"

  cd /workspace/vllm-plugin-FL || return 1
  local SERVER_LOG="$OUT/server.log"
  : > "$SERVER_LOG"
  nohup vllm serve $MODEL --port $PORT --served-model-name $SERVED \
    --gpu-memory-utilization 0.85 --max-model-len 131072 > "$SERVER_LOG" 2>&1 &
  local SRV=$!

  local READY=0
  for _ in $(seq 1 90); do
    grep -qa "Application startup complete" "$SERVER_LOG" 2>/dev/null && { READY=1; break; }
    kill -0 "$SRV" 2>/dev/null || { say "ARM $ARM | server 退出"; tail -10 "$SERVER_LOG"; return 1; }
    sleep 5
  done
  [ "$READY" = 1 ] || { say "ARM $ARM | 就绪超时（可能启动阶段就在 autotune 风暴里）"; return 1; }
  say "ARM $ARM | READY"

  local BENCH_LOG="$OUT/bench.log"
  : > "$BENCH_LOG"
  cd /workspace || return 1
  nohup vllm bench serve --backend vllm --model "$SERVED" --tokenizer "$MODEL" \
    --endpoint /v1/completions --host localhost --port "$PORT" \
    --dataset-name random --ignore-eos \
    --random-input-len 4096 --random-output-len 1024 --max-concurrency 64 --num-prompts 256 \
    > "$BENCH_LOG" 2>&1 &
  local BENCH=$!
  say "ARM $ARM | bench pid=$BENCH"

  local VERDICT="TIMEOUT" LOW=0
  for i in $(seq 1 40); do
    sleep 15
    local gt waitn
    gt=$(tr '\r' '\n' < "$SERVER_LOG" | grep -ao "Avg generation throughput: [0-9.]*" | tail -1 | grep -ao "[0-9.]*$")
    waitn=$(tr '\r' '\n' < "$SERVER_LOG" | grep -ao "Waiting: [0-9]*" | tail -1 | grep -ao "[0-9]*$")
    grep -qa "Total token throughput" "$BENCH_LOG" 2>/dev/null && { VERDICT="DONE"; break; }
    kill -0 "$BENCH" 2>/dev/null || { VERDICT="BENCH_EXIT"; break; }
    if [ -n "$gt" ]; then
      if awk "BEGIN{exit !($gt < 100)}"; then LOW=$((LOW+1)); else LOW=0; fi
      say "  t=$((i*15))s gen=${gt} tok/s waiting=${waitn:-?} low=$LOW"
      [ "$LOW" -ge 3 ] && { VERDICT="STORM"; break; }
    else
      say "  t=$((i*15))s (尚无吞吐行)"
    fi
  done

  say "ARM $ARM | 判定=$VERDICT"
  db_snap "$OUT/db_after.txt"
  paste "$OUT/db_before.txt" "$OUT/db_after.txt" 2>/dev/null | sed 's/^/    /'
  grep -aE "Total token throughput|Mean TTFT|Mean TPOT" "$BENCH_LOG" 2>/dev/null | sed 's/^/    /'

  kill_all
  sleep 3
}

cp "$MM_SRC" "$MM_FIXED" 2>/dev/null

run_arm D0 reverted "白名单含 mm，策略未修复（预期 STORM）"
run_arm D  fixed    "白名单含 mm，strategy=align32（预期健康）"

cp "$MM_FIXED" "$MM_SRC"

echo
echo "================ mm strategy A/B 汇总 ================"
for A in D0 D; do
  echo "--- ARM $A ---"
  grep -aE "判定|Total token throughput|Mean TTFT|Mean TPOT|mm_kernel" "$ROOT_OUT/$A/bench.log" 2>/dev/null | sed 's/^/  /'
  grep -a "判定" "$ROOT_OUT/$A/server.log" >/dev/null 2>&1
done
grep -a "判定=" /root/bench_results/mm_strategy_ab_driver.log 2>/dev/null | sed 's/^/  /'
