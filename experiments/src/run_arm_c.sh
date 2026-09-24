#!/bin/bash
# ARM C —— 验证「linear 走通用 linear_kernel（无补丁）」是否也 autotune 风暴。
#
# 假设：autotune 风暴源于「key 含运行期 M」这一通用机制，而非我那个 patch。
#   mm_kernel_nt     key=["M","N","K","stride_am","stride_bk"]
#   linear_kernel    key=["M","N","K"]          <- ARM C 走这条
#   rms_norm         key=["N"]                  <- 白名单内，安全
# 若 ARM C 同样塌陷 => 机制是通用的，与 patch 无关 => 正确修法是改 autotune key。
#
# 与 ARM A/B 的差异：只有白名单（多了 linear），linear.py 是**原始版**
# （M>1 回落通用 kernel）。site-packages 当前已是原始版，无需切换。
#
# 防护：ARM B 跑了 5:40 仍未结束。这里加看门狗 —— 每 15s 读一次 engine 的
# generation 吞吐，连续 3 次 < 100 tok/s 即判定 STORM 并早停；bench 硬上限 10 分钟。
set -u

export PATH=/opt/conda/envs/mx/bin:$PATH
export VLLM_PLUGINS=fl
unset VLLM_FL_CUDAGRAPH_ONLY

PORT=9031
MODEL=/workspace/MiniCPM5-2B
SERVED=minicpm
OUT=/root/bench_results/arm_c
PY=/opt/conda/envs/mx/bin/python
DB=/root/.flaggems/config_cache/TunedConfig_metax_triton_3_0.db

ts() { date '+%H:%M:%S'; }
say() { echo "[$(ts)] $*"; }
mkdir -p "$OUT"

db_snap() {
  $PY - "$DB" "$1" <<'PYEOF'
import sqlite3, sys
db, out = sys.argv[1], sys.argv[2]
c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
tabs = [r[0] for r in c.execute("select name from sqlite_master where type='table'")]

def fam(t):
    for k in ("mm_kernel_nt", "mm_kernel_splitk", "mm_kernel_nn", "linear_kernel", "gemv_kernel"):
        if t.startswith(k):
            return k
    return None

from collections import Counter
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

cleanup_all() {
  say "TEARDOWN"
  # 注意：不要 pkill -f "run_arm_c" —— 本脚本自己的 cmdline 就含这个字符串，会自杀。
  for p in $(pgrep -f "vllm bench serve" 2>/dev/null); do kill -9 $p 2>/dev/null; done
  for p in $(pgrep -f "VLLM::EngineCore" 2>/dev/null); do kill -9 $p 2>/dev/null; done
  for p in $(pgrep -f "bin/vllm serve" 2>/dev/null); do kill -9 $p 2>/dev/null; done
  sleep 2
}
trap cleanup_all EXIT

say "PART 0 | 清理"
cleanup_all

say "快照 DB（前）"
db_snap "$OUT/db_before.txt"
cat "$OUT/db_before.txt" | sed 's/^/  before  /'

say "启动 server：白名单 = silu_and_mul,rms_norm,rotary_embedding,linear（linear.py = 原始版）"
export VLLM_FL_FLAGOS_WHITELIST=silu_and_mul,rms_norm,rotary_embedding,linear
export FLAG_GEMS_METAX_GEMV_TRACE="$OUT/gemv_trace.txt"
: > "$FLAG_GEMS_METAX_GEMV_TRACE"

cd /workspace/vllm-plugin-FL || exit 1
SERVER_LOG="$OUT/server.log"
: > "$SERVER_LOG"
nohup vllm serve $MODEL --port $PORT --served-model-name $SERVED \
  --gpu-memory-utilization 0.85 --max-model-len 131072 > "$SERVER_LOG" 2>&1 &
SRV=$!

READY=0
for _ in $(seq 1 60); do
  grep -qa "Application startup complete" "$SERVER_LOG" 2>/dev/null && { READY=1; break; }
  kill -0 "$SRV" 2>/dev/null || { say "server 退出"; tail -12 "$SERVER_LOG"; exit 1; }
  sleep 5
done
[ "$READY" = 1 ] || { say "就绪超时"; tail -12 "$SERVER_LOG"; exit 1; }
say "READY"

SRV_PID=$(ps -eo pid,cmd | grep "[v]llm serve" | awk '{print $1}' | head -1)
if tr '\0' '\n' < "/proc/$SRV_PID/environ" 2>/dev/null | grep -qF "VLLM_FL_FLAGOS_WHITELIST=silu_and_mul,rms_norm,rotary_embedding,linear"; then
  say "白名单已确认（含 linear）"
else
  say "警告：白名单未确认"
fi

say "启动 4k bench（硬上限 10 分钟，看门狗盯着 generation 吞吐）"
BENCH_LOG="$OUT/bench.log"
: > "$BENCH_LOG"
cd /workspace || exit 1
nohup vllm bench serve --backend vllm --model "$SERVED" --tokenizer "$MODEL" \
  --endpoint /v1/completions --host localhost --port "$PORT" \
  --dataset-name random --ignore-eos \
  --random-input-len 4096 --random-output-len 1024 --max-concurrency 64 --num-prompts 256 \
  > "$BENCH_LOG" 2>&1 &
BENCH=$!
say "bench pid=$BENCH"

VERDICT="TIMEOUT"
LOW=0
for i in $(seq 1 40); do
  sleep 15
  gt=$(tr '\r' '\n' < "$SERVER_LOG" | grep -ao "Avg generation throughput: [0-9.]*" | tail -1 | grep -ao "[0-9.]*$")
  waitn=$(tr '\r' '\n' < "$SERVER_LOG" | grep -ao "Waiting: [0-9]*" | tail -1 | grep -ao "[0-9]*$")
  if grep -qa "Total token throughput" "$BENCH_LOG" 2>/dev/null; then VERDICT="DONE"; break; fi
  if ! kill -0 "$BENCH" 2>/dev/null; then VERDICT="BENCH_EXIT"; break; fi
  if [ -n "$gt" ]; then
    if awk "BEGIN{exit !($gt < 100)}"; then LOW=$((LOW+1)); else LOW=0; fi
    say "  t=$((i*15))s gen=${gt} tok/s waiting=${waitn:-?}  (连续低值=$LOW)"
    if [ "$LOW" -ge 3 ]; then VERDICT="STORM"; break; fi
  else
    say "  t=$((i*15))s (尚未产出吞吐行)"
  fi
done

say "判定: $VERDICT"
[ "$VERDICT" = "STORM" ] && say "  => generation 吞吐持续 < 100 tok/s，与 ARM B 同型风暴"

say "快照 DB（后）"
db_snap "$OUT/db_after.txt"
echo "  --- DB 变化 ---"
paste "$OUT/db_before.txt" "$OUT/db_after.txt" 2>/dev/null | sed 's/^/  /'

say "linear 进入的形状数"
[ -s "$OUT/gemv_trace.txt" ] && { echo "  去重形状数: $(sort -u $OUT/gemv_trace.txt | wc -l)"; echo "  M 值: $(grep -o 'M=[0-9]*' $OUT/gemv_trace.txt | sort -t= -k2 -n -u | tr '\n' ' ')"; } || echo "  (空)"

echo
echo "================ ARM C 结论 ================"
echo "  verdict: $VERDICT"
grep -aE "Total token throughput|Mean TTFT|Mean TPOT" "$BENCH_LOG" 2>/dev/null | sed 's/^/  /' || echo "  (bench 未完成)"
echo
echo "  对照 ARM A(基线)  : 8876.55 tok/s / gen 819-2399 tok/s"
echo "  对照 ARM B(有补丁): 未完成 / gen 7.3-22.5 tok/s"
