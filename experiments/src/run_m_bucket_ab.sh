#!/bin/bash
# 决定性 A/B：align32 vs align32_geometric，同一 M 范围。
#
# 步长 8 覆盖 1..2048：align32 的桶都是 32 的倍数、geometric 的桶是 2 的幂，
# 步长 8 都能命中，所以两臂都能走完各自的桶集，而 M 值数量只有 256 个。
# 每个冷桶一次完整 autotune（实测 7~9 s）。
#
# align32          -> 桶集 69 个，预期 ~9~10 分钟
# align32_geo      -> 桶集 13 个，预期 ~2 分钟
set -u

MM=/workspace/FlagGems/src/flag_gems/runtime/backend/_metax/ops/mm.py
PY=/opt/conda/envs/mx/bin/python
OUT=/root/bench_results/m_bucket_ab
mkdir -p "$OUT"

say() { echo "[$(date '+%H:%M:%S')] $*"; }

set_mode() {
  if [ "$1" = "align32" ]; then
    sed -i 's/strategy=\["align32_geometric"/strategy=["align32"/' "$MM"
  else
    sed -i 's/strategy=\["align32", "align32", "align32", "align32", "align32"\]/strategy=["align32_geometric", "align32", "align32", "align32", "align32"]/;
            s/strategy=\["align32", "align32", "align32"\]/strategy=["align32_geometric", "align32", "align32"]/;
            s/strategy=\["align32", "align32", "align32", "default"\]/strategy=["align32_geometric", "align32", "align32", "default"]/' "$MM"
  fi
  echo "    mm.py M-strategy: $(grep -o 'strategy=\["[a-z0-9_]*"' "$MM" | sort -u | head -3 | tr '\n' ' ')"
}

run_arm() {
  local ARM=$1
  say "======== ARM $ARM ========"
  set_mode "$ARM"
  rm -f /tmp/mb_$ARM.db
  FLAGGEMS_DB_URL="sqlite:////tmp/mb_$ARM.db" \
    MM_MAX=2048 MM_STEP=8 MM_LABEL="$ARM" \
    $PY -u /root/src/m_bucket_measure.py > "$OUT/$ARM.log" 2>&1
  grep -aE "tunings|total_bench_s|wall_s|\"M_swept\"" "$OUT/$ARM.log" | sed 's/^/    /'
  # 顺带报告桶集大小（纯数学，和 DB 无关）
  $PY - "$ARM" <<'PYEOF' | sed 's/^/    /'
import sys, math
arm = sys.argv[1]
def geo(k):
    if k == 0: return 0
    if k < 32: return 2 ** math.ceil(math.log2(k))
    if k <= 128: return math.ceil(k/32)*32
    return 2 ** math.ceil(math.log2(k))
def a32(k):
    if k == 0: return 0
    if k < 32: return 2 ** math.ceil(math.log2(k))
    return math.ceil(k/32)*32
fn = geo if arm != "align32" else a32
b = sorted({fn(m) for m in range(1, 2049)})
print(f"桶集大小 = {len(b)}")
PYEOF
}

run_arm align32
run_arm align32_geometric

# 恢复为修复版
$PY /root/src/patch_m_bucket.py "$MM" >/dev/null 2>&1 || set_mode align32_geometric
say "已恢复"

echo
echo "======== M 分桶上界 A/B 汇总 ========"
for A in align32 align32_geometric; do
  echo "--- $A ---"
  grep -aE '"tunings"|"total_bench_s"|"wall_s"|"by_kernel"|mm_kernel' "$OUT/$A.log" | sed 's/^/  /'
done
