#!/usr/bin/env bash
# Interleaved A/B: cudagraph-only vs compile, SAME prompt, alternating order.
#
# Both arms start from the same base config (MODE=compile) so the only
# difference is the VLLM_FL_CUDAGRAPH_ONLY rewrite. Order is A,B,B,A so any
# monotonic drift (thermal, neighbour load) cancels instead of favouring one arm.
#
# Usage:  bash ab_cudagraph_vs_compile.sh
# Result: /tmp/ab_<arm>_<n>.log per run + /tmp/ab_summary.log one line per run.
set -u
source /opt/conda/etc/profile.d/conda.sh && conda activate mx
cd /workspace/vllm-plugin-FL

# 0.45 keeps us under a neighbour's footprint; the box is shared.
export PROMPT=short GPU_MEM=0.45 MODE=compile
SUMMARY=/tmp/abx_summary.log
: > "$SUMMARY"

run_one() {
  local arm="$1"
  local n="$2"
  local log="/tmp/abx_${arm}_${n}.log"
  local cg=0
  [ "$arm" = "cgonly" ] && cg=1
  echo ">>> arm=$arm run=$n cg_only=$cg" >> "$SUMMARY"
  VLLM_FL_CUDAGRAPH_ONLY=$cg timeout 600 python /root/src/cudagraph_no_compile.py \
      > "$log" 2>&1
  local rc=$?
  local line
  line=$(grep -oE "=> [0-9.]+ \(median\) / [0-9.]+ \(best\) tok/s" "$log" | tail -1)
  local uniq
  uniq=$(grep -oE "uniq=[0-9]+" "$log" | tail -1)
  local mode
  mode=$(grep -oE "^  mode            = .*" "$log" | tail -1 | tr -s ' ')
  echo "  rc=$rc $uniq $mode" >> "$SUMMARY"
  echo "  ${line:-NO RESULT}" >> "$SUMMARY"
  sleep 30
}

run_one cgonly 1
run_one compile 1
run_one compile 2
run_one cgonly 2

echo "AB_DONE" > /tmp/abx_DONE
echo "AB_DONE" >> "$SUMMARY"
