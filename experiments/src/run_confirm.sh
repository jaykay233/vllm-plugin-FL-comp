#!/bin/bash
# Interleaved A/B/A confirmation of decode-attention num_splits at the
# MATH-500-shaped workload (short prompt, long CoT output), batch=1.
#
# Interleaving lets us tell signal from drift: the two ns=0 runs should agree.
# Retries guard against transient GPU contention from other processes.
set -u

E=/root/bench_results/attn_num_splits
L=$E/confirm_b1.log
RUNS=$E/confirm_runs
mkdir -p "$RUNS"
: > "$L"

echo "start $(date '+%F %T')" >> "$L"

for ns in 0 16 32 0 16; do
  ok=0
  for a in 1 2 3; do
    echo "=== RUN ns=$ns attempt=$a  $(date '+%T') ===" >> "$L"
    if python /root/src/bench_attn_num_splits.py \
         --splits "$ns" --ctx 60 --batch 1 --out 1024 >> "$L" 2>&1; then
      rf="$E/result_ns${ns}_ctx60_b1_o1024.json"
      if [ -f "$rf" ]; then
        cp "$rf" "$RUNS/ns${ns}_a${a}_$(date '+%H%M%S').json"
        ok=1
        break
      fi
    fi
    echo "    -> failed, retrying" >> "$L"
    sleep 12
  done
  echo "RESULT ns=$ns ok=$ok" >> "$L"
done

echo "done $(date '+%F %T')" >> "$L"
echo CONFIRM_DONE > "$E/CONFIRM_DONE"
