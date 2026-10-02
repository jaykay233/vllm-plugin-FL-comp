#!/bin/bash
set -u
OUT=/root/bench_results/eval_iluvatar_mathL3_fix_ext
PORT=9048
MODEL=/workspace/MiniCPM5-2B
SERVED=minicpm
FRAMEWORK=/workspace/vllm-plugin-FL-comp
DATASET=/workspace/evalscope-datasets/math_500
LEVEL_DIR=/workspace/evalscope-datasets/level3_fix_ext
CGRAPH_CFG='{"cudagraph_mode": "FULL_DECODE_ONLY"}'
SERVER_LOG=$OUT/server.log
EVAL_LOG=$OUT/evalscope.log

unset VLLM_FL_CUDAGRAPH_ONLY
unset VLLM_FL_ILUVATAR_IXFORMER_ATTN
unset VLLM_FL_ILUVATAR_IXFORMER_SPLIT
unset VLLM_FL_ILUVATAR_MAX_NUM_BATCHED_TOKENS
export VLLM_FL_ILUVATAR_IXFORMER_ATTN=1
export VLLM_FL_ILUVATAR_IXFORMER_SPLIT=1
export VLLM_FL_ILUVATAR_IXFORMER_PREFILL_MIN_Q=256
export VLLM_PLUGINS=fl
echo "ATTN=1 SPLIT=1 MIN_Q=256 once_per_step_meta FlagGems_UNUSED=ext graph_pages_fix batch=8" | tee "$OUT/env_check.txt"

cd "$FRAMEWORK" || exit 1
: > "$SERVER_LOG"
nohup vllm serve "$MODEL" --port "$PORT" \
  --compilation-config "$CGRAPH_CFG" \
  --served-model-name "$SERVED" \
  --gpu-memory-utilization 0.85 \
  --max-model-len 131072 \
  > "$SERVER_LOG" 2>&1 &
SRV=$!
echo $SRV > "$OUT/server.pid"
echo "[$(date +%H:%M:%S)] server pid=$SRV"

READY=0
for i in $(seq 1 96); do
  if curl -sf -m 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
    READY=1
    break
  fi
  if ! kill -0 $SRV 2>/dev/null; then
    echo "[$(date +%H:%M:%S)] server died"
    tail -30 "$SERVER_LOG"
    exit 1
  fi
  sleep 5
done
if [ "$READY" != 1 ]; then
  echo "[$(date +%H:%M:%S)] server not ready"
  tail -40 "$SERVER_LOG"
  exit 1
fi
echo "[$(date +%H:%M:%S)] READY"
rg -a "once-per-step|mbt_for_iluvatar|IXFORMER_SPLIT" "$SERVER_LOG" | head -8 || true

echo "[$(date +%H:%M:%S)] PART C | math_500 Level 3"
rm -rf "$LEVEL_DIR"
/usr/local/bin/evalscope eval \
  --model "$SERVED" \
  --api-url "http://127.0.0.1:$PORT/v1/chat/completions" \
  --api-key EMPTY \
  --eval-type openai_api \
  --datasets math_500 \
  --dataset-args "{\"math_500\": {\"dataset_id\": \"$DATASET\", \"subset_list\": [\"Level 3\"]}}" \
  --eval-batch-size 8 \
  --timeout 3600 \
  --generation-config '{"temperature": 1.0, "top_p": 0.95, "max_tokens": 32768}' \
  --work-dir "$LEVEL_DIR" \
  --ignore-errors \
  2>&1 | tee "$EVAL_LOG"
echo "[$(date +%H:%M:%S)] PART C done"
rg -ao "stats=\{[^}]*\}" "$SERVER_LOG" | tail -1

echo "[$(date +%H:%M:%S)] TEARDOWN"
python3 - "$SRV" <<'PY'
import os, signal, sys, time
root = int(sys.argv[1])
pids = {root}
changed = True
while changed:
    changed = False
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            ppid = int(open(f"/proc/{pid}/stat").read().split()[3])
        except Exception:
            continue
        if ppid in pids and int(pid) not in pids:
            pids.add(int(pid))
            changed = True
for p in sorted(pids, reverse=True):
    try:
        os.kill(p, signal.SIGKILL)
    except Exception:
        pass
time.sleep(1)
print("killed", sorted(pids))
PY
