#!/bin/bash
# Current-stack math_500 Level 3 correctness (105 problems), concurrency 8.
#
# This is the gate before any weight-only quantisation work: the P0 stack scored
# 98.1% (103/105) in FINDINGS 12.3, and the pass threshold is 0.95. Quantising
# weights to 4 bits risks landing under it, so measure the headroom first.
#
# Settings follow the official generation config (FINDINGS 12.3 / 15):
#   temperature 1.0, top_p 0.95, max_tokens 32768
# Client concurrency is 32 here rather than the official eval-batch-size 8, which
# the user asked for to cut wall time. Server max_num_seqs defaults to 128, so 32
# in flight is within capacity. Note concurrency only affects speed and batching,
# not decoding: temperature 1.0 sampling is per-request and unaffected by how many
# requests share a batch, and prefill/decode interleaving does not change the
# distribution because sampling happens per row.
# The client MUST run with CUDA_VISIBLE_DEVICES= set to the EMPTY STRING and
# VLLM_PLUGINS= empty. evalscope imports torch (via datasets/modelscope) during
# startup, and if any GPU is visible it binds it and the serve's EngineCore is
# SIGKILLed 20 s later with no Python traceback (FINDINGS 12.4).
#
# Verified on this box: `env -u CUDA_VISIBLE_DEVICES` still reports
# torch.cuda.is_available() == True and reproduces the kill; only the empty
# string reports False. An earlier version of this script used `env -u` and lost
# all 105 problems that way.
set -u
export PATH=/opt/conda/bin:$PATH
export VLLM_PLUGINS=fl
PY=/opt/conda/bin/python
PORT=9041
MODEL=/workspace/MiniCPM5-2B
SERVED=minicpm
OUT=/workspace/vllm-plugin-FL-comp/experiments/bench_results/matheval_l3_densekv
mkdir -p "$OUT"
ts(){ date '+%H:%M:%S'; }
say(){ echo "[$(ts)] $*"; }

stop_server(){
  local pid; pid=$(pgrep -f "vllm.entrypoints.openai.api_server.*$MODEL" | head -1)
  [ -n "${pid:-}" ] && { kill -TERM "$pid"; say "stop server $pid"; }
  for _ in $(seq 1 20); do [ -z "$(pgrep -f "vllm.entrypoints.openai.api_server.*$MODEL")" ] && break; sleep 3; done
  pkill -9 -f "vllm.entrypoints.openai.api_server.*$MODEL" 2>/dev/null
  pkill -9 -f 'VLLM::EngineCore' 2>/dev/null
  sleep 8
}

stop_server
say "start serve"
cd /workspace/vllm-plugin-FL || exit 1
setsid env VLLM_FL_METAX_MAX_NUM_BATCHED_TOKENS=8192 \
  $PY -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --served-model-name "$SERVED" --port "$PORT" \
  --gpu-memory-utilization 0.85 --max-model-len 131072 \
  > "$OUT/server.log" 2>&1 < /dev/null &
disown 2>/dev/null
cd - >/dev/null

t0=$SECONDS
until [ "$(curl -s -m 3 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" 2>/dev/null)" = "200" ]; do
  [ $((SECONDS-t0)) -gt 900 ] && { say "TIMEOUT"; tail -25 "$OUT/server.log"; exit 1; }
  [ -z "$(pgrep -f "vllm.entrypoints.openai.api_server.*$MODEL")" ] && { say "SERVER DIED"; tail -25 "$OUT/server.log"; exit 1; }
  sleep 5
done
say "HEALTHY after $((SECONDS-t0))s"
say "topp_fast: $(rg -o '\[fl\] topp_fast coverage[^\n]*' "$OUT/server.log" | tail -1)"

rm -rf /workspace/evalscope-datasets/level3
say "start evalscope (105 problems, batch 32)"
( cd /workspace && env CUDA_VISIBLE_DEVICES= VLLM_PLUGINS= \
  /opt/conda/bin/evalscope eval \
    --model "$SERVED" \
    --api-url "http://127.0.0.1:$PORT/v1/chat/completions" \
    --api-key EMPTY \
    --eval-type openai_api \
    --datasets math_500 \
    --dataset-args '{"math_500": {"dataset_id": "/workspace/evalscope-datasets/math_500", "subset_list": ["Level 3"]}}' \
    --eval-batch-size 32 \
    --timeout 3600 \
    --generation-config '{"temperature": 1.0, "top_p": 0.95, "max_tokens": 32768}' \
    --work-dir /workspace/evalscope-datasets/level3 \
    --ignore-errors \
    2>&1 | tee "$OUT/evalscope.log" )

say "--- result ---"
rg -ai -e 'accuracy|acc:|score computed|errored' "$OUT/evalscope.log" | tail -10
say "serve still healthy: $(curl -s -m 3 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health")"
stop_server
say DONE
