#!/bin/bash
# One arm of the KV block-size A/B at the official 4k shape.
#
# MetaX resolves block_size to 16 (vllm_fl/platform.py:409).  The paged
# flash_attn kernel accepts any multiple of 16, so 32 is legal.  A bigger page
# means fewer page-table entries to walk per attention call -- cheap to test,
# expected effect <1%, which is why both arms run in the same session: the
# session-to-session band is +-0.94% (FINDINGS 23.5), i.e. larger than the
# effect being measured.
#
# Usage: run_blocksize_arm.sh <block_size> <label>
set -u
export PATH=/opt/conda/bin:$PATH
export VLLM_PLUGINS=fl
PY=/opt/conda/bin/python
BS="${1:?block_size}"
LABEL="${2:?label}"
OUT=/workspace/vllm-plugin-FL-comp/experiments/bench_results/blocksize_ab_20261024
PORT=9031
mkdir -p "$OUT"
say(){ echo "[$(date +%H:%M:%S)] $*"; }

server_pid(){ pgrep -f "vllm serve /workspace/MiniCPM5-2B" | head -1; }

stop_server(){
  local pid; pid=$(server_pid)
  [ -n "${pid:-}" ] && { kill -TERM "$pid"; say "stop server $pid"; }
  for _ in $(seq 1 20); do [ -z "$(server_pid)" ] && break; sleep 3; done
  pid=$(server_pid); [ -n "${pid:-}" ] && kill -9 "$pid"
  pkill -9 -f 'VLLM::EngineCore' 2>/dev/null
  sleep 3
}

stop_server
say "start [bs=$BS] label=$LABEL"
cd /workspace/vllm-plugin-FL || exit 1
setsid /opt/conda/bin/vllm serve /workspace/MiniCPM5-2B \
  --port "$PORT" --served-model-name minicpm \
  --gpu-memory-utilization 0.85 --max-model-len 131072 \
  --no-enable-prefix-caching \
  --block-size "$BS" \
  > "$OUT/server_${LABEL}.log" 2>&1 < /dev/null &
disown 2>/dev/null
cd - >/dev/null

t0=$SECONDS
until [ "$(curl -s -m 3 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" 2>/dev/null)" = "200" ]; do
  [ $((SECONDS-t0)) -gt 900 ] && { say "TIMEOUT"; exit 1; }
  [ -z "$(server_pid)" ] && { say "SERVER DIED -- block_size=$BS may be unsupported"; tail -20 "$OUT/server_${LABEL}.log"; exit 1; }
  sleep 5
done
say "HEALTHY after $((SECONDS-t0))s"

# what actually took effect: block size, KV capacity, attention block size
say "kv blocks: $(rg -ao '# GPU blocks: [0-9]+|GPU KV cache size: [0-9,]+ tokens' "$OUT/server_${LABEL}.log" | tr '\n' ' ')"
say "cache: $(rg -ao 'block_size[^,)]{0,12}' "$OUT/server_${LABEL}.log" | head -2 | tr '\n' ' ')"

say "bench [bs=$BS]"
( cd /workspace && env VLLM_PLUGINS= RUNS=2 SKIP_FIRST=0 \
    PYTHONPATH=/workspace/vllm-plugin-FL-comp/experiments/src \
    $PY /workspace/vllm-plugin-FL-comp/experiments/src/runs2_bench.py \
    --model /workspace/MiniCPM5-2B --served-model-name minicpm --port "$PORT" \
    --test-cases '[[4096,1024,64,256]]' > "$OUT/bench_${LABEL}.log" 2>&1 )

rg -o 'Total token throughput \(tok/s\): +[0-9.]+|Output token throughput \(tok/s\): +[0-9.]+|Mean (TTFT|TPOT) \(ms\): +[0-9.]+|Successful requests: +[0-9.]+|Failed requests: +[0-9.]+' "$OUT/bench_${LABEL}.log"

stop_server
say "DONE [bs=$BS]"
