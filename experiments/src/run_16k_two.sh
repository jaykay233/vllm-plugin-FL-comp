#!/bin/bash
# 16k official case, two runs, on the current default stack.
#
# Official 16k shape is [[16384,1024,64,128]] (see FINDINGS 15).  Client A/B
# conventions from section 15 are kept: --no-enable-prefix-caching on the
# server, VLLM_PLUGINS= empty on the client so the bench process does not pull
# FlagGems + inductor into its own RSS (section 11 / 12.4 double-init SIGKILL).
#
# Nothing else is overridden: this measures what the plugin ships. That means
# MBT 8192 (mbt_defaults.py), whitelist silu/rms/rope, lm_head on, MLP silu
# fusion off, fused_add_rms_norm -> vllm_c, MACA light launch, and the
# align32_geometric cap of 8192 that FlagGems now defaults to.
#
# A single 4k run follows as an anchor.  Section 22.5 recorded that absolute
# numbers move ~1.6% between sessions with no code change, so comparing this
# 16k figure against section 15's 11650.80 (2026-10-22) needs a same-session
# 4k point -- section 15 has one too (11956.89).
set -u
export PATH=/opt/conda/bin:$PATH
export VLLM_PLUGINS=fl
PY=/opt/conda/bin/python
OUT=/workspace/vllm-plugin-FL-comp/experiments/bench_results/official_16k_20261024
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
say "gpu idle: $(mx-smi | sed -n '15p' | tr -s ' ')"

say "start server (current defaults)"
cd /workspace/vllm-plugin-FL || exit 1
setsid /opt/conda/bin/vllm serve /workspace/MiniCPM5-2B \
  --port "$PORT" --served-model-name minicpm \
  --gpu-memory-utilization 0.85 --max-model-len 131072 \
  --no-enable-prefix-caching \
  > "$OUT/server.log" 2>&1 < /dev/null &
disown 2>/dev/null
cd - >/dev/null

t0=$SECONDS
until [ "$(curl -s -m 3 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" 2>/dev/null)" = "200" ]; do
  [ $((SECONDS-t0)) -gt 900 ] && { say "TIMEOUT"; exit 1; }
  [ -z "$(server_pid)" ] && { say "SERVER DIED"; exit 1; }
  sleep 5
done
say "HEALTHY after $((SECONDS-t0))s"

# record what actually took effect, so the entry is self-describing
say "MBT: $(rg -ao 'max_num_batched_tokens[^,]{0,40}' "$OUT/server.log" | head -2 | tr '\n' ' ')"
say "IR priority: $(rg -ao 'Final IR op priority[^)]*\)' "$OUT/server.log" | tail -1)"

run_case(){
  local label="$1" cases="$2" runs="$3"
  say "bench [$label] runs=$runs cases=$cases"
  ( cd /workspace && env VLLM_PLUGINS= RUNS="$runs" SKIP_FIRST=0 \
      PYTHONPATH=/workspace/vllm-plugin-FL-comp/experiments/src \
      $PY /workspace/vllm-plugin-FL-comp/experiments/src/runs2_bench.py \
      --model /workspace/MiniCPM5-2B --served-model-name minicpm --port "$PORT" \
      --test-cases "$cases" > "$OUT/bench_${label}.log" 2>&1 )
  rg -o 'Total token throughput \(tok/s\): +[0-9.]+|Output token throughput \(tok/s\): +[0-9.]+|Mean (TTFT|TPOT|ITL) \(ms\): +[0-9.]+|Successful requests: +[0-9.]+|Failed requests: +[0-9.]+' "$OUT/bench_${label}.log"
}

# --- the ask: 16k, two runs ---
run_case 16k '[[16384,1024,64,128]]' 2

# --- same-session anchor for cross-session comparison ---
run_case 4k_anchor '[[4096,1024,64,256]]' 1

stop_server
say "DONE"
