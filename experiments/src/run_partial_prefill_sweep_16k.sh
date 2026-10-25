#!/bin/bash
# Does vLLM's own long/short prefill split fix the 16k -3.6%?
#
# Established (FINDINGS-mx 27.2/27.4), post-fix stack, official shapes, nopc:
#   4k : MBT  8192 -> 12542      MBT 16384 -> 12657   (+0.92%)
#   16k: MBT  8192 -> 11921      MBT 16384 -> 11496   (-3.6%)
# At 16k both TTFT and TPOT degrade, no preemptions in the log, so it is not KV
# pressure. The likely mechanism is that a 16384-token prompt exactly fills the
# whole token budget, so that step is pure prefill and decode is starved.
#
# vLLM already has the machinery to split long from short prompts, but it is off
# by default: SchedulerConfig.max_num_partial_prefills defaults to 1, and
# __post_init__ only sets long_prefill_token_threshold (=4% of max_model_len)
# when max_num_partial_prefills > 1. With 1, the threshold is forced to 0 and the
# per-request chunk cap in Scheduler.schedule is dead code.
#
# This sweep is pure configuration -- no code change to vLLM or the plugin -- to
# find out whether enabling it recovers 16k, and whether it costs anything at 4k.
#
# Arms (16k, official [[16384,1024,64,256]]):
#   P1  16384 + partial=2                    long threshold defaults to 5242
#   P2  16384 + partial=4 long=2             shorter prompts may jump ahead
#   P3   8192 + partial=2                    does it also help the safe default?
#   P4  16384 + partial=2  (anchor, repeat P1)
set -u
export PATH=/opt/conda/bin:$PATH
export VLLM_PLUGINS=fl
PY=/opt/conda/bin/python
OUT=/workspace/vllm-plugin-FL-comp/experiments/bench_results/partial_prefill_sweep_16k_20261025
PORT=9040
mkdir -p "$OUT"
say(){ echo "[$(date +%H:%M:%S)] $*"; }

server_pid(){ pgrep -f "vllm serve /workspace/MiniCPM5-2B" | head -1; }

stop_server(){
  local pid; pid=$(server_pid)
  [ -n "${pid:-}" ] && { kill -TERM "$pid"; say "stop server $pid"; }
  for _ in $(seq 1 20); do [ -z "$(server_pid)" ] && break; sleep 3; done
  pid=$(server_pid); [ -n "${pid:-}" ] && kill -9 "$pid"
  pkill -9 -f 'VLLM::EngineCore' 2>/dev/null
  sleep 8
}

run_arm(){
  local label="$1" mbt="$2" extra="$3"
  stop_server
  say "start [$label] MBT=$mbt  extra='$extra'"
  cd /workspace/vllm-plugin-FL || exit 1
  setsid env VLLM_FL_METAX_MAX_NUM_BATCHED_TOKENS="$mbt" \
    /opt/conda/bin/vllm serve /workspace/MiniCPM5-2B \
    --port "$PORT" --served-model-name minicpm \
    --gpu-memory-utilization 0.85 --max-model-len 131072 \
    --no-enable-prefix-caching $extra \
    > "$OUT/server_${label}.log" 2>&1 < /dev/null &
  disown 2>/dev/null
  cd - >/dev/null

  local t0=$SECONDS
  until [ "$(curl -s -m 3 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" 2>/dev/null)" = "200" ]; do
    [ $((SECONDS-t0)) -gt 900 ] && { say "TIMEOUT"; return 1; }
    [ -z "$(server_pid)" ] && { say "SERVER DIED"; tail -25 "$OUT/server_${label}.log"; return 1; }
    sleep 5
  done
  say "HEALTHY after $((SECONDS-t0))s"
  # what the scheduler actually resolved
  rg -q 'Concurrent partial prefills enabled' "$OUT/server_${label}.log" \
    && say "partial: $(rg -o 'max_num_partial_prefills=[0-9]+, max_long_partial_prefills=[0-9]+, long_prefill_token_threshold=[0-9]+' "$OUT/server_${label}.log" | head -1)" \
    || say "partial: DISABLED (threshold forced to 0)"
  say "MBT: $(rg -ao 'max_num_batched_tokens -> [0-9]+' "$OUT/server_${label}.log" | head -1)"

  ( cd /workspace && env VLLM_PLUGINS= RUNS=1 SKIP_FIRST=0 \
      $PY /workspace/vllm-plugin-FL-comp/experiments/src/runs2_bench.py \
      --model /workspace/MiniCPM5-2B --served-model-name minicpm --port "$PORT" \
      --test-cases '[[16384,1024,64,256]]' > "$OUT/bench_${label}.log" 2>&1 )

  say "results [$label] MBT=$mbt $extra"
  rg -o 'Total token throughput \(tok/s\): +[0-9.]+|Output token throughput \(tok/s\): +[0-9.]+|Mean (TTFT|TPOT) \(ms\): +[0-9.]+' "$OUT/bench_${label}.log"
  stop_server
}

run_arm P1 16384 "--max-num-partial-prefills 2"
run_arm P2 16384 "--max-num-partial-prefills 4 --max-long-partial-prefills 2"
run_arm P3 8192  "--max-num-partial-prefills 2"
run_arm P4 16384 "--max-num-partial-prefills 2"

say DONE
for f in "$OUT"/bench_*.log; do
  echo "$(basename "$f"): $(rg -o 'Total token throughput \(tok/s\): +[0-9.]+' "$f" | tr -s ' ' | tr '\n' ' ')"
done
