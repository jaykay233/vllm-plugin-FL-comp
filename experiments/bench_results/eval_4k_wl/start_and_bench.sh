#!/bin/bash
# One-shot: start whitelist server + run 4k serve bench
set -u
source /opt/conda/etc/profile.d/conda.sh
conda activate mx

OUT=/workspace/vllm-plugin-FL-comp/experiments/bench_results/eval_4k_wl
mkdir -p "$OUT"
PY=/opt/conda/envs/mx/bin/python
MODEL=/workspace/MiniCPM5-2B
PORT=9031
SERVED=minicpm

export PATH=/opt/conda/envs/mx/bin:$PATH
export VLLM_PLUGINS=fl
export VLLM_FL_FLAGOS_WHITELIST=silu_and_mul,rms_norm,rotary_embedding
unset VLLM_FL_CUDAGRAPH_ONLY FL_METAX_ATTN_NUM_SPLITS

ts() { date '+%H:%M:%S'; }
say() { echo "[$(ts)] $*"; }

say "cleanup old server on $PORT"
$PY - <<'PY'
import os, signal, time
me = os.getpid()
killed = []
for pid in os.listdir('/proc'):
    if not pid.isdigit() or int(pid) == me:
        continue
    try:
        cl = open(f'/proc/{pid}/cmdline', 'rb').read().decode('utf8', 'replace')
    except Exception:
        continue
    try:
        cwd = os.readlink(f'/proc/{pid}/cwd')
    except Exception:
        cwd = ''
    is_server = 'vllm' in cl and 'serve' in cl and '9031' in cl
    is_orphan = cl.startswith('VLLM::EngineCore') and cwd.startswith('/workspace')
    if is_server or is_orphan:
        try:
            os.kill(int(pid), signal.SIGKILL)
            killed.append(pid)
        except Exception:
            pass
print('  killed', killed or 'none')
time.sleep(3)
PY

: > "$OUT/server.log"
: > "$OUT/bench.log"
say "starting vllm serve"
cd /workspace/vllm-plugin-FL-comp
setsid vllm serve "$MODEL" --port "$PORT" --served-model-name "$SERVED" \
  --gpu-memory-utilization 0.85 --max-model-len 131072 \
  > "$OUT/server.log" 2>&1 < /dev/null &
echo $! > "$OUT/server.pid"
say "server pid=$(cat "$OUT/server.pid")"

READY=0
for i in $(seq 1 180); do
  if grep -qa "Application startup complete" "$OUT/server.log"; then
    READY=1
    break
  fi
  if ! kill -0 "$(cat "$OUT/server.pid")" 2>/dev/null; then
    say "server died early; tail:"
    tr '\r' '\n' < "$OUT/server.log" | tail -40
    exit 1
  fi
  if (( i % 6 == 0 )); then
    say "waiting... (${i}0s) log_bytes=$(wc -c < "$OUT/server.log")"
  fi
  sleep 10
done

if [[ "$READY" != 1 ]]; then
  say "NOT READY; tail:"
  tr '\r' '\n' < "$OUT/server.log" | tail -40
  exit 1
fi
say "READY"
tr '\r' '\n' < "$OUT/server.log" | grep -aE "Loading weights took|GPU KV cache size|Maximum concurrency|VLLM_FL_FLAGOS|whitelist" | head -20

say "run 4k bench (RUNS=4, SKIP_FIRST=1)"
cd /workspace
$PY -u /workspace/vllm-plugin-FL-comp/benchmarks/benchmark_throughput_serve.py \
  --model "$MODEL" --served-model-name "$SERVED" --port "$PORT" \
  --test-cases '[[4096,1024,64,256]]' \
  2>&1 | tee "$OUT/bench.log"

say "DONE"
echo "======== SUMMARY ========"
grep -a "^  Prefill=" "$OUT/bench.log" || true
grep -a "Total token throughput" "$OUT/bench.log" || true
grep -a "Mean TTFT\|Mean TPOT\|Request throughput" "$OUT/bench.log" | head -40 || true
