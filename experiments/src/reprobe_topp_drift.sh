#!/bin/bash
# Reproduction probe for the 10.8% gap between the top-p A/B arms.
#
# Timeline that raised the question: arm A (VLLM_FL_TOPP_SKIP_P1=0) gave
# 12165.05 / 12157.43 tok/s, arm B (default) gave 10847.93 / 10875.17 on an
# identical workload (256 reqs, 1048576 in, 262144 out, peak conc ~73).  Arm B
# was also 7.4% slower on TTFT, which prefill dominates and the sampler barely
# touches -- pointing at a global slowdown rather than the sampler change.
#
# The change has since been reverted (it was dead code: vLLM already passes
# p=None when no request has top_p < 1), so this run uses the committed code
# and asks a simpler question: what does this machine produce *now*?
#
# Three runs, fresh server, official 4k case.  If it lands near 12150 the arm B
# numbers were a transient; if it lands near 10850 the regression is real and
# the revert alone will not have fixed it.
set -u
export PATH=/opt/conda/bin:$PATH
export VLLM_PLUGINS=fl
PY=/opt/conda/bin/python
OUT=/workspace/vllm-plugin-FL-comp/experiments/bench_results/topp_reprobe_20261023
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

say "stopping anything left over"
pkill -9 -f 'run_topp_arm_b.sh' 2>/dev/null
pkill -9 -f 'ab_topp_skip_4k.sh' 2>/dev/null
pkill -9 -f 'benchmark_throughput_serve' 2>/dev/null
sleep 2
stop_server
say "rss before start: $(rg '^rss ' /sys/fs/cgroup/memory/memory.stat)"
say "gpu: $(mx-smi 2>/dev/null | rg -o 'MetaX C500|[0-9]+C +P[0-9]|[0-9]+W / [0-9]+W' | tr '\n' ' ')"

# --- server, committed code (no SKIP_P1 logic exists any more) ---
cd /workspace/vllm-plugin-FL || exit 1
setsid /opt/conda/bin/vllm serve /workspace/MiniCPM5-2B \
  --port "$PORT" --served-model-name minicpm \
  --gpu-memory-utilization 0.85 --max-model-len 131072 \
  > "$OUT/server.log" 2>&1 < /dev/null &
disown 2>/dev/null
cd - >/dev/null

t0=$SECONDS
while [ $((SECONDS-t0)) -lt 900 ]; do
  code=$(curl -s -m 3 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" 2>/dev/null)
  [ "$code" = "200" ] && { say "HEALTHY after $((SECONDS-t0))s"; break; }
  [ -z "$(server_pid)" ] && { say "SERVER DIED"; exit 1; }
  sleep 5
done
say "gpu after load: $(mx-smi 2>/dev/null | rg -o '[0-9]+C +P[0-9]|[0-9]+W / [0-9]+W' | tr '\n' ' ')"

say "bench 4k RUNS=3 (skip first kept at 0 so all three count)"
( cd /workspace && env VLLM_PLUGINS= $PY /workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py \
    --model /workspace/MiniCPM5-2B --served-model-name minicpm --port "$PORT" \
    --test-cases '[[4096,1024,64,256]]' > "$OUT/bench.log" 2>&1 )

$PY - "$OUT" <<'PY'
import pathlib, re, sys
d = pathlib.Path(sys.argv[1])
t = (d/"bench.log").read_text(errors="ignore")
b = re.split(r'Running: (\S+) \| Run (\d+)/(\d+)', t)
g = lambda s, p: (float(re.search(p, s).group(1)) if re.search(p, s) else None)
rows, i = [], 1
while i + 3 < len(b):
    body = b[i+3]
    rows.append((int(b[i+1]),
                 g(body, r'Total token throughput \(tok/s\): +([0-9.]+)'),
                 g(body, r'Mean TPOT \(ms\): +([0-9.]+)'),
                 g(body, r'Mean TTFT \(ms\): +([0-9.]+)')))
    i += 4
for r in rows:
    print(f"  run{r[0]}  {r[1]} tok/s  TPOT {r[2]} ms  TTFT {r[3]} ms")
v = [r[1] for r in rows if r[1]]
if v:
    print(f"\nmean {sum(v)/len(v):.2f} tok/s")
PY

say "gpu after bench: $(mx-smi 2>/dev/null | rg -o '[0-9]+C +P[0-9]|[0-9]+W / [0-9]+W' | tr '\n' ' ')"
stop_server
say "DONE"
