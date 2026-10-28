#!/bin/bash
# A/B: fused_add_rms_norm provider order, end to end.
#
# A microbenchmark of the two providers says the opposite of what §19 recorded:
#
#   vllm_c   FusedAddRMSNormKernelOpt    1 kernel   10.21 us/call
#   flagos   split (add + gems rms_norm) 2 kernels   7.42 us/call
#
# §19 had vllm_c at 4.44 us and flagos at 7.43, and picked vllm_c partly on that
# 4.44.  The flagos number reproduces exactly, so the method is consistent and
# only the vllm_c figure disagrees -- either §19 measured it differently or the
# kernel behaves differently now.
#
# Neither provider clones any more (copy_count = 0 for both), so the +3.12% in
# §19 was about the clone, not about which provider wins per call.  That makes
# the per-call ordering an open question, and §21 established that
# microbenchmarks cannot settle hot-path questions.  Hence this run:
#
#   A  priority vllm_c first   (current default)
#   B  priority flagos first   (2 kernels, 2.8 us/call cheaper in isolation)
#
# if B wins, one kernel fewer is not the same as faster.
set -u
export PATH=/opt/conda/bin:$PATH
export VLLM_PLUGINS=fl
PY=/opt/conda/bin/python
OUT=/workspace/vllm-plugin-FL-comp/experiments/bench_results/norm_provider_ab_20261023
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

start_server(){ # $1=label  $2=kernel-config json
  local label=$1 kc=$2
  say "start [$label] kernel-config=$kc"
  cd /workspace/vllm-plugin-FL || return 1
  setsid /opt/conda/bin/vllm serve /workspace/MiniCPM5-2B \
    --port "$PORT" --served-model-name minicpm \
    --gpu-memory-utilization 0.85 --max-model-len 131072 \
    --kernel-config "$kc" \
    > "$OUT/server_$label.log" 2>&1 < /dev/null &
  disown 2>/dev/null
  cd - >/dev/null
  local t0=$SECONDS
  while [ $((SECONDS-t0)) -lt 900 ]; do
    local code
    code=$(curl -s -m 3 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" 2>/dev/null)
    [ "$code" = "200" ] && { say "HEALTHY [$label] after $((SECONDS-t0))s"; return 0; }
    [ -z "$(server_pid)" ] && { say "SERVER DIED [$label]"; return 1; }
    sleep 5
  done
  say "TIMEOUT [$label]"; return 1
}

bench(){ # $1=label
  local label=$1
  say "bench [$label]"
  ( cd /workspace && env VLLM_PLUGINS= $PY /workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py \
      --model /workspace/MiniCPM5-2B --served-model-name minicpm --port "$PORT" \
      --test-cases '[[4096,1024,64,256]]' > "$OUT/bench_$label.log" 2>&1 )
}

parse(){ # $1=label
  $PY - "$OUT/bench_$1.log" "$1" <<'PY'
import re, sys
path, label = sys.argv[1], sys.argv[2]
t = open(path, errors="ignore").read()
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
valid = [r for r in rows if r[1]]
print(f"  [{label}] " + ", ".join(f"run{r[0]}={r[1]:.1f}" for r in valid))
if not valid:
    print(f"  [{label}] NO VALID RUNS"); sys.exit(1)
m  = sum(r[1] for r in valid)/len(valid)
tp = sum(r[2] for r in valid)/len(valid)
tf = sum(r[3] for r in valid)/len(valid)
print(f"  [{label}] MEAN {m:.2f} tok/s  TPOT {tp:.2f}  TTFT {tf:.0f}")
if m <= 0:
    print(f"  [{label}] ZERO -- server dead"); sys.exit(1)
open(path + ".mean", "w").write(f"{m:.4f} {tp:.4f}\n")
PY
}

stop_server
say "rss: $(rg '^rss ' /sys/fs/cgroup/memory/memory.stat)"

KC_A='{"ir_op_priority": {"fused_add_rms_norm": ["vllm_c", "flagos", "native"]}}'
KC_B='{"ir_op_priority": {"fused_add_rms_norm": ["flagos", "vllm_c", "native"]}}'

start_server A_vllmc "$KC_A" || exit 1
bench A_vllmc
parse A_vllmc || { stop_server; exit 1; }
stop_server

start_server B_flagos "$KC_B" || exit 1
bench B_flagos
parse B_flagos || { stop_server; exit 1; }
stop_server

$PY - "$OUT" <<'PY'
import pathlib, sys
d = pathlib.Path(sys.argv[1])
a = float((d/"bench_A_vllmc.log.mean").read_text().split()[0])
b = float((d/"bench_B_flagos.log.mean").read_text().split()[0])
ta = float((d/"bench_A_vllmc.log.mean").read_text().split()[1])
tb = float((d/"bench_B_flagos.log.mean").read_text().split()[1])
print(f"\nA vllm_c first {a:8.2f} tok/s  TPOT {ta:.3f}")
print(f"B flagos first {b:8.2f} tok/s  TPOT {tb:.3f}")
print(f"delta {b-a:+.2f} tok/s = {100*(b-a)/a:+.2f}%   TPOT {tb-ta:+.3f} ms")
(d/"SUMMARY.txt").write_text(
    f"A vllm_c first {a:.2f} tok/s TPOT {ta:.3f}\n"
    f"B flagos first {b:.2f} tok/s TPOT {tb:.3f}\n"
    f"delta {b-a:+.2f} = {100*(b-a)/a:+.2f}%  TPOT {tb-ta:+.3f} ms\n")
PY
say "DONE -> $OUT"
