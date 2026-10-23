#!/bin/bash
set -u
OUT=/workspace/vllm-plugin-FL-comp/experiments/bench_results/hostgap_opt_20261023
LAUNCH=$OUT/launch_serve.py
kill_serve() {
  pkill -9 -f 'vllm serve' 2>/dev/null || true
  pkill -9 -f 'VLLM::EngineCore' 2>/dev/null || true
  sleep 2
}
wait_health() {
  local PID=$1
  for i in $(seq 1 90); do
    [[ -f $OUT/server.exit ]] && { echo DIED; cat $OUT/server.exit; return 1; }
    kill -0 "$PID" 2>/dev/null || { echo pid_gone; return 1; }
    curl -sf http://127.0.0.1:9031/health >/dev/null && { echo HEALTHY; return 0; }
    (( i % 10 == 0 )) && echo wait_$i
    sleep 10
  done
  echo NOT_HEALTHY; return 1
}
run_one() {
  local name=$1; shift
  echo "===== ARM $name env=$* $(date -Iseconds) ====="
  kill_serve
  rm -f "$OUT/server.pid" "$OUT/server.exit"
  : > "$OUT/stage_${name}.txt"
  # clear and set env
  unset MACA_GRAPH_LAUNCH_MODE MACA_LAUNCH_MODE MACA_DIRECT_DISPATCH MACA_CPU_THREAD_POLICY MACA_SYNC_POLICY
  export "$@"
  # point stage file via editing? launch hardcodes stage_prof.txt — symlink swap
  ln -sfn "stage_${name}.txt" "$OUT/stage_prof.txt"
  # actually launch writes to STAGEF path absolute stage_prof.txt — overwrite after by copying
  # Better: export won't change hardcode. Patch: stage file is OUT/stage_prof.txt — we'll mv after.
  python3 "$LAUNCH"
  sleep 1
  PID=$(cat "$OUT/server.pid")
  wait_health "$PID" || return 1
  # if launch used stage_prof.txt, rename after to stage_name — redirect by replacing file content sink
  # The env VLLM_ITER_STAGE_FILE is hardcoded in launch to stage_prof.txt. Copy during bench.
  export CUDA_VISIBLE_DEVICES= VLLM_PLUGINS=
  /opt/conda/bin/vllm bench serve \
    --backend vllm --model minicpm --tokenizer /workspace/MiniCPM5-2B \
    --endpoint /v1/completions --host localhost --port 9031 \
    --dataset-name random --ignore-eos \
    --random-input-len 4096 --random-output-len 1024 \
    --max-concurrency 64 --num-prompts 256 \
    > "$OUT/bench_${name}.log" 2>&1
  cp -f "$OUT/stage_prof.txt" "$OUT/stage_${name}.txt"
  # summarize
  python3 - <<PY
from pathlib import Path
import re, subprocess, sys
name="$name"
b=Path("$OUT/bench_${name}.log").read_text(errors="ignore")
tot=re.search(r"Total token throughput \(tok/s\):\s+([\d.]+)", b)
dur=re.search(r"Benchmark duration \(s\):\s+([\d.]+)", b)
ok=re.search(r"Successful requests:\s+(\d+)", b)
print(f"ARM {name}: tok/s={tot.group(1) if tot else '?'} dur={dur.group(1) if dur else '?'} ok={ok.group(1) if ok else '?'}")
subprocess.check_call([sys.executable,
  "/workspace/vllm-plugin-FL-comp/experiments/src/stage_prof_scan.py",
  f"$OUT/stage_{name}.txt"])
PY
  kill_serve
}

# baseline: no extra MACA knobs
run_one baseline
# host-launch pack (no PDL / no MODE=2)
run_one light \
  MACA_DIRECT_DISPATCH=1 \
  MACA_CPU_THREAD_POLICY=-1 \
  MACA_LAUNCH_MODE=4
# graph mode 3 (in-graph parallelism) — may fail capture; try separately
run_one gmode3 \
  MACA_GRAPH_LAUNCH_MODE=3 \
  MACA_DIRECT_DISPATCH=1 \
  MACA_CPU_THREAD_POLICY=-1 \
  MACA_LAUNCH_MODE=4

echo ALL_DONE
