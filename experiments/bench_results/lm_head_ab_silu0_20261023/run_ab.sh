#!/bin/bash
# lm_head FlagGems on/off A/B on the silu=0 baseline (silu fusion rejected).
set -u
OUT=/workspace/vllm-plugin-FL-comp/experiments/bench_results/lm_head_ab_silu0_20261023
LAUNCH=$OUT/launch_serve.py
RUNS=${RUNS:-2}

kill_serve() {
  pkill -9 -f 'vllm serve' 2>/dev/null || true
  pkill -9 -f 'VLLM::EngineCore' 2>/dev/null || true
  sleep 3
}
wait_health() {
  local PID=$1
  for i in $(seq 1 120); do
    [[ -f $OUT/server.exit ]] && { echo DIED; cat $OUT/server.exit; return 1; }
    kill -0 "$PID" 2>/dev/null || { echo pid_gone; return 1; }
    curl -sf http://127.0.0.1:9031/health >/dev/null && { echo HEALTHY; return 0; }
    (( i % 10 == 0 )) && echo wait_$i
    sleep 10
  done
  echo NOT_HEALTHY; return 1
}

run_arm() {
  local name=$1
  local lm=$2
  echo "===== ARM $name LM_HEAD=$lm SILU=0 $(date -Iseconds) ====="
  kill_serve
  rm -f "$OUT/server.pid" "$OUT/server.exit"
  export VLLM_METAX_LM_HEAD_GEMM="$lm"
  if [[ "$lm" == "1" ]]; then
    export VLLM_METAX_LM_HEAD_GEMM_STATS=1
  else
    export VLLM_METAX_LM_HEAD_GEMM_STATS=0
  fi
  python3 "$LAUNCH"
  sleep 1
  PID=$(cat "$OUT/server.pid")
  if ! wait_health "$PID"; then
    cp -f "$OUT/server.log" "$OUT/server_${name}_fail.log" 2>/dev/null || true
    return 1
  fi
  rg -n "lm_head GEMM|MLP fusion|Failed to import maca" "$OUT/server.log" | head -20 | tee "$OUT/wrap_${name}.txt"
  cp -f "$OUT/server.log" "$OUT/server_${name}.log"
  : > "$OUT/bench_${name}.log"
  for r in $(seq 1 "$RUNS"); do
    echo "----- $name run $r/$RUNS -----" | tee -a "$OUT/bench_${name}.log"
    env -u CUDA_VISIBLE_DEVICES VLLM_PLUGINS= /opt/conda/bin/vllm bench serve \
      --backend vllm --model minicpm --tokenizer /workspace/MiniCPM5-2B \
      --endpoint /v1/completions --host localhost --port 9031 \
      --dataset-name random --ignore-eos \
      --random-input-len 4096 --random-output-len 1024 \
      --max-concurrency 64 --num-prompts 256 \
      >> "$OUT/bench_${name}.log" 2>&1
  done
  python3 - <<PY
import re
from pathlib import Path
name="$name"
b=Path("$OUT/bench_${name}.log").read_text(errors="ignore")
tots=re.findall(r"Total token throughput \(tok/s\):\s+([\d.]+)", b)
ttft=re.findall(r"Mean TTFT \(ms\):\s+([\d.]+)", b)
tpot=re.findall(r"Mean TPOT \(ms\):\s+([\d.]+)", b)
oks=re.findall(r"Successful requests:\s+(\d+)", b)
print(f"ARM {name}: toks={tots} ttft={ttft} tpot={tpot} oks={oks}")
if tots:
    vals=[float(x) for x in tots]
    print(f"ARM {name}: avg={sum(vals)/len(vals):.2f}")
PY
  kill_serve
}

run_arm off 0
run_arm on 1
echo ALL_DONE $(date -Iseconds)
