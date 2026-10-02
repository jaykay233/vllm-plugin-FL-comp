#!/bin/bash
unset VLLM_FL_ILUVATAR_IXFORMER_ATTN
export VLLM_FL_ILUVATAR_IXFORMER_ATTN=1
export VLLM_FL_ILUVATAR_IXFORMER_PREFILL_MIN_Q=512
export FRAMEWORK=/workspace/vllm-plugin-FL-comp
export EVAL_OUT=/root/bench_results/eval_iluvatar_4k_gate_nosync
export PORT=9031
export CASES='[[4096,1024,64,256]]'
export BENCH_RUNS=2
export BENCH_SKIP_FIRST=0
export SKIP_EVAL=1
echo "ATTN=${VLLM_FL_ILUVATAR_IXFORMER_ATTN} MIN_Q=${VLLM_FL_ILUVATAR_IXFORMER_PREFILL_MIN_Q}" > "$EVAL_OUT/env_check.txt"
exec bash /workspace/vllm-plugin-FL-comp/experiments/src/launch_official_eval_iluvatar.sh
