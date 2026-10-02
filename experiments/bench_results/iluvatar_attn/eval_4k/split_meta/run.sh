#!/bin/bash
unset VLLM_FL_ILUVATAR_IXFORMER_ATTN
unset VLLM_FL_ILUVATAR_IXFORMER_SPLIT
export VLLM_FL_ILUVATAR_IXFORMER_ATTN=1
export VLLM_FL_ILUVATAR_IXFORMER_SPLIT=1
export VLLM_FL_ILUVATAR_IXFORMER_PREFILL_MIN_Q=256
export FRAMEWORK=/workspace/vllm-plugin-FL-comp
export EVAL_OUT=/root/bench_results/eval_iluvatar_4k_split_meta
export PORT=9038
export CASES='[[4096,1024,64,256]]'
export BENCH_RUNS=2
export BENCH_SKIP_FIRST=0
export SKIP_EVAL=1
echo "ATTN=1 SPLIT=1 MIN_Q=256 decode_guard=ctx_aware once_per_step_meta=on" > "$EVAL_OUT/env_check.txt"
exec bash /workspace/vllm-plugin-FL-comp/experiments/src/launch_official_eval_iluvatar.sh
