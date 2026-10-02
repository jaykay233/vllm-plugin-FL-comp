#!/bin/bash
unset VLLM_FL_ILUVATAR_IXFORMER_ATTN
unset VLLM_FL_ILUVATAR_IXFORMER_SPLIT
unset VLLM_FL_ILUVATAR_MAX_NUM_BATCHED_TOKENS
export VLLM_FL_ILUVATAR_IXFORMER_ATTN=1
export VLLM_FL_ILUVATAR_IXFORMER_SPLIT=1
export VLLM_FL_ILUVATAR_IXFORMER_PREFILL_MIN_Q=256
export VLLM_FL_ILUVATAR_MAX_NUM_BATCHED_TOKENS=0
export FRAMEWORK=/workspace/vllm-plugin-FL-comp
export EVAL_OUT=/root/bench_results/eval_iluvatar_4k_noadd
export PORT=9041
export CASES='[[4096,1024,64,256]]'
export BENCH_RUNS=2
export BENCH_SKIP_FIRST=0
export SKIP_EVAL=1
echo "ATTN=1 SPLIT=1 MIN_Q=256 once_per_step_meta mbt=2048 FlagGems_UNUSED=add" > "$EVAL_OUT/env_check.txt"
exec bash /workspace/vllm-plugin-FL-comp/experiments/src/launch_official_eval_iluvatar.sh
