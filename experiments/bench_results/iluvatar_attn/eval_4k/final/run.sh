#!/bin/bash
# Final configuration: ctx-aware decode guard, split disabled.
unset VLLM_FL_ILUVATAR_IXFORMER_ATTN
unset VLLM_FL_ILUVATAR_IXFORMER_SPLIT
export VLLM_FL_ILUVATAR_IXFORMER_ATTN=1
export VLLM_FL_ILUVATAR_IXFORMER_SPLIT=0
export VLLM_FL_ILUVATAR_IXFORMER_PREFILL_MIN_Q=256
export FRAMEWORK=/workspace/vllm-plugin-FL-comp
export EVAL_OUT=/root/bench_results/eval_iluvatar_4k_final
export PORT=9037
export CASES='[[4096,1024,64,256]]'
export BENCH_RUNS=2
export BENCH_SKIP_FIRST=0
export SKIP_EVAL=1
echo "ATTN=1 SPLIT=0 MIN_Q=256 decode_guard=ctx_aware(batch<32 and ctx>=2048)" > "$EVAL_OUT/env_check.txt"
exec bash /workspace/vllm-plugin-FL-comp/experiments/src/launch_official_eval_iluvatar.sh
