#!/bin/bash
unset VLLM_FL_ILUVATAR_IXFORMER_ATTN VLLM_FL_ILUVATAR_IXFORMER_SPLIT VLLM_FL_ILUVATAR_MAX_NUM_BATCHED_TOKENS VLLM_FL_ILUVATAR_IX_RMSNORM
export VLLM_FL_ILUVATAR_IXFORMER_ATTN=1
export VLLM_FL_ILUVATAR_IXFORMER_SPLIT=1
export VLLM_FL_ILUVATAR_IXFORMER_PREFILL_MIN_Q=256
export VLLM_CACHE_ROOT=/root/vllm_cache_cold/cold_16k_mbt4096
rm -rf "$VLLM_CACHE_ROOT"
cp /root/bench_results/fg_variants/iluvatar_init_ext.py /usr/local/lib/python3.12/site-packages/flag_gems/runtime/backend/_iluvatar/__init__.py
export FRAMEWORK=/workspace/vllm-plugin-FL-comp
export EVAL_OUT=/root/bench_results/eval_iluvatar_16k_cold_mbt4096
export PORT=9052
export CASES='[[16384,1024,64,128]]'
export BENCH_RUNS=2
export BENCH_SKIP_FIRST=0
export SKIP_EVAL=1
echo "ATTN=1 SPLIT=1 MIN_Q=256 meta FlagGems_UNUSED=ext graph_pages_fix mbt=4096(plugin default) cold VLLM_CACHE_ROOT 16k" > "$EVAL_OUT/env_check.txt"
exec bash /workspace/vllm-plugin-FL-comp/experiments/src/launch_official_eval_iluvatar.sh
