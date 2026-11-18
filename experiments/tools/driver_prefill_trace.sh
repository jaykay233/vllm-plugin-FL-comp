#!/bin/bash
# Pure-prefill trace driver: start the server, wait for readiness, then run a
# prefill-dominated benchmark (output_len=4) so nearly every engine step is a
# prefill chunk.  Prefill kernels are NOT captured in the CUDA graph, so this
# gives full per-kernel attribution.
cd /workspace/vllm-plugin-FL-comp || exit 1

bash /tmp/trace16k/inner_pf.sh > /tmp/trace16k/pf_serve.log 2>&1 &
SRV=$!

for i in $(seq 1 150); do
    if curl -s -m 2 localhost:9068/v1/models >/dev/null 2>&1; then
        echo "ready at t=$((i * 2))s"
        break
    fi
    sleep 2
done

BENCH_RUNS=1 python3 benchmarks/benchmark_throughput_serve.py \
    --model /workspace/MiniCPM5-2B --served-model-name minicpm --port 9068 \
    --test-cases '[[4096,4,64,256]]' > /tmp/trace16k/pf_bench.log 2>&1

kill $SRV 2>/dev/null
wait $SRV 2>/dev/null
echo "driver done"
