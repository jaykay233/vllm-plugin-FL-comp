# Kernel microbenchmarks (MetaX C500, MiniCPM5-2B, 2026-11-20)

These are the one-command reproductions behind the negative results in
`experiments/RESULTS-metax-fused-sampler-20261120.md`.  They were originally
run ad hoc and only their stdout was read, so the numbers below are the
recorded observations -- the scripts are here so they can be re-run and teed.

Every one needs `source /tmp/metax_env.sh` (MACA_PATH etc.) and, for the ones
that import the vendor attention, `PYTHONPATH=/workspace/vllm-plugin-FL`.

| script | what it establishes | observed |
|---|---|---|
| `kv_read_ceiling.py` | HBM read ceiling at the exact per-layer KV payload | bf16 268.4 MB in 169.3 us = 1585 GB/s; int8 134.2 MB in 105.7 us = 1270 GB/s |
| `attn_bf16_splits.py` | vendor decode attention `num_splits` is already optimal at c=64 | splits=0 213 us/layer; 8 -> 275; 16 -> 224; 1 -> 572 |
| `attn_kv_dtype.py` | int8-KV kernel vs fp8-KV vs vendor (needs `int8_decode.cu` built) | int8 0.3136, fp8 0.3143, vendor 0.2136 ms/layer |
| `attn_int8_vs_fp8_vs_vendor.py` | same comparison driven through the tuned harness | int8 BEST 0.3136 ms/layer (S=16); ratio 1.47x |
| `int8_gemm_bare.py` | C500 INT8 tensor cores are fast in isolation | prefill 282-340 TFLOPS vs BF16 194-225 |
| `int8_gemm_full_pipeline.py` | quant + `_int_mm` + dequant is a loss | prefill gate_up 0.48x (3910 -> 8214 us) |
| `dequant_fused_silu.py` | best int8 structure: dequant folded into the existing SiLU | silu from int32 704 us vs 536 from bf16; gate_up 4448 -> 3628 us = 1.23x |
| `int8_gemm_fused_triton.py` | a fused int8 GEMM cannot be built in Triton here | 12-13 TFLOPS = 0.06x mcblas |
| `bf16_gemm_peak.py` | C500 BF16 dense peak, and mcblas reaches it | 227.9 TFLOPS at 8192^3, 228.2 at 16384x16384x8192 |

`attn_kv_dtype.py` and `attn_int8_vs_fp8_vs_vendor.py` need the two shared
objects, which are deliberately not committed (build them first):

    mxcc -shared -fPIC -std=c++17 -O3 --offload-arch=xcore1000 \
      -I/opt/maca/include -I/opt/maca/include/mcr -I/opt/maca/include/common \
      -L/opt/maca/lib -lmxc-runtime64 \
      -o libfp8_decode.so fp8_decode.cu
    mxcc ... -DKV_INT8 -o libint8_decode.so int8_decode.cu

The scripts expect those in `experiments/fp8kv/csrc/`.
