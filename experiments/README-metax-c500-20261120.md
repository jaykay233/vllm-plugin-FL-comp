# MetaX C500 / MiniCPM5-2B — one-page answer (2026-11-20)

Full analysis: `RESULTS-metax-fused-sampler-20261120.md`.
Evidence: `bench_results/` (three directories, see below).

## Question

Raise 4k (and 16k) throughput while keeping MATH-500 Level 3 >= 95%.
Target: **4k > 14,000 tok/s at concurrency 64**.

## Answer

**Not reachable at c=64 on this part.** The official configuration is
`--gpu-memory-utilization 0.85`, `max_model_len 131072`, default
`max_num_batched_tokens` (16384), chunked prefill on.

| case | tok/s | mean TTFT | note |
|---|---:|---:|---|
| 4k c=64, official config | **13,058.96** | 2.48 s | prefix hit rate 0.0-0.4% |
| 4k c=64, MBT 131072 | 13,091.68 | 5.28 s | eligible, but 2.1x TTFT |
| 16k c=64, official config | **12,733.78** | 13.94 s | prefix hit rate 0.0% |
| 4k c=128 | 14,579.49 | 5.09 s | target met, different concurrency |
| MATH-500 L3 | **97.1%** (105) | -- | gate 0.95, PASS |

Shipped code change: fused top-p sampler (`+1.37%` 4k, `+0.55%` 16k as
measured, though only the fused sampler's own microbench and the L3 gate are
unambiguous; read section 19 before quoting those deltas).

## Why 14,000 is out of reach at c=64

Measured, not estimated:

* decode step = 14.96 ms kernel + ~1.2 ms idle.  Attention is 8.51 ms and
  reads 11.3 GB at **1.32 TB/s = 82% of the 1.585 TB/s HBM read ceiling**.
  Projections are 4.44 ms for 5 GB of weights (vendor mcblas).
* prefill projections run at **244 TFLOPS against a measured 228 TFLOPS BF16
  hardware peak** -- i.e. at peak.
* zero-idle floor = 91.9 s = 14,260 tok/s.  14,000 would need 98% of an
  unreachable optimum.

Eleven avenues were closed by measurement, including two kernels written for
this: an int8-KV decode attention kernel (0.3136 vs vendor 0.2136 ms/layer,
1.47x slower) and a fused int8 GEMM attempt (Triton int8 dot is 12 TFLOPS, so
it cannot be built there; the vendor int8 GEMM must write int32, which makes
the full pipeline 0.48x).

## Evidence layout

| directory | contents |
|---|---|
| `bench_results/throughput_20261120/` | throughput logs + server logs; README marks which blocks are warm-cache artefacts (14,029.43 and 15,082.87 are **not** results) |
| `bench_results/matheval_l3_20261120/` | evalscope log for the 97.1% L3 gate |
| `bench_results/kernel_micro_20261120/` | the nine microbenchmark scripts behind the negative results, with observed values |

## Two traps worth knowing before re-measuring

1. **Repeated runs are inflated.** `vllm bench serve` re-seeds its random
   dataset identically, so `BENCH_RUNS=2, SKIP_FIRST=1` sends byte-identical
   prompts to a warm prefix cache: +15% and a physically impossible 0.6 s TTFT.
   Restart the server before every timed run and check the logged hit rate.
2. **The throughput benchmark samples** (the model's `generation_config.json`
   sets `temperature 1.0, top_p 0.95`), and `topp_fast coverage: fast=0` in the
   logs means the fused path handled everything -- the counter belongs to a
   superseded patch.
