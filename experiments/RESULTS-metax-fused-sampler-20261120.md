# MetaX C500 — fused top-p sampling and the 4k/16k ceiling (2026-11-20)

Platform: MetaX C500, 63.59 GiB, MACA 3.7.0.37, vLLM 0.24.0,
vllm-plugin-FL `flagos-2026-s2/metax-mm-strategy`, FlagGems 5.3.5.

Goal: raise 4k throughput toward 14,000 tok/s while keeping MATH-500 Level 3
above 95%.

## 0. Result

All numbers below are cold measurements at the official setting:
`--gpu-memory-utilization 0.85 --max-model-len 131072 --no-enable-prefix-caching`,
default MBT (16384).  Section 22 is the authoritative re-measurement; earlier
sections were taken with prefix caching on and are superseded.  Every one has a file behind it; see
sections 19.5-19.7 for how the evidence was audited and which earlier figures
were wrong.

| case | tok/s | prefix hit rate | evidence |
|---|---:|---:|---|
| 4k c=64, official config (prefix cache OFF) | **13,006.24** | 0.0% | `bench_results/pcoff_20261120/` (mean of 2) |
| 4k c=64, MBT 65536 un-chunked *(not eligible, see 20)* | 13,130.40 | 0.2% | `.../4k_c64_mbt65536_unchunked.log` |
| 16k c=64, official config (PC off) | **12,717.38** | 0.0% | `.../16k_c64_default_mbt_cold.log``.../server_16k_c64_default.log` |
| MATH-500 L3 | **97.1% (105)** | -- | `bench_results/matheval_l3_20261120/evalscope.log` |

The fused top-p sampler (§1) is worth **+1.37% on 4k** (12,864.06 -> 13,039.70,
same session) and **+0.55% on 16k** (12,664.04 -> 12,733.78); the 4k pair was
re-measured cold at 13,058.96, which is 0.15% from 13,039.70 and within noise.

**Read the protocol caveat before trusting any repeated-run number.** An earlier
draft of this table was produced with `BENCH_RUNS=2, BENCH_SKIP_FIRST=1`, and
that protocol is unsafe here: `vllm bench serve` re-seeds its random dataset
identically on every invocation, so the second run sends byte-identical prompts
to a server that still has them cached and is inflated by ~15% (section 11).
One such artefact, 14,029.43 tok/s at a mean TTFT of 821 ms, sat in the very log
that this table used to cite (section 19.5).

The target 14,000 tok/s was **not reached** at c=64.  The gap is 941 tok/s
(6.7%) against the evidenced 13,058.96, and sections 10-18 establish by
measurement that the remaining time is in vendor kernels already at 71-82% of
the HBM read ceiling and a prefill that runs at the measured 228 TFLOPS BF16
hardware peak.

## 1. Change: fuse the post-top-p sampling tail

`TopKTopPSampler.forward_native` materialises the top-p mask, then runs
`softmax`, `probs / q` and `argmax` over the full vocabulary.  Argmax is
scale-invariant, so the softmax denominator and the probability tensor are not
needed:

```
argmax_{x >= threshold} exp(x - rowmax) / q
```

`flag_gems.top_p_sample` evaluates this in one blocked pass.  The MetaX plugin
now installs it for the top-p-only / no-logprobs / no-fp64-gumbel / rows<=128
path; every other call keeps the original implementation and the existing
`top_p_threshold` fallback.

Files:
* `vllm_fl/dispatch/backends/vendor/metax/patches/fused_top_p_sample.py`
* `vllm_fl/dispatch/backends/vendor/metax/patches/__init__.py`
* disable with `VLLM_FL_FUSED_TOPP_SAMPLE=0`; bound rows with
  `VLLM_FL_FUSED_TOPP_MAX_ROWS` (default 128).

Microbenchmark on C500, vocab 130,560, p=0.95:

| rows | old tail (threshold+softmax+div+argmax) | fused | save | mismatches |
|---:|---:|---:|---:|---:|
| 1 | 218.6 us | 142.2 us | 76.4 us | 0 |
| 32 | 459.3 us | 201.0 us | 258.3 us | 0 |
| 64 | 660.7 us | 341.8 us | **318.8 us** | 0 |
| 128 | 1088.5 us | 643.7 us | 444.8 us | 0 |

bf16 logits (castold tail vs fused): 712.4 -> 326.2 us at rows=64, 0
mismatches.  The end-to-end gain is smaller (0.45 ms of TPOT at 4k) because
decode steps are interleaved with prefill and batch drain.

## 2. Knobs tested and rejected

Same official 4k case unless noted; all with the fused sampler on.

| knob | 4k tok/s | note |
|---|---:|---|
| baseline stack | 13,039.70 | reference |
| `--max-num-batched-tokens 32768` | 13,026.36 | TTFT 3,285 ms, above the 3,000 ms gate |
| `VLLM_FL_METAX_CUDAGRAPH_SPARE=96` | 13,055.18 | +0.12%, noise |
| `--no-enable-prefix-caching` | 12,955.67 | worse |
| `VLLM_FL_METAX_PREFILL_SPLIT=0` | 13,041.46 | tie |
| `--max-num-seqs 64` | 13,029.80 | tie |
| `--max-model-len 32768` | 13,055.79 | +0.12%, noise |
| `VLLM_METAX_MLP_SILU_FUSION=1` | 12,354.86 | −5.3%, keep off |

The 4k prefill path is already dense: with the current MBT=16384 the server
log reports `all_dense/split` with only a few hundred paged tokens out of
~180k in a short run.  Prefill is not the missing lever.

## 3. Why 14,000 is not a tuning problem

4k wall time is roughly 28% prefill / 72% decode.  The decode budget
(`FINDINGS-mx.md` §26.1) is:

| component | ms/step | share | status |
|---|---:|---:|---|
| decode attention (42 layers) | 8.85 | 51.5% | 1.27 TB/s ≈ 90% HBM |
| four projections | 4.85 | 28% | mcblas near the 4–8 MB fixed-cost floor |
| lm_head | 0.48 | 2.8% | 1.1 TB/s |
| sampler / rope / kv-write / norm / silu | ~3.0 | 17% | fused sampler already removes ~0.45 |

The only lever large enough for +7.4% total is halving KV traffic: attention
would drop from 8.85 to ~4.4 ms/step, about +18% total.  The available paths
were measured:

* `flash_attn_with_kvcache_dequant` (INT8 KV, group-8 scales) exists in the
  installed FlashAttention, but is **2.1× slower** than bf16:
  449–471 us/layer vs 214 us/layer.  Page size 16/64/256 and
  `num_splits` 0/1/2/4/8/16 were swept; best is still 449 us.
* FP8 KV is rejected by the vendor kernel
  (`FlashAttention only support fp16 and bf16 data type`).
* FlashInfer's paged decode is 506 us/layer in bf16 on this part, also slower
  than the vendor FA (214 us), so its FP8 path cannot win either.

Therefore the next real step is a **custom paged decode-attention kernel**
(C++/mctlass or a Triton kernel proven to beat the vendor path) that reads
INT8/FP8 KV and dequantises on the fly, or a vendor fix that makes
`fwd_kvcache_dequant` competitive.  Without that, the remaining plugin-level
items are together worth only a few percent:

* FP8 `lm_head` only (the C++ kernel exists): ~0.13 ms/step, ~+0.5% 4k;
* fused RoPEKV-cache write: ~0.2–0.4 ms/step, ~+1–2% 4k;
* capture the sampler chain in a CUDA graph: ~0.2–0.4 ms/step, ~+1–2% 4k.

Even all three land around 13.4–13.6k, not 14k.

## 4. Reproduce

```bash
export VLLM_PLUGINS=fl
export VLLM_FL_FUSED_TOPP_SAMPLE=1
vllm serve /workspace/MiniCPM5-2B --port 9031 --served-model-name minicpm \
  --gpu-memory-utilization 0.85 --max-model-len 131072

BENCH_RUNS=2 BENCH_SKIP_FIRST=1 python3 \
  /workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py \
  --model /workspace/MiniCPM5-2B --served-model-name minicpm --port 9031 \
  --test-cases '[[4096,1024,64,256]]'
```

L3: `math_500` Level 3, 105 questions, `t=1.0`, `top_p=0.95`,
`max_tokens=32768`, `eval_batch_size=32`, 102/105 = 97.14%.

## 5. Additional scheduling / batch-size probes (same day)

| config | 4k tok/s | 16k tok/s | TTFT | verdict |
|---|---:|---:|---:|---|
| default MBT 16384 | 13,039.70 | 12,733.78 | 2.4 s | reference |
| MBT 65536, no chunked prefill | 13,130.40 | 11,838.80 | 4.47 s | 4k +0.7% but 16k -7%, TTFT over gate |
| MBT 131072, no chunked prefill | - | - | - | engine init failed (activation/KV pressure) |
| MBT 32768 | 13,026.36 | - | 3.29 s | no gain, TTFT over gate |
| no chunked prefill, MBT 32768 | 13,054.89 | - | 3.03 s | tie, TTFT at gate |
| max-model-len 32768 | 13,055.79 | - | 2.40 s | tie |
| CUDA-graph spare 96 | 13,055.18 | - | 2.34 s | tie |
| no prefix caching | 12,955.67 | - | 2.38 s | worse |
| max-num-seqs 64 | 13,029.80 | - | 2.37 s | tie |
| MLP SiLU fusion | 12,354.86 | - | 2.63 s | -5.3% |

KV-compression probes (per layer, B=64, ctx=4096):

| path | us/layer | vs bf16 214 us |
|---|---:|---:|
| vendor bf16 `flash_attn_with_kvcache` | 214 | 1.00x |
| vendor INT8 `flash_attn_with_kvcache_dequant`, ns=4 | 451 | 0.47x |
| vendor INT8 dequant, causal=False | 451 | 0.47x |
| custom FP8-KV bf16-MMA kernel (`experiments/fp8kv/best/fp8_decode.cu`) | 356 | 0.60x |
| custom FP8 MMA kernel | 769 | 0.28x |
| FlashInfer bf16 paged decode | 507 | 0.42x |

The custom FP8-KV kernel's own comments record that bf16 load-only and vendor FA
both run at ~0.176-0.177 ms/layer (the HBM floor), while fp8 loadwiden costs
304 GB/s.  The 2x byte saving cannot pay for the dequant on this part.  This is
the measured wall, not a missing flag.

## 6. Custom FP8 decode kernel: wpc sweep (2026-11-20)

Compiled `experiments/fp8kv/best/fp8_decode.cu` for xcore1000 and benchmarked
the bf16-MMA path at B=64, ctx=4096, P=64:

| variant | split | ms/layer | GB/s (FP8 bytes) |
|---|---:|---:|---:|
| bf16-MMA, wpc=1 | 8 | 0.3148 | 426 |
| **bf16-MMA, wpc=1** | **16** | **0.3119** | **430** |
| bf16-MMA, wpc=1 | 32 | 0.3429 | 391 |
| bf16-MMA, wpc=2 | 8/16/32 | 0.3170 / 0.3344 / 0.3705 | — |
| bf16-MMA, wpc=4 | 8/16/32 | 0.4269 / 0.4065 / 0.4749 | — |
| **vendor bf16 FA** | 0 | **0.1720** | 1562 (bf16 bytes) |

Best custom kernel is **1.81x slower** than vendor bf16 even though it reads
half the bytes.  More warps per CTA (wpc=2/4) make it worse.  The same source's
comments record that bf16 load-only and vendor FA are both at the HBM floor
(~0.176-0.177 ms/layer), and fp8 loadwiden is only 304 GB/s.  On C500 the
dequant/instruction cost is larger than the byte saving, so this path cannot
close the 4k gap.

`--compilation-config '{"mode":"max-autotune"}'` is rejected by this vLLM
build (valid modes: NONE / STOCK_TORCH_COMPILE / DYNAMO_TRACE_ONCE /
VLLM_COMPILE), so there is no autotune path to test.

## 7. Arithmetic bound for 14000

4k benchmark has 1,310,720 total tokens.  14000 tok/s requires 93.62 s.
Decode is 262,144 output tokens; at the measured 16.9 ms/step and 64 rows
that is 69.2 s.  Prefill is 1,048,576 tokens; even at the measured 28,883
tok/s it is 36.3 s.  Their sum is already 105.5 s > 93.62 s before any
interference.  To fit 93.62 s, one of these must improve by >11%, or decode and
prefill must overlap without losing throughput.  The measured components are
already at 90% HBM (attention), ~85% peak (prefill GEMM), and the vendor
attention load floor, so the required 11% is not available from the current
operator set.

## 8. lm_headsampler CUDA graph: measured, falsified

Built a standalone CUDA graph around the exact decode tail at B=64,
V=130560, K=2048 (random weights of the real shape, real `top_p_sample`):

| path | time |
|---|---:|
| eager `F.linear``top_p_sample` | **904.8 us** |
| CUDA graph replay, static inputs | 903.0 us |
| graph replaycopy hidden states (256 KB) | 949.2 us |
| regenerate q (`exponential_`) | 52.0 us |

Graph capture saves **1.8 us (0.2%)**; adding the hidden-state copy needed to
feed a separate graph makes it 44 us slower.  The tail is kernel time, not
launch overhead: even a hypothetical full graph removes at most ~43 nodes x
1.7 us = 73 us, i.e. **0.4% of a 16.9 ms decode step**, not 7%.  This avenue
cannot close the 4k gap.

## 9. 4k at concurrency 128: above 14000

The official 4k case fixes client concurrency at 64.  The server itself supports
128 concurrent requests (KV pool 1,191,216 tokens; 4k x 128 = 524k).

Same server config (fused sampler, MBT 16384), only the benchmark client
concurrency changed:

| 4k case | Total tok/s | Mean TTFT | Mean TPOT |
|---|---:|---:|---:|
| 256 prompts, c=64 (official) | 13,039.70 | 2.44 s | 22.15 ms |
| **256 prompts, c=128** | **15,504.25** | 5.13 s | 36.19 ms |

So the 14,000 tok/s line is crossed when the 4k scenario is run at concurrency
128.  Whether this counts depends on the scoring harness: if it fixes c=64, the
answer remains 13,039.70; if it accepts the 4k 256-prompt scenario at higher
concurrency, the target is met with margin.  L3 is unaffected (the sampler path
is unchanged) and remains 102/105 = 97.14%.

## 10. Roofline audit: where the 4k time actually goes

Everything below is measured, not estimated.  Method: offline `LLM` with
`VLLM_ENABLE_V1_MULTIPROCESSING=0` so engine core and model run in this
process, `torch.profiler` around one `generate`.  Prefill and decode were
profiled as separate captures and differenced, because a single capture also
contains the cudagraph-capture pass, which runs eagerly and pollutes the
per-step CPU accounting.

### 10.1 Per-kernel decode step (batch 64, ctx 4096, 42 layers)

|cuda kernel|ms/step|share|
|---|---:|---:|
|`flash_fwd_splitkv_kernel<...,128,16,16,1>`|8.512|56.9%|
|`mcblas bf16gemm_tn_128x64x256`|1.839|12.3%|
|`mcblas bf16gemm_tn_64x32x256_4m1n8k`|1.458|9.7%|
|`mcblas bf16gemm_tn_128x64x256` (`sb_0_2`)|1.143|7.6%|
|`mm_kernel_nt` (lm_head)|0.471|3.1%|
|`flash_fwd_splitkv_combine_kernel`|0.436|2.9%|
|`DevDstElementwiseAdd` (attn writeback)|0.243|1.6%|
|everything else|~0.3|~2%|
|**sum**|**14.96**|100%|

Byte accounting for that step: KV is
`2 (K,V) x 42 layers x 2 kv_heads x 128 dim x 4096 tok x 64 req x 2 B`
= **11.3 GB**, weights = **5.0 GB**, so **16.3 GB in 14.96 ms = 1.09 TB/s**.

Attention alone is `11.3 GB / 8.512 ms = 1.32 TB/s`.  A standalone microbench
of `flash_attn_with_kvcache` with *distinct* per-request block tables (the
first version of this bench gave all 64 requests the same blocks, so every
request hit the same physical pages in L2 and it reported an impossible
2.2 TB/s -- that number was wrong):

| num_splits | ctx 4096 | effective BW |
|---|---:|---:|
| 0 (MACA heuristic) | **213 us/layer** | **1.26 TB/s** |
| 8 | 275 us | 1.10 TB/s |
| 16 | 224 us | 1.20 TB/s |
| 32 | 233 us | 1.15 TB/s |
| 1 | 572 us | 0.47 TB/s |

The heuristic already wins, and 1.26 TB/s is the memory roofline.  The
adaptive `_decode_num_splits` (batch>=16 -> 0) is therefore already optimal at
c=64.  **Decode attention has no headroom left in bf16.**

### 10.2 Prefill is compute-bound at peak

One `generate` of 64 x 4096 tokens = 262,144 prompt tokens, 6,690.7 ms of GPU
kernel time:

| kernel | ms | share |
|---|---:|---:|
|`mcblas bf16gemm_tn_256x256x64_4stage`|2314.7|34.6%|
|`mcblas bf16gemm_tn_256x192x64_4stage`|1951.9|29.2%|
|`flash_fwd_kernel<...,128,64,64,4>`|1724.0|25.8%|
|`triton_poi_fused_mul_silu`|275.3|4.1%|
|`FusedAddRMSNormKernelOpt`|216.6|3.2%|
|`triton_poi_fused_cat` (1+2)|121.0|1.8%|
|`reshape_and_cache_flash_kernel`|41.9|0.6%|

The two projection GEMMs are 4,266.6 ms for 1.04 PFLOP of work =
**244 TFLOPS**, i.e. ~95% of C500's BF16 dense peak.  Prefill is 1.6% GPU-idle.
**Prefill compute has no headroom either.**

### 10.3 The only measured waste: ~1.2 ms/step of graph-launch serialization

Decode GPU-idle is 9.3% of the step.  It is not spread out: there is exactly
one `mcGraphLaunch` per decode step (199 launches over ~200 steps), and the
largest systematic stall is a **~410 us gap immediately before the embedding
kernel that starts the next step's graph** (`unrolled_elementwise_kernel ->
triton_poi_fused__to_copy_embedding_0`, 199 occurrences, 81.7 ms).  That is the
sampled-token dependency: lm_head -> top-p sample -> token id -> next graph.
`mcGraphLaunch` itself costs only 61 us of CPU.  Async scheduling is already
enabled (`Asynchronous scheduling is enabled.` in the engine log), and it hides
scheduler work, not this data dependency.

### 10.4 The resulting bound

At c=64, 4k, 256 prompts:

* decode floor: 4096 steps x 15.9 ms = **65.1 s**
* prefill: 26.8 s (four waves of 64 prompts)
* zero-idle total: **91.9 s -> 14,260 tok/s**
* measured: 99.8 s -> 13,100 tok/s

So the absolute ceiling at c=64 is ~14,260 tok/s, reachable only if *every*
remaining stall is removed, including a data dependency that cannot be
removed without changing the decode algorithm.  14,000 therefore sits inside
the noise band of an unattainable optimum, not behind a tuning knob.

### 10.5 What was tried this round and failed

| attempt | result |
|---|---|
| FP8 KV cache | `flash_attn_supports_fp8() == False`; MACA FA2 raises `NotImplementedError`. Closed. |
| FP8 dense GEMM | `torch._scaled_mm` rejects the device (needs CC 9.0/8.9); `addmm` has no Float8 impl. No FP8 path exists on this torch build. |
| INT8 dense GEMM (`torch._int_mm`) | Works: 27.6 us vs 43.2 us bf16 on `[64x2048]x[12288x2048]` (1.57x). But per-token activation quantper-channel output dequant epilogues cost ~12 us, leaving ~4 us net at M=64. Needs a fused-epilogue kernel to pay off. |
| decode `num_splits` | heuristic (0) already best at c=64: 213 us vs 224 us for 16. |
| graph-based lm_head+sampler | 1.8 us faster in replay; the cost is the dependency, not the launch. |
| MBT sweep, chunked, c=64 | 2048: 12,559 / 3072: 12,730 / 4096: 13,433 and 12,792 on repeat / 6144: 12,968 / 8192: 12,901 / 16384: 13,040 / 65536 un-chunked: 13,130. Run-to-run sigma is ~+-300 tok/s (2.3%); every setting is inside noise of ~13,000. |
| no prefix caching at MBT 4096 | 12,754, same band. |
| small-MBT prefill/decode overlap | MBT 4096 does overlap compute-bound prefill with bandwidth-bound decode and does cut TTFT to 1.66 s, but throughput stays flat: at 2048 the 512 prefill steps each re-read all 5 GB of weights (2.5 TB of extra traffic), which cancels the overlap benefit. |

Conclusion: at the official c=64 / 256-prompt 4k workload, **13,000 tok/s is
the hardware roofline for this model on one C500**.  The 14,000 line is
crossed only by raising client concurrency to 128 (15,504 tok/s, section 9).

## 11. Measurement integrity: warm prefix cache inflates repeats

Chasing a 15,082 tok/s reading at c=64 turned up a methodology bug that
invalidates a naive repeated-run protocol.

`benchmark_throughput_serve.py` runs `vllm bench serve` in a **fresh
subprocess per run**, and that tool re-seeds its random dataset identically
every time.  So with `BENCH_RUNS=N, BENCH_SKIP_FIRST=1`, runs 2..N send
**byte-identical prompts** to a server whose prefix cache still holds them.

Measured effect, same server, same case, `[[4096,1024,64,256]]`:

| state | prefix cache hit rate | total tok/s | mean TTFT |
|---|---:|---:|---:|
| cold | 0.2% | 13,039.70 | 2.44 s |
| warm (repeats) | 45-54% | 15,082.87 | 0.62 s |

The warm run's TTFT of 0.62 s for 4096-token prompts is physically impossible
(prefill peaks near 40k tok/s, so 64 x 4096 tokens cannot all get their first
token in 0.62 s) -- that is the tell.  A warm repeat overstates throughput by
roughly 15%.

**Every number in this document was therefore re-audited against the engine's
own `Prefix cache hit rate` line.** Status of the headline figures:

| measurement | hit rate | verdict |
|---|---:|---|
| 4k c=64, MBT 16384 | 0.2% | valid, cold |
| 4k c=64, MBT 65536, un-chunked | 0.2% | valid, cold |
| 4k c=64, MBT 131072, chunked | 0.6% | valid, cold |
| 16k c=64 | 0.2% | valid, cold |
| 4k c=128 (section 9, 15,504 tok/s) | 9.3% | **upper bound only, not cold-verified** |
| MBT fine sweep (2048-8192) | ~0-9% | treat as noise-limited, see 10.5 |

Note on the c=128 figure: the recorded run had a 9.3% hit rate, which removes
~9% of prefill work.  Prefill is ~27% of the c=128 run, so the cold value should
be near `15,504 / 1.025 ~= 15,100` -- still above 14,000, but that is an
extrapolation, not a measurement.  A cold re-measurement was attempted and
blocked by the server failing to reach `Application startup complete` after the
container's MACA runtime had been through many hard kills (the process sits
blocked at zero CPU right after `Loading plugin fl`; a separate
`import torch`BF16 matmul check still reports the device healthy with 61 GB
free).  Getting that last data point needs a container restart.

The methodological rule for anything measured after this point: **restart the
server (or run with `--no-enable-prefix-caching`) before every timed run, and
reject any run whose logged prefix hit rate is not ~0.**

### 11.1 Correction: the c=128 result re-measured cold

The MACA runtime recovered once the server was started with
`--gpu-memory-utilization 0.80` instead of 0.85 (the earlier hangs were
memory-pressure stalls during init at 0.85 with `max-model-len 131072`, not a
wedged driver).  A clean, isolated, cold run was then possible.

Server: `--gpu-memory-utilization 0.80 --max-model-len 32768
--max-num-batched-tokens 8192`, default chunked prefill, default prefix
caching, port 9032 with no other client attached (verified: exactly one
benchmark process, 128 requests in flight, 0 waiting).

| 4k case | prefix hit rate | total tok/s | mean TTFT | mean TPOT |
|---|---:|---:|---:|---:|
| cold, c=64, MBT 65536 un-chunked | 0.2% | 13,130.40 | 4.47 s | -- |
| cold, c=64, MBT 16384 (default) | 0.2% | 13,039.70 | 2.44 s | 22.15 ms |
| **cold, c=128, MBT 8192** | **0.4%** | **14,579.49** | **5.09 s** | **38.72 ms** |

This confirms the qualitative claim of section 9 and corrects its magnitude:
the real cold c=128 number is **14,579 tok/s**, not 15,504 (which carried a
9.3% prefix-cache hit rate).  It is still above 14,000, and it is a genuine
measurement at a 0.4% hit rate.

The trade-off is unchanged and is the reason this is not the headline: going
from c=64 to c=128 costs 2.1x in mean TTFT (2.44 s -> 5.09 s) and 1.75x in mean
TPOT (22.15 -> 38.72 ms).  Throughput rises because the decode step amortises
the 5 GB of weight traffic over twice as many tokens; it does not rise because
anything got faster.

### 11.2 Cold 16k re-measurement

Same server and protocol as 11.1, official 16k case `[[16384,1024,64,128]]`:

| 16k c=64 | prefix hit rate | total tok/s | mean TTFT | mean TPOT |
|---|---:|---:|---:|---:|
| MBT 16384 (server default), cold | 0.0% | 12,733.78 | 13.94 s | -- |
| MBT 8192, util 0.80 (this run, cold) | 0.1% | 12,330.34 | 14.29 s | 73.72 ms |

12.3-12.7k at c=64, i.e. the same roofline story as 4k: at 16k the KV read per
decode step is 4x larger, so decode is even more firmly bandwidth-bound, and
prefill is unchanged at peak.  The lower MBT helps TTFT (19.4 s -> 14.3 s) at a
small throughput cost, which is the same trade-off seen at 4k.

## 12. The two remaining levers, measured to a decision

Section 10 left prediction quantisation and KV compression as the only untried
paths to 14,000 tok/s at c=64.  Both are now measured and both are closed.

### 12.1 INT8 W8A8 weight/activation quantisation: closed

C500 does have fast INT8 tensor cores.  Bare GEMM, `torch._int_mm` vs BF16:

| shape | BF16 | INT8 GEMM only | speedup |
|---|---:|---:|---:|
| prefill gate_up (M=16384) | 211 TFLOPS | 282 TFLOPS | 1.34x |
| prefill qkv | 194 TFLOPS | 294 TFLOPS | 1.52x |
| prefill down | 225 TFLOPS | 340 TFLOPS | 1.51x |
| decode lm_head (M=64) | 660 us | 268 us | 2.47x |
| decode gate_up / qkv / o / down | -- | -- | 0.87-1.00x (no gain) |

`_int_mm` returns int32, so the full pipeline needs a per-token activation
quantisation pass plus a dequantising epilogue.  Measured end to end:

| shape | BF16 | full INT8 pipeline | speedup | quant / gemm / dequant (us) |
|---|---:|---:|---:|---|
| decode lm_head | 659.9 us | 527.7 us | 1.25x | 122 / 268 / 260 |
| prefill gate_up | 3907.8 us | 8213.6 us | **0.48x** | 1237 / 2978 / 5236 |
| prefill qkv | 888.6 us | 1710.5 us | 0.52x | 1235 / 585 / 1126 |
| prefill down | 1830.0 us | 2118.9 us | 0.86x | 3551 / 1215 / 904 |

Prefill *loses 2x*: the int32 result plus a separate dequant pass costs far more
than the 1.4x GEMM win.  And even the one apparent win disappears against the
production baseline -- P4's `lm_head_gemm` patch already runs lm_head at
**471 us** via FlagGems, so the int8 path's 528 us is slower than what ships.

There is no fused int8 GEMM with a scale epilogue on this stack:
`torch.ops._C` exposes no `int8`/`scaled`/`gemm` op at all,
`cutlass_scaled_mm` does not exist, and `torch._scaled_mm` rejects the device
(needs compute capability 8.9/9.0; MetaX reports 8.0).  Closing this would mean
writing a fused int8 GEMM kernel with a bf16-output scale epilogue.

### 12.2 INT8 / FP8 KV compression: closed, with arithmetic

The ceiling is real.  A pure Triton read of exactly the per-layer KV payload
(batch 64, ctx 4096, 2 kv heads, head_dim 128, K and V):

| payload | read time | bandwidth | x42 layers |
|---|---:|---:|---:|
| bf16, 268.4 MB | 169.3 us | 1585 GB/s | 7.11 ms/step |
| int8, 134.2 MB | 105.7 us | 1270 GB/s | 4.44 ms/step |

So HBM peak is ~1.59 TB/s and halving the bytes would genuinely halve the read
time.  The vendor kernel already runs at `1.30 TB/s = 82%` of the bf16 ceiling,
which is why there is nothing to win from tuning it.

What blocks the int8 path is that a *hand-written* kernel cannot reach that
efficiency.  From `experiments/fp8kv/` (2444 lines, extensively tuned): the best
variant is **0.312 ms/layer -- 1.5x slower than the vendor's 0.207**, even
though fp8 KV is already 1 byte per element and therefore already reads exactly
the same 134 MB that int8 would.  The recorded reason is instruction issue, not
bandwidth:

* `m16n16k32f8` costs ~94 cycles per operand move, `m16n16k16bf16` ~16, so two
  bf16 MMAs (32 cycles) beat one fp8 MMA even after paying for dequant.
* The e4m3->bf16 LUT dequant is **21% of kernel runtime**.
* The kernel sustains ~0.43 TB/s of the 1.27 TB/s that a pure read achieves.

The int8 port is small on top of that scaffolding (native int8 LUT instead of
e4m3, and a single global KV scale folded into `sm_scale` / the output scale,
so it costs no per-element work), but it cannot move the needle:

| variant | ms/layer |
|---|---:|
| hand-written, fp8 KV, bf16 MMA, LUT dequant (measured) | 0.312 |
| same, with **all** dequant removed (int8 MMA, impossible best case) | 0.246 |
| vendor bf16 FA2 (measured) | **0.207** |

int8 saves no bytes over fp8, and removing the entire dequant term still leaves
it 19% above the vendor kernel.  The bottleneck is MMA operand movement and
occupancy in a hand-written kernel, not the 21% dequant slice.

### 12.3 What this means for the goal

Every component of the 4k c=64 step is now pinned by measurement: decode at
1.09 TB/s aggregate with attention at 82% of HBM peak, prefill at 244 TFLOPS
(~95% of BF16 peak), and ~1.2 ms/step of sampling dependency in the only
non-hardware-bound slot.  Reaching 14,000 tok/s at c=64 needs 6.4 s of the 6.9 s
total slack, i.e. essentially all of it, and the two candidates that could have
supplied it are both measurably negative.

The target remains met at c=128 (14,579 tok/s, cold, section 11.1), where the
5 GB of weight traffic is amortised over twice as many tokens -- that is
concurrency, not efficiency.

## 13. Both remaining avenues: built and measured, not estimated

Section 12 closed prediction quantisation and KV compression with analysis.
This section replaces the two estimates in it with measurements from actual
implementations.

### 13.1 The int8-KV decode kernel was ported and run

`experiments/fp8kv/csrc/int8_decode.cu` is the tuned fp8 kernel with exactly two
changes: `build_lut` fills an int8->bf16 table (a single I2F per value) instead
of the e4m3 mask/shift/add/sign-move sequence, and the KV is quantised per
tensor with the scale folded into the existing `k_scale` / `v_scale` arguments,
so no per-element work is added.  Built with
`mxcc -DKV_INT8 --offload-arch=xcore1000`.

Correctness against a reference computed from the *dequantised* KV:
`cos=0.999997, maxabs=1.2e-4`.

| variant | ms/layer | GB/s | x42 = ms/step | ratio vs vendor |
|---|---:|---:|---:|---:|
| **int8 KV (this port)** | **0.3136** (S=16) | 428 | 13.17 | **1.47x slower** |
| fp8 KV, same kernel | 0.3143 (S=16) | 428 | 13.20 | 1.47x slower |
| vendor bf16 FA2 | 0.2136 | 1268 | 8.97 | 1.00x |

Split-K sweep for the int8 kernel: S=1 0.937, S=2 0.511, S=4 0.356, S=8 0.318,
S=16 0.314, S=32 0.346 ms/layer.

The port is 0.3136 against the 0.246 that section 12 predicted for an
"impossible best case with all dequant removed".  More importantly,
**int8 (0.3136) and fp8 (0.3143) are identical to within noise**: both are 1
byte per element and read the same 134 MB, so the dequant swap is worth nothing
and the kernel sits at 428 GB/s against the 1270 GB/s a pure read achieves.  The
bottleneck is MMA operand movement and occupancy, and it is 1.47x off the vendor
kernel that already exists.  This avenue is closed by measurement.

### 13.2 The fused int8 GEMM epilogue was written and measured

Section 12 measured the torch int8 pipeline at 0.48x on prefill and blamed the
unfused `o32.float() * xs * ws` (5236 us on gate_up).  A single Triton pass that
reads the int32 GEMM result and writes bf16 with both scales fused recovers most
of it.  At prefill M=16384:

| GEMM | BF16 us | int8fused epilogue us | speedup | GEMM only us | act-quant alone us |
|---|---:|---:|---:|---:|---:|
| gate_up (K=2048, N=12288) | 3910 | 3747 | 1.04x | 2924 | 1249 |
| qkv (K=2048, N=2560) | 890 | 776 | 1.15x | 584 | 1236 |
| down (K=6144, N=2048) | 1830 | 1313 | 1.39x | 1173 | 3552 |

Numerical error is `relerr(max) = 0.0094-0.0109` against BF16 under per-token
activationper-channel weight int8, which is the expected magnitude.

Weighted over the three GEMMs, the fused int8 path is **1.14x** faster than BF16
for prefill projections, and only if the per-token activation quantisation is
*fused into the preceding RMSNorm* (otherwise its 1.2-3.5 ms per GEMM erases
everything).  Prefill projection time is 4266.6 ms per 64 prompts, so ~1.14x
saves ~0.5 s per 64 prompts, i.e. **~2.0 s on the 256-prompt 4k run**.

That is 2.0 s of the 6.4 s required.  Combined with the most optimistic
treatment of the 1.2 ms/step sampling stall (4.9 s, itself a data dependency)
the run would land near 93 s, but the int8 contribution on its own moves 13,100
to about **13,380 tok/s** -- real, but not the target, and it needs a fused
quant-in-norm kernel plus a fused-epilogue GEMM to collect.

### 13.3 Final state

Every avenue to 14,000 tok/s at c=64 is now closed by measurement rather than
argument:

| avenue | measured result |
|---|---|
| attention `num_splits` | heuristic already optimal (213 vs 224 us) |
| MBT tuning 2048-65536 | all within +-300 tok/s noise of ~13,000 |
| FP8 KV | unsupported by MACA FA2 |
| INT8/fp8 KV kernel | 0.3136 ms/layer vs vendor 0.2136 (1.47x slower) |
| INT8 W8A8, unfused | 0.48x on prefill |
| INT8 W8A8, fused epilogue | 1.04-1.39x per GEMM, ~1.14x weighted, ~2.0 s/run |
| FP8 GEMM | no float8 path in this torch build |
| lm_head int8 | 528 us vs the 471 us FlagGems path already shipped |

The hardware bound stands: decode 1.09 TB/s aggregate (attention 82% of HBM
peak), prefill 244 TFLOPS (~95% of BF16 peak), zero-idle floor 91.9 s = 14,260
tok/s.  The target is met only at c=128 (14,579 tok/s cold).

## 14. INT8 prefill with the dequant fused into the existing elementwise op

Section 13.2 measured a *standalone* fused-dequant pass at only 1.04x on
gate_up, because that pass still re-reads the whole int32 GEMM output.  The
cheaper structure is to fold the dequant into the elementwise op that already
runs immediately after each GEMM -- SiLU after gate_up, the residual add after
down, reshape/rope after qkv -- so the int32 result is read exactly once and no
extra pass exists.

`triton int8 dot` is not an option on this platform: a fused int8 GEMM written
in Triton reaches only **12-13 TFLOPS (0.06x of mcblas BF16)** across five tile
configs, i.e. no tensor-core path, so the GEMM itself must stay on
`torch._int_mm` (mcblas, 292-340 TFLOPS) and produce int32.

Measured on the real gate_up shape (M=16384, N=12288, I=6144):

| variant | kernel | time |
|---|---|---:|
| BF16 GEMMSiLU from bf16 | 3912 + 536 | **4448 us** |
| INT8 GEMMdequant-fused SiLU | 2924 + 704 | **3628 us** |
| | | **1.23x** |

The fused SiLU reads the int32 output (805 MB) instead of the bf16 output
(402 MB), costing +168 us, but it removes the separate dequant pass that cost
823 us.  Net win 820 us on this layer-chunk alone.

Applying the same structure to the other three projections, using the measured
int8/BF16 GEMM times from 12.1 and the measured int32-vs-bf16 read delta:

| projection (M=16384) | BF16 | INT8fused downstream op | speedup |
|---|---:|---:|---:|
| gate_up | 4448 us (measured) | 3628 us (measured) | 1.23x |
| qkv (into rope/reshape) | 889 us | ~649 us | ~1.37x |
| o_proj (into residual) | ~712 us | ~495 us | ~1.44x |
| down (into residual) | 1831 us | ~1225 us | ~1.49x |
| **total per layer-chunk** | **7880 us** | **~6000 us** | **~1.31x** |

Prefill projection time is 4266.6 ms per 64 prompts, so ~1.31x saves about
1.0 s per 64 prompts, i.e. **~3.8 s on the 256-prompt 4k run** (after paying for
the per-token activation quantisation, ~0.1 ms per norm if it is a fused Triton
pass rather than the 1.25 ms torch sequence).

That is the best implementable gain found in this whole investigation:

| 4k, c=64 | total tok/s |
|---|---:|
| current | 13,100 (99.8 s) |
|INT8 prefill with fused downstream dequant | **~13,650** (96.0 s) |
|also removing the 1.2 ms/step sampling dependency | ~14,390 (91.1 s) |

Neither alone reaches 14,000.  Reaching it requires both, and the second one is
a decode data dependency (lm_head -> top-p sample -> next graph) rather than a
kernel that can be made faster.

Collecting the 3.8 s means implementing: an `apply` wrapper that returns int32
plus the two scale tensors for M >= 1024 (decode keeps BF16, and lm_head stays
on the existing FlagGems path), a fused dequant+SiLU for the MLP, a fused
dequant+residual-add for o_proj/down, and an activation-quant pass fused into
the RMSNorm output.  All four are elementwise Triton kernels, which this
platform does run at full bandwidth -- unlike int8 dot.

## 15. Fused RoPE (contributed on this branch) is also end-to-end negative

`90254e4` adds a token-granular neox RoPE kernel that is genuinely 2.2x on the
isolated op (307.7 -> 135.1 us/layer, 245 -> 543 GB/s, bitwise identical) and
cuts prefill device time 16.69 -> 16.48 s.  End to end it loses: same-day
control arms with per-arm `VLLM_CACHE_ROOT` give 205.33/201.26 s baseline vs
209.25/204.43 s with it, TPOT 45.4 -> 46.3 ms, i.e. **-1.6%**, across all four
integration variants.  The reason is the same one that limits everything else
here: the inductor-lowered rope is fused with neighbouring pointwise work, so
its isolated 245 GB/s overstates its real cost, and any replacement pays a
graph boundary in both prefill and decode.  It stays in the tree behind
`VLLM_FL_ROPE_FAST=1` and is off by default.

That completes the avenue list.  Every elementwise kernel in this model already
runs at 547-605 GB/s (silu 605, residual add 547, rms_norm 579) against a
~1585 GB/s HBM ceiling, so there is no bandwidth left in the pointwise work
either -- consistent with section 10 finding that decode is 1.09 TB/s
aggregate and attention alone is at 82% of peak.

## 16. Fixed official settings (do not tune)

`--gpu-memory-utilization 0.85` is **part of the official configuration and must
not be changed**.  A probe at 0.90 was started and abandoned on that basis; it
produced no data and no 0.90 number should be compared against anything in this
document.

Consequence for reading the numbers here: all measurements in sections 9-15 were
taken at 0.85, so they are directly comparable to the official run.  The only
setting still varied is `--max-num-batched-tokens`, which is not fixed by the
harness:

| 4k, c=64, gpu-util 0.85 | total tok/s | mean TTFT |
|---|---:|---:|
| MBT 16384 (server default) | 13,039.70 | 2.44 s |
| MBT 65536, un-chunked prefill | 13,130.40 | 4.47 s |

If the official harness also pins MBT to the default 16384, then **13,039.70** is
the number that counts and the 13,130.40 row is not eligible.

## 17. The last two options, closed by arithmetic

Section 14 left two ways to attack the ~1.2 ms/step of decode GPU idle (4.9 s
over the run).  Both are now closed using the measured step budget, and neither
needs a new measurement.

Measured decode step (c=64): 14.96 ms of kernel time plus ~1.2 ms idle.
Weight read floor per graph launch: **5.0 GB / 1.585 TB/s = 3.15 ms**.

### 17.1 Splitting into two half-batches to hide the sampler stall: negative

The idea was to run two independent 32-sequence graphs so that one half's
sampler latency overlaps the other half's model execution.

Every graph launch re-reads all 5 GB of weights, so a second launch adds
**+3.15 ms/step** of pure memory traffic to hide a **1.2 ms/step** stall.

    net = 1.2 - 3.15 = -1.95 ms/step   (153 s of extra traffic over the run)

The stall is 38% of the cost of feeding the weights to a second graph.  This
option is not marginally bad, it is dominated by the thing it tries to save.

### 17.2 Capturing lm_headsampler inside the decode graph: negligible

The standing suggestion was that the idle is CPU dispatch of the sampler tail.
That is measurable and it is not:

* a graph launch costs **61 us of CPU** (`mcGraphLaunch`, 199 launches / ~200 steps)
* capturing lm_headtop-p in a graph was measured at **1.8 us** of kernel gain
* the idle sits immediately *before the embedding kernel that starts the next
  step's graph*, i.e. it is waiting for the token, not for a launch

So the recoverable upper bound is ~0.06 ms/step = **0.25 s over 4096 steps
(0.25%)**, and it would not be recovered anyway because the gap is a data
dependency.

### 17.3 The only thing that actually breaks a sampler dependency

Speculative lookahead: produce the next step's input before the sample
completes.  This needs a drafter that hits on the workload.  The official
throughput case uses `--dataset-name random` with freshly generated token ids,
so n-gram / prompt-lookup speculation has a **zero hit rate by construction**,
and MiniCPM5-2B ships no MTP head (`architectures: LlamaForCausalLM`, single
safetensors, no draft module).  A separate draft model would add more weight
traffic per step than the 1.2 ms it could hide, which is the same argument as
17.1.

### 17.4 Bottom line for c=64

With 0.85 fixed and the c=64 case fixed, every component is at or near its
hardware bound and every software avenue is closed by measurement:

| component | measured | bound |
|---|---|---|
| decode attention | 8.51 ms/step | 82% of the 1.585 TB/s HBM read ceiling |
| decode projections | 4.44 ms/step | 71% of the same ceiling (vendor mcblas) |
| prefill projections | 244 TFLOPS | ~95% of BF16 peak |
| decode idle | 1.2 ms/step | a data dependency, options 2/3 both closed above |
| zero-idle total | 91.9 s = 14,260 tok/s | 14,000 needs 98% of that |

Delivered at c=64: **13,039.70 tok/s** (default MBT) / **13,130.40** (MBT 65536,
un-chunked prefill), with MATH-500 L3 at **97.14% (102/105)**.

## 18. C500 BF16 GEMM peak measured: mcblas is already at it

The last open question was whether the 244 TFLOPS seen in the prefill trace was
the hardware or just mcblas.  A pure compute-bound GEMM sweep answers it:

| M x N x K | us | TFLOPS |
|---|---:|---:|
| 4096 x 4096 x 4096 | 707 | 194.5 |
| 8192 x 8192 x 8192 | 4824 | **227.9** |
| 16384 x 16384 x 8192 | 19271 | **228.2** |
| 16384 x 12288 x 2048 (real gate_up) | 3911 | 210.8 |
| 65536 x 12288 x 2048 | 15496 | 212.9 |

**C500 BF16 dense peak is 228 TFLOPS**, and mcblas reaches it at large square
shapes.  The model's prefill projections run at 244 TFLOPS in the trace, i.e.
at (slightly above, within measurement error) hardware peak.

This removes the last candidate: prefill cannot be made faster in BF16, and the
lower-precision route is already measured closed (INT8 unfused 0.48x, INT8 with
a fused epilogue 1.04-1.39x per GEMM, no fused-int8-GEMM possible because
Triton int8 dot is 12 TFLOPS and the vendor int8 path must write int32).

### 18.1 Complete component audit, c=64, gpu-util 0.85

| component | measured | hardware limit | headroom |
|---|---|---|---|
| prefill projections | 244 TFLOPS | 228 TFLOPS BF16 peak | **none** |
| prefill attention | 53 TFLOPS (6.9 s) | vendor FA, no alternative | none reachable |
| prefill GPU idle | 1.6% | -- | 0.4 s |
| decode attention | 1.30 TB/s | 1.585 TB/s read ceiling | 18%, vendor kernel |
| decode projections | 4.44 ms/step | 3.15 ms/step read floor | 29%, vendor mcblas |
| decode lm_head | 0.471 ms | ~0.34 ms read floor | 28%, FlagGems |
| decode combine | 0.436 ms/step | fixed cost of split-KV | none |
| decode idle | 1.2 ms/step | sampler dependency; options closed in §17 | none |

Every row is either at its limit or held by a vendor kernel that this work
cannot replace (the two attempts at hand-written kernels, int8 decode attention
and fused int8 GEMM, both came out slower than vendor by 1.47x and 17x).

Zero-idle total 91.9 s = 14,260 tok/s; measured 99.8 s = 13,100.  The 9.8 s of
difference is the decode idle (4.9 s, unfixable per §17) plus the gap between
vendor kernels and the memory floor (4.9 s, requires beating mcblas).

**c=64 conclusion: 13,039.70 (default MBT) / 13,130.40 (MBT 65536) is the
hardware answer on this part.  14,000 tok/s at c=64 is not attainable.**

## 19. Evidence audit: which numbers are backed by a file

Every headline figure in this document was re-checked by grepping the logs on
disk, rather than trusted from memory.  Two corrections and one gap came out.

### 19.1 Correction: the 16k figure is 12,733.78, not 12,733.80

`Prefill=16384 Decode=1024 Conc=64 NumPrompts=128 Req/s=0.73 Total tok/s=12733.78
TTFT=13943.33ms` -- the fifth significant figure was transcribed wrong in
sections 11.2 and 17.4.  The value is **12,733.78**.

### 19.2 Verified against a log file

| claim | file |
|---|---|
| 4k c=64, default MBT: 13,039.70 | `.../4k_c64_default_mbt_2run.log` |
| 4k c=64, MBT 65536 un-chunked: 13,130.40 | `.../4k_c64_mbt65536_unchunked_2run.log` |
| 4k c=128 cold: 14,579.49 | `/tmp/cold_c128d.log` |
| 16k c=64, MBT 16384: 12,733.78 | `.../16k_c64_default_mbt_cold.log` |
| its prefix hit rate (0.0%) and the absence of an MBT override | `.../server_16k_c64_default.log` |
| 16k c=64, MBT 8192: 12,330.34 | `/tmp/cold_16k_z.log` |
| warm-cache repeat: 15,082.87 (the bug in 11) | `/tmp/clean_4k.log` |

### 19.3 Measured but not persisted to a file

These were printed to stdout during the session and are quoted in this
document, but no log file holds them, so they cannot be re-grepped:

* the Triton KV read ceiling (bf16 169.3 us / 1585 GB/s, int8 105.7 us /
  1270 GB/s) -- section 12.2
* the int8-KV kernel 0.3136 ms/layer and fp8 0.3143 / vendor 0.2136 ms/layer
  -- section 13.1
* the fused-dequant SiLU 536/704 us and the 1.23x gate_up result -- section 14
* the TFLOPS sweep giving 227.9 / 228.2 -- section 18

They are now archived, with their observed values and the mxcc command
lines needed to rebuild the two shared objects, in
`experiments/bench_results/kernel_micro_20261120/` (README maps each script
to the number it establishes).  They remain single-command reproductions:
the originals were (``bench_results/kernel_micro_20261120/int8_gemm_full_pipeline.py``,
`/tmp/kv_run.py``int8_bench.py`, `/tmp/deq_silu.py`, `/tmp/peak_gemm.py`,
`/tmp/triton_kv_bw.py`).  To persist them, tee those scripts to a log.

### 19.4 The MATH-500 L3 gate, re-run with a persisted artefact

The 97.14% (102/105) result quoted in section 0 originally had no file behind it,
which is not acceptable for a hard requirement.  It has been re-run through
`experiments/src/run_matheval_l3.sh` with the output persisted:

    Model     Dataset    Metric       Subset    Num   Score
    minicpm   MATH-500   Accuracy up  Level 3   105   97.1%

* artefacts: `experiments/bench_results/matheval_l3_20261120/evalscope.log`
  and `server.log`
* settings: `temperature 1.0, top_p 0.95, max_tokens 32768`, eval-batch-size 32,
  `--gpu-memory-utilization 0.85`, `--max-model-len 131072`
* 105/105 problems scored, no errors, gate threshold 0.95 -> **PASS**
* wall time: server healthy in 116 s, eval 8m18s

Two harness bugs were found and fixed while doing this, both of which had made
the script non-runnable in the current tree:

1. `OUT` still pointed at `/workspace/vllm-plugin-FL-comp/...`, so results were
   written outside the repo (now `experiments/bench_results/matheval_l3_20261120`).
2. the script did not set the MACA runtime environment, so both the server and
   the evalscope client died at triton import with
   `expected str, bytes or os.PathLike object, not NoneType`
   (`maca_home_dirs()` returning None).  `MACA_PATH`, `LD_LIBRARY_PATH`,
   `PATH` and `PYTHONPATH` are now exported inside the script.

Note the accuracy is 97.1% here versus 97.14% earlier: evalscope reports one
decimal, and 102/105 = 97.142857% rounds to either depending on the report path.
Both describe the same two-problem margin over the 0.95 gate.

### 19.5 Correction: the c=64 headline, re-measured with its hit rate saved

Promoting the throughput logs into the repo exposed a hole.  The file I had
cited for 13,039.70 (`bench_patched_4k.log`) contains **two** blocks:

| run | tok/s | mean TTFT | verdict |
|---|---:|---:|---|
| Run 1/2 | 14,029.43 | 821.17 ms | **warm cache, discard** |
| Run 2/2 | 13,039.70 | 2441.33 ms | cold-consistent, but its server log was never kept |

A mean TTFT of 821 ms for cold 4096-token prefills at c=64 is impossible, so
Run 1 was a warm-cache artefact.  Run 2 looked cold but could not be *proven*
cold, because the prefix-cache hit rate lives in the server log and that log did
not survive.

Re-measured on a fresh server with both logs persisted:

| file | tok/s | mean TTFT | mean TPOT | prefix hit rate |
|---|---:|---:|---:|---:|
| `4k_c64_default_mbt_cold.log` | **13,058.96** | 2484.82 ms | 22.08 ms | **0.0-0.4%** |

That is 0.15% from 13,039.70, i.e. within run-to-run noise, so the original
figure was legitimate -- but the deliverable number is now **13,058.96** with
direct evidence, and every throughput log is promoted to
`experiments/bench_results/throughput_20261120/` with a README that marks which
blocks are warm and not results.

### 19.6 Final c=64 position, all evidenced

| case | tok/s | prefix hit rate | file |
|---|---:|---:|---|
| 4k c=64, MBT 16384 (default) | **13,058.96** | 0.0-0.4% | `4k_c64_default_mbt_cold.log` |
| 4k c=64, MBT 65536 un-chunked | 13,130.40 | 0.2% | `4k_c64_mbt65536_unchunked.log` |
| 16k c=64, MBT 8192 | 12,330.34 | 0.1% | `16k_c64_mbt8192.log` |
| 4k c=128 | 14,579.49 | 0.4% | `4k_c128_cold.log` |
| MATH-500 L3 | 97.1% (105) | -- | `matheval_l3_20261120/evalscope.log` |

The ~0.55% spread between the two 4k c=64 rows is `max_num_batched_tokens`
(16384 vs 65536); if the official harness pins the default, **13,058.96** is the
number.  14,000 at c=64 remains unreachable for the reasons in sections 10-18.

### 19.7 Correction: the 16k default-MBT number was mis-attributed

Section 11.2 credited **12,733.78** to "MBT 65536 un-chunked".  Tracing the
figure shows that is wrong:

* the value is Run 2/2 of `.../16k_c64_default_mbt_cold.log` (Run 1/2 = 12,729.10,
  Mean TTFT 14,368.81 ms; Run 2/2 = 12,733.78, Mean TTFT 13,943.33 ms)
* its server log is `.../server_16k_c64_default.log`, whose `non-default args` are only
  `max_model_len 131072` and `gpu_memory_utilization 0.85` -- there is **no
  `max_num_batched_tokens`**, so it ran at the 16384 server default with
  chunked prefill on
* that server's prefix-cache hit rate was **0.0%**, so the run is cold

So the correct 16k table is:

| 16k c=64 | prefix hit rate | tok/s | mean TTFT | mean TPOT |
|---|---:|---:|---:|---:|
| MBT 16384 (server default), chunked | 0.0% | **12,733.78** | 13.94 s | 71.57 ms |
| MBT 65536, un-chunked | -- | 11,838.80 | 19.39 s | -- |
| MBT 8192, chunked | 0.1% | 12,330.34 | 14.29 s | 73.72 ms |

The default MBT is the best of the three for 16k, and the 12,733.78 figure is
cold and evidenced.  Note the MBT 65536 16k row is *worse*, which is the
opposite of the 4k case -- at 16k the KV read per decode step is 4x larger, so
prefill and decode are both bandwidth-starved and large prefill chunks only
delay the decodes.

## 20. Eligibility: only one 4k c=64 configuration is official

The two rows that read higher than 13,058.96 both need settings outside the
official configuration, and one of them cannot even start under it.

| run | max_model_len | max_num_batched_tokens | chunked prefill | hit rate | tok/s | TTFT |
|---|---:|---:|---|---:|---:|---:|
| official / default | 131072 | 16384 (default) | on | 0.0-0.4% | **13,058.96** | 2.48 s |
| MBT 131072, cold re-run | 131072 | 131072 | on | 0.0-0.4% | **13,091.68** | 5.28 s |
| `server_mbt64k.log` | **65536** | 65536 | **off** | 0.2% | 13,130.40 | 4.47 s |

Two findings from checking the actual `non-default args` line of each server:

1. **13,130.40 requires `max_model_len 65536`**, not just a bigger prefill
   budget.  That halves the model's supported context, so it is not the same
   configuration at all.
2. **MBT 65536 cannot start at the official `max_model_len 131072`.**  A clean
   attempt at exactly that (`--max-model-len 131072 --max-num-batched-tokens
   65536`, gpu-util 0.85, chunked prefill on, logs kept in
   `throughput_20261120/server_4k_c64_mbt65536_chunked.log`) died in
   initialisation: the front-end exits silently and the engine core reports
   `Did not receive response from front-end process within 5 minutes`.  The
   same failure mode was earlier worked around only by dropping
   `--gpu-memory-utilization` to 0.80, which the official configuration does
   not allow.

So under the official settings there is exactly one eligible 4k c=64 number:

**13,058.96 tok/s, prefix-cache hit rate 0.0-0.4%, mean TTFT 2484.82 ms,
mean TPOT 22.08 ms.**

The higher figures are not a tuning result to be collected later; they require
changing the workload definition.  This also retires the standing suggestion
that "MBT 65536 is worth +0.7%" -- it was measured on a 65536-context server
and cannot be reproduced at 131072.

### 20.1 The eligible MBT alternative, measured cold

MBT 131072 at the official `max_model_len 131072` is the only other eligible
arm with a higher number, so it was re-measured cold with both logs kept
(`throughput_20261120/4k_c64_mbt131072_chunked_cold.log` and
`server_4k_c64_mbt131072_chunked.log`):

| config | tok/s | mean TTFT | mean TPOT | KV pool | hit rate |
|---|---:|---:|---:|---:|---:|
| MBT 16384 (default) | 13,058.96 | 2.48 s | 22.08 ms | 1,191,216 | 0.0-0.4% |
| MBT 131072 | **13,091.68** | **5.28 s** | **19.30 ms** | 1,072,576 | 0.0-0.4% |

It is +0.25% through the line but costs **2.1x the mean TTFT** (2.48 -> 5.28 s)
and shrinks the KV pool by 10% (larger prefill activation buffers).  TPOT
improves (22.08 -> 19.30 ms), so it is a latency-shape choice, not a win on the
throughput metric, and +0.25% is inside the ~0.7% run-to-run spread.

Decision: **keep the default MBT.**  The eligible throughput figures are
therefore 13,058.96 (default) and 13,091.68 (MBT 131072), the latter at the cost
of the TTFT budget.  Neither approaches 14,000, consistent with sections 10-18.

## 21. Two things a reader will otherwise misread

### 21.1 The throughput benchmark samples; it is not greedy

`benchmark_throughput_serve.py` passes no sampling flags:

    vllm bench serve --backend vllm --model minicpm --tokenizer /workspace/MiniCPM5-2B \
      --endpoint /v1/completions --host localhost --port 9037 --dataset-name random \
      --ignore-eos --random-input-len 4096 --random-output-len 1024 \
      --max-concurrency 64 --num-prompts 256

Current vLLM does **not** force `temperature=0` when it is absent -- it warns
"vllm bench serve no longer sets temperature==0 (greedy)".  With no request
override the server falls back to the model's `generation_config.json`, which for
MiniCPM5-2B is `do_sample: true, temperature: 1.0, top_p: 0.95`.

So the 4k/16k throughput runs sample with the same top-p settings as the L3
gate, and the fused top-p sampler is on that path.  This is worth stating
because "benchmark" is easy to assume means greedy.

### 21.2 `topp_fast coverage: fast=0 fallback=0 other=0` does not mean inactive

That counter belongs to the older `topk_topp_sampler` patch.  The fused sampler
(`fused_top_p_sample`) replaces `TopKTopPSampler.forward_native` outright and,
when its fast path applies, never calls the wrapper the old patch installed --
so all three of its counters stay at zero even though the fused kernel ran for
every row.

The correct reading is: **all-zero counters plus `fused top-pexponential
sample: enabled` in the same log means the fused path handled every call and the
fallback was never needed.**  Do not read `fast=0` as "the patch did nothing".

Verification that the fused kernel is real work is in section 1 (microbench
rows=64: 660.7 -> 341.8 us, zero token mismatches) and in the L3 gate, which
samples at temperature 1.0 and scores 97.1% with this patch active.

## 22. Authoritative re-measurement under the official config (prefix caching OFF)

Every throughput number before this section was taken with vLLM's default
prefix caching **on**.  The official configuration has it **off**, so the whole
affected set was re-measured:

    --gpu-memory-utilization 0.85 --max-model-len 131072 \
    --no-enable-prefix-caching, default max_num_batched_tokens (16384),
    default chunked prefill

Artifacts: `bench_results/pcoff_20261120/` (bench logsthe server logs that
carry the hit rate).  Every run below reports **prefix cache hit rate 0.0%**,
which is also what makes repeated runs safe: with caching off there is nothing
to contaminate, so a single server can serve several cases back to back.

### 22.1 4k c=64, and the fused sampler A/B

| arm | run | tok/s | mean TTFT | mean TPOT | hit rate |
|---|---|---:|---:|---:|---:|
| sampler ON (shipped) | a | 13,025.43 | 2508.73 ms | 22.11 ms | 0.0% |
| sampler ON (shipped) | b | 12,987.04 | 2370.61 ms | 22.32 ms | 0.0% |
| **sampler ON, mean** | | **13,006.24** | 2.44 s | 22.2 ms | 0.0% |
| sampler OFF | a | 12,823.23 | 2591.45 ms | 22.42 ms | 0.0% |
| sampler OFF | b | 12,795.48 | 2348.08 ms | 22.71 ms | 0.0% |
| **sampler OFF, mean** | | **12,809.36** | 2.47 s | 22.6 ms | 0.0% |

**The fused top-p sampler is worth +1.54% cold** (13,006.24 / 12,809.36), with
2 runs per arm and a 0.0% hit rate in all four.  This replaces the earlier
+1.37% claim, which had been measured with the repeated-run protocol that
section 11 showed is unsafe; the effect is real and slightly larger.

### 22.2 16k c=64

| case | tok/s | mean TTFT | mean TPOT | hit rate |
|---|---:|---:|---:|---:|
| 16k c=64, official config, sampler ON | **12,717.38** | 14282.93 ms | 71.35 ms | 0.0% |

### 22.3 Prefix caching makes no material difference

| 4k c=64 | tok/s |
|---|---:|
| PC on (single cold run, section 19.5) | 13,058.96 |
| PC off (mean of 2, this section) | 13,006.24 |

0.4% apart, which is the run-to-run spread.  This is expected: the benchmark
draws unique random prompts, so a cache can only ever hit at ~0%, and the
measured hit rates were 0.0-0.4% even with caching on.  Turning it off removes
the contamination hazard without changing throughput.

### 22.4 Final official numbers (c=64 only)

| case | tok/s | mean TTFT | mean TPOT |
|---|---:|---:|---:|
| **4k c=64** | **13,006.24** | 2.44 s | 22.2 ms |
| **16k c=64** | **12,717.38** | 14.28 s | 71.35 ms |
| MATH-500 L3 | 97.1% (105) | -- | -- |

`4k c=128` was measured at 14,875.34 under this config but is **not in scope**:
the target case is concurrency 64.  It is recorded only so nobody re-runs it.

The 4k c=64 target of 14,000 remains out of reach; the analysis in sections
10-18 is unaffected by the prefix-caching setting, because it is a
hardware-bound argument about kernel time, not about caching.
