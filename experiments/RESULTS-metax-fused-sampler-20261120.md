# MetaX C500 — fused top-p sampling and the 4k/16k ceiling (2026-11-20)

Platform: MetaX C500, 63.59 GiB, MACA 3.7.0.37, vLLM 0.24.0,
vllm-plugin-FL `flagos-2026-s2/metax-mm-strategy`, FlagGems 5.3.5.

Goal: raise 4k throughput toward 14,000 tok/s while keeping MATH-500 Level 3
above 95%.

## 0. Result

| case | baseline | fused sampler | change |
|---|---:|---:|---:|
| 4k `[[4096,1024,64,256]]` | 12,864.06 | **13,039.70** | **+1.37%** |
| 16k `[[16384,1024,64,128]]` | 12,664.04 | **12,733.78** | **+0.55%** |
| MATH-500 L3 | 97.14% (102/105) | **97.14% (102/105)** | PASS |

4k and 16k are two-run summaries (`BENCH_RUNS=2`, `BENCH_SKIP_FIRST=1`).
4k baseline and fused are the same session; 16k baseline is the 2026-10-31
default-stack result from `FINDINGS-mx.md` §32.

The target 14,000 tok/s was **not reached**. The remaining gap is 960 tok/s
(7.4%), and the measured decode-side floor says it is not reachable by tuning
the current operator set.

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

bf16 logits (cast + old tail vs fused): 712.4 -> 326.2 us at rows=64, 0
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
* fused RoPE + KV-cache write: ~0.2–0.4 ms/step, ~+1–2% 4k;
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
both run at ~0.176-0.177 ms/layer (the HBM floor), while fp8 load + widen costs
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
(~0.176-0.177 ms/layer), and fp8 load + widen is only 304 GB/s.  On C500 the
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

## 8. lm_head + sampler CUDA graph: measured, falsified

Built a standalone CUDA graph around the exact decode tail at B=64,
V=130560, K=2048 (random weights of the real shape, real `top_p_sample`):

| path | time |
|---|---:|
| eager `F.linear` + `top_p_sample` | **904.8 us** |
| CUDA graph replay, static inputs | 903.0 us |
| graph replay + copy hidden states (256 KB) | 949.2 us |
| regenerate q (`exponential_`) | 52.0 us |

Graph capture saves **1.8 us (0.2%)**; adding the hidden-state copy needed to
feed a separate graph makes it 44 us slower.  The tail is kernel time, not
launch overhead: even a hypothetical full graph removes at most ~43 nodes x
1.7 us = 73 us, i.e. **0.4% of a 16.9 ms decode step**, not 7%.  This avenue
cannot close the 4k gap.
