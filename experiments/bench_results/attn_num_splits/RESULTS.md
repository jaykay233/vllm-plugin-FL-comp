# MetaX decode attention: `num_splits` tuning

## TL;DR

For the competition workload (MATH-500: short prompt + long CoT output, batch=1),
pinning the decode `num_splits` to **16** instead of letting the MACA heuristic
pick it gives a reproducible **-7.3% TPOT / +7.8% throughput** with **bit-identical
greedy output**. Nearly all of the gain is the disappearance of the split-reduction
pass for 15 of the 42 layers.

## Root cause

`vllm_fl/.../vendor/metax/impl/attention/flash_attn.py::FlashAttentionImpl.forward`
routes decode through `flash_attn_with_kvcache()` **without passing `num_splits`**,
so it defaults to `0` = "MACA heuristic decides".

The heuristic is effectively **blind to the real sequence length**: it splits the
KV cache into a fixed **8 chunks** regardless of context (confirmed by the constant
`8` in the `flash_fwd_splitkv_combine_kernel<..., 8, N, ...>` template). Every
decode step therefore pays a cross-split reduction:

    flash_fwd_splitkv_kernel           ~351 us/step  (42 layers)
    flash_fwd_splitkv_combine_kernel  ~1007 us/step  (42 layers)  <-- pure overhead

The ~1007 us is almost identical at 13 KV tokens and at 1201 KV tokens, which is
the tell: it is a fixed cost, not work proportional to the context.

## Why it cannot be chosen per step

Attention executes **inside** the FULL decode CUDA graph, not eagerly. Verified by
wrapping `flash_attn_with_kvcache` in the vendor module namespace: a 16-token
generation produced exactly **42** calls, all with a frozen `max_seqlen=9` — i.e.
capture-time only, then graph replays. `num_splits` is therefore a capture-time
constant. (`splitting_ops` lists `vllm::unified_attention_with_output`, which
suggests eager attention, but the trace proves otherwise.)

## Measurement

Workload matches the eval set: `math_500` (`epochs: 4`, prompt template asks for
step-by-step reasoning -> long output).

| | |
|---|---|
| prompt | 49 tokens |
| generation | 1024 tokens |
| batch | 1 |
| model | MiniCPM5-2B (42 layers, 16 q-heads / 2 kv-heads, head_dim 128, bf16) |
| metric | TPOT = (t(N tokens) - t(1 token)) / (N-1), **min of 3 repeats** |

Interleaved A/B/A (`0,16,32,0,16`) in separate processes, since `num_splits` is
baked into the captured graph. Both `ns=0` runs reproduced to within 1.2 us, so
the numbers are not drift.

| num_splits | TPOT mean (us) | samples | combine (us/step) | attention (us/step) |
|---|---|---|---|---|
| 0 (heuristic) | 7522.6 | [7523,7522,7523], [7523,7524,7524] | 1007.1 | 1547.6 |
| **16** | **6975.8** | [6992,6988,6988], [6966,6964,6966] | **399.8** | **1046.2** |
| 32 | 6980.2 | [6982,6980,6980] | 474.0 | 1050.6 |

| config | latency | throughput | attention |
|---|---|---|---|
| ns=16 vs ns=0 | **-7.27%** | **+7.84%** | **-32.4%** |
| ns=32 vs ns=0 | -7.21% | +7.77% | -32.1% |

Greedy token ids: identical across all 5 runs -> numerically equivalent, not an
accuracy trade.

ns=16 and ns=32 are within 0.1% of each other; the gain plateaus at 16.

## Workload dependence (why the shipped default is adaptive, not a constant)

Kernel-level sweep over (batch x seqlen x num_splits), metric = pure GPU time from
the profiler (`device_time`), reproduced as us/call:

| batch | seqlen | heuristic | best ns | gain |
|---|---|---|---|---|
| 1 | 32 | 13.5 | 1 | 37.1% |
| 1 | 1024 | 22.3 | 32 | 13.5% |
| 1 | 2048 | 31.0 | 32 | 28.1% |
| 8 | 1024 | 27.9 | 16 | 6.5% |
| 32 | 1024 | 44.3 | 0 | 0.0% |
| 32 | 2048 | 68.7 | 0 | 0.0% |

No single constant wins everywhere: short context wants 1, batch-1 long context
wants 32, and heavily batched long context (batch>=32) wants the stock heuristic.
The competition shape (batch=1, long CoT) is the batch-1 long-context column,
which is why 16 wins there.

Since no constant is right everywhere, the shipped default is not a constant:
`_decode_num_splits(batch_size)` returns 16 splits below
`FL_METAX_ATTN_ADAPTIVE_BATCH` (default 16) and the stock heuristic at or above
it. Batch size is known at capture time, which is the only moment the choice can
be made. Verified end-to-end on the ctx=60 / 1024-output workload:

| config | effective ns | TPOT (us/token) |
|---|---|---|
| adaptive, batch=1 | 16 | 6970.9 |
| adaptive, batch=16 | 0 (heuristic) | 488.0 |

Setting `FL_METAX_ATTN_NUM_SPLITS` overrides the adaptive rule entirely
(`0` = stock heuristic, `N` = exactly N splits).

## Pitfalls hit (so they are not repeated)

1. **CUDA-event timing of ~30us kernels measures launch overhead, not GPU time.**
   An early version of the fast sweep was off by ~30x versus end-to-end; switching
   to the profiler's `device_time` made it agree. Same trap as an earlier
   `rms_norm` investigation.
2. **The fast sweep underestimates `combine` in absolute terms** (~14 vs ~24 us):
   under CUDA graphs the split workspace is sized from the capture-time bound, not
   the live sequence length. Only end-to-end numbers are authoritative.
3. **A single TPOT sample is noisy enough to lie.** One ns=16 run reported 7750 us
   while the profiler said GPU time had *dropped*; three repeats of that config
   gave 6988 us.
4. `PYTHONPATH` had been left pointing at `/root/src/probe_site` from earlier
   tracing experiments; combined with the presence of
   `/workspace/vllm-plugin-FL/vllm_fl/platform.py` it hijacked the stdlib
   `platform` module. Unset it when benchmarking.

## Files

* `vllm_fl/dispatch/backends/vendor/metax/impl/attention/flash_attn.py` — forwards
  `_decode_num_splits(batch_size)` to `flash_attn_with_kvcache`; the default is the
  adaptive batch rule above (`FL_METAX_ATTN_NUM_SPLITS` overrides it, unset =
  adaptive, `0` = stock).
* `/root/src/bench_attn_num_splits.py` — end-to-end A/B harness.
* `/root/src/fast_attn_split_sweep.py` — kernel-level sweep (GPU-time metric).
* `/root/src/run_confirm.sh` — interleaved confirmation driver.
* `/root/src/trace_decode_attn.py` — proves attention is captured in the graph.
* `/root/bench_results/attn_num_splits/confirm_runs/` — raw per-run JSON.
