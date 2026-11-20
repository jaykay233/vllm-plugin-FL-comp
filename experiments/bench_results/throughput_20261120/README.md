# Throughput artefacts (MetaX C500, MiniCPM5-2B, 2026-11-20)

Promoted out of `/tmp` so the numbers in
`experiments/RESULTS-metax-fused-sampler-20261120.md` have evidence that
outlives the machine.  Every run used `--gpu-memory-utilization 0.85`, which is
part of the official configuration and must not be tuned.

## Read the prefix-cache hit rate before believing any number here

`benchmark_throughput_serve.py` launches `vllm bench serve` in a fresh
subprocess per run, and that tool re-seeds its random dataset identically every
time.  So with `BENCH_RUNS=2, BENCH_SKIP_FIRST=1` the two runs send byte
identical prompts to a server that still has them cached, and the second one is
inflated by ~15%.

**Several logs in this directory therefore contain one warm block and one cold
block.**  A warm block is identifiable by its mean TTFT: a cold 4096-token
prefill at c=64 cannot deliver first tokens in under ~2 s.

## The headline number, measured cold with its hit rate saved

| file | tok/s | mean TTFT | mean TPOT | prefix hit rate |
|---|---:|---:|---:|---:|
| `4k_c64_default_mbt_cold.log` | **13,058.96** | 2484.82 ms | 22.08 ms | **0.0-0.4%** (`server_4k_c64_default.log`) |

Single timed run on a fresh server (KV pool 1,191,216 tokens), MBT 16384
(server default), chunked prefill default.  This supersedes 13,039.70, which
differed by 0.15% and whose server log was never kept.

## Superseded / conditional blocks, kept for transparency

| file | run | tok/s | mean TTFT | verdict |
|---|---|---:|---:|---|
| `4k_c64_default_mbt.log` | Run 1/2 | 14,029.43 | 821.17 ms | **warm, discard** (TTFT impossible) |
| `4k_c64_default_mbt.log` | Run 2/2 | 13,039.70 | 2441.33 ms | cold-consistent, but server log lost |
| `4k_c64_mbt65536_unchunked.log` | Run 1/2 | 13,087.68 | 4543.65 ms | cold-consistent |
| `4k_c64_mbt65536_unchunked.log` | Run 2/2 | 13,130.40 | 4468.72 ms | cold-consistent, hit rate 0.2% in `4k_c64_mbt65536_server.log` |
| `4k_c128_cold.log` | 1/1 | 14,579.49 | 5088.15 ms | **cold, hit rate 0.4%** |
| `16k_c64_mbt8192.log` | 1/1 | 12,330.34 | 14290.82 ms | cold, hit rate 0.1% |
| `4k_warm_cache_repeat.log` | 1/1 | 15,082.87 | 619.48 ms | **the bug, kept as a counter-example** |

The last row is why this file exists: 15,082.87 tok/s with a 45-54% hit rate and
0.62 s TTFT for 4096-token prompts is the signature of a warm cache, not a
faster server.

## Bottom line for c=64 at 0.85

Cold measurements span **13,039.70 - 13,130.40** tok/s depending on
`max_num_batched_tokens` (16384 default vs 65536 un-chunked), i.e. within the
~0.7% run-to-run spread.  The 14,029.43 / 15,082.87 figures in this directory
are warm-cache artefacts and are not results.
