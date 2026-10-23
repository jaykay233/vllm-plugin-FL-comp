# decode64 mcProfiler — 2026-10-23

## Attention num_splits @ batch=64 (device_time)

| seqlen | best splits | us/call | vs heuristic0 |
|--------|-------------|---------|---------------|
| 512 | **0** | 45.8 | 0% |
| 2048 | **0** | 117.8 | 0% |
| 4096 | **0** | 209.8 | 0% |
| 5120 | **0** | 258.5 | 0% |

Adaptive `num_splits=0` for batch≥16 remains correct for conc-64.

## GEMM M=64 (device_time)

| shape | native µs | FlagGems µs | ratio |
|-------|-----------|-------------|-------|
| qkv | 16.6 | 36.4 | 2.19 |
| o_proj | 15.5 | 35.3 | 2.28 |
| gate_up | 42.3 | 60.0 | 1.42 |
| down | 41.0 | 94.3 | 2.30 |
| lm_head | 655.9 | 619.9 | **0.95** |

Native kernels: `mcblas__Mck_bf16gemm_tn_*`. Gems: `mm_kernel_general`.
