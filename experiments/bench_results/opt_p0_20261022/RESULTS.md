# P0 opt (MBT 4096 + silu fusion) — 2026-10-22

Serve: `VLLM_PLUGINS=fl`, `--max-num-batched-tokens 4096`,
`VLLM_METAX_MLP_SILU_FUSION=1`, `FLAG_GEMS_METAX_MM_GEMV=1`.

## Short A/B (Total tok/s)

| Case | baseline | MBT only | MBT+silu |
|------|----------|----------|----------|
| 4k/256 c64 n64 | 12900.63 | 16072.30 | 16666.71 |
| 16k/64 c32 n32 | 16284.97 | 18493.23 | 18950.13 |

## math_500 Level 3 (105)

- Accuracy: **98.1%** (103/105), threshold ≥0.95 → PASS
- Client must use `VLLM_PLUGINS=` and `CUDA_VISIBLE_DEVICES=`
- Report: `evalscope_level3/20261022_221204/reports/minicpm/math_500.json`
