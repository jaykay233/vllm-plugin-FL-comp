# MetaX 16k throughput (2026-10-22)

## Config
- Model: `/workspace/MiniCPM5-2B`, serve with official args + `VLLM_PLUGINS=fl`
- Case: `[[16384,1024,64,128]]`, `RUNS=4`, `SKIP_FIRST=1`
- No flagos whitelist (`align32_geometric` FlagGems branch)
- Bench client: `VLLM_PLUGINS=` (empty) — see `FINDINGS-mx.md` §11

## Summary (runs 2–4)
| Metric | Value |
|---|---|
| Total tok/s | **9891.78** |
| Output tok/s | 581.87 |
| Peak Output tok/s | 1536 |
| Req/s | 0.57 |
| Mean TTFT (ms) | 21003.51 |
| Failed | 0 |

Raw/summary CSV under `/workspace/benchmark_results/*_20261022_203415.csv`.

## Harness
- `launch_serve.py` — double-fork serve, records `rc=-N` on signal kill
- `run_16k_daemon.sh` — isolates bench client plugins; no CUDA health while serve is up
