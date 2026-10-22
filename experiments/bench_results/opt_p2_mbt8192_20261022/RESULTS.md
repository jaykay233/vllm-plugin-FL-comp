# P2 MBT 8192 (nopc fair A/B) — 2026-10-22

Official random throughput: prefix cache irrelevant; A/B with `--no-enable-prefix-caching`, seed=7.

| Case | MBT 4096 | MBT 8192 | Δ |
|------|----------|----------|---|
| 4k/256 c64 | 20400 | 21275 | +4.3% |
| 16k/64 c32 | 18446 | 18982 | +2.9% |

Default `VLLM_FL_METAX_MAX_NUM_BATCHED_TOKENS` raised to 8192.
