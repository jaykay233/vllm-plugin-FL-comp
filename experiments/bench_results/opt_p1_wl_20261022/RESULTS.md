# P1 + FlagOS whitelist — 2026-10-22

On top of P0 (MBT 4096 + silu fusion):

- cudagraph capture sizes trimmed to ≤64 + spare 72
- `align32_geometric` hard cap 4096 (FlagGems)
- MetaX default `flagos_whitelist: silu_and_mul,rms_norm,rotary_embedding`

## Short A/B (Total tok/s)

| Case | bottleneck | P0 | P1 (trim) | P1+whitelist |
|------|------------|----|-----------|--------------|
| 4k/256 c64 | 12900.63 | 16666.71 | 16855.13 | **20390.63** |
| 16k/64 c32 | 16284.97 | 18950.13 | 18938.08 | **21276.28** |

Whitelist vs P1: 4k **+21.0%**, 16k **+12.3%**.
Whitelist vs bottleneck: 4k **+58.1%**, 16k **+30.7%**.
