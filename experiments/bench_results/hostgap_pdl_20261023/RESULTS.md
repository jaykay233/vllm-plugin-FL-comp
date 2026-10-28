# Host-gap / PDL A/B — 2026-10-23

Official 4k: `[[4096,1024,64,256]]` ×2, P2 env (GEMV+silu fusion+MBT8192+cg spare72), `VLLM_METAX_LM_HEAD_GEMM=0`.

## Baseline (`MACA_GRAPH_LAUNCH_MODE` unset)

| run | total tok/s | output tok/s |
|-----|-------------|--------------|
| 1 | 11752.06 | 2350.41 |
| 2 | 11746.26 | 2349.25 |
| **avg** | **11749.16** | 2349.83 |

## `MACA_GRAPH_LAUNCH_MODE=2` (PDL graph launch)

**FAIL / blocked** — serve never reached `/health`.

- Stuck at `Capturing CUDA graphs (mixed prefill-decode, PIECEWISE): 0/12`
- mxcc optimizer crashes in pass `function(metaxgpu-pdl)` (`Recompile fail` 1008/1011, `ErrorNoBitcodeForGpu`)
- `server.log` frozen from 10:36:12; EngineCore kept ~70% CPU but no progress for >2 min
- Process killed; **no 4k A/B number**. Do **not** enable MODE=2 by default on this stack (MACA 3.7 / C500).

### Verdict
PDL via `MACA_GRAPH_LAUNCH_MODE=2` is **not usable** for closing host-gap on current MiniCPM5-2B + cudagraph path. Host-gap remains unmeasured-clear; need MetaX-side PDL/recompile fix or a different overlap strategy (async scheduling already on).
