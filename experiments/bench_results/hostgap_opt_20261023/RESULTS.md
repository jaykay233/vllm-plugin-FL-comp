# Host-overhead opt A/B — 2026-10-23

P2 MiniCPM5-2B, official 4k `[[4096,1024,64,256]]` ×1 per arm (+ stage-prof).

## Results

| arm | env | total tok/s | Δ vs baseline | notes |
|-----|-----|-------------|---------------|-------|
| baseline | (unset) | **11760.72** | — | exec 2.69 ms/it, wait 82% |
| **light** | `DIRECT_DISPATCH=1` `CPU_THREAD_POLICY=-1` `LAUNCH_MODE=4` | **11896.95** | **+1.16%** | **landed** |
| gmode3 | light + `GRAPH_LAUNCH_MODE=3` | **9975.92** | **−15.2%** | **reject** |

## Landed

MetaX setdefaults in `vllm_fl/.../metax/patches/maca_launch_defaults.py` (also from `register()`):

- `MACA_DIRECT_DISPATCH=1`
- `MACA_CPU_THREAD_POLICY=-1`
- `MACA_LAUNCH_MODE=4`

Opt-out: `VLLM_METAX_MACA_LAUNCH_DEFAULTS=0`. User-set keys win.

## Rejected

- `MACA_GRAPH_LAUNCH_MODE=2` (PDL): cudagraph capture hang (`hostgap_pdl_20261023`)
- `MACA_GRAPH_LAUNCH_MODE=3`: e2e **−15%**, do not enable
