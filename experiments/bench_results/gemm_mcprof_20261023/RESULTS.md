# GEMM mcProfiler analysis (2026-10-23)

Tool: **torch.profiler / MCPTI** (MetaX has no standalone `mcProfiler` CLI;
FINDINGS historically call this path “mcProfiler”). Scripts:
`gemm_mcprofiler.py`, `lm_head_msweep.log`.

Serve stack today (P2): FlagOS whitelist **excludes** `mm`/`linear` → projections
go **native mcblas**. That choice is correct.

## Device time: FlagGems `mm` vs native `F.linear` (bf16)

Ratio = gems/native (**&lt;1 = FlagGems faster**).

| shape | M=64 | M=256 | M=1024 | M=2048 | M=8192 |
|---|---|---|---|---|---|
| qkv (2560×2048) | 1.31 | 1.36 | **2.00** | 1.32 | 1.24 |
| o_proj | 1.33 | 1.34 | 1.59 | 1.70 | 1.22 |
| gate_up | 1.23 | 1.44 | 1.72 | 1.33 | 1.33 |
| down | 1.29 | 1.78 | 1.69 | 1.69 | 1.44 |
| **lm_head (130560×2048)** | **0.74** | 1.41 | 1.33 | 1.33 | 1.33 |

Host launch: FlagGems ~**105 µs**/call vs native ~**12 µs** (≈9×). Under CUDA
graph capture this is amortized; in eager it would erase small device wins.

Native top kernels: `mcblas__Mck_bf16gemm_tn_*` (M&gt;1), `b16gemvt_*` (M=1).
FlagGems: `mm_kernel_nt` / `_gemv_kernel`.

## lm_head M sweep (only niche win)

| M | native µs | gems µs | ratio | save |
|---|---|---|---|---|
| 8–32 | ~650 | ~360–410 | **0.55–0.63** | **+246–290 µs** |
| 40–64 | ~656 | ~480–485 | **0.74** | **+170 µs** |
| ≥72 | ~656 | ~870+ | **1.32+** | lose |

At official `--max-concurrency 64`, decode steps are typically **M≈64** →
~170 µs/step on lm_head only. Against ~40 ms decode steps that is **≲0.5%**
e2e — below the 2% scoring bar unless stacked with something else.

## Proton-style dig (Triton proton unavailable on MetaX)

`triton 3.0.0+metax3.7.0.7` **does not ship** `triton.profiler` / proton
(`pip install 'triton[proton]'` → *does not provide the extra 'proton'*).
Upstream proton needs a Triton build with `TRITON_BUILD_PROTON` + CUPTI/rocprof;
MetaX has MCPTI instead.

Stand-in (`gemm_proton_sweep.py`): same metrics proton scopes expose —
`flops16` / `bytes` → TFLOP/s + GB/s + % of soft peaks (1600 GB/s, 300 TFLOPS) —
plus a **fixed-config** launch of `mm_kernel_nt` sweeping
`(BLOCK_M/N/K, num_stages, pipeline∈{basic,cpasync}, scenario∈{-,fullstage})`.

### Roofline @ M=2048 (native vs best Triton)

| shape | native µs / TFLOPS / %BW | best Triton | ratio | best cfg |
|---|---|---|---|---|
| qkv | 131 / 164T / 14%BW | 174 µs / 124T | **1.32** | 256×256×32/s2/basic/fullstage |
| o_proj | 100 / 172T / 16%BW | 171 µs / 100T | **1.71** | 128×128×64/s2/basic/fullstage |
| gate_up | 524 / 197T / 13%BW | 696 µs / 148T | **1.33** | 256×256×32/s2/basic/fullstage |
| down | 276 / 187T / 13%BW | 466 µs / 111T | **1.69** | 128×128×64/s2/basic/fullstage |
| lm_head | 5223 / 210T / 13%BW | 6982 µs / 157T | **1.34** | 256×256×32/s2/basic/fullstage |

Prefill is **compute-bound** (~55–70% soft TFLOPS, only ~14% BW). Triton tops out
~20–30% lower TFLOPS than mcblas on the same shapes — not a bandwidth miss.

### What the sweep ruled out

- **`cpasync`**: uniformly worse than `basic` here (often 1.5–2× the already-losing
  basic time).
- **`fullstage`**: ~1% help on the best basic tiles; does not close the gap.
- **Bigger tiles matching mcblas (256×256×64, 128×128×128 basic)**:
  **OOM shared memory** (need 96–128 KB; C500 cap **64 KB/block**). That is the
  hard ceiling: mcblas can use denser MMA tiling; Triton `mm_kernel_nt` cannot
  without MetaX compiler/smem changes.
- Decode M=64: best Triton still **1.20–1.29×** slower on all projections except
  lm_head.

### Only confirmed Triton win (pinned)

`lm_head` M=64 → **64×128×64 / num_stages=3 / basic** = **0.733×** native
(481 vs 656 µs). Matches the earlier autotune path.

## Conclusion / next action

1. **Do not** whitelist `mm`/`linear` for MiniCPM on C500.
2. Prefill GEMM gap is **not** a missing yaml tile — 64 KB smem + current
   `mm_kernel_nt` cannot reach mcblas’s effective MMA intensity. Next real lever
   would be MetaX-side MMA codegen / larger smem, not more autotune keys.
3. **Landed**: selective `lm_head` route for **M∈[8,64]** via
   `patches/lm_head_gemm.py` (wraps `quant_method.apply`, default **on**;
   `VLLM_METAX_LM_HEAD_GEMM=0` to disable). Expected e2e ≲0.5%.
4. If a future MetaX Triton ships proton: re-run with
   `proton.start(..., backend=...)` for intra-kernel cycles; until then this
   roofline + fixed-config sweep is the substitute.
