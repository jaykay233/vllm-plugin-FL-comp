# FP8 on MetaX C500: what actually works, and what does not

All numbers measured on this box at `M=64` (the MiniCPM5-2B decode shape) against
`F.linear` in bf16, which dispatches to `mcblas`.

## Headline

**fp8 weights now beat bf16 by 3.2% on the decode step** (0.773 ms vs 0.798 ms),
after three separate fixes. Three of the five projections still lose; the win comes
entirely from `lm_head`, which is 86% of the weight bytes.

## 1. The hardware fp8 MMA is not a hardware path

`mma_peak2.cu` saturates the issue rate of every MMA intrinsic (416 blocks, 16
independent accumulator chains), so these are ceilings, not estimates.

| intrinsic | peak |
|---|---|
| `__builtin_mxc_mma_16x16x16bf16` | 149 TFLOPS |
| `__builtin_mxc_mma_16x16x16f16` | 151 TFLOPS |
| bf16 issued as two `16x16x16` (k=32) | 195 TFLOPS |
| `__builtin_mxc_mma_16x16x32f8` | **17.8 TFLOPS** |

The fp8 intrinsic is 8.4x slower than bf16. `mctlass/arch/maca_fp8_tmp.h` shows
why: it software-converts fp8 to f16 and issues two `16x16x16f16` MMAs. cute's
`MACA_16x16x32_F32BF16BF16F32` is likewise two bf16 calls. There is no wide fp8
tensor-core instruction on this part; `16x16x16` is the only primitive. Anything
built on fp8 MMA at `M=64` is capped at ~139 GB/s, while matching mcblas needs 1080.

## 2. The real bottleneck was the global load

`diag_split.cu` decomposed the working bf16-MMA kernel on `gate_up`: staging 62%,
MMA 8%. `stage3.cu` isolated staging and the single change that mattered:

| load | staging throughput |
|---|---|
| `__builtin_memcpy(dst, src, 16)` | 168 GB/s |
| `__builtin_mxc_ldg_b128` | **619 - 832 GB/s** |

`memcpy` of 16 bytes lowers to scalar loads on this toolchain.
`__builtin_mxc_ldg_b128` is what `mcflashinfer`'s `load_128b` and mctlass's
`maca_ldg128` use. Occupancy and barriers were not factors: adding or removing the
per-chunk barrier changed staging by less than 7%.

## 3. Dequantize B at fragment read, never A

`gemm_fp8_v7.cu` established the dataflow:

* A stays bf16 and is never converted. `M=64` makes A tiny, and every warp reads
  the whole A tile, so converting A multiplies the work by the warp count (this is
  what made v4 slower than v3).
* B stays fp8 in gmem *and* smem, so smem halves and bigger tiles fit. Each B
  element is consumed by exactly one warp, so conversions equal unique elements.
* B is dequantized at fragment read. e4m3 to bf16 is exact for all 256 encodings;
  a LUT does this correctly including subnormals (rel err 2.5e-07).

## 4. The LUT was the next bottleneck; arithmetic beats it

The LUT costs 4 shared-memory reads per B fragment, which doubles the kernel's
smem traffic. The conversion needs no table at all: widening the e4m3 byte into a
16-bit lane and shifting the `(exp|mant)` field left by 4 lines the bits up
exactly, then the exponent is rebased by `+0x3C00`. Two lanes per 32-bit word, so
a fragment costs ~18 ALU ops and zero smem reads.

| lm_head | time | ratio |
|---|---|---|
| LUT, WPB4 NT4 | 0.6617 ms | 1.00x |
| arithmetic, WPB4 NT4 | **0.5245 ms** | **1.26x** |

Subnormals are not special-cased by the arithmetic path, so accuracy goes from
2.5e-07 to 2.2e-06 -- still far inside tolerance, and it removes the scaling
workaround entirely.

Two dead ends worth recording:

* `__builtin_mxc_ldg_b128_bsm` (async global-to-shared) benchmarked at 1019 GB/s but
  produced wrong data under every completion protocol tried (`arrive_gvmcnt(0)` +
  `barrier_inst`, `__builtin_mxc_arrive(64)`, with and without `__syncthreads`,
  flattened and per-row indexing). Only a fully linear smem layout was ever
  correct, which looks like a race, not a layout rule. Not used.
* `__builtin_mxc_byte_perm` does not behave like PTX `prmt` on this toolchain
  (`byte_perm(x, 0, 0x4140)` returns `0xff0000ff`-style sign-extended output), so
  the 2-instruction lane expansion it would allow is unavailable.

## 5. Split-K only where the N dimension starves the SMs

`gemm_fp8_v8.cu` partitions K across `grid.y` and accumulates with `atomicAdd`.
Model decode launches only `N/(WPB*NT*16)` blocks, so `o_proj` gets 16 blocks for
104 SMs. Split-K fixes the starvation; single-split blocks keep a plain store
because `atomicAdd` is pure overhead when there is no contention.

| shape | s1 | s4 | s8 |
|---|---|---|---|
| o_proj (16 blocks) | 0.28x | **0.51x** | 0.41x |
| gate_up (96 blocks) | **0.63x** | 0.33x | 0.29x |
| lm_head (510 blocks) | **1.20x** | 0.56x | 0.43x |

## 6. Result

Best config per shape, `M=64`, all numerically validated (rel err 1.6e-06 - 3.2e-06):

| proj | N | K | fp8 ms | bf16 ms | ratio | config |
|---|---|---|---|---|---|---|
| qkv | 2560 | 2048 | 0.0405 | 0.0215 | 0.53x | WPB2 NT2 KC64 s4 |
| gate_up | 12288 | 2048 | 0.0748 | 0.0473 | 0.63x | WPB2 NT2 KC64 s1 |
| o_proj | 2048 | 2048 | 0.0399 | 0.0204 | 0.51x | WPB2 NT2 KC64 s4 |
| down | 2048 | 6144 | 0.0693 | 0.0481 | 0.69x | WPB2 NT2 KC64 s4 |
| lm_head | 130560 | 2048 | 0.5487 | 0.6608 | **1.20x** | WPB4 NT4 KC64 s1 |
| **decode step** | | | **0.7732** | **0.7982** | **1.03x** | |

Trajectory of the kernel: 91 GB/s (v1, per-chunk barrier + smem-dequant) ->
274 GB/s (ldg128) -> 415 GB/s (split-K on small N) -> **510 GB/s** (arithmetic
dequant, lm_head).

## 7. Why the small shapes still lose

`lm_head` moves 267 MB of fp8 weights and writes 33 MB in 0.55 ms (~547 GB/s
actual). mcblas moves 535 MB of bf16 in 0.66 ms (809 GB/s). So mcblas is already
at 57% of the ~1416 GB/s HBM peak, and the fp8 kernel is at 39%.

The small shapes lose for a different reason: they are latency- and launch-bound,
not bandwidth-bound. `o_proj` streams only 4 MB of weights; at 104 SMs that is
~40 KB per SM, so the whole kernel runs for 40 us and is dominated by wave
scheduling and the 32 block barriers. Checked and rejected: `KC=32` to fit two
blocks per SM (0.46x vs 0.51x -- the extra barriers cost more than occupancy
gains), and `NT=8` for more accumulators (register spill, 0.78x).

The remaining lever for small shapes is fusing them, not optimizing them
individually: `qkv` + `o_proj` + `gate_up` + `down` are 0.225 ms of fp8 time but
only 0.138 ms of bf16 time, and they are launched back to back.

## Corrected: those four cannot be fused

The fusing idea above is wrong and was withdrawn. In a real layer the four are a
**serial dependency chain**, not siblings:

```
x -> qkv -> attention -> o_proj -> gate_up -> act -> down -> x'
```

`o_proj` consumes attention's output, which consumes `qkv`'s output. There is no
launch to merge without restructuring the layer, which is a different project.

## What actually moved the number: A-side bank conflicts

A row stride of `KC*2 + PADA` decides which shared-memory bank each lane hits,
since a lane reads `(row*RSB + kb)/4 mod 32`. `PADA=32` looked like a safe pad but
at `KC=64` gives `RSB=160B = 40 words` and `gcd(40,32)=8`, so the 16 rows of a lane
group collapse onto **4 bank sets -- a 4-way conflict on every A fragment read**.
`PADA=16` (36 words) or `48` (44 words) both give `gcd=4` and 8 distinct sets.

| PADA | RSB | words | gcd | bank sets | lm_head |
|------|-----|-------|-----|-----------|---------|
| 32 | 160 | 40 | 8 | 4 (conflict) | 0.548 ms |
| 16 | 144 | 36 | 4 | 8 | **0.529 ms** |
| 48 | 176 | 44 | 4 | 8 | 0.529 ms |

This also fixed `down`, which had been running an accidentally-bad config: it
takes the same shape as `o_proj` and now times the same (0.038 vs 0.069 ms).

## Final numbers (gemm_fp8_v13.cu)

| proj | N | K | config | split | fp8 ms | bf16 ms | ratio |
|------|---|---|--------|-------|--------|---------|-------|
| qkv | 2560 | 2048 | WPB2 NT2 KC64 PADA16 | 4 | 0.03875 | 0.02022 | 0.52x |
| gate_up | 12288 | 2048 | WPB2 NT2 KC64 PADA16 | 1 | 0.07058 | 0.04537 | 0.64x |
| o_proj | 2048 | 2048 | WPB2 NT2 KC64 PADA16 | 4 | 0.03782 | 0.01926 | 0.51x |
| down | 2048 | 2048 | WPB2 NT2 KC64 PADA16 | 4 | 0.03805 | 0.01919 | 0.50x |
| lm_head | 130560 | 2048 | WPB4 NT4 KC64 PADA16 | 1 | 0.52860 | 0.65983 | **1.25x** |

```
DECODE STEP  fp8 0.7219 ms | bf16 0.7643 ms | 1.059x  (+5.9%)
```

Up from 1.032x. Split-K on the small shapes was re-swept (s1/2/4/8/16/32) and 4
is the optimum for `qkv`/`o_proj`/`down`: s2 leaves too few blocks and s16+ pays
more in `atomicAdd` traffic (s4 already writes 2 MB of atomics) than it gains.

## Two more dead ends, closed by measurement

Both were plausible on paper and both lost. They are recorded so they are not
retried.

**Multi-stage pipelining (v13, `STG=3,4`).** Aimed at the overlap gap: staging
alone is 0.444 ms, compute alone 0.436 ms, full 0.522 ms, so perfect overlap
would give 0.444. Prefetching `STG-1` chunks ahead to give the copy room was
worth nothing -- `STG=3` best case 0.92x and `STG=4` 0.82x against `STG=2`'s
1.20x. The barrier is not the problem; issue throughput is.

**Dequantize during staging (v14).** Computed that ~72 ALU ops ride along with
16 MMAs per k-step, so ~82% of issue slots go to dequantization, capping compute
at 52% of MMA peak. Moving the conversion into the memory-bound staging path
should have been free. It scored 0.83x. Reason: a bf16 B tile doubles both the
shared-memory write during staging *and* the read per fragment in the inner
loop, and **smem bandwidth is scarcer than ALU on this part**. Dequantizing at
fragment read, which touches 4 bytes per fragment, is the right side of that
trade.

## Files

* `mma_peak2.cu` - MMA issue-rate ceilings per intrinsic
* `diag_split.cu` - staging vs compute split
* `stage3.cu` - staging only; ldg128 vs memcpy, occupancy, barriers
* `bw_stage.cu`, `cpasync_test.cu` - async `ldg_b128_bsm` probing (fast, wrong)
* `gemm_fp8_v13.cu` - **current best**; v12 plus configurable A padding and
  stage count. `tc=20` is the lm_head config, `tc=23` the small-shape one
* `gemm_fp8_v12.cu` - previous best (LUT and arithmetic dequant both selectable,
  split-K, staging-compute decomposition modes)
* `gemm_fp8_v14.cu` - dequant-during-staging variant; 0.83x, kept as a dead end
* `gemm_fp8_v15.cu` - occupancy experiment (A single-buffered, B pad sweep).
  Exposes `v15_occ_blocks`/`v15_smem_bytes` via the occupancy runtime API
* `gemm_fp8_v16.cu` - **current best**; identical dataflow to v13 but codes to 146
  registers instead of 154, worth a consistent 0.7%. Also carries the streamed-A
  dead end. Exposes `v16_probe(tc, which)` for smem / regs / spill / blocks-per-SM
* `gemm_fp8_v17.cu` - wide-N-tile and persistent-grid variants; both lose, kept as
  dead ends. Exposes `v17_probe(tc, which)`
* `ab17.py`, `ab_final.py` - interleaved same-process A/B between library versions
  (run-to-run spread is ~2%, so cross-run comparison cannot resolve sub-1% gaps)
* `probe16.py`, `occ15.py` - resource and occupancy probes
* `decode_step13.py` - end-to-end decode step, fp8 chain vs bf16 chain
* `final_v13.py` - one shape/config per process, with a correctness check
* `bench_one13.py` - one config per process, no validation (survives OOM)
* `final_sweep.py` - per-shape config search with numerical validation
* `run_v12.py check|bench` - quick validation and benchmarking
* `lut_e4m3_bf16.inc` - exact 256-entry e4m3 to bf16 table
* `gemm_fp8_mma.cu` .. `gemm_fp8_v11.cu` - superseded, kept for the trail


## v15 / v16: five hypotheses for the last 19% of lm_head, all measured negative

`lm_head` sits at 0.524-0.529 ms. The staging/compute split says staging alone is
0.444 ms, so perfect overlap would be 1.49x. The remaining loss is 0.08 ms of
non-overlap, and every attempt to recover it lost. Recorded with the number that
refuted it so none of this gets retried.

A useful tool came out of this: `mcOccupancyMaxActiveBlocksPerMultiprocessor` and
`mcFuncGetAttributes` are available, so smem, registers, spill bytes and blocks
per SM can be read back per config instead of inferred. The table for the best
config pair:

| config | smem | regs | spill | blk/SM | lm_head |
|--------|------|------|-------|--------|---------|
| WPB4 NT4 KC64, A buffered in regs | 59392 | 154 | 0 | 1 | **0.524 ms (1.26x)** |
| WPB4 NT4 KC64, A streamed to smem | 59392 | 146 | 0 | 1 | 0.756 ms (0.87x) |
| WPB4 NT4 KC32, A single-buffered | 29696 | -- | -- | 2 | 0.794 ms (0.83x) |
| WPB4 NT4 KC32, B pad dropped | 26624 | -- | -- | 2 | -- (not timed) |
| WPB2 NT2 KC128, A buffered | 53248 | -- | -- | 1 | 1.018 ms (0.65x) |

**1. Two blocks per SM. Fails.** Shared memory was never the binding constraint;
registers are. 154 regs x 256 threads = 39424 registers per block, so a second
block cannot be resident regardless of smem. Forcing it by shrinking the tile
(`KC=32` + single-buffered A = 29696 B, 2 blocks/SM confirmed) gives 0.794 ms. The
extra chunks double the barrier count and that costs far more than the added
parallelism returns -- the same conclusion the earlier `KC=32` test reached, now
with the occupancy number to prove the mechanism.

**2. Register pressure in the stager. Fails.** With `NPA=32`, every thread holds 32
`v4u` (128 registers) of prefetched A before storing any of it. Interleaving load
and store per chunk drops that to roughly one vector and only 8 registers overall
(154 -> 146), and it is 44% *slower* (0.756 vs 0.524 ms). So the register buffer is
not waste -- it is the mechanism that covers DRAM latency. Deep software prefetch
into registers is doing the work that `cp.async` was supposed to do, and `cp.async`
is unusable here (see the sync-race note above).

**3. Cache-line granularity. Partly right, but not cashable -- see the correction
below.** Each chunk reads 64 contiguous bytes from each of 256 B rows at stride
2048, so if the memory system fetches 128 B lines, half of every fetch is
discarded. Measured 1.018 ms (0.65x) at `WPB2 NT2 KC128` against 0.594 ms for the
same tile shape at `KC=64`, which was recorded here as a falsification. **That was
premature**: the same-shape comparison held the tile fixed but the pattern probe
later showed the run length alone is worth 2x (719 -> 1471 GB/s), so the two effects
were cancelling. The hypothesis is real; the reason it cannot be used is A traffic,
quantified in "The row probe, and why KC=128 still loses".

**4. More pipeline stages. Fails** (v13, `STG=3/4`: 0.92x / 0.82x). At `KC=64 NT=4`
a third stage needs 89088 B, so this was only ever testable at smaller tiles, which
confounds it -- but combined with 2 and 3 the picture is consistent: depth in
registers works, depth in shared memory does not.

**5. Single-instruction smem to fragment loads. Not available.** The toolchain
exposes no `ldmatrix` equivalent -- only scalar and vector global loads
(`ldg_b8/b32/b64/b96/b128` and their `_bsm`/`_predicator` forms). So the four
8-byte A fragment loads per k-step cannot be collapsed.

Conclusion: 0.524 ms / 1.26x is where this dataflow lands. Staging (0.444) and
compute (0.436) are already balanced to within 2%, and the overlap between them is
as good as a barrier-per-chunk structure allows. Beating it needs a different
dataflow, not another parameter.


## v17: the last two dataflow ideas, both measured negative

After v13-16 the only remaining proposals were structural. Both were built and are
now closed. The A/B numbers come from `ab_final.py`, which interleaves the variants
inside one process (25 iterations x 14 rounds) because cross-run spread is ~2% and
cannot resolve sub-1% differences.

| variant | median | min | vs v13 |
|---------|--------|-----|--------|
| v13 tc20 | 0.5305 | 0.5274 | -- |
| **v16 tc20** | **0.5269** | **0.5240** | **+0.7%** |
| v17 tc21 (persistent) | 0.5346 | 0.5322 | -0.8% |
| v16 tc24 (NT2) | 0.9226 | 0.9103 | -42.5% |

### (A) Wider N tile -- fails

Keeping A resident in shared memory is impossible (256 KB against 64 KB), so the
only form the idea can take is covering more N columns per K loop, which amortises
each block's A read over more output. A correction to how I framed this earlier: I
expected to free registers by streaming A and spend them on accumulators. **That
trade does not exist.** At `KC=32` the A tile is `64*4 = 256` chunks over 256
threads, so `NPA = 1` -- the A prefetch is already a single vector and there is
nothing to free. Wide tiles are purely an accumulator question, and accumulators
cost `NT*4*4 = 16*NT` registers because every warp covers all four m-fragments
(M=64 decomposes as exactly 4x16).

| config | regs | spill | blk/SM | lm_head |
|--------|------|-------|--------|---------|
| NT4 KC32 (base for this comparison) | 134 | 0 | 1 | 0.718 ms |
| NT6 KC32 | 172 | 0 | 1 | 0.858 ms |
| NT8 KC32 | 214 | 0 | 1 | 0.922 ms |
| NT8 KC32, persistent | 242 | 0 | 1 | 0.920 ms |

Note there is **no register spill** in any of them -- the earlier note that `NT=8`
spilled was wrong; it is simply slower. Wider tiles lose against their own `KC=32`
baseline, so the halved A traffic does not pay for the larger accumulator set at
all. `KC=32` itself remains ~35% behind `KC=64`, so this axis was never going to
win. Wide tiles also need `KC=32` to fit shared memory (at `KC=64 NT=8` the B tile
alone is 81920 B).

### (B) Persistent grid -- fails, but only just

510 tiles over 104 SMs is 4.90 waves, so the theoretical prize was the ~1.9% idle
tail plus ~1% launch overhead, i.e. ~3% at most. A persistent grid of one block per
SM, each looping over a strided tile set, delivers 1.0% against its own non-
persistent twin (v17 tc20 0.5330 -> tc21 0.5276) -- the idea works, and lands in
the predicted ballpark. But the tile loop costs 18 registers (146 -> 164), which is
1.5%, so the net against v16 is **-0.8%**. Hoisting the loop-invariant split-K
terms (`kbeg`, `NCH` do not depend on the tile) did not reduce it: the cost is the
loop state itself.

Worth recording as a positive result for the mechanism even though it is a net
loss: a persistent kernel is the right way to remove wave quantisation on this
part, it is simply worth less than the register pressure it introduces at a config
that is already at 1 block/SM. If some future version drops under 146 registers,
persistence should be re-measured -- the 1% is real and additive.

## Current best: gemm_fp8_v16.cu

| proj | config | split | fp8 ms | bf16 ms | ratio |
|------|--------|-------|--------|---------|-------|
| qkv | WPB2 NT2 KC64 PADA16 | 4 | 0.03843 | 0.02013 | 0.52x |
| gate_up | same | 1 | 0.06905 | 0.04574 | 0.66x |
| o_proj | same | 4 | 0.03827 | 0.01939 | 0.51x |
| down | same | 4 | 0.03920 | 0.01908 | 0.49x |
| lm_head | WPB4 NT4 KC64 PADA16 | 1 | 0.52321 | 0.66014 | **1.26x** |

```
DECODE STEP  fp8 0.7123 ms | bf16 0.7662 ms | 1.076x  (+7.6%)
```

Up from 1.059x. `v16` is dataflow-identical to `v13` -- same staging, same
dequantize-at-fragment-read, same tiles -- and differs only in how the compiler
schedules it (146 registers against 154). The 0.7% is free but it is codegen, not
architecture. The rest of the step to 1.076x is the `down` projection no longer
running its accidental bad config.


## The floor probe: why the small shapes cannot win

`floor_probe.cu` measures what a kernel that does *nothing but the traffic* costs,
before deciding whether more effort on `o_proj`/`qkv`/`gate_up`/`down` is worth it.

Read bandwidth against transfer size, contiguous 128-bit loads, grid swept:

| size | best GB/s | best time |
|------|-----------|-----------|
| 1 MB | 64 | 15.66 us |
| 2 MB | 128 | 15.62 us |
| 4 MB | 244 | 16.37 us |
| 8 MB | 454 | 17.64 us |
| 16 MB | 664 | 24.11 us |
| 32 MB | 934 | 34.25 us |
| 64 MB | 1159 | 55.21 us |

Fitting `t = t0 + bytes/BW` over 16-64 MB gives **BW ~= 1550 GB/s marginal and
t0 ~= 13.3 us fixed**. An empty kernel with the same grid costs 6.45 us, so ~7 us of
the intercept is L2 cold-start and ramp. The empty-kernel cost itself is flat at
6.45 us from 52 to 256 blocks and only rises past 512 (9.41 us at 1024, 13.32 at
2048).

Consequence: **4 MB cannot be streamed in less than ~16.4 us on this machine**, not
the 2.96 us that 1416 GB/s suggests. At small sizes the fixed cost dominates
everything.

Isolating the GEMM's access pattern at the o_proj shape (4 MB in, 512 KB out):

| probe | time | GB/s |
|-------|------|------|
| empty kernel | 6.67 us | -- |
| contiguous read of 4 MB | 16.37 us | 244 |
| GEMM's tiled read pattern | 24.51 us | 171 |
| tiled read + fp32 output write | 27.41 us | 172 |
| **actual fp8 GEMM** | **38.30 us** | **105** |

Two conclusions, and the second one is the important one:

1. The strided pattern costs 8 us over contiguous (244 -> 171 GB/s). The fp8 GEMM's
   remaining gap, 38.3 against 27.4, is compute plus barriers on a kernel that only
   has 8 chunks per block to amortise its prologue over.

2. **bf16 mcblas takes 19 us for o_proj, which is 93% of the contiguous-stream floor
   for the 8 MB it reads.** The library kernel is already essentially at the memory
   floor for this shape. A *perfect* fp8 kernel reading 4 MB at the 4 MB floor
   (16.4 us, with zero compute and zero strided-access penalty) would be only 1.16x
   over bf16 -- and the realistic floor with the GEMM's own access pattern plus the
   epilogue is 27.4 us, which **loses** to bf16's 19 us.

So the small shapes are closed for good, not by parameter tuning but by the
machine's fixed cost at 4 MB. This also reframes the earlier table: the 105 GB/s
that looked like a 13x anomaly is a 1.4x gap against a real floor, and the floor
itself is above what bf16 achieves.

## The correction that pays: stop using fp8 where it loses

The four small projections cost 0.185 ms in fp8 against 0.104 ms in bf16 -- a
0.081 ms loss, larger than nothing but real. `lm_head` saves 0.137 ms. Running fp8
everywhere therefore nets only 0.056 ms. Running fp8 *only where it wins* keeps the
whole 0.137 ms.

Measured on the same chained decode step:

| variant | decode step | vs bf16 |
|---------|-------------|---------|
| all bf16 | 0.7660 ms | -- |
| all fp8 | 0.7125 ms | 1.075x |
| **fp8 lm_head only, bf16 for the rest** | **0.6303 ms** | **1.215x (+21.5%)** |

This is the configuration to ship: a single fp8 kernel for `lm_head` (73% of the
step, 1.26x) and the vendor bf16 path for the four serial projections that fp8
cannot help. It is 0.14x better than the fully-fp8 version and needs no new kernel,
only a dispatch decision.


## The row probe, and why KC=128 still loses

`rowprobe.cu` measures the staging access pattern alone, at lm_head scale, with no
shared memory and no MMA. Block covers ROWS output columns, reads KC contiguous
bytes from each, strides K=2048:

| per-row run | KC=64 (current) | KC=128 | KC=256 | KC=512 | KC=2048 |
|-------------|-----------------|--------|--------|--------|---------|
| GB/s | **719** | 1471 | 1560 | 1563 | 1577 |

And at KC=64 fixed, varying rows per block:

| rows/block | 256 | 128 | 64 | 32 |
|------------|-----|-----|----|----|
| GB/s | 719 | 804 | 1371 | 1564 |

So the current geometry's own pattern floors at 719 GB/s against a contiguous floor
of 1505-1577, and reading 128 B per row instead of 64 B doubles it. That much of the
earlier cache-line reasoning was right after all -- the 4 MB-scale test that
"falsified" it was confounded by halving the N tile at the same time.

**But it does not translate into a faster kernel**, and the reason is the term that
was missing from every earlier analysis: A traffic. Every block re-reads the whole
256 KB A tile, so block count multiplies A:

| config | N tile | blocks | A traffic | B traffic | total | time | raw GB/s |
|--------|--------|--------|-----------|-----------|-------|------|----------|
| KC64 NT4 | 256 | 510 | 130 MB | 267 MB | 397 MB | 523 us | 759 |
| KC128 NT2 | 128 | 1020 | 261 MB | 267 MB | 528 MB | 660 us | 800 |
| KC64 NT2 | 64 | 2040 | 522 MB | 267 MB | 789 MB | 660 us | 1195 |

The small tiles achieve the *highest* raw bandwidth (1195 GB/s) and are still the
slowest, because they move twice the data. Time is traffic divided by the pattern
efficiency, and the two move in opposite directions.

KC=128 needs 144 B of B-row per output column per stage, so shared memory caps it
at ~160 columns; that pushes blocks to 816 and A traffic to 209 MB, versus 130 MB
for the current KC=64/256-column config. Measured at 128 columns: 0.660 ms against
0.527. The pattern gain is real but smaller than the traffic loss.

### Correcting the headroom claim

It would be easy to read the 719 GB/s pattern floor against the 1505 GB/s
contiguous floor and conclude staging has 2.5x of headroom. **That comparison is
invalid and the resulting claim was wrong.** The staging pass cannot be held to the
contiguous number, because the pattern it is required to use only reaches 719.

Re-deriving the stack with B's pattern floor as the reference:

| step | time | accounting |
|------|------|------------|
| pattern-only floor, B's 267 MB at KC=64 | 372 us | 719 GB/s |
| staging pass (adds A's 130 MB) | 444 us | 397 MB total = 797 GB/s |
| full kernel (adds MMA + barriers) | 523 us | -- |

So the staging pass runs at 797 GB/s against a 719 GB/s *pattern* floor, i.e. it is
already **above** the pure-B floor once A is counted, and the only real losses in the
whole kernel are the 79 us of compute that the barrier-per-chunk structure fails to
overlap. There is no 2.5x sitting in staging; that number came from comparing against
a floor the access pattern does not allow.

The one genuine opening the probe found is that B's own 64 B row read is worth 2x
(719 -> 1471 at 128 B), which is what motivated the KC=128 attempt -- and the two
sections above are why it still cannot be cashed in.

### Reading A directly from global: catastrophic

The obvious escape is to stop staging A, since it is 256 KB and L2-resident anyway,
which would free enough shared memory for large N tiles at KC=128. Implemented in
`gemm_fp8_v20.cu`: A fragments read at the MMA site, shared memory holds only B.

| config | time | ratio |
|--------|------|-------|
| KC64 256col (identical to the 0.527 ms config otherwise) | 5.199 ms | 0.13x |
| KC128 192col | 6.837 ms | 0.10x |
| KC128 128col | 9.619 ms | 0.07x |

A 10x regression. The A fragment read is 4 x 8 B per thread per k-step at row
stride 4096 B -- uncoalesced, and issued inside the MMA dependency chain with
nothing to hide it. The shared-memory staging of A is not overhead, it is what makes
the fragments cheap to load. This closes the "free the shared memory" idea entirely.

### Where that leaves it

Shared memory, A traffic, B read granularity and block count form a closed loop: the
only way to lengthen B's per-row read is a larger KC, which costs B shared memory,
which forces fewer columns, which multiplies A traffic, which is worse than the
pattern gain. At 64 KB there is no point in that loop that beats KC=64 / 256 columns.
The config that wins is the one the constraint graph pins, which is why every
neighbour of it loses.

So the deliverable is the dispatch decision, not another kernel: fp8 `lm_head` at
1.26x plus the vendor bf16 path for the four small projections, measured at **1.215x**
on the decode step. `gemm_fp8_v16.cu` remains the best kernel and its lm_head config
(`WPB4 NT4 KC64 PADA16`, split 1) is unchanged.

Files: `rowprobe.cu` / `run_rows.py` (pattern-only floor at scale),
`gemm_fp8_v19.cu` (KC=128 experiments), `gemm_fp8_v20.cu` (global-A, dead end),
`bw_sweep.py` and `bw_big.py` (bandwidth against size and grid).


## Accuracy: what the fp8 lm_head actually costs

Every `rel=2e-06` figure elsewhere in this document compares the kernel against the
**exact result for already-quantized weights**. That is a correctness check, not an
accuracy check, and the two are four orders of magnitude apart. The numbers below are
the accuracy side.

Measured on Qwen3-0.6B, a real checkpoint with a real vocabulary projection
(`lm_head` is `[151936, 1024]`), because MiniCPM5-2B weights were not available on
this machine.

### Per-projection error

Real hidden states entering `lm_head`, five prompts:

| quantization | relative Frobenius | max abs dlogit | top-1 agree | cosine |
|--------------|--------------------|----------------|-------------|--------|
| per-tensor | 0.02649 | 0.549 | 5/5 | 0.999281 |
| per-row | 0.02641 | 0.635 | 5/5 | 0.999252 |
| per-128-group | 0.02572 | 0.612 | 5/5 | 0.999295 |

Two useful negatives: **finer scale granularity buys nothing here**, and the reason is
in the weight statistics -- this `lm_head` has an unusually even row-magnitude
distribution, max/median = 3.156, so there are no outlier rows for a finer scale to
rescue. Per-tensor is therefore sufficient, and the kernel needs no per-row scale
support. That saves a real amount of complexity.

### End-to-end perplexity

40 windows of 1024 tokens from a fineweb-edu shard, paired (both configurations see
the identical tokens), all other layers bf16:

| `lm_head` | mean ppl |
|-----------|----------|
| bf16 | 23.5393 (sd 8.17) |
| fp8 per-tensor | 23.5912 (sd 8.21) |

**Paired relative regression +0.2143%, 95% CI [+0.0956%, +0.3331%], t=3.65, df=39.**
8 of 40 windows improved, 32 regressed.

For scale: +0.21% is roughly an order of magnitude below the +0.5-2% typical of
full-model W8A8, and it is paid on a single layer.

### A methodological trap worth recording

The first run of this experiment used 10 windows and measured +0.0889% with
sem 0.0950% -- t of about 0.94, i.e. indistinguishable from noise, with 3 of 10
windows improving. That was read as "no significant degradation". **It was an
underpowered experiment, not a null result.** At 40 windows the standard error halves
to 0.059% and the same effect becomes 3.65 sigma.

The lesson generalizes: the window-to-window spread of absolute ppl is sd 8.17 while
the effect being measured is 0.05 ppl, a ratio of 40. Comparing two corpus means can
never resolve this; only the paired difference can, and even then it needs tens of
windows. Any future accuracy claim here should quote the paired CI, not a pair of
means.

### Caveats

* Qwen3-0.6B is not the target model. Its hidden size is 1024 against MiniCPM5-2B's
  2048, and a wider hidden dimension divides each element's contribution further, so
  MiniCPM5-2B should land at or below this regression -- but that is an argument, not a
  measurement. Re-running `ppl_paired.py` against the real checkpoint is the open item.
* Only `lm_head` is quantized, which matches the shipped configuration (the four small
  projections stay bf16 because fp8 loses on them -- see the floor probe).
