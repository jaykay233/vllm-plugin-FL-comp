// Fused FP8 (e4m3) GQA paged-decode attention for MetaX C500 / MACA 3.7.
//
// Design
// ------
//   * One 64-lane warp per (batch, kv_head, split).  GQA: the warp owns all
//     `group = H/HKV` query heads of one KV head, so K/V are read once.
//   * Scores and PV both use the *hardware* fp8 MMA
//         __builtin_mxc_mma_16x16x32f8   (m16n16k32, e4m3 x e4m3 -> f32)
//     so there is no elementwise fp8->float dequant (the 4.4x Triton slowdown
//     documented in experiments/fp8kv/ is avoided entirely).
//   * Fragment layout (calibrated on C500, see csrc/calib.cu):
//         A : row = lane%16, k = {g*4+0..3} (reg0) and {16+g*4+0..3} (reg1)
//         B : n   = lane%16, k = {g*4+0..3} (reg0) and {16+g*4+0..3} (reg1)
//         C : row = lane%16, col = g*4 + reg
//     with g = lane/16 and warpSize = 64.
//   * Q is quantised to e4m3 per query head inside the kernel (scale folded
//     into the softmax base-2 scale).  P is scaled by 256 before the e4m3
//     pack to keep the softmax weights in the normal range.
//   * KV cache layout: [num_blocks, page, HKV, D] fp8, page/gather via
//     block_table.  Online softmax with split-K; a combine kernel finishes.
//
// CUDA-graph safe: no host sync, no .item(), all bounds are device values.
//
// Measured on C500 (B=64, seq=4096, HKV=2, H=16, D=128, P=64, split-K=16)
// ---------------------------------------------------------------------------
//   bf16 tensor-core kernel (default)  0.356 ms/layer   377 GB/s
//   combine kernel only                0.028 ms
//   vendor FlashAttention bf16         0.214 ms/layer  1257 GB/s   (1.66x)
//   Triton fp8 (first attempt)         3.40 ms/layer              (9.6x)
//   fp8 MMA kernel (previous default)  0.769 ms/layer   174 GB/s   (2.17x slower)
//   direct-from-global kernel          1.118 ms/layer
//
// Journey: 3.40 ms (Triton) -> 1.118 ms (hardware fp8 MMA) -> 0.966 ms
//          (fragment staging) -> 0.832 ms (async g2s K pipeline) -> 0.769 ms
//          (bank-conflict swizzle) -> 0.356 ms (bf16 tensor core).
//
// The decisive result: on C500 the fp8 tensor core is the *slow* path.  An
// m16n16x32f8 whose operands have to be rebuilt from general-purpose registers
// for every instruction costs ~94 cycles; m16n16k16bf16 costs ~16.  Covering
// the same K=32 of work therefore needs two bf16 instructions (32 cycles),
// still 2.9x cheaper than one fp8 instruction even after paying for the
// e4m3->bf16 dequant (mma_cmp.cu measures 94.5 vs 32.1 cycles).  This is
// exactly why the vendor FlashAttention binary contains no fp8 MMA at all: it
// dequantises in registers and uses the bf16 tensor core.  We now do the same,
// with a branchless ALU conversion so shared-memory banks stay free for the K
// pipeline.  A side effect is 5.6x better accuracy than the fp8 path, since
// only the KV cache is quantised -- Q and P stay bf16.
//
// Where the time goes (ablation, S=16):
//   full                               0.352 ms
//   no QK, no PV                       0.126 ms   <- pure K pipeline, 613 GB/s
//   no PV (QK only)                    0.220 ms   -> PV costs 0.132
//   no QK (PV only)                    0.272 ms   -> QK costs 0.080
// Split further with probe builds: V gather 0.135, PV MMA 0.100, QK MMA 0.084.
//
// Pipeline depth: copy_bw.cu shows the g2s engine going 291 -> 625 -> 2398 GB/s
// for one warp as the number of allowed outstanding copies rises 0 -> 8 ->
// unbounded, so the natural next move was more K stages.  It does not work:
// 4 stages need 16 KB of shared memory, which halves how many 1-warp CTAs stay
// resident per SM, and the kernel measured 0.435 ms instead of 0.352.  With one
// warp per CTA the residency *is* the pipeline, so 2 stages is the sweet spot.
// Loosening the wait from arrive_gvmcnt(0) to arrive_gvmcnt(4) was still worth
// having: it keeps the second stage in flight instead of draining it.
// Register caps (112/96/80) and a structurally lighter PV loop were both tried
// and neither helps -- 162 registers with no spills is already the compiler's
// best point.
//
// Two changes account for the earlier fp8 improvements, both measured:
//   1. Fragment staging.  Interleaving ld4(smem) -> mma serialises a full
//      shared-memory round trip into every MMA.  Gathering a whole tile's
//      fragments into registers first and then firing the MMAs is worth 12%
//      (1.118 -> 0.966, and it also dropped the register count 240 -> 234).
//      A dependent-chain operand costs ~130-330 cyc/mma versus ~22 when the
//      operand is already resident.
//   2. Async g2s K pipeline.  K needs no transpose, so the hardware copy engine
//      moves it: __builtin_mxc_ldg_b128_bsm into a double buffer, waited on
//      with __builtin_mxc_arrive(64) (== arrive_gvmcnt(0)).  That removes the
//      32-register prefetch buffer (240 -> 182 registers) and overlaps the copy
//      with compute: 0.966 -> 0.832.  V deliberately stays a direct global byte
//      gather -- the smem-transposed PV loop measured 0.84 ms alone vs 0.57 ms
//      for the gather, so only K goes through shared memory.
//
// Calibrated constraint (see g2s_calib.cu): __builtin_mxc_ldg_b128_bsm requires
// the 64 lanes' shared destinations to be contiguous and lane-linear
// (dst == base + lane*16).  A padded stride silently scribbles elsewhere, so
// the async K tile must use the unpadded 128-byte stride and eats 16-way bank
// conflicts on the MMA read.  That is still a net win here.
//
// Ruled out by direct measurement (bw_probe2.cu, mma_probe.cu, mma_dyn.cu):
//   * DRAM/transaction cost -- the identical paged K+V address stream with
//     coalesced 16B/lane loads and no compute runs at 4.5 TB/s (0.0297 ms).
//   * Coalescing -- the padded-smem kernel (bit-for-bit identical output) is
//     not faster than the direct one.
//   * fp32->e4m3 conversion (+0.04 ms), exp2f (+0.01 ms), the accumulator
//     rescale (now conditional, no change), the combine kernel (0.028 ms).
//   * Occupancy via register caps -- --maxrregcount=96 spills, 2.4x slower.
//   * Full double buffering of K *and* V in shared memory (fp8_decode_split_
//     kernel_pipe) -- 17664 B/CTA against a 64 KB/SM budget leaves 3 CTAs/SM
//     and is 1.5x slower despite the prefetch.
//
// Note __builtin_mxc_mma_16x16x32f8 is real hardware on xcore1000 (the
// *f32_16x16x32f8_e4m3* spelling in mctlass/arch/maca_fp8_tmp.h is a software
// emulation: fp8->fp16 then 2x m16n16k16f16).
//
// Vendor FlashAttention on C500 was reverse-engineered from the unstripped
// mangled symbols in /opt/maca/lib/libmcFlashAttn.so's .mc_fatbin: it has no
// fp8 path at all (zero hits for 16x16x32f8 / e4m3 / e5m2), uses
// mma.f32.16x16x16bf16/f16 with byte.perm-based fp8 expansion, and runs a
// multi-stage pipeline built on ldg.predicator.bsm / arriv / barrier.inst, with
// kBlockN=64 and 4 warps per CTA for decode.
//
// Remaining structural headroom: only 8 of the MMA's 16 rows are live
// (GQA group = 8 of m16), i.e. a theoretical 2x.

#include <mc_runtime.h>
#include <maca_bfloat16.h>
#include <cstdint>
#include <cfloat>

typedef __NATIVE_VECTOR__(2, uint32_t) v2u;
typedef __NATIVE_VECTOR__(4, float) v4f;

namespace {

constexpr int kD = 128;       // head_dim (template-fixed)
constexpr int kRows = 16;     // MMA M rows (query heads, padded)
constexpr int kStep = 32;     // MMA K per call
constexpr int kDt = 128 / 16; // 8 output d-tiles
constexpr float kLog2e = 1.4426950408889634f;
constexpr float kPscale = 256.0f;

__device__ __forceinline__ uint32_t ld4(const uint8_t* p) {
  uint32_t v;
  __builtin_memcpy(&v, p, 4);
  return v;
}

__device__ __forceinline__ v4f mma_f8(v2u b, v2u a, v4f c) {
#ifdef FP8_PROBE_NOMMA
  // Timing probe: keep every load and every dependency, drop only the tensor
  // core instruction, so the measured delta is the MMA itself.
  c[0] = fmaf((float)(b[0] & 0xffu), 1e-30f, c[0]);
  c[1] = fmaf((float)(a[0] & 0xffu), 1e-30f, c[1]);
  c[2] = fmaf((float)(b[1] & 0xffu), 1e-30f, c[2]);
  c[3] = fmaf((float)(a[1] & 0xffu), 1e-30f, c[3]);
  return c;
#else
  return __builtin_mxc_mma_16x16x32f8(b, a, c);
#endif
}

// C500 (xcore1000) exposes the fp8 MMA but *not* the packed f32->e4m3
// conversion (that builtin needs xcore1600/1700).  P/Q are a tiny fraction of
// the traffic, so a small bit-twiddling converter is fine here; K/V never get
// converted -- they feed the MMA as raw fp8.
__device__ __forceinline__ uint32_t f32_to_e4m3_u(float x) {
  uint32_t u;
  __builtin_memcpy(&u, &x, 4);
  const uint32_t sign = (u >> 24) & 0x80u;
  const uint32_t mag = u & 0x7FFFFFFFu;
  if (mag > 0x7F800000u) return sign | 0x7Fu;          // NaN
  if (mag >= 0x43E00000u) return sign | 0x7Eu;         // >= 448 -> saturate
  const uint32_t e = mag >> 23;
  if (e < 121u) return sign;                           // < 2^-6 (incl 0/denorm)
  uint32_t mant = (mag >> 20) & 0x7u;
  mant += (mag >> 19) & 0x1u;                          // round to nearest
  uint32_t exp = e - 120u;                             // e4m3 biased exponent
  if (mant & 0x8u) {
    mant = 0;
    ++exp;
  }
  if (exp >= 16u) return sign | 0x7Eu;                 // overflow -> 448
  if (exp == 15u && mant > 6u) return sign | 0x7Eu;    // 0x7F would be NaN
  return sign | (exp << 3) | mant;
}

__device__ __forceinline__ uint32_t pack_f8(float x0, float x1, float x2,
                                            float x3) {
  return f32_to_e4m3_u(x0) | (f32_to_e4m3_u(x1) << 8) |
         (f32_to_e4m3_u(x2) << 16) | (f32_to_e4m3_u(x3) << 24);
}

// ---------------------------------------------------------------------------
// BF16 MMA path
// ---------------------------------------------------------------------------
// The fp8 MMA is fast only when its operands are already resident in the
// operand registers.  Inside this kernel a fresh operand has to be moved out of
// general-purpose registers for every instruction, and that move costs ~94
// cycles for m16n16k32f8 against ~16 for m16n16k16bf16.  Even though bf16 needs
// twice the instructions to cover the same K, it is ~2.9x cheaper per unit of
// work once the fp8->bf16 dequant is included (measured in mma_cmp.cu):
//
//     f8    m16n16k32, raw operand          94.5 cyc / K=32
//     bf16  m16n16k16 + ALU dequant         32.1 cyc / K=32   -> 2.9x
//
// This is exactly why the vendor FlashAttention binary contains no fp8 MMA at
// all: it dequantises in registers and uses the bf16 tensor core.  We do the
// same, but with a branchless ALU conversion so the shared-memory banks stay
// free for the K pipeline.

typedef __NATIVE_VECTOR__(4, __fp16) v4h;

// Argument order is (b, a, c) -- the k x n operand first -- exactly like the
// fp8 builtin, even though mctlass mma_sync() spells its call (a, b, c).
// Verified end to end in mma_bf16_gemm.cu: mma(B, A, C) reproduces A @ B with
// zero error, while mma(A, B, C) does not.
__device__ __forceinline__ v4f mma_bf16(v4h b, v4h a, v4f c) {
#ifdef FP8_PROBE_NOMMA
  c[0] = fmaf((float)(unsigned short&)b[0], 1e-30f, c[0]);
  c[1] = fmaf((float)(unsigned short&)a[0], 1e-30f, c[1]);
  return c;
#else
  return __builtin_mxc_mma_16x16x16bf16(b, a, c);
#endif
}

#ifdef KV_INT8
// int8 -> bf16 bits.  int8 values are exact small integers, so this is just an
// I2F followed by the usual round-to-nearest-even fp32 -> bf16 truncation.  The
// measured cost is a single I2F per value, versus mask/shift/add/sign-move for
// e4m3, which is why the int8 variant should beat the fp8 one on this kernel.
__device__ __forceinline__ uint16_t i8_to_bf16_bits(uint32_t u) {
  const float f = (float)(int)(signed char)(u & 0xffu);
  uint32_t b;
  __builtin_memcpy(&b, &f, 4);
  b += 0x7fffu + ((b >> 16) & 1u);
  return (uint16_t)(b >> 16);
}
#endif

// e4m3 -> bf16, branchless.  e4m3 is s(1) e(4) m(3); bf16 is s(1) E(8) M(7).
//   u<<4  lands s at bit 11, e at 7..10, m at 4..6  (m is already in place)
//   +0x3C00 rebases the exponent from bias 7 to bias 127  (120<<7)
//   +0x7800 walks the sign from bit 11 up to bit 15
//   e == 0 (zero and subnormals, which are negligible in a KV cache) -> 0
__device__ __forceinline__ uint32_t f8_to_bf16_bits(uint32_t u) {
  const uint32_t mag = u & 0x7fu;
  const uint32_t v = (mag << 4) + 0x3C00u;
  const uint32_t keep = ((mag >> 3) == 0u) ? 0u : 0xffffffffu;
  return (((u & 0x80u) << 8) | v) & keep;
}

// e4m3 x4 (little-endian bytes) -> four bf16.  Each byte must be extracted by
// shifting, not by adding to the word: `b + 1` would carry into the next byte
// whenever the low byte is 0xFF, and the low byte is not byte 1 anyway.
// deq4 via a shared-memory table.  The ALU form below costs ~35 instructions
// per four values (four times: mask, shift, add, sign move, zero test);
// the table form is four 16-bit loads plus three combines.  Measured at ~21% of
// the kernel's runtime, so this is worth doing.
__device__ __forceinline__ void build_lut(uint16_t* lut) {
  for (int i = threadIdx.x; i < 256; i += blockDim.x)
  #ifdef KV_INT8
  lut[i] = i8_to_bf16_bits((uint32_t)i);
#else
  lut[i] = (uint16_t)f8_to_bf16_bits((uint32_t)i);
#endif
}

__device__ __forceinline__ v4h deq4_lut(const uint16_t* lut, uint32_t b) {
  uint32_t w[2];
  w[0] = (uint32_t)lut[b & 0xffu] | ((uint32_t)lut[(b >> 8) & 0xffu] << 16);
  w[1] = (uint32_t)lut[(b >> 16) & 0xffu] | ((uint32_t)lut[b >> 24] << 16);
  v4h r;
  __builtin_memcpy(&r, w, 8);
  return r;
}

__device__ __forceinline__ v4h deq4(uint32_t b) {
#ifdef FP8_PROBE_CHEAP_DEQ
  // Timing probe only (numerically wrong): bare byte unpack, no exponent
  // rebase and no zero handling.  Isolates how much of the runtime is the
  // e4m3->bf16 conversion ALU rather than the MMA or the memory traffic.
  uint32_t w[2];
  w[0] = (b & 0xffu) | ((b << 8) & 0xff0000u);
  w[1] = ((b >> 16) & 0xffu) | ((b >> 8) & 0xff000000u);
  v4h r;
  __builtin_memcpy(&r, w, 8);
  return r;
#else
  uint32_t w[2];
  w[0] = f8_to_bf16_bits(b & 0xffu) |
         (f8_to_bf16_bits((b >> 8) & 0xffu) << 16);
  w[1] = f8_to_bf16_bits((b >> 16) & 0xffu) |
         (f8_to_bf16_bits(b >> 24) << 16);
  v4h r;
  __builtin_memcpy(&r, w, 8);
  return r;
#endif
}

// float -> bf16 bits, round to nearest even.
__device__ __forceinline__ uint32_t f2bf_bits(float x) {
  uint32_t u;
  __builtin_memcpy(&u, &x, 4);
  u += 0x7fffu + ((u >> 16) & 1u);
  return u >> 16;
}

__device__ __forceinline__ v4h pack_bf16(float x0, float x1, float x2,
                                         float x3) {
  uint32_t w[2];
  w[0] = f2bf_bits(x0) | (f2bf_bits(x1) << 16);
  w[1] = f2bf_bits(x2) | (f2bf_bits(x3) << 16);
  v4h r;
  __builtin_memcpy(&r, w, 8);
  return r;
}

__device__ __forceinline__ float bf2f(const __maca_bfloat16 x) {
  return __bfloat162float(x);
}

__device__ __forceinline__ float warp_max(float x) {
  const uint64_t m = ~0ull;
  x = fmaxf(x, __shfl_xor_sync(m, x, 16));
  x = fmaxf(x, __shfl_xor_sync(m, x, 32));
  return x;
}

__device__ __forceinline__ float warp_sum(float x) {
  const uint64_t m = ~0ull;
  x += __shfl_xor_sync(m, x, 16);
  x += __shfl_xor_sync(m, x, 32);
  return x;
}

// ---------------------------------------------------------------------------
// Stage 1: partial attention over one split of the sequence.
// ---------------------------------------------------------------------------
__global__ void fp8_decode_split_kernel(
    const __maca_bfloat16* __restrict__ q,   // [B, H, D]
    const uint8_t* __restrict__ k_cache,     // [nb, P, HKV, D] fp8
    const uint8_t* __restrict__ v_cache,     // [nb, P, HKV, D] fp8
    const int32_t* __restrict__ block_table, // [B, stride_bt]
    const int32_t* __restrict__ seq_lens,    // [B]
    float* __restrict__ mid_o,               // [S, B, H, D]
    float* __restrict__ mid_m,               // [S, B, H]
    float* __restrict__ mid_l,               // [S, B, H]
    int B, int H, int HKV, int P, int stride_bt, int num_splits,
    float sm_scale, float k_scale, int mode) {
  const int lane = threadIdx.x & 63;
  const int pid = blockIdx.x;
  const int split = pid % num_splits;
  const int rest = pid / num_splits;
  const int kvh = rest % HKV;
  const int b = rest / HKV;

  const int row = lane & 15;   // query head within the group
  const int g = lane >> 4;     // 0..3, k/col block
  const int group = H / HKV;

  const int seqlen = seq_lens[b];
  const int npages = (seqlen + P - 1) / P;
  const int chunk = (npages + num_splits - 1) / num_splits;
  int p0 = split * chunk;
  int p1 = p0 + chunk;
  if (p1 > npages) p1 = npages;

  // ---- quantise this row of Q to e4m3 (per-head scale) ----
  const int qhead = kvh * group + row;
  uint32_t aq[4][2];
  float qscale = 1.0f;
  const bool row_ok = (row < group);
  if (row_ok) {
    const __maca_bfloat16* qrow =
        q + (static_cast<size_t>(b) * H + qhead) * kD;
    float amax = 0.0f;
#pragma unroll 4
    for (int d = 0; d < kD; ++d) amax = fmaxf(amax, fabsf(bf2f(qrow[d])));
    float sc = amax * (1.0f / 448.0f);
    qscale = (sc > 1e-8f) ? sc : 1e-8f;
    const float inv = 1.0f / qscale;
#pragma unroll
    for (int ks = 0; ks < 4; ++ks) {
      const int k0 = ks * kStep + g * 4;
      aq[ks][0] = pack_f8(bf2f(qrow[k0 + 0]) * inv, bf2f(qrow[k0 + 1]) * inv,
                          bf2f(qrow[k0 + 2]) * inv, bf2f(qrow[k0 + 3]) * inv);
      const int k1 = k0 + 16;
      aq[ks][1] = pack_f8(bf2f(qrow[k1 + 0]) * inv, bf2f(qrow[k1 + 1]) * inv,
                          bf2f(qrow[k1 + 2]) * inv, bf2f(qrow[k1 + 3]) * inv);
    }
  } else {
#pragma unroll
    for (int ks = 0; ks < 4; ++ks) {
      aq[ks][0] = 0u;
      aq[ks][1] = 0u;
    }
  }

  // score scale: qscale * k_scale * sm_scale, expressed in base-2
  const float sscale = qscale * k_scale * sm_scale * kLog2e;

  float m_prev = row_ok ? -FLT_MAX : 0.0f;
  float l_acc = 0.0f;
  v4f o_acc[kDt];
#pragma unroll
  for (int i = 0; i < kDt; ++i) o_acc[i] = {0.f, 0.f, 0.f, 0.f};

  const size_t kv_head_stride = kD;
  const size_t tok_stride = static_cast<size_t>(HKV) * kD;
  const int tok_in_page = lane & 15;
  const int tstep = (P % kStep == 0) ? kStep : 16;
  const int nt_count = tstep / 16;

  if (p0 < p1) {
    for (int p = p0; p < p1; ++p) {
      const int page_id = block_table[static_cast<size_t>(b) * stride_bt + p];
      const size_t page_base =
          (static_cast<size_t>(page_id) * P) * tok_stride +
          static_cast<size_t>(kvh) * kv_head_stride;
      const int page_valid = seqlen - p * P;  // >0

      for (int tl = 0; tl < P; tl += tstep) {
        const int nvalid = min(tstep, page_valid - tl);
        if (nvalid <= 0) break;

        float s[2][4];
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
          if (nt >= nt_count || (mode & 2)) {
#pragma unroll
            for (int r = 0; r < 4; ++r) s[nt][r] = 0.f;
            continue;
          }
          const int btok = tl + nt * 16 + tok_in_page;
          const bool bvalid = btok < tl + nvalid;
          const uint8_t* kp = k_cache + page_base +
                              static_cast<size_t>(btok) * tok_stride;
          v4f acc[4];
#ifdef FP8_QK_SPLIT
#pragma unroll
          for (int i = 0; i < 4; ++i) acc[i] = {0.f, 0.f, 0.f, 0.f};
#else
          acc[0] = {0.f, 0.f, 0.f, 0.f};
#endif
#pragma unroll
          for (int ks = 0; ks < 4; ++ks) {
            const int d0 = ks * kStep + g * 4;
            uint32_t k0 = bvalid ? ld4(kp + d0) : 0u;
            uint32_t k1 = bvalid ? ld4(kp + d0 + 16) : 0u;
            v2u vb = {k0, k1};
            v2u va = {aq[ks][0], aq[ks][1]};
            #ifdef FP8_QK_SPLIT
            acc[ks] = mma_f8(vb, va, acc[ks]);
#else
            acc[0] = mma_f8(vb, va, acc[0]);
#endif
          }
#pragma unroll
          for (int r = 0; r < 4; ++r) {
#ifdef FP8_QK_SPLIT
            s[nt][r] = (acc[0][r] + acc[1][r]) + (acc[2][r] + acc[3][r]);
#else
            s[nt][r] = acc[0][r];
#endif
          }
        }

        // ---- online softmax (row = lane%16, cols spread over g) ----
        float m_local = -FLT_MAX;
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
#pragma unroll
          for (int r = 0; r < 4; ++r) {
            if (nt >= nt_count) {
              s[nt][r] = -FLT_MAX;
              continue;
            }
            const int token = tl + nt * 16 + g * 4 + r;
            const bool ok = row_ok && (token < tl + nvalid);
            float sv = ok ? s[nt][r] * sscale : -FLT_MAX;
            s[nt][r] = sv;
            m_local = fmaxf(m_local, sv);
          }
        }
        uint32_t pf[2];
        if (mode & 8) {
          // timing probe: drop the whole softmax math
          pf[0] = 0x38383838u;
          pf[1] = 0x38383838u;
        } else {
          const float m_row = warp_max(m_local);
          const float m_new = fmaxf(m_prev, m_row);
          // Lazy rescale: alpha is 1 whenever the running max did not move,
          // and m_new is warp-uniform, so this is a uniform branch.  It matters
          // a lot here: o_acc lives in the MMA accumulator registers, and an
          // unconditional rescale forces 32 accumulator read-modify-writes per
          // tile (~330 cyc/mma instead of ~22).
          if (m_new != m_prev) {
            const float alpha =
                (m_prev == -FLT_MAX) ? 0.0f : exp2f(m_prev - m_new);
            m_prev = m_new;
#pragma unroll
            for (int i = 0; i < kDt; ++i) {
              o_acc[i][0] *= alpha;
              o_acc[i][1] *= alpha;
              o_acc[i][2] *= alpha;
              o_acc[i][3] *= alpha;
            }
            l_acc *= alpha;
          }

          float lsum = 0.0f;
#pragma unroll
          for (int nt = 0; nt < 2; ++nt) {
            float pv[4];
#pragma unroll
            for (int r = 0; r < 4; ++r) {
              const float ex = (mode & 16) ? 1.5f : exp2f(s[nt][r] - m_new);
              const float e = (s[nt][r] == -FLT_MAX) ? 0.0f : ex * kPscale;
              pv[r] = e;
              lsum += e;
            }
            pf[nt] = (mode & 32) ? 0x38383838u
                                 : pack_f8(pv[0], pv[1], pv[2], pv[3]);
          }
          l_acc += warp_sum(lsum) * (1.0f / kPscale);
        }

        // ---- PV: one fp8 MMA per 16-d tile, K = 32 kv tokens ----
        v2u pa = {pf[0], pf[1]};
        if (!(mode & 1)) {
#pragma unroll
          for (int dt = 0; dt < kDt; ++dt) {
            const int dv = dt * 16 + (lane & 15);
            const uint8_t* vp0 = v_cache + page_base +
                                 static_cast<size_t>(tl + g * 4) * tok_stride + dv;
            const uint8_t* vp1 = vp0 + 16 * tok_stride;
            uint32_t b0 = (uint32_t)vp0[0] |
                          ((uint32_t)vp0[tok_stride] << 8) |
                          ((uint32_t)vp0[2 * tok_stride] << 16) |
                          ((uint32_t)vp0[3 * tok_stride] << 24);
            uint32_t b1 = 0u;
            if (nt_count == 2) {
              b1 = (uint32_t)vp1[0] | ((uint32_t)vp1[tok_stride] << 8) |
                   ((uint32_t)vp1[2 * tok_stride] << 16) |
                   ((uint32_t)vp1[3 * tok_stride] << 24);
            }
            v2u vb = {b0, b1};
            o_acc[dt] = mma_f8(vb, pa, o_acc[dt]);
          }
        }
      }
    }
  }

  const size_t mid_idx =
      (static_cast<size_t>(split) * B * H + static_cast<size_t>(b) * H + qhead);
  if (row_ok) {
    float* op = mid_o + mid_idx * kD;
#pragma unroll
    for (int dt = 0; dt < kDt; ++dt) {
      const int d = dt * 16 + g * 4;
      op[d + 0] = o_acc[dt][0];
      op[d + 1] = o_acc[dt][1];
      op[d + 2] = o_acc[dt][2];
      op[d + 3] = o_acc[dt][3];
    }
    mid_m[mid_idx] = m_prev;
    mid_l[mid_idx] = l_acc;
  }
}

// ---------------------------------------------------------------------------
// Shared-memory staged split kernel.
//
// The direct version above fetches every MMA operand straight from global
// memory.  The fragment layout forces each lane to grab 4 *bytes* at a time
// with a 256-byte stride (K) or 4 separate bytes 256 bytes apart (V), so a
// single fragment fetch degenerates into a 16-way strided gather.  The total
// DRAM traffic is right, but the transaction count is ~10x what the memory
// system sustains, which is exactly the 0.99 ms vs 0.21 ms gap to vendor FA.
//
// Here each 32-token tile of K and V is first copied into shared memory with
// fully coalesced 16-byte per-lane loads (64 lanes -> 1 KiB per instruction),
// V is transposed on the fly so the PV B-fragment becomes 4 contiguous bytes,
// and the MMA operands are then read from smem (padded strides keep both the
// ld4 (K) and the PV (V^T) accesses bank-conflict free).
// ---------------------------------------------------------------------------
constexpr int kKStride = 132;  // 128 + 4 : 33 words/row -> ld4 conflict free
constexpr int kVtStride = 36;  //  32 + 4 :  9 words/row -> PV  conflict free

__global__ void fp8_decode_split_kernel_smem(
    const __maca_bfloat16* __restrict__ q, const uint8_t* __restrict__ k_cache,
    const uint8_t* __restrict__ v_cache, const int* __restrict__ block_table,
    const int* __restrict__ seq_lens, float* __restrict__ mid_o,
    float* __restrict__ mid_m, float* __restrict__ mid_l,
    int B, int H, int HKV, int P, int stride_bt, int num_splits,
    float sm_scale, float k_scale, int mode) {
  __shared__ uint8_t sk[32 * kKStride];
  __shared__ uint8_t svt[kD * kVtStride];

  const int lane = threadIdx.x & 63;
  const int pid = blockIdx.x;
  const int split = pid % num_splits;
  const int rest = pid / num_splits;
  const int kvh = rest % HKV;
  const int b = rest / HKV;

  const int row = lane & 15;   // query head within the group
  const int g = lane >> 4;     // 0..3, k/col block
  const int group = H / HKV;

  const int seqlen = seq_lens[b];
  const int npages = (seqlen + P - 1) / P;
  const int chunk = (npages + num_splits - 1) / num_splits;
  int p0 = split * chunk;
  int p1 = p0 + chunk;
  if (p1 > npages) p1 = npages;

  // ---- quantise this row of Q to e4m3 (per-head scale) ----
  const int qhead = kvh * group + row;
  uint32_t aq[4][2];
  float qscale = 1.0f;
  const bool row_ok = (row < group);
  if (row_ok) {
    const __maca_bfloat16* qrow =
        q + (static_cast<size_t>(b) * H + qhead) * kD;
    float amax = 0.0f;
#pragma unroll 4
    for (int d = 0; d < kD; ++d) amax = fmaxf(amax, fabsf(bf2f(qrow[d])));
    float sc = amax * (1.0f / 448.0f);
    qscale = (sc > 1e-8f) ? sc : 1e-8f;
    const float inv = 1.0f / qscale;
#pragma unroll
    for (int ks = 0; ks < 4; ++ks) {
      const int k0 = ks * kStep + g * 4;
      aq[ks][0] = pack_f8(bf2f(qrow[k0 + 0]) * inv, bf2f(qrow[k0 + 1]) * inv,
                          bf2f(qrow[k0 + 2]) * inv, bf2f(qrow[k0 + 3]) * inv);
      const int k1 = k0 + 16;
      aq[ks][1] = pack_f8(bf2f(qrow[k1 + 0]) * inv, bf2f(qrow[k1 + 1]) * inv,
                          bf2f(qrow[k1 + 2]) * inv, bf2f(qrow[k1 + 3]) * inv);
    }
  } else {
#pragma unroll
    for (int ks = 0; ks < 4; ++ks) {
      aq[ks][0] = 0u;
      aq[ks][1] = 0u;
    }
  }

  const float sscale = qscale * k_scale * sm_scale * kLog2e;

  float m_prev = row_ok ? -FLT_MAX : 0.0f;
  float l_acc = 0.0f;
  v4f o_acc[kDt];
#pragma unroll
  for (int i = 0; i < kDt; ++i) o_acc[i] = {0.f, 0.f, 0.f, 0.f};
  if (!row_ok) m_prev = 0.0f;

  const size_t tok_stride = static_cast<size_t>(HKV) * kD;
  const int tstep = (P % kStep == 0) ? kStep : 16;
  const int nt_count = tstep / 16;

  // coalesced staging: lane loads 16 contiguous bytes of a 32-token tile
  const int stok = lane >> 3;         // 0..7  token within the 1 KiB chunk
  const int soff = (lane & 7) * 16;   // 0..112 d offset inside the token

  if (p0 < p1) {
    for (int p = p0; p < p1; ++p) {
      const int page_id = block_table[static_cast<size_t>(b) * stride_bt + p];
      const uint8_t* kbase =
          k_cache + (static_cast<size_t>(page_id) * P) * tok_stride +
          static_cast<size_t>(kvh) * kD;
      const uint8_t* vbase =
          v_cache + (static_cast<size_t>(page_id) * P) * tok_stride +
          static_cast<size_t>(kvh) * kD;
      const int page_valid = seqlen - p * P;

      for (int tl = 0; tl < P; tl += tstep) {
        const int nvalid = min(tstep, page_valid - tl);
        if (nvalid <= 0) break;

        // ---- stage K tile (32 x 128) with coalesced 16B loads ----
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const int t = j * 8 + stok;
          const bool ok = (t < nvalid);
          uint32_t w0 = 0u, w1 = 0u, w2 = 0u, w3 = 0u;
          if (ok) {
            const uint8_t* src = kbase + static_cast<size_t>(tl + t) * tok_stride + soff;
            uint4 v = *reinterpret_cast<const uint4*>(src);
            w0 = v.x; w1 = v.y; w2 = v.z; w3 = v.w;
          }
          uint32_t* dst = reinterpret_cast<uint32_t*>(sk + t * kKStride + soff);
          dst[0] = w0; dst[1] = w1; dst[2] = w2; dst[3] = w3;
        }
        // ---- stage V tile transposed (128 x 32) ----
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const int t = j * 8 + stok;
          const bool ok = (t < nvalid);
          uint4 v = make_uint4(0u, 0u, 0u, 0u);
          if (ok) {
            const uint8_t* src = vbase + static_cast<size_t>(tl + t) * tok_stride + soff;
            v = *reinterpret_cast<const uint4*>(src);
          }
#pragma unroll
          for (int i = 0; i < 16; ++i) {
            const uint32_t w = (i < 4) ? v.x : ((i < 8) ? v.y : ((i < 12) ? v.z : v.w));
            svt[(soff + i) * kVtStride + t] = (uint8_t)(w >> ((i & 3) * 8));
          }
        }
        __syncthreads();

        float s[2][4];
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
          if (nt >= nt_count || (mode & 2)) {
#pragma unroll
            for (int r = 0; r < 4; ++r) s[nt][r] = 0.f;
            continue;
          }
          v4f acc[4];
#ifdef FP8_QK_SPLIT
#pragma unroll
          for (int i = 0; i < 4; ++i) acc[i] = {0.f, 0.f, 0.f, 0.f};
#else
          acc[0] = {0.f, 0.f, 0.f, 0.f};
#endif
          const uint8_t* krow = sk + (nt * 16 + row) * kKStride;
#ifdef FP8_STAGE_FRAG
          v2u kf[4];
#pragma unroll
          for (int ks = 0; ks < 4; ++ks) {
            const int d0 = ks * kStep + g * 4;
            kf[ks] = {ld4(krow + d0), ld4(krow + d0 + 16)};
          }
#pragma unroll
          for (int ks = 0; ks < 4; ++ks) {
            v2u va = {aq[ks][0], aq[ks][1]};
            #ifdef FP8_QK_SPLIT
            acc[ks] = mma_f8(kf[ks], va, acc[ks]);
#else
            acc[0] = mma_f8(kf[ks], va, acc[0]);
#endif
          }
#else
#pragma unroll
          for (int ks = 0; ks < 4; ++ks) {
            const int d0 = ks * kStep + g * 4;
            uint32_t k0 = ld4(krow + d0);
            uint32_t k1 = ld4(krow + d0 + 16);
            v2u vb = {k0, k1};
            v2u va = {aq[ks][0], aq[ks][1]};
            #ifdef FP8_QK_SPLIT
            acc[ks] = mma_f8(vb, va, acc[ks]);
#else
            acc[0] = mma_f8(vb, va, acc[0]);
#endif
          }
#endif
#pragma unroll
          for (int r = 0; r < 4; ++r) {
#ifdef FP8_QK_SPLIT
            s[nt][r] = (acc[0][r] + acc[1][r]) + (acc[2][r] + acc[3][r]);
#else
            s[nt][r] = acc[0][r];
#endif
          }
        }

        // ---- online softmax ----
        float m_local = -FLT_MAX;
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
#pragma unroll
          for (int r = 0; r < 4; ++r) {
            if (nt >= nt_count) {
              s[nt][r] = -FLT_MAX;
              continue;
            }
            const int token = tl + nt * 16 + g * 4 + r;
            const bool ok = row_ok && (token < tl + nvalid);
            float sv = ok ? s[nt][r] * sscale : -FLT_MAX;
            s[nt][r] = sv;
            m_local = fmaxf(m_local, sv);
          }
        }
        uint32_t pf[2];
        if (mode & 8) {
          // timing probe: drop the whole softmax math
          pf[0] = 0x38383838u;
          pf[1] = 0x38383838u;
        } else {
          const float m_row = warp_max(m_local);
          const float m_new = fmaxf(m_prev, m_row);
          // Lazy rescale: alpha is 1 whenever the running max did not move,
          // and m_new is warp-uniform, so this is a uniform branch.  It matters
          // a lot here: o_acc lives in the MMA accumulator registers, and an
          // unconditional rescale forces 32 accumulator read-modify-writes per
          // tile (~330 cyc/mma instead of ~22).
          if (m_new != m_prev) {
            const float alpha =
                (m_prev == -FLT_MAX) ? 0.0f : exp2f(m_prev - m_new);
            m_prev = m_new;
#pragma unroll
            for (int i = 0; i < kDt; ++i) {
              o_acc[i][0] *= alpha;
              o_acc[i][1] *= alpha;
              o_acc[i][2] *= alpha;
              o_acc[i][3] *= alpha;
            }
            l_acc *= alpha;
          }

          float lsum = 0.0f;
#pragma unroll
          for (int nt = 0; nt < 2; ++nt) {
            float pv[4];
#pragma unroll
            for (int r = 0; r < 4; ++r) {
              const float ex = (mode & 16) ? 1.5f : exp2f(s[nt][r] - m_new);
              const float e = (s[nt][r] == -FLT_MAX) ? 0.0f : ex * kPscale;
              pv[r] = e;
              lsum += e;
            }
            pf[nt] = (mode & 32) ? 0x38383838u
                                 : pack_f8(pv[0], pv[1], pv[2], pv[3]);
          }
          l_acc += warp_sum(lsum) * (1.0f / kPscale);
        }

        // ---- PV from the transposed smem tile: 4 contiguous bytes ----
        v2u pa = {pf[0], pf[1]};
        if (!(mode & 1)) {
#ifdef FP8_STAGE_FRAG
          v2u vf[kDt];
#pragma unroll
          for (int dt = 0; dt < kDt; ++dt) {
            const int d = dt * 16 + (lane & 15);
            const uint8_t* vrow = svt + d * kVtStride;
            vf[dt] = {ld4(vrow + g * 4),
                      (nt_count == 2) ? ld4(vrow + 16 + g * 4) : 0u};
          }
#pragma unroll
          for (int dt = 0; dt < kDt; ++dt)
            o_acc[dt] = mma_f8(vf[dt], pa, o_acc[dt]);
#else
#pragma unroll
          for (int dt = 0; dt < kDt; ++dt) {
            const int d = dt * 16 + (lane & 15);
            const uint8_t* vrow = svt + d * kVtStride;
            uint32_t b0 = ld4(vrow + g * 4);
            uint32_t b1 = (nt_count == 2) ? ld4(vrow + 16 + g * 4) : 0u;
            v2u vb = {b0, b1};
            o_acc[dt] = mma_f8(vb, pa, o_acc[dt]);
          }
#endif
        }
        __syncthreads();
      }
    }
  }

  const size_t mid_idx =
      (static_cast<size_t>(split) * B * H + static_cast<size_t>(b) * H + qhead);
  if (row_ok) {
    float* op = mid_o + mid_idx * kD;
#pragma unroll
    for (int dt = 0; dt < kDt; ++dt) {
      const int d = dt * 16 + g * 4;
      op[d + 0] = o_acc[dt][0];
      op[d + 1] = o_acc[dt][1];
      op[d + 2] = o_acc[dt][2];
      op[d + 3] = o_acc[dt][3];
    }
    mid_m[mid_idx] = m_prev;
    mid_l[mid_idx] = l_acc;
  }
}

// ---------------------------------------------------------------------------
// Double-buffered software-pipelined split kernel.
//
// The direct kernel alternates load -> mma -> load -> mma inside one tile, so
// with ~8 resident warps/SM (138 regs) each warp exposes two full memory
// latencies per tile and nothing hides them.  Registers cannot be traded for
// occupancy (capping at 96 spills and is 2.4x slower), so the only lever left
// is to overlap: the next tile's K/V are staged into the other smem buffer
// *before* the current tile is computed, so the copies are in flight while the
// MMA/softmax of the current tile runs.  Same pattern as mctlass's
// mma_multistage.h (prefetch into registers, then store to smem), with one
// warp per CTA so the hand-off is a __syncwarp().
// ---------------------------------------------------------------------------
__global__ void fp8_decode_split_kernel_pipe(
    const __maca_bfloat16* __restrict__ q, const uint8_t* __restrict__ k_cache,
    const uint8_t* __restrict__ v_cache, const int* __restrict__ block_table,
    const int* __restrict__ seq_lens, float* __restrict__ mid_o,
    float* __restrict__ mid_m, float* __restrict__ mid_l,
    int B, int H, int HKV, int P, int stride_bt, int num_splits,
    float sm_scale, float k_scale, int mode) {
  __shared__ uint8_t sk[2][32 * kKStride];
  __shared__ uint8_t svt[2][kD * kVtStride];

  const int lane = threadIdx.x & 63;
  const int pid = blockIdx.x;
  const int split = pid % num_splits;
  const int rest = pid / num_splits;
  const int kvh = rest % HKV;
  const int b = rest / HKV;

  const int row = lane & 15;
  const int g = lane >> 4;
  const int group = H / HKV;

  const int seqlen = seq_lens[b];
  const int npages = (seqlen + P - 1) / P;
  const int chunk = (npages + num_splits - 1) / num_splits;
  int p0 = split * chunk;
  int p1 = p0 + chunk;
  if (p1 > npages) p1 = npages;

  const int qhead = kvh * group + row;
  uint32_t aq[4][2];
  float qscale = 1.0f;
  const bool row_ok = (row < group);
  if (row_ok) {
    const __maca_bfloat16* qrow =
        q + (static_cast<size_t>(b) * H + qhead) * kD;
    float amax = 0.0f;
#pragma unroll 4
    for (int d = 0; d < kD; ++d) amax = fmaxf(amax, fabsf(bf2f(qrow[d])));
    float sc = amax * (1.0f / 448.0f);
    qscale = (sc > 1e-8f) ? sc : 1e-8f;
    const float inv = 1.0f / qscale;
#pragma unroll
    for (int ks = 0; ks < 4; ++ks) {
      const int k0 = ks * kStep + g * 4;
      aq[ks][0] = pack_f8(bf2f(qrow[k0 + 0]) * inv, bf2f(qrow[k0 + 1]) * inv,
                          bf2f(qrow[k0 + 2]) * inv, bf2f(qrow[k0 + 3]) * inv);
      const int k1 = k0 + 16;
      aq[ks][1] = pack_f8(bf2f(qrow[k1 + 0]) * inv, bf2f(qrow[k1 + 1]) * inv,
                          bf2f(qrow[k1 + 2]) * inv, bf2f(qrow[k1 + 3]) * inv);
    }
  } else {
#pragma unroll
    for (int ks = 0; ks < 4; ++ks) {
      aq[ks][0] = 0u;
      aq[ks][1] = 0u;
    }
  }

  const float sscale = qscale * k_scale * sm_scale * kLog2e;

  float m_prev = row_ok ? -FLT_MAX : 0.0f;
  float l_acc = 0.0f;
  v4f o_acc[kDt];
#pragma unroll
  for (int i = 0; i < kDt; ++i) o_acc[i] = {0.f, 0.f, 0.f, 0.f};

  const size_t tok_stride = static_cast<size_t>(HKV) * kD;
  const int tstep = (P % kStep == 0) ? kStep : 16;
  const int nt_count = tstep / 16;

  const int stok = lane >> 3;
  const int soff = (lane & 7) * 16;

  // linear tile index across the whole split, so the pipeline can look ahead
  // past a page boundary without special-casing it
  const int tiles_per_page = (P + tstep - 1) / tstep;
  const int ntiles = (p1 - p0) * tiles_per_page;

  // stage tile `t` into buffer `buf` (no barrier: caller syncs)
  auto stage = [&](int t, int buf) {
    const int p = p0 + t / tiles_per_page;
    const int tl = (t % tiles_per_page) * tstep;
    const int page_id = block_table[static_cast<size_t>(b) * stride_bt + p];
    const size_t base = (static_cast<size_t>(page_id) * P) * tok_stride +
                        static_cast<size_t>(kvh) * kD;
    const int nvalid = min(tstep, seqlen - p * P - tl);
    uint8_t* kb = sk[buf];
    uint8_t* vb = svt[buf];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int t2 = j * 8 + stok;
      const bool ok = (t2 < nvalid);
      uint4 kv = make_uint4(0u, 0u, 0u, 0u);
      uint4 vv = make_uint4(0u, 0u, 0u, 0u);
      if (ok) {
        const size_t off = static_cast<size_t>(tl + t2) * tok_stride + soff;
        kv = *reinterpret_cast<const uint4*>(k_cache + base + off);
        vv = *reinterpret_cast<const uint4*>(v_cache + base + off);
      }
      uint32_t* dst = reinterpret_cast<uint32_t*>(kb + t2 * kKStride + soff);
      dst[0] = kv.x; dst[1] = kv.y; dst[2] = kv.z; dst[3] = kv.w;
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        const uint32_t w = (i < 4) ? vv.x : ((i < 8) ? vv.y : ((i < 12) ? vv.z : vv.w));
        vb[(soff + i) * kVtStride + t2] = (uint8_t)(w >> ((i & 3) * 8));
      }
    }
  };

  if (ntiles > 0) {
    stage(0, 0);
    __syncwarp();

    for (int t = 0; t < ntiles; ++t) {
      const int cur = t & 1;
      if (t + 1 < ntiles) stage(t + 1, cur ^ 1);  // in flight during compute

      const int p = p0 + t / tiles_per_page;
      const int tl = (t % tiles_per_page) * tstep;
      const int nvalid = min(tstep, seqlen - p * P - tl);
      const uint8_t* kb = sk[cur];
      const uint8_t* vb = svt[cur];

      float s[2][4];
#pragma unroll
      for (int nt = 0; nt < 2; ++nt) {
        if (nt >= nt_count || (mode & 2)) {
#pragma unroll
          for (int r = 0; r < 4; ++r) s[nt][r] = 0.f;
          continue;
        }
        v4f acc[4];
#ifdef FP8_QK_SPLIT
#pragma unroll
          for (int i = 0; i < 4; ++i) acc[i] = {0.f, 0.f, 0.f, 0.f};
#else
          acc[0] = {0.f, 0.f, 0.f, 0.f};
#endif
        const uint8_t* krow = kb + (nt * 16 + row) * kKStride;
#pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
          const int d0 = ks * kStep + g * 4;
          v2u vb2 = {ld4(krow + d0), ld4(krow + d0 + 16)};
          v2u va = {aq[ks][0], aq[ks][1]};
          #ifdef FP8_QK_SPLIT
            acc[ks] = mma_f8(vb2, va, acc[ks]);
#else
            acc[0] = mma_f8(vb2, va, acc[0]);
#endif
        }
#pragma unroll
        for (int r = 0; r < 4; ++r) {
#ifdef FP8_QK_SPLIT
          s[nt][r] = (acc[0][r] + acc[1][r]) + (acc[2][r] + acc[3][r]);
#else
          s[nt][r] = acc[0][r];
#endif
        }
      }

      float m_local = -FLT_MAX;
#pragma unroll
      for (int nt = 0; nt < 2; ++nt) {
#pragma unroll
        for (int r = 0; r < 4; ++r) {
          if (nt >= nt_count) {
            s[nt][r] = -FLT_MAX;
            continue;
          }
          const int token = tl + nt * 16 + g * 4 + r;
          const bool ok = row_ok && (token < tl + nvalid);
          float sv = ok ? s[nt][r] * sscale : -FLT_MAX;
          s[nt][r] = sv;
          m_local = fmaxf(m_local, sv);
        }
      }
      const float m_row = warp_max(m_local);
      const float m_new = fmaxf(m_prev, m_row);
      uint32_t pf[2];
      if (mode & 8) {
        pf[0] = 0x38383838u;
        pf[1] = 0x38383838u;
      } else {
        if (m_new != m_prev) {
          const float alpha =
              (m_prev == -FLT_MAX) ? 0.0f : exp2f(m_prev - m_new);
          m_prev = m_new;
#pragma unroll
          for (int i = 0; i < kDt; ++i) {
            o_acc[i][0] *= alpha;
            o_acc[i][1] *= alpha;
            o_acc[i][2] *= alpha;
            o_acc[i][3] *= alpha;
          }
          l_acc *= alpha;
        }
        float lsum = 0.0f;
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
          float pv[4];
#pragma unroll
          for (int r = 0; r < 4; ++r) {
            const float e = (s[nt][r] == -FLT_MAX)
                                ? 0.0f
                                : exp2f(s[nt][r] - m_new) * kPscale;
            pv[r] = e;
            lsum += e;
          }
          pf[nt] = pack_f8(pv[0], pv[1], pv[2], pv[3]);
        }
        l_acc += warp_sum(lsum) * (1.0f / kPscale);
      }

      v2u pa = {pf[0], pf[1]};
      if (!(mode & 1)) {
#pragma unroll
        for (int dt = 0; dt < kDt; ++dt) {
          const int d = dt * 16 + (lane & 15);
          const uint8_t* vrow = vb + d * kVtStride;
          uint32_t b0 = ld4(vrow + g * 4);
          uint32_t b1 = (nt_count == 2) ? ld4(vrow + 16 + g * 4) : 0u;
          v2u vb2 = {b0, b1};
          o_acc[dt] = mma_f8(vb2, pa, o_acc[dt]);
        }
      }
      __syncwarp();  // buffer staged at the top of this iteration is now valid
    }
  }

  const size_t mid_idx =
      (static_cast<size_t>(split) * B * H + static_cast<size_t>(b) * H + qhead);
  if (row_ok) {
    float* op = mid_o + mid_idx * kD;
#pragma unroll
    for (int dt = 0; dt < kDt; ++dt) {
      const int d = dt * 16 + g * 4;
      op[d + 0] = o_acc[dt][0];
      op[d + 1] = o_acc[dt][1];
      op[d + 2] = o_acc[dt][2];
      op[d + 3] = o_acc[dt][3];
    }
    mid_m[mid_idx] = m_prev;
    mid_l[mid_idx] = l_acc;
  }
}

// ---------------------------------------------------------------------------
// Register-prefetch, single-buffer smem split kernel.
//
// The double-buffered version above is correct but loses: 17.6 KB/CTA against a
// 64 KB/SM shared budget drops residency to 3 CTAs/SM.  This version keeps the
// SAME smem footprint as the non-pipelined smem kernel (8832 B) and instead
// prefetches the next tile into registers, so the copies are in flight during
// the current tile's compute without paying for a second smem buffer.
//
// Order inside one iteration:
//     fetch(t+1) -> registers      (no smem touched, overlaps with compute)
//     compute(t) from smem
//     __syncwarp()
//     commit()  registers -> smem  (overwrites the buffer just consumed)
//     __syncwarp()
// ---------------------------------------------------------------------------
__global__ void fp8_decode_split_kernel_pipe1(
    const __maca_bfloat16* __restrict__ q, const uint8_t* __restrict__ k_cache,
    const uint8_t* __restrict__ v_cache, const int* __restrict__ block_table,
    const int* __restrict__ seq_lens, float* __restrict__ mid_o,
    float* __restrict__ mid_m, float* __restrict__ mid_l,
    int B, int H, int HKV, int P, int stride_bt, int num_splits,
    float sm_scale, float k_scale, int mode) {
  __shared__ uint8_t sk[32 * kKStride];
  __shared__ uint8_t svt[kD * kVtStride];

  const int lane = threadIdx.x & 63;
  const int pid = blockIdx.x;
  const int split = pid % num_splits;
  const int rest = pid / num_splits;
  const int kvh = rest % HKV;
  const int b = rest / HKV;

  const int row = lane & 15;
  const int g = lane >> 4;
  const int group = H / HKV;

  const int seqlen = seq_lens[b];
  const int npages = (seqlen + P - 1) / P;
  const int chunk = (npages + num_splits - 1) / num_splits;
  int p0 = split * chunk;
  int p1 = p0 + chunk;
  if (p1 > npages) p1 = npages;

  const int qhead = kvh * group + row;
  uint32_t aq[4][2];
  float qscale = 1.0f;
  const bool row_ok = (row < group);
  if (row_ok) {
    const __maca_bfloat16* qrow =
        q + (static_cast<size_t>(b) * H + qhead) * kD;
    float amax = 0.0f;
#pragma unroll 4
    for (int d = 0; d < kD; ++d) amax = fmaxf(amax, fabsf(bf2f(qrow[d])));
    float sc = amax * (1.0f / 448.0f);
    qscale = (sc > 1e-8f) ? sc : 1e-8f;
    const float inv = 1.0f / qscale;
#pragma unroll
    for (int ks = 0; ks < 4; ++ks) {
      const int k0 = ks * kStep + g * 4;
      aq[ks][0] = pack_f8(bf2f(qrow[k0 + 0]) * inv, bf2f(qrow[k0 + 1]) * inv,
                          bf2f(qrow[k0 + 2]) * inv, bf2f(qrow[k0 + 3]) * inv);
      const int k1 = k0 + 16;
      aq[ks][1] = pack_f8(bf2f(qrow[k1 + 0]) * inv, bf2f(qrow[k1 + 1]) * inv,
                          bf2f(qrow[k1 + 2]) * inv, bf2f(qrow[k1 + 3]) * inv);
    }
  } else {
#pragma unroll
    for (int ks = 0; ks < 4; ++ks) {
      aq[ks][0] = 0u;
      aq[ks][1] = 0u;
    }
  }

  const float sscale = qscale * k_scale * sm_scale * kLog2e;

  float m_prev = row_ok ? -FLT_MAX : 0.0f;
  float l_acc = 0.0f;
  v4f o_acc[kDt];
#pragma unroll
  for (int i = 0; i < kDt; ++i) o_acc[i] = {0.f, 0.f, 0.f, 0.f};

  const size_t tok_stride = static_cast<size_t>(HKV) * kD;
  const int tstep = (P % kStep == 0) ? kStep : 16;
  const int nt_count = tstep / 16;
  const int tiles_per_page = (P + tstep - 1) / tstep;
  const int ntiles = (p1 - p0) * tiles_per_page;

  const int stok = lane >> 3;
  const int soff = (lane & 7) * 16;

  uint4 rk[4], rv[4];

  auto fetch = [&](int t) {
    const int p = p0 + t / tiles_per_page;
    const int tl = (t % tiles_per_page) * tstep;
    const int page_id = block_table[static_cast<size_t>(b) * stride_bt + p];
    const size_t base = (static_cast<size_t>(page_id) * P) * tok_stride +
                        static_cast<size_t>(kvh) * kD;
    const int nvalid = min(tstep, seqlen - p * P - tl);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int t2 = j * 8 + stok;
      const bool ok = (t2 < nvalid);
      rk[j] = make_uint4(0u, 0u, 0u, 0u);
      rv[j] = make_uint4(0u, 0u, 0u, 0u);
      if (ok) {
        const size_t off = static_cast<size_t>(tl + t2) * tok_stride + soff;
        rk[j] = *reinterpret_cast<const uint4*>(k_cache + base + off);
        rv[j] = *reinterpret_cast<const uint4*>(v_cache + base + off);
      }
    }
  };

  auto commit = [&] {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int t2 = j * 8 + stok;
      uint32_t* dst = reinterpret_cast<uint32_t*>(sk + t2 * kKStride + soff);
      dst[0] = rk[j].x; dst[1] = rk[j].y; dst[2] = rk[j].z; dst[3] = rk[j].w;
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        const uint32_t w =
            (i < 4) ? rv[j].x : ((i < 8) ? rv[j].y : ((i < 12) ? rv[j].z : rv[j].w));
        svt[(soff + i) * kVtStride + t2] = (uint8_t)(w >> ((i & 3) * 8));
      }
    }
  };

  if (ntiles > 0) {
    fetch(0);
    commit();
    __syncwarp();

    for (int t = 0; t < ntiles; ++t) {
      if (t + 1 < ntiles) fetch(t + 1);  // in flight during the compute below

      const int p = p0 + t / tiles_per_page;
      const int tl = (t % tiles_per_page) * tstep;
      const int nvalid = min(tstep, seqlen - p * P - tl);

      float s[2][4];
#pragma unroll
      for (int nt = 0; nt < 2; ++nt) {
        if (nt >= nt_count || (mode & 2)) {
#pragma unroll
          for (int r = 0; r < 4; ++r) s[nt][r] = 0.f;
          continue;
        }
        v4f acc[4];
#ifdef FP8_QK_SPLIT
#pragma unroll
          for (int i = 0; i < 4; ++i) acc[i] = {0.f, 0.f, 0.f, 0.f};
#else
          acc[0] = {0.f, 0.f, 0.f, 0.f};
#endif
        const uint8_t* krow = sk + (nt * 16 + row) * kKStride;
#ifdef FP8_STAGE_FRAG
        // Issue all four fragment loads first so they are independent and can
        // overlap; interleaving ld4 -> mma serialises a full smem round trip
        // into every MMA (~330 cyc/mma measured, vs ~22 with resident operands).
        v2u kf[4];
#pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
          const int d0 = ks * kStep + g * 4;
          kf[ks] = {ld4(krow + d0), ld4(krow + d0 + 16)};
        }
#pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
          v2u va = {aq[ks][0], aq[ks][1]};
          #ifdef FP8_QK_SPLIT
            acc[ks] = mma_f8(kf[ks], va, acc[ks]);
#else
            acc[0] = mma_f8(kf[ks], va, acc[0]);
#endif
        }
#else
#pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
          const int d0 = ks * kStep + g * 4;
          v2u vb2 = {ld4(krow + d0), ld4(krow + d0 + 16)};
          v2u va = {aq[ks][0], aq[ks][1]};
          #ifdef FP8_QK_SPLIT
            acc[ks] = mma_f8(vb2, va, acc[ks]);
#else
            acc[0] = mma_f8(vb2, va, acc[0]);
#endif
        }
#endif
#pragma unroll
        for (int r = 0; r < 4; ++r) {
#ifdef FP8_QK_SPLIT
          s[nt][r] = (acc[0][r] + acc[1][r]) + (acc[2][r] + acc[3][r]);
#else
          s[nt][r] = acc[0][r];
#endif
        }
      }

      float m_local = -FLT_MAX;
#pragma unroll
      for (int nt = 0; nt < 2; ++nt) {
#pragma unroll
        for (int r = 0; r < 4; ++r) {
          if (nt >= nt_count) {
            s[nt][r] = -FLT_MAX;
            continue;
          }
          const int token = tl + nt * 16 + g * 4 + r;
          const bool ok = row_ok && (token < tl + nvalid);
          float sv = ok ? s[nt][r] * sscale : -FLT_MAX;
          s[nt][r] = sv;
          m_local = fmaxf(m_local, sv);
        }
      }
      const float m_row = warp_max(m_local);
      const float m_new = fmaxf(m_prev, m_row);
      uint32_t pf[2];
      if (mode & 8) {
        pf[0] = 0x38383838u;
        pf[1] = 0x38383838u;
      } else {
        if (m_new != m_prev) {
          const float alpha =
              (m_prev == -FLT_MAX) ? 0.0f : exp2f(m_prev - m_new);
          m_prev = m_new;
#pragma unroll
          for (int i = 0; i < kDt; ++i) {
            o_acc[i][0] *= alpha;
            o_acc[i][1] *= alpha;
            o_acc[i][2] *= alpha;
            o_acc[i][3] *= alpha;
          }
          l_acc *= alpha;
        }
        float lsum = 0.0f;
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
          float pv[4];
#pragma unroll
          for (int r = 0; r < 4; ++r) {
            const float e = (s[nt][r] == -FLT_MAX)
                                ? 0.0f
                                : exp2f(s[nt][r] - m_new) * kPscale;
            pv[r] = e;
            lsum += e;
          }
          pf[nt] = pack_f8(pv[0], pv[1], pv[2], pv[3]);
        }
        l_acc += warp_sum(lsum) * (1.0f / kPscale);
      }

      v2u pa = {pf[0], pf[1]};
      if (!(mode & 1)) {
#ifdef FP8_STAGE_FRAG
        // Same idea for PV: gather all eight V fragments, then fire the MMAs.
        v2u vf[kDt];
#pragma unroll
        for (int dt = 0; dt < kDt; ++dt) {
          const int d = dt * 16 + (lane & 15);
          const uint8_t* vrow = svt + d * kVtStride;
          vf[dt] = {ld4(vrow + g * 4),
                    (nt_count == 2) ? ld4(vrow + 16 + g * 4) : 0u};
        }
#pragma unroll
        for (int dt = 0; dt < kDt; ++dt) o_acc[dt] = mma_f8(vf[dt], pa, o_acc[dt]);
#else
#pragma unroll
        for (int dt = 0; dt < kDt; ++dt) {
          const int d = dt * 16 + (lane & 15);
          const uint8_t* vrow = svt + d * kVtStride;
          uint32_t b0 = ld4(vrow + g * 4);
          uint32_t b1 = (nt_count == 2) ? ld4(vrow + 16 + g * 4) : 0u;
          v2u vb2 = {b0, b1};
          o_acc[dt] = mma_f8(vb2, pa, o_acc[dt]);
        }
#endif
      }

      if (t + 1 < ntiles) {
        __syncwarp();  // compute above must finish before the buffer is reused
        commit();
        __syncwarp();
      }
    }
  }

  const size_t mid_idx =
      (static_cast<size_t>(split) * B * H + static_cast<size_t>(b) * H + qhead);
  if (row_ok) {
    float* op = mid_o + mid_idx * kD;
#pragma unroll
    for (int dt = 0; dt < kDt; ++dt) {
      const int d = dt * 16 + g * 4;
      op[d + 0] = o_acc[dt][0];
      op[d + 1] = o_acc[dt][1];
      op[d + 2] = o_acc[dt][2];
      op[d + 3] = o_acc[dt][3];
    }
    mid_m[mid_idx] = m_prev;
    mid_l[mid_idx] = l_acc;
  }
}

// ---------------------------------------------------------------------------
// Async g2s split kernel: K double-buffered through shared memory with the
// hardware copy engine, V read straight from global.
//
// Why this split (all measured on C500):
//   * The MMA operand path is the dominant cost, and staging a whole tile's
//     fragments into registers before firing the MMAs is worth ~12% on its own.
//   * V benefits from *not* being staged in smem: the smem-transposed PV loop
//     measured 0.84 ms standalone vs 0.57 ms for the direct global byte gather,
//     so only K goes through smem.
//   * K needs no transpose, so it can be moved by __builtin_mxc_ldg_b128_bsm
//     (predicated 16B global->shared) and waited on with __builtin_mxc_arrive,
//     which is the same mechanism mctlass mma_multistage.h and the vendor
//     FlashAttention kernels use.  That removes the 32-register register
//     prefetch buffer (freeing occupancy) and lets the copy overlap compute.
//   * Shared memory stays at 2 x 32 x 132 = 8448 B, so residency is still
//     register-limited rather than shared-memory-limited (the earlier
//     double-buffered-everything attempt needed 17664 B and lost 1.5x).
// ---------------------------------------------------------------------------
__device__ __forceinline__ void g2s16(void* dst_smem, const void* src_global) {
  __builtin_mxc_ldg_b128_bsm(dst_smem, src_global, 0, -1, false, true, false,
                             false);
}

// Wait until every outstanding global->shared copy has landed.
__device__ __forceinline__ void g2s_wait() {
  __builtin_mxc_arrive(64 + 0);  // arrive_gvmcnt(0)
}

// Wait until at most kN copies are still in flight (arrive_gvmcnt(kN)).
// Draining all the way to zero stalls the engine at one stage of depth;
// copy_bw.cu measures one warp at 291 / 625 / 2398 GB/s for a limit of
// 0 / 8 / unbounded outstanding 16-byte global->shared copies.
template <int kN>
__device__ __forceinline__ void g2s_wait_upto() {
  __builtin_mxc_arrive(64 + kN);
}

// Calibrated constraint (see g2s_calib.cu): __builtin_mxc_ldg_b128_bsm requires
// the 64 lanes' shared destinations to be *contiguous and lane-linear* (each
// lane 16B, dst == base + lane*16).  Any padded stride silently lands the data
// somewhere else.  So the async K tile must use the unpadded 128-byte stride,
// which costs bank conflicts on the MMA read (16 lanes hit one bank) -- that is
// the price of using the copy engine here.
constexpr int kKStride16 = 128;

__global__ void fp8_decode_split_kernel_async(
    const __maca_bfloat16* __restrict__ q, const uint8_t* __restrict__ k_cache,
    const uint8_t* __restrict__ v_cache, const int* __restrict__ block_table,
    const int* __restrict__ seq_lens, float* __restrict__ mid_o,
    float* __restrict__ mid_m, float* __restrict__ mid_l,
    int B, int H, int HKV, int P, int stride_bt, int num_splits,
    float sm_scale, float k_scale, int mode) {
  __shared__ uint8_t sk[2][32 * kKStride16];

  const int lane = threadIdx.x & 63;
  const int pid = blockIdx.x;
  const int split = pid % num_splits;
  const int rest = pid / num_splits;
  const int kvh = rest % HKV;
  const int b = rest / HKV;

  const int row = lane & 15;
  const int g = lane >> 4;
  const int group = H / HKV;

  const int seqlen = seq_lens[b];
  const int npages = (seqlen + P - 1) / P;
  const int chunk = (npages + num_splits - 1) / num_splits;
  int p0 = split * chunk;
  int p1 = p0 + chunk;
  if (p1 > npages) p1 = npages;

  const int qhead = kvh * group + row;
  uint32_t aq[4][2];
  float qscale = 1.0f;
  const bool row_ok = (row < group);
  if (row_ok) {
    const __maca_bfloat16* qrow =
        q + (static_cast<size_t>(b) * H + qhead) * kD;
    float amax = 0.0f;
#pragma unroll 4
    for (int d = 0; d < kD; ++d) amax = fmaxf(amax, fabsf(bf2f(qrow[d])));
    float sc = amax * (1.0f / 448.0f);
    qscale = (sc > 1e-8f) ? sc : 1e-8f;
    const float inv = 1.0f / qscale;
#pragma unroll
    for (int ks = 0; ks < 4; ++ks) {
      const int k0 = ks * kStep + g * 4;
      aq[ks][0] = pack_f8(bf2f(qrow[k0 + 0]) * inv, bf2f(qrow[k0 + 1]) * inv,
                          bf2f(qrow[k0 + 2]) * inv, bf2f(qrow[k0 + 3]) * inv);
      const int k1 = k0 + 16;
      aq[ks][1] = pack_f8(bf2f(qrow[k1 + 0]) * inv, bf2f(qrow[k1 + 1]) * inv,
                          bf2f(qrow[k1 + 2]) * inv, bf2f(qrow[k1 + 3]) * inv);
    }
  } else {
#pragma unroll
    for (int ks = 0; ks < 4; ++ks) {
      aq[ks][0] = 0u;
      aq[ks][1] = 0u;
    }
  }

  const float sscale = qscale * k_scale * sm_scale * kLog2e;

  float m_prev = row_ok ? -FLT_MAX : 0.0f;
  float l_acc = 0.0f;
  v4f o_acc[kDt];
#pragma unroll
  for (int i = 0; i < kDt; ++i) o_acc[i] = {0.f, 0.f, 0.f, 0.f};

  const size_t tok_stride = static_cast<size_t>(HKV) * kD;
  const int tstep = (P % kStep == 0) ? kStep : 16;
  const int nt_count = tstep / 16;
  const int tiles_per_page = (P + tstep - 1) / tstep;
  const int ntiles = (p1 - p0) * tiles_per_page;

  // Lane -> copy-engine role.  The engine only requires the *destination* to be
  // lane-linear (dst == base + lane*16); each lane picks its own source.  So we
  // are free to swap the two roles, and doing so spreads the later MMA fragment
  // reads over all 32 banks:
  //   old  (tok = lane>>3, chunk = lane&7):  dst = tok*128 + chunk*16
  //         fragment word = row*32 + ..  -> row*32 % 32 == 0  => 16-way conflict
  //   new  (tok = lane&7, chunk = lane>>3):  element (t,d) lives at
  //         (t/8)*1024 + (d/16)*128 + (t%8)*16 + (d%16)
  //         fragment bank = (row%8)*4 + g, which spans 0..31 => 2-way conflict
  const int stok = lane & 7;          // token within the 8-token copy group
  const int soff = (lane >> 3) * 16;  // d-chunk offset inside the token

  auto tile_base = [&](int t, int& tl, int& nvalid) {
    const int p = p0 + t / tiles_per_page;
    tl = (t % tiles_per_page) * tstep;
    nvalid = min(tstep, seqlen - p * P - tl);
    const int page_id = block_table[static_cast<size_t>(b) * stride_bt + p];
    return (static_cast<size_t>(page_id) * P) * tok_stride +
           static_cast<size_t>(kvh) * kD;
  };

  // Issue the async K copy for tile `t` into buffer `buf`.
  auto stage_k = [&](int t, int buf) {
    int tl, nvalid;
    const size_t base = tile_base(t, tl, nvalid);
    uint8_t* kb = sk[buf];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int t2 = j * 8 + stok;
      // Clamp instead of branching: out-of-range tokens are masked out by the
      // softmax anyway, and clamping keeps the async copy in bounds.
      const int tok = (t2 < nvalid) ? t2 : (nvalid - 1);
      // destination is lane-linear by construction: j*1024 + lane*16
      g2s16(kb + j * 1024 + lane * 16,
            k_cache + base + static_cast<size_t>(tl + tok) * tok_stride + soff);
    }
  };

  if (ntiles > 0) {
    stage_k(0, 0);
    g2s_wait();

    for (int t = 0; t < ntiles; ++t) {
      const int cur = t & 1;
      if (t + 1 < ntiles) stage_k(t + 1, cur ^ 1);  // in flight during compute

      int tl, nvalid;
      const size_t base = tile_base(t, tl, nvalid);
      (void)base;

      // ---- QK: stage all four K fragments, then fire the MMAs ----
      float s[2][4];
#pragma unroll
      for (int nt = 0; nt < 2; ++nt) {
        if (nt >= nt_count || (mode & 2)) {
#pragma unroll
          for (int r = 0; r < 4; ++r) s[nt][r] = 0.f;
          continue;
        }
        v4f acc[4];
#ifdef FP8_QK_SPLIT
#pragma unroll
        for (int i = 0; i < 4; ++i) acc[i] = {0.f, 0.f, 0.f, 0.f};
#else
        acc[0] = {0.f, 0.f, 0.f, 0.f};
#endif
        // Swizzled layout: element (t,d) lives at
        //   (t/8)*1024 + (d/16)*128 + (t%8)*16 + (d%16)
        // For the fragment at d0 = ks*32 + g*4 (d0%16 == g*4, d0/16 == 2ks):
        //   w0 = (t/8)*1024 + (t%8)*16 + g*4   ->  + ks*256
        //   w1 (d0+16)                          ->  + ks*256 + 128
        const int tt = nt * 16 + row;
        const int kw = (tt >> 3) * 1024 + (tt & 7) * 16 + g * 4;
        const uint8_t* krow = sk[cur] + kw;
        v2u kf[4];
#pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
#ifdef FP8_PROBE_CONST_FRAG
          // Timing probe: keep the async copy and every dependency, drop only
          // the shared-memory fragment reads.
          kf[ks] = {0x38383838u + (uint32_t)(kw & 0xffu), 0x38383838u};
#else
          kf[ks] = {ld4(krow + ks * 256), ld4(krow + ks * 256 + 128)};
#endif
        }
#pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
          v2u va = {aq[ks][0], aq[ks][1]};
#ifdef FP8_QK_SPLIT
          acc[ks] = mma_f8(kf[ks], va, acc[ks]);
#else
          acc[0] = mma_f8(kf[ks], va, acc[0]);
#endif
        }
#pragma unroll
        for (int r = 0; r < 4; ++r) {
#ifdef FP8_QK_SPLIT
          s[nt][r] = (acc[0][r] + acc[1][r]) + (acc[2][r] + acc[3][r]);
#else
          s[nt][r] = acc[0][r];
#endif
        }
      }

      // ---- online softmax ----
      float m_local = -FLT_MAX;
#pragma unroll
      for (int nt = 0; nt < 2; ++nt) {
#pragma unroll
        for (int r = 0; r < 4; ++r) {
          if (nt >= nt_count) {
            s[nt][r] = -FLT_MAX;
            continue;
          }
          const int token = tl + nt * 16 + g * 4 + r;
          const bool ok = row_ok && (token < tl + nvalid);
          float sv = ok ? s[nt][r] * sscale : -FLT_MAX;
          s[nt][r] = sv;
          m_local = fmaxf(m_local, sv);
        }
      }
      const float m_row = warp_max(m_local);
      const float m_new = fmaxf(m_prev, m_row);
      uint32_t pf[2];
      if (mode & 8) {
        pf[0] = 0x38383838u;
        pf[1] = 0x38383838u;
      } else {
        if (m_new != m_prev) {
          const float alpha =
              (m_prev == -FLT_MAX) ? 0.0f : exp2f(m_prev - m_new);
          m_prev = m_new;
#pragma unroll
          for (int i = 0; i < kDt; ++i) {
            o_acc[i][0] *= alpha;
            o_acc[i][1] *= alpha;
            o_acc[i][2] *= alpha;
            o_acc[i][3] *= alpha;
          }
          l_acc *= alpha;
        }
        float lsum = 0.0f;
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
          float pv[4];
#pragma unroll
          for (int r = 0; r < 4; ++r) {
            const float e = (s[nt][r] == -FLT_MAX)
                                ? 0.0f
                                : exp2f(s[nt][r] - m_new) * kPscale;
            pv[r] = e;
            lsum += e;
          }
          pf[nt] = pack_f8(pv[0], pv[1], pv[2], pv[3]);
        }
        l_acc += warp_sum(lsum) * (1.0f / kPscale);
      }

      // ---- PV: V straight from global, fragments staged then MMAs ----
      v2u pa = {pf[0], pf[1]};
      if (!(mode & 1)) {
#ifdef FP8_PROBE_VFIX
        // Timing probe: same instruction count / same dependency structure,
        // but every lane reads the tile's first 16 bytes -> minimal traffic.
        const int vt = 0;
#else
        const int vt = min(tl + g * 4, tl + nvalid - 1);
#endif
        const int vt2 = min(vt + 16, tl + nvalid - 1);
        const uint8_t* vp0 =
            v_cache + base + static_cast<size_t>(vt) * tok_stride;
        const uint8_t* vp1 =
            v_cache + base + static_cast<size_t>(vt2) * tok_stride;
        v2u vf[kDt];
#pragma unroll
        for (int dt = 0; dt < kDt; ++dt) {
          const int dv = dt * 16 + (lane & 15);
          uint32_t b0 = (uint32_t)vp0[dv] |
                        ((uint32_t)vp0[dv + tok_stride] << 8) |
                        ((uint32_t)vp0[dv + 2 * tok_stride] << 16) |
                        ((uint32_t)vp0[dv + 3 * tok_stride] << 24);
          uint32_t b1 = 0u;
          if (nt_count == 2) {
            b1 = (uint32_t)vp1[dv] | ((uint32_t)vp1[dv + tok_stride] << 8) |
                 ((uint32_t)vp1[dv + 2 * tok_stride] << 16) |
                 ((uint32_t)vp1[dv + 3 * tok_stride] << 24);
          }
          vf[dt] = {b0, b1};
        }
#pragma unroll
        for (int dt = 0; dt < kDt; ++dt)
          o_acc[dt] = mma_f8(vf[dt], pa, o_acc[dt]);
      }

      if (t + 1 < ntiles) g2s_wait();  // buffer for t+1 must be complete
    }
  }

  const size_t mid_idx =
      (static_cast<size_t>(split) * B * H + static_cast<size_t>(b) * H + qhead);
  if (row_ok) {
    float* op = mid_o + mid_idx * kD;
#pragma unroll
    for (int dt = 0; dt < kDt; ++dt) {
      const int d = dt * 16 + g * 4;
      op[d + 0] = o_acc[dt][0];
      op[d + 1] = o_acc[dt][1];
      op[d + 2] = o_acc[dt][2];
      op[d + 3] = o_acc[dt][3];
    }
    mid_m[mid_idx] = m_prev;
    mid_l[mid_idx] = l_acc;
  }
}

// ---------------------------------------------------------------------------
// Stage 1 (bf16 path): split-K decode on the bf16 tensor core.
//
// Same swizzled ldg_b128_bsm K pipeline, same direct-from-global V gather and
// the same split-K reduction as the fp8 kernel; only the two MMA cores differ:
//   QK: 4 x m16n16k32f8  ->  8 x m16n16k16bf16   (D = 128)
//   PV: 8 x m16n16k32f8  -> 16 x m16n16k16bf16   (32 tokens -> 2 x 16)
// Q stays bf16, so there is no fp8 range squeeze and no qscale; P stays bf16,
// so only the KV cache quantisation is left.  That makes this path both faster
// and 5.6x more accurate than the fp8 one.
//
// kStages is the K pipeline depth in shared memory.  The wait is
// arrive_gvmcnt(4*(kStages-1)) -- "at most one stage set still in flight" --
// not arrive_gvmcnt(0).  Draining to zero is what pinned the engine at a
// single stage of depth and 583 GB/s; copy_bw.cu shows one warp going 291 ->
// 625 -> 2398 GB/s as the allowed outstanding count goes 0 -> 8 -> unbounded.
// ---------------------------------------------------------------------------
// kStages must be a power of two: the ring buffer index uses a mask.
template <int kStages, int kWPC>
__global__ void fp8_decode_split_kernel_bf16(
    const __maca_bfloat16* __restrict__ q, const uint8_t* __restrict__ k_cache,
    const uint8_t* __restrict__ v_cache, const int* __restrict__ block_table,
    const int* __restrict__ seq_lens, float* __restrict__ mid_o,
    float* __restrict__ mid_m, float* __restrict__ mid_l,
    int B, int H, int HKV, int P, int stride_bt, int num_splits,
    float sm_scale, float k_scale, int mode) {
  __shared__ uint8_t sk[kWPC][kStages][32 * kKStride16];
  __shared__ uint16_t lut[256];

  // Built before the early-return below, because __syncthreads() must be
  // reached by every warp in the CTA.
  build_lut(lut);
  __syncthreads();

  // kWPC independent warps share one CTA.  They touch disjoint global data and
  // disjoint shared-memory K buffers, so no barrier is needed between them;
  // packing them only changes how many warps the SM can hold (measured: no
  // difference at all, so the SM is throughput- not slot-limited).
  const int wid = threadIdx.x >> 6;
  const int lane = threadIdx.x & 63;
  const int pid = (kWPC == 1) ? (int)blockIdx.x : (int)blockIdx.x * kWPC + wid;
  if (pid >= B * HKV * num_splits) return;
  const int split = pid % num_splits;
  const int rest = pid / num_splits;
  const int kvh = rest % HKV;
  const int b = rest / HKV;

  const int row = lane & 15;
  const int g = lane >> 4;
  const int group = H / HKV;

  const int seqlen = seq_lens[b];
  const int npages = (seqlen + P - 1) / P;
  const int chunk = (npages + num_splits - 1) / num_splits;
  int p0 = split * chunk;
  int p1 = p0 + chunk;
  if (p1 > npages) p1 = npages;

  const int qhead = kvh * group + row;
  const bool row_ok = (row < group);

  // Q in bf16, already in operand layout: instruction i covers d in
  // [i*16, i*16+16) and this lane holds that half-warp's four d values.
  v4h aqb[8];
  if (row_ok) {
    const __maca_bfloat16* qrow =
        q + (static_cast<size_t>(b) * H + qhead) * kD;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      __builtin_memcpy(&aqb[i], qrow + i * 16 + g * 4, 8);
    }
  } else {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      uint32_t z = 0u;
      __builtin_memcpy(&aqb[i], &z, 4);
      __builtin_memcpy(((char*)&aqb[i]) + 4, &z, 4);
    }
  }

  const float sscale = k_scale * sm_scale * kLog2e;

  float m_prev = row_ok ? -FLT_MAX : 0.0f;
  float l_acc = 0.0f;
  v4f o_acc[kDt];
#pragma unroll
  for (int i = 0; i < kDt; ++i) o_acc[i] = {0.f, 0.f, 0.f, 0.f};

  const size_t tok_stride = static_cast<size_t>(HKV) * kD;
  const int tstep = (P % kStep == 0) ? kStep : 16;
  const int nt_count = tstep / 16;
  const int tiles_per_page = (P + tstep - 1) / tstep;
  const int ntiles = (p1 - p0) * tiles_per_page;

  // Lane -> copy-engine role.  The engine only requires the *destination* to be
  // lane-linear (dst == base + lane*16); each lane chooses its own source, so
  // the two roles can be swapped, and that swap takes the later MMA fragment
  // read from a 16-way to a 2-way bank conflict:
  //   element (t,d) lives at (t/8)*1024 + (d/16)*128 + (t%8)*16 + (d%16)
  const int stok = lane & 7;
  const int soff = (lane >> 3) * 16;

  auto tile_base = [&](int t, int& tl, int& nvalid) {
    const int p = p0 + t / tiles_per_page;
    tl = (t % tiles_per_page) * tstep;
    nvalid = min(tstep, seqlen - p * P - tl);
    const int page_id = block_table[static_cast<size_t>(b) * stride_bt + p];
    return (static_cast<size_t>(page_id) * P) * tok_stride +
           static_cast<size_t>(kvh) * kD;
  };

  auto stage_k = [&](int t, int buf) {
    int tl, nvalid;
    const size_t base = tile_base(t, tl, nvalid);
    uint8_t* kb = sk[wid][buf];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int t2 = j * 8 + stok;
      // Clamping is safe: out-of-range tokens are masked out by the softmax and
      // it keeps the async copy inside the page.
      const int tok = (t2 < nvalid) ? t2 : (nvalid - 1);
      g2s16(kb + j * 1024 + lane * 16,
            k_cache + base + static_cast<size_t>(tl + tok) * tok_stride + soff);
    }
  };

  if (ntiles > 0) {
    // Prologue: fill kStages-1 buffers before the first tile is consumed.
#pragma unroll
    for (int s = 0; s < kStages - 1; ++s)
      if (s < ntiles) stage_k(s, s & (kStages - 1));
    g2s_wait_upto<4 * (kStages - 1)>();

    for (int t = 0; t < ntiles; ++t) {
      const int cur = t & (kStages - 1);
      // Issue one tile ahead.  The slot it overwrites held tile t-1, whose
      // reads this warp finished in the previous iteration, so reuse is safe.
      if (t + kStages - 1 < ntiles)
        stage_k(t + kStages - 1, (t + kStages - 1) & (kStages - 1));
      // Stage t must have landed: every newer request is still counted, so at
      // most 4*(kStages-1) outstanding copies cannot include it.
      g2s_wait_upto<4 * (kStages - 1)>();

      int tl, nvalid;
      const size_t base = tile_base(t, tl, nvalid);

      // ---- QK: 8 bf16 k-steps per 16-token group ----
      float s[2][4];
#pragma unroll
      for (int nt = 0; nt < 2; ++nt) {
        if (nt >= nt_count || (mode & 2)) {
#pragma unroll
          for (int r = 0; r < 4; ++r) s[nt][r] = 0.f;
          continue;
        }
        // Four independent accumulators stop the 8-deep k-chain from
        // serialising; they are summed once at the end.
        v4f acc[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) acc[i] = {0.f, 0.f, 0.f, 0.f};

        const int tt = nt * 16 + row;
        // Fragment at d0 = ks*16 + g*4 -> one 4-byte word per k-step.
        const int kw = (tt >> 3) * 1024 + (tt & 7) * 16 + g * 4;
        const uint8_t* krow = sk[wid][cur] + kw;
        v4h kf[8];
#pragma unroll
        for (int ks = 0; ks < 8; ++ks) kf[ks] = deq4_lut(lut, ld4(krow + ks * 128));
#pragma unroll
        for (int ks = 0; ks < 8; ++ks)
          acc[ks & 3] = mma_bf16(kf[ks], aqb[ks], acc[ks & 3]);
        s[nt][0] = acc[0][0] + acc[1][0] + acc[2][0] + acc[3][0];
        s[nt][1] = acc[0][1] + acc[1][1] + acc[2][1] + acc[3][1];
        s[nt][2] = acc[0][2] + acc[1][2] + acc[2][2] + acc[3][2];
        s[nt][3] = acc[0][3] + acc[1][3] + acc[2][3] + acc[3][3];
      }

      // ---- online softmax ----
      float m_local = -FLT_MAX;
#pragma unroll
      for (int nt = 0; nt < 2; ++nt) {
#pragma unroll
        for (int r = 0; r < 4; ++r) {
          if (nt >= nt_count) {
            s[nt][r] = -FLT_MAX;
            continue;
          }
          const int token = tl + nt * 16 + g * 4 + r;
          const bool ok = row_ok && (token < tl + nvalid);
          float sv = ok ? s[nt][r] * sscale : -FLT_MAX;
          s[nt][r] = sv;
          m_local = fmaxf(m_local, sv);
        }
      }
      const float m_row = warp_max(m_local);
      const float m_new = fmaxf(m_prev, m_row);
      v4h pb[2];
      if (mode & 8) {
        uint32_t z = 0x3f803f80u;
        __builtin_memcpy(&pb[0], &z, 4);
        __builtin_memcpy(&pb[1], &z, 4);
        __builtin_memcpy(((char*)&pb[0]) + 4, &z, 4);
        __builtin_memcpy(((char*)&pb[1]) + 4, &z, 4);
      } else {
        if (m_new != m_prev) {
          const float alpha =
              (m_prev == -FLT_MAX) ? 0.0f : exp2f(m_prev - m_new);
          m_prev = m_new;
#pragma unroll
          for (int i = 0; i < kDt; ++i) {
            o_acc[i][0] *= alpha;
            o_acc[i][1] *= alpha;
            o_acc[i][2] *= alpha;
            o_acc[i][3] *= alpha;
          }
          l_acc *= alpha;
        }
        float lsum = 0.0f;
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
          float pv[4];
#pragma unroll
          for (int r = 0; r < 4; ++r) {
            const float e =
                (s[nt][r] == -FLT_MAX) ? 0.0f : exp2f(s[nt][r] - m_new);
            // Same convention as the fp8 path: P carries the kPscale factor and
            // lsum accumulates the *scaled* value, so the combine kernel's
            // 1/(kPscale * lsum) recovers the true sum of exp2.
            pv[r] = e * kPscale;
            lsum += pv[r];
          }
          pb[nt] = pack_bf16(pv[0], pv[1], pv[2], pv[3]);
        }
        l_acc += warp_sum(lsum) * (1.0f / kPscale);
      }

      // ---- PV: V byte-gathered straight from global, dequantised in place ----
      if (!(mode & 1)) {
        const int vt = min(tl + g * 4, tl + nvalid - 1);
        const int vt2 = min(vt + 16, tl + nvalid - 1);
        const uint8_t* vp0 =
            v_cache + base + static_cast<size_t>(vt) * tok_stride;
        const uint8_t* vp1 =
            v_cache + base + static_cast<size_t>(vt2) * tok_stride;
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
          if (nt >= nt_count) break;
          const uint8_t* vp = nt ? vp1 : vp0;
          // Two passes of four d-tiles instead of one pass of eight: the same
          // instruction count, but only four V fragments are ever live, which
          // takes ~8 registers off the peak.
#pragma unroll
          for (int half = 0; half < 2; ++half) {
            v4h vb[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
              const int dt = half * 4 + i;
              const int dv = dt * 16 + (lane & 15);
              const uint32_t w0 = (uint32_t)vp[dv] |
                                  ((uint32_t)vp[dv + tok_stride] << 8) |
                                  ((uint32_t)vp[dv + 2 * tok_stride] << 16) |
                                  ((uint32_t)vp[dv + 3 * tok_stride] << 24);
              vb[i] = deq4_lut(lut, w0);
            }
#pragma unroll
            for (int i = 0; i < 4; ++i) {
              const int dt = half * 4 + i;
              o_acc[dt] = mma_bf16(vb[i], pb[nt], o_acc[dt]);
            }
          }
        }
      }
    }
  }

  const size_t mid_idx =
      (static_cast<size_t>(split) * B * H + static_cast<size_t>(b) * H + qhead);
  if (row_ok) {
    float* op = mid_o + mid_idx * kD;
#pragma unroll
    for (int dt = 0; dt < kDt; ++dt) {
      const int d = dt * 16 + g * 4;
      op[d + 0] = o_acc[dt][0];
      op[d + 1] = o_acc[dt][1];
      op[d + 2] = o_acc[dt][2];
      op[d + 3] = o_acc[dt][3];
    }
    mid_m[mid_idx] = m_prev;
    mid_l[mid_idx] = l_acc;
  }
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Stage 1 (bf16 + async V): the same bf16 core, but V goes through the same
// async global->shared engine as K.
//
// Why: a byte gather builds each 32-bit V word from four one-byte global loads
// whose addresses are 256 B apart, so a warp's load touches a single sector and
// the latency sits right on the consumer.  Prefetching V into registers earlier
// in the loop hides the latency but lengthens live ranges (176 -> 192 registers,
// measured 0.803 ms), so instead we let the async engine move it -- the copy
// engine costs no registers and is fully coalesced.
//
// The transpose still happens, but in shared memory, where a byte read has no
// coalescing penalty.  Cost: 16 KB of shared memory (2 x K + 2 x V).
// ---------------------------------------------------------------------------
__global__ void fp8_decode_split_kernel_bf16_av(
    const __maca_bfloat16* __restrict__ q, const uint8_t* __restrict__ k_cache,
    const uint8_t* __restrict__ v_cache, const int* __restrict__ block_table,
    const int* __restrict__ seq_lens, float* __restrict__ mid_o,
    float* __restrict__ mid_m, float* __restrict__ mid_l,
    int B, int H, int HKV, int P, int stride_bt, int num_splits,
    float sm_scale, float k_scale, int mode) {
  __shared__ uint8_t sk[2][32 * kKStride16];
  __shared__ uint8_t sv[2][32 * kKStride16];

  const int lane = threadIdx.x & 63;
  const int pid = blockIdx.x;
  const int split = pid % num_splits;
  const int rest = pid / num_splits;
  const int kvh = rest % HKV;
  const int b = rest / HKV;

  const int row = lane & 15;
  const int g = lane >> 4;
  const int group = H / HKV;

  const int seqlen = seq_lens[b];
  const int npages = (seqlen + P - 1) / P;
  const int chunk = (npages + num_splits - 1) / num_splits;
  int p0 = split * chunk;
  int p1 = p0 + chunk;
  if (p1 > npages) p1 = npages;

  const int qhead = kvh * group + row;
  const bool row_ok = (row < group);

  v4h aqb[8];
  if (row_ok) {
    const __maca_bfloat16* qrow =
        q + (static_cast<size_t>(b) * H + qhead) * kD;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      __builtin_memcpy(&aqb[i], qrow + i * 16 + g * 4, 8);
    }
  } else {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      uint32_t z = 0u;
      __builtin_memcpy(&aqb[i], &z, 4);
      __builtin_memcpy(((char*)&aqb[i]) + 4, &z, 4);
    }
  }

  const float sscale = k_scale * sm_scale * kLog2e;

  float m_prev = row_ok ? -FLT_MAX : 0.0f;
  float l_acc = 0.0f;
  v4f o_acc[kDt];
#pragma unroll
  for (int i = 0; i < kDt; ++i) o_acc[i] = {0.f, 0.f, 0.f, 0.f};

  const size_t tok_stride = static_cast<size_t>(HKV) * kD;
  const int tstep = (P % kStep == 0) ? kStep : 16;
  const int nt_count = tstep / 16;
  const int tiles_per_page = (P + tstep - 1) / tstep;
  const int ntiles = (p1 - p0) * tiles_per_page;

  const int stok = lane & 7;
  const int soff = (lane >> 3) * 16;

  auto tile_base = [&](int t, int& tl, int& nvalid) {
    const int p = p0 + t / tiles_per_page;
    tl = (t % tiles_per_page) * tstep;
    nvalid = min(tstep, seqlen - p * P - tl);
    const int page_id = block_table[static_cast<size_t>(b) * stride_bt + p];
    return (static_cast<size_t>(page_id) * P) * tok_stride +
           static_cast<size_t>(kvh) * kD;
  };

  // Both halves of the tile (K for QK, V for PV) are issued together, so one
  // g2s_wait() covers the pair.
  auto stage = [&](int t, int buf) {
    int tl, nvalid;
    const size_t base = tile_base(t, tl, nvalid);
    uint8_t* kb = sk[buf];
    uint8_t* vb = sv[buf];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int t2 = j * 8 + stok;
      const int tok = (t2 < nvalid) ? t2 : (nvalid - 1);
      const size_t src = base + static_cast<size_t>(tl + tok) * tok_stride + soff;
      const int dst = j * 1024 + lane * 16;
      g2s16(kb + dst, k_cache + src);
      g2s16(vb + dst, v_cache + src);
    }
  };

  if (ntiles > 0) {
    stage(0, 0);
    g2s_wait();

    for (int t = 0; t < ntiles; ++t) {
      const int cur = t & 1;
      if (t + 1 < ntiles) stage(t + 1, cur ^ 1);

      int tl, nvalid;
      const size_t base = tile_base(t, tl, nvalid);
      (void)base;

      float s[2][4];
#pragma unroll
      for (int nt = 0; nt < 2; ++nt) {
        if (nt >= nt_count || (mode & 2)) {
#pragma unroll
          for (int r = 0; r < 4; ++r) s[nt][r] = 0.f;
          continue;
        }
        v4f acc[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) acc[i] = {0.f, 0.f, 0.f, 0.f};

        const int tt = nt * 16 + row;
        const int kw = (tt >> 3) * 1024 + (tt & 7) * 16 + g * 4;
        const uint8_t* krow = sk[cur] + kw;
        v4h kf[8];
#pragma unroll
        for (int ks = 0; ks < 8; ++ks) kf[ks] = deq4(ld4(krow + ks * 128));
#pragma unroll
        for (int ks = 0; ks < 8; ++ks)
          acc[ks & 3] = mma_bf16(kf[ks], aqb[ks], acc[ks & 3]);
        s[nt][0] = acc[0][0] + acc[1][0] + acc[2][0] + acc[3][0];
        s[nt][1] = acc[0][1] + acc[1][1] + acc[2][1] + acc[3][1];
        s[nt][2] = acc[0][2] + acc[1][2] + acc[2][2] + acc[3][2];
        s[nt][3] = acc[0][3] + acc[1][3] + acc[2][3] + acc[3][3];
      }

      float m_local = -FLT_MAX;
#pragma unroll
      for (int nt = 0; nt < 2; ++nt) {
#pragma unroll
        for (int r = 0; r < 4; ++r) {
          if (nt >= nt_count) {
            s[nt][r] = -FLT_MAX;
            continue;
          }
          const int token = tl + nt * 16 + g * 4 + r;
          const bool ok = row_ok && (token < tl + nvalid);
          float sv_ = ok ? s[nt][r] * sscale : -FLT_MAX;
          s[nt][r] = sv_;
          m_local = fmaxf(m_local, sv_);
        }
      }
      const float m_row = warp_max(m_local);
      const float m_new = fmaxf(m_prev, m_row);
      v4h pb[2];
      if (mode & 8) {
        uint32_t z = 0x3f803f80u;
        __builtin_memcpy(&pb[0], &z, 4);
        __builtin_memcpy(&pb[1], &z, 4);
        __builtin_memcpy(((char*)&pb[0]) + 4, &z, 4);
        __builtin_memcpy(((char*)&pb[1]) + 4, &z, 4);
      } else {
        if (m_new != m_prev) {
          const float alpha =
              (m_prev == -FLT_MAX) ? 0.0f : exp2f(m_prev - m_new);
          m_prev = m_new;
#pragma unroll
          for (int i = 0; i < kDt; ++i) {
            o_acc[i][0] *= alpha;
            o_acc[i][1] *= alpha;
            o_acc[i][2] *= alpha;
            o_acc[i][3] *= alpha;
          }
          l_acc *= alpha;
        }
        float lsum = 0.0f;
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
          float pv[4];
#pragma unroll
          for (int r = 0; r < 4; ++r) {
            const float e =
                (s[nt][r] == -FLT_MAX) ? 0.0f : exp2f(s[nt][r] - m_new);
            // Same convention as the fp8 path: P carries the kPscale factor
            // and lsum accumulates the *scaled* value, so the combine kernel's
            // 1/(kPscale * lsum) recovers the true sum of exp2.  Scaling P also
            // keeps the products off the low end of the bf16 range.
            pv[r] = e * kPscale;
            lsum += pv[r];
          }
          pb[nt] = pack_bf16(pv[0], pv[1], pv[2], pv[3]);
        }
        l_acc += warp_sum(lsum) * (1.0f / kPscale);
      }

      // ---- PV: transposed read out of the staged V tile ----
      if (!(mode & 1)) {
        const uint8_t* vtile = sv[cur];
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
          if (nt >= nt_count) break;
          v4h vb[kDt];
#pragma unroll
          for (int dt = 0; dt < kDt; ++dt) {
            const int d = dt * 16 + (lane & 15);
            // four consecutive tokens, same d -> 16 B apart in the swizzle
            const int t0 = nt * 16 + g * 4;
            const uint8_t* p0 =
                vtile + (t0 >> 3) * 1024 + (d >> 4) * 128 + (d & 15);
            const int i0 = (t0 & 7) * 16;
            const uint32_t w0 = (uint32_t)p0[i0] |
                                ((uint32_t)p0[i0 + 16] << 8) |
                                ((uint32_t)p0[i0 + 32] << 16) |
                                ((uint32_t)p0[i0 + 48] << 24);
            vb[dt] = deq4(w0);
          }
#pragma unroll
          for (int dt = 0; dt < kDt; ++dt)
            o_acc[dt] = mma_bf16(vb[dt], pb[nt], o_acc[dt]);
        }
      }

      if (t + 1 < ntiles) g2s_wait();
    }
  }

  const size_t mid_idx =
      (static_cast<size_t>(split) * B * H + static_cast<size_t>(b) * H + qhead);
  if (row_ok) {
    float* op = mid_o + mid_idx * kD;
#pragma unroll
    for (int dt = 0; dt < kDt; ++dt) {
      const int d = dt * 16 + g * 4;
      op[d + 0] = o_acc[dt][0];
      op[d + 1] = o_acc[dt][1];
      op[d + 2] = o_acc[dt][2];
      op[d + 3] = o_acc[dt][3];
    }
    mid_m[mid_idx] = m_prev;
    mid_l[mid_idx] = l_acc;
  }
}

// ---------------------------------------------------------------------------
// Stage 2: combine splits -> bf16 output.
// ---------------------------------------------------------------------------
__global__ void fp8_decode_combine_kernel(const float* __restrict__ mid_o,
                                          const float* __restrict__ mid_m,
                                          const float* __restrict__ mid_l,
                                          __maca_bfloat16* __restrict__ out,
                                          int B, int H, int num_splits,
                                          float v_scale) {
  const int bh = blockIdx.x;
  const int b = bh / H;
  const int qh = bh - b * H;
  const int t = threadIdx.x;

  float M = -FLT_MAX;
  for (int s = 0; s < num_splits; ++s)
    M = fmaxf(M, mid_m[(static_cast<size_t>(s) * B * H + bh)]);

  float wsum = 0.0f;
  float lsum = 0.0f;
  float acc[2] = {0.f, 0.f};
  for (int s = 0; s < num_splits; ++s) {
    const size_t idx = static_cast<size_t>(s) * B * H + bh;
    const float w = (mid_m[idx] == -FLT_MAX) ? 0.f : exp2f(mid_m[idx] - M);
    wsum += w;
    lsum += w * mid_l[idx];
    const float* op = mid_o + idx * kD;
    const int d = t;
    if (d < kD) acc[0] += w * op[d];
    if (d + 64 < kD) acc[1] += w * op[d + 64];
  }
  const float inv = (lsum > 0.f) ? (v_scale / (kPscale * lsum)) : 0.f;
  __maca_bfloat16* op = out + static_cast<size_t>(bh) * kD;
  if (t < kD) op[t] = __float2bfloat16(acc[0] * inv);
  if (t + 64 < kD) op[t + 64] = __float2bfloat16(acc[1] * inv);
}

}  // namespace

extern "C" int launch_fp8_decode(const void* q, const void* k_cache,
                                 const void* v_cache, const int* block_table,
                                 const int* seq_lens, void* out, float* mid_o,
                                 float* mid_m, float* mid_l, int B, int H,
                                 int HKV, int P, int stride_bt,
                                 int num_splits, float sm_scale,
                                 float k_scale, float v_scale, int mode,
                                 void* stream) {
  if (B <= 0 || H <= 0 || HKV <= 0) return 0;
  mcStream_t s = (mcStream_t)stream;
  dim3 grid(B * HKV * num_splits), block(64);
  const bool do_split = (mode & 1024) == 0, do_comb = (mode & 512) == 0;
  const int kflags = (mode & 11) | (mode & 48);
  if (do_split && (mode & 8192)) {  // bf16 tensor core + async-staged V
    fp8_decode_split_kernel_bf16_av<<<grid, block, 0, s>>>(
        (const __maca_bfloat16*)q, (const uint8_t*)k_cache,
        (const uint8_t*)v_cache, block_table, seq_lens, mid_o, mid_m, mid_l, B,
        H, HKV, P, stride_bt, num_splits, sm_scale, k_scale, kflags);
  } else if (do_split && (mode & 4096)) {  // bf16 tensor-core path
    const int total = B * HKV * num_splits;
    if (mode & 16384) {
      const int wpc = 2;
      fp8_decode_split_kernel_bf16<2, 2>
          <<<(total + wpc - 1) / wpc, 64 * wpc, 0, s>>>(
              (const __maca_bfloat16*)q, (const uint8_t*)k_cache,
              (const uint8_t*)v_cache, block_table, seq_lens, mid_o, mid_m,
              mid_l, B, H, HKV, P, stride_bt, num_splits, sm_scale, k_scale,
              kflags);
    } else if (mode & 32768) {
      const int wpc = 4;
      fp8_decode_split_kernel_bf16<2, 4>
          <<<(total + wpc - 1) / wpc, 64 * wpc, 0, s>>>(
              (const __maca_bfloat16*)q, (const uint8_t*)k_cache,
              (const uint8_t*)v_cache, block_table, seq_lens, mid_o, mid_m,
              mid_l, B, H, HKV, P, stride_bt, num_splits, sm_scale, k_scale,
              kflags);
    } else {
      fp8_decode_split_kernel_bf16<2, 1><<<grid, block, 0, s>>>(
          (const __maca_bfloat16*)q, (const uint8_t*)k_cache,
          (const uint8_t*)v_cache, block_table, seq_lens, mid_o, mid_m, mid_l,
          B, H, HKV, P, stride_bt, num_splits, sm_scale, k_scale, kflags);
    }
  } else if (do_split && (mode & 2048)) {  // legacy direct-from-global kernel
    fp8_decode_split_kernel<<<grid, block, 0, s>>>(
        (const __maca_bfloat16*)q, (const uint8_t*)k_cache,
        (const uint8_t*)v_cache, block_table, seq_lens, mid_o, mid_m, mid_l, B,
        H, HKV, P, stride_bt, num_splits, sm_scale, k_scale, mode);
  } else if (do_split && !(mode & (64 | 4 | 128 | 256))) {
    // Default path: the bf16 tensor core beats the fp8 tensor core here by
    // 2.17x because fp8 MMA operand assembly costs ~94 cycles per instruction
    // against ~16 for bf16 (see mma_cmp.cu).  Kept first so that mode == 0,
    // which is what callers and benchmarks use, picks the fast kernel.
    const int total = B * HKV * num_splits;
    if (mode & 16384) {
      const int wpc = 2;
      fp8_decode_split_kernel_bf16<2, 2>
          <<<(total + wpc - 1) / wpc, 64 * wpc, 0, s>>>(
              (const __maca_bfloat16*)q, (const uint8_t*)k_cache,
              (const uint8_t*)v_cache, block_table, seq_lens, mid_o, mid_m,
              mid_l, B, H, HKV, P, stride_bt, num_splits, sm_scale, k_scale,
              kflags);
    } else if (mode & 32768) {
      const int wpc = 4;
      fp8_decode_split_kernel_bf16<2, 4>
          <<<(total + wpc - 1) / wpc, 64 * wpc, 0, s>>>(
              (const __maca_bfloat16*)q, (const uint8_t*)k_cache,
              (const uint8_t*)v_cache, block_table, seq_lens, mid_o, mid_m,
              mid_l, B, H, HKV, P, stride_bt, num_splits, sm_scale, k_scale,
              kflags);
    } else {
      fp8_decode_split_kernel_bf16<2, 1><<<grid, block, 0, s>>>(
          (const __maca_bfloat16*)q, (const uint8_t*)k_cache,
          (const uint8_t*)v_cache, block_table, seq_lens, mid_o, mid_m, mid_l,
          B, H, HKV, P, stride_bt, num_splits, sm_scale, k_scale, kflags);
    }
  } else if (do_split && (mode & 64)) {
    fp8_decode_split_kernel_async<<<grid, block, 0, s>>>(
        (const __maca_bfloat16*)q, (const uint8_t*)k_cache,
        (const uint8_t*)v_cache, block_table, seq_lens, mid_o, mid_m, mid_l, B,
        H, HKV, P, stride_bt, num_splits, sm_scale, k_scale,
        (mode & 11) | (mode & 48));
  } else if (do_split && (mode & 128)) {
    fp8_decode_split_kernel_pipe1<<<grid, block, 0, s>>>(
        (const __maca_bfloat16*)q, (const uint8_t*)k_cache,
        (const uint8_t*)v_cache, block_table, seq_lens, mid_o, mid_m, mid_l, B,
        H, HKV, P, stride_bt, num_splits, sm_scale, k_scale,
        (mode & 11) | (mode & 48));
  } else if (do_split && (mode & 256)) {
    fp8_decode_split_kernel_pipe<<<grid, block, 0, s>>>(
        (const __maca_bfloat16*)q, (const uint8_t*)k_cache,
        (const uint8_t*)v_cache, block_table, seq_lens, mid_o, mid_m, mid_l, B,
        H, HKV, P, stride_bt, num_splits, sm_scale, k_scale,
        (mode & 11) | (mode & 48));
  } else if (do_split && (mode & 4)) {
    fp8_decode_split_kernel_smem<<<grid, block, 0, s>>>(
        (const __maca_bfloat16*)q, (const uint8_t*)k_cache,
        (const uint8_t*)v_cache, block_table, seq_lens, mid_o, mid_m, mid_l, B,
        H, HKV, P, stride_bt, num_splits, sm_scale, k_scale,
        (mode & 11) | (mode & 48));
  } else if (do_split) {
    fp8_decode_split_kernel<<<grid, block, 0, s>>>(
        (const __maca_bfloat16*)q, (const uint8_t*)k_cache,
        (const uint8_t*)v_cache, block_table, seq_lens, mid_o, mid_m, mid_l, B,
        H, HKV, P, stride_bt, num_splits, sm_scale, k_scale, mode);
  }
  if (do_comb)
    fp8_decode_combine_kernel<<<B * H, 64, 0, s>>>(mid_o, mid_m, mid_l,
                                                 (__maca_bfloat16*)out, B, H,
                                                 num_splits, v_scale);
  return 0;
}
