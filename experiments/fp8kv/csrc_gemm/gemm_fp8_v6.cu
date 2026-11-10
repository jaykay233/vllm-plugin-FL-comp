// fp8-weight GEMM on C500 -- v6. async staging + fragment-time dequant on B only.
//
// Established by measurement, in order:
//   mma_peak2.cu     fp8 __builtin_mxc_mma_16x16x32f8 peaks at 17.8 TFLOPS while
//                    bf16 16x16x16 hits 149 and fp16 151. The fp8 intrinsic is a
//                    software emulation (see mctlass/arch/maca_fp8_tmp.h), so the
//                    "hardware fp8 MMA" route cannot win. Dead end, closed.
//   diag_split.cu    In the bf16-MMA kernel, 62% of time was global->smem staging
//                    (147 GB/s) and only 8% was MMA.
//   bw_stage.cu      Plain 16B loads reach 158 GB/s; __builtin_mxc_ldg_b128_bsm
//                    (async 16B global->smem, i.e. cp.async) reaches 1019 GB/s on
//                    the same tile geometry. That is the fix.
//   gemm_fp8_v4.cu   Fragment-time dequant was right in principle, but it did it
//                    for A too. Every warp reads the whole A tile, so A got
//                    converted nwarps times for nothing.
//
// v6 therefore:
//   * A stays bf16 in gmem. M=64 makes A tiny, and weight-only fp8 is what decode
//     bandwidth actually needs. cp.async'd straight to smem, never converted.
//   * B stays fp8 in gmem and in smem. Halves smem, so bigger tiles fit.
//   * B is dequantized at fragment read via a 256-entry exact LUT. Each B element
//     is consumed by exactly one warp, so conversions equal unique elements.
//   * One block barrier per k-chunk; the next chunk's cp.async is issued before
//     the MMA so the copy overlaps compute.
//
// Layouts must keep the smem row stride a multiple of 16 for cp.async alignment,
// hence PADA/PADF instead of a byte padding. Padding also decides bank spread on
// the fragment reads, so it is swept rather than derived.
//
// Fragment layout (calibrated in csrc/calib.cu):
//   A : row = lane%16, k = {g*4+0..3};  B : n = lane%16, k = same
//   C : row = lane%16, col = g*4 + reg;  g = lane/16, warpSize = 64

#include <mc_runtime.h>
#include <cstdint>
#include <maca_bfloat16.h>

typedef __NATIVE_VECTOR__(2, uint32_t) v2u;
typedef __NATIVE_VECTOR__(4, uint32_t) v4u;
typedef __NATIVE_VECTOR__(4, float) v4f;

static __device__ __constant__ unsigned short kLut[256] = {
#include "lut_e4m3_bf16.inc"
};

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
  __builtin_mxc_ldg_b128_bsm((v4u*)smem, (const v4u*)gmem, 0, -1, true, true, false, true);
}

template <int WPB, int KC, int NT, int PADA, int PADF>
__global__ void gemm_v6_kernel(const __maca_bfloat16* __restrict__ A,
                               const uint8_t* __restrict__ Bn,
                               float* __restrict__ C,
                               int M, int N, int K) {
  constexpr int RSB = KC * 2 + PADA;        // bf16 A row stride, bytes
  constexpr int RSF = KC + PADF;            // fp8  B row stride, bytes
  static_assert(RSB % 16 == 0, "A smem stride must be 16B aligned for cp.async");
  static_assert(RSF % 16 == 0, "B smem stride must be 16B aligned for cp.async");

  constexpr int BROWS = WPB * NT * 16;
  constexpr int NTHR = WPB * 64;
  constexpr int KSTEPS = KC / 16;

  // 16B chunk counts are driven by KC, never by the padded stride.
  constexpr int ACL = KC * 2 / 16;          // A chunks per row (8 bf16 each)
  constexpr int BCL = KC / 16;              // B chunks per row
  constexpr int ACH = 64 * ACL;
  constexpr int BCH = BROWS * BCL;

  // cp.async needs a 16B-aligned destination; without the attribute the arrays
  // are laid out at arbitrary alignment and the copies land in the wrong place.
  __shared__ __align__(16) uint8_t sAb[2][64 * RSB];
  __shared__ __align__(16) uint8_t sBf[2][BROWS * RSF];
  __shared__ __align__(16) unsigned short lut[256];

  const int tid = threadIdx.x;
  const int warp = tid >> 6;
  const int lane = tid & 63;
  const int row = lane % 16;
  const int g = lane / 16;

  const int n0 = blockIdx.x * (WPB * NT * 16);
  if (n0 >= N) return;

  for (int i = tid; i < 256; i += NTHR) lut[i] = kLut[i];

  v4f acc[NT][4];
#pragma unroll
  for (int t = 0; t < NT; ++t)
#pragma unroll
    for (int m = 0; m < 4; ++m) acc[t][m] = v4f{0.f, 0.f, 0.f, 0.f};

  // prologue: chunk 0 -> buffer 0
  for (int j = tid; j < ACH; j += NTHR) {
    const int r = j / ACL, c = (j - r * ACL) * 16;
    cp_async16(sAb[0] + r * RSB + c, A + (size_t)r * K + c / 2);
  }
  for (int j = tid; j < BCH; j += NTHR) {
    const int r = j / BCL, c = (j - r * BCL) * 16;
    cp_async16(sBf[0] + r * RSF + c, Bn + (size_t)(n0 + r) * K + c);
  }
  __builtin_mxc_arrive_gvmcnt(0);
  __builtin_mxc_barrier_inst();

  const int NCH = K / KC;
  for (int ch = 0; ch < NCH; ++ch) {
    const int cur = ch & 1, nxt = cur ^ 1;

    if (ch + 1 < NCH) {
      const int k0 = (ch + 1) * KC;
      for (int j = tid; j < ACH; j += NTHR) {
        const int r = j / ACL, c = (j - r * ACL) * 16;
        cp_async16(sAb[nxt] + r * RSB + c, A + (size_t)r * K + k0 + c / 2);
      }
      for (int j = tid; j < BCH; j += NTHR) {
        const int r = j / BCL, c = (j - r * BCL) * 16;
        cp_async16(sBf[nxt] + r * RSF + c, Bn + (size_t)(n0 + r) * K + k0 + c);
      }
      __builtin_mxc_arrive_gvmcnt(0);
    }

#pragma unroll
    for (int s = 0; s < KSTEPS; ++s) {
      const int kb = s * 32 + g * 8;         // A: bf16, 16 k = 32 B
      uint64_t q0, q1, q2, q3;
      __builtin_memcpy(&q0, sAb[cur] + (0 + row) * RSB + kb, 8);
      __builtin_memcpy(&q1, sAb[cur] + (16 + row) * RSB + kb, 8);
      __builtin_memcpy(&q2, sAb[cur] + (32 + row) * RSB + kb, 8);
      __builtin_memcpy(&q3, sAb[cur] + (48 + row) * RSB + kb, 8);
      v2u va0 = {(uint32_t)q0, (uint32_t)(q0 >> 32)};
      v2u va1 = {(uint32_t)q1, (uint32_t)(q1 >> 32)};
      v2u va2 = {(uint32_t)q2, (uint32_t)(q2 >> 32)};
      v2u va3 = {(uint32_t)q3, (uint32_t)(q3 >> 32)};

      const int kf = s * 16 + g * 4;         // B: fp8, 16 k = 16 B
#pragma unroll
      for (int t = 0; t < NT; ++t) {
        uint32_t bq;
        __builtin_memcpy(&bq, sBf[cur] + (warp * NT * 16 + t * 16 + row) * RSF + kf, 4);
        v2u vb = {(uint32_t)(lut[bq & 0xFF] | (lut[(bq >> 8) & 0xFF] << 16)),
                  (uint32_t)(lut[(bq >> 16) & 0xFF] | (lut[(bq >> 24) & 0xFF] << 16))};
        acc[t][0] = __builtin_mxc_mma_16x16x16bf16(vb, va0, acc[t][0]);
        acc[t][1] = __builtin_mxc_mma_16x16x16bf16(vb, va1, acc[t][1]);
        acc[t][2] = __builtin_mxc_mma_16x16x16bf16(vb, va2, acc[t][2]);
        acc[t][3] = __builtin_mxc_mma_16x16x16bf16(vb, va3, acc[t][3]);
      }
    }
    __builtin_mxc_barrier_inst();
  }

  const int col = g * 4;
#pragma unroll
  for (int t = 0; t < NT; ++t) {
    const int no = n0 + (warp * NT + t) * 16 + col;
    if (no >= N) continue;
    float* p;
    p = C + (size_t)(0 + row) * N + no;
    p[0] = acc[t][0][0]; p[1] = acc[t][0][1]; p[2] = acc[t][0][2]; p[3] = acc[t][0][3];
    p = C + (size_t)(16 + row) * N + no;
    p[0] = acc[t][1][0]; p[1] = acc[t][1][1]; p[2] = acc[t][1][2]; p[3] = acc[t][1][3];
    p = C + (size_t)(32 + row) * N + no;
    p[0] = acc[t][2][0]; p[1] = acc[t][2][1]; p[2] = acc[t][2][2]; p[3] = acc[t][2][3];
    p = C + (size_t)(48 + row) * N + no;
    p[0] = acc[t][3][0]; p[1] = acc[t][3][1]; p[2] = acc[t][3][2]; p[3] = acc[t][3][3];
  }
}

#define LAUNCH_V6(WPB, KC, NT, PADA, PADF)                                     \
  do {                                                                         \
    const int per = (WPB) * (NT) * 16;                                         \
    const int tiles = (N + per - 1) / per;                                     \
    gemm_v6_kernel<WPB, KC, NT, PADA, PADF>                                    \
        <<<tiles, (WPB) * 64, 0, (mcStream_t)stream>>>(                        \
            (const __maca_bfloat16*)A, Bn, C, M, N, K);                          \
  } while (0)

extern "C" int launch_gemm_v6(const void* A, const uint8_t* Bn, float* C,
                              int M, int N, int K, int cfg, void* stream) {
  switch (cfg) {
    case 0: LAUNCH_V6(4, 64, 2, 32, 16); break;
    case 1: LAUNCH_V6(4, 64, 1, 32, 16); break;
    case 2: LAUNCH_V6(4, 96, 2, 32, 16); break;
    case 3: LAUNCH_V6(4, 32, 2, 16, 16); break;
    case 4: LAUNCH_V6(2, 64, 2, 32, 16); break;
    case 5: LAUNCH_V6(4, 64, 4, 32, 16); break;
    case 6: LAUNCH_V6(8, 64, 1, 32, 16); break;
    case 7: LAUNCH_V6(2, 96, 4, 32, 16); break;
    case 8: LAUNCH_V6(4, 64, 2, 16, 32); break;
    case 9: LAUNCH_V6(4, 64, 2, 48, 32); break;
    default: return -1;
  }
  return 0;
}
