// fp8-weight GEMM on MetaX C500 -- v3. THE design that can actually win.
//
// Measured peak issue rates on this box (mma_peak2.cu, 416 blocks, saturating):
//
//     bf16 __builtin_mxc_mma_16x16x16bf16      149 TFLOPS
//     f16  __builtin_mxc_mma_16x16x16f16       151 TFLOPS
//     bf16 as 2x16x16 (k=32)                   195 TFLOPS
//     fp8  __builtin_mxc_mma_16x16x32f8         17.8 TFLOPS   <-- 8.4x slower
//
// The fp8 intrinsic is not a hardware path. `/opt/maca/include/mctlass/arch/
// maca_fp8_tmp.h` shows why: it converts fp8->f16 in software and issues two
// 16x16x16f16 MMAs. cute's MACA_16x16x32_F32BF16BF16F32 does the same for bf16.
// So "use the hardware fp8 MMA" is a dead end -- v2 proved it caps at 139 GB/s
// when a M=64 decode GEMM needs 1080 GB/s to match bf16 mcblas.
//
// What DOES work: keep the weight fp8 in DRAM (half the bytes of bf16), convert
// fp8->bf16 on the fly, and run bf16 MMAs at 149-195 TFLOPS. At M=64 the compute
// ceiling becomes 195e12/(2*64) = 1523 GB/s, comfortably above the 1080 GB/s
// needed, so the kernel can be bandwidth bound where it belongs.
//
// Conversion is exact, not approximate: e4m3 has a 3-bit mantissa and bf16 has
// 7, so fp8->bf16 is a bit shuffle (exponent rebias + mantissa shift). Subnormal
// e4m3 is mapped to zero, matching the hardware MMA's own flush behaviour -- with
// proper scaling those are ~0.01% of values (see csrc/dbg_subnormal.py).
//
// Layout (unchanged from v1/v2, calibrated in csrc/calib.cu):
//     A : row = lane%16, k = {g*4+0..3} (lower 4 bf16) ...
//     B : n   = lane%16, k = same
//     C : row = lane%16, col = g*4 + reg
//   g = lane/16, warpSize = 64, intrinsic called as (B, A, C).

#include <mc_runtime.h>
#include <cstdint>

typedef __NATIVE_VECTOR__(2, uint32_t) v2u;
typedef __NATIVE_VECTOR__(4, float) v4f;

__device__ __forceinline__ uint64_t ld8(const uint8_t* p) {
  uint64_t v;
  __builtin_memcpy(&v, p, 8);
  return v;
}

// e4m3 -> bf16, exact for normals. e==0 (zero or subnormal) -> signed zero,
// which is what the MMA does anyway.
__device__ __forceinline__ uint16_t e4m3_to_bf16(uint32_t b) {
  const uint32_t e = (b >> 3) & 0xFu;
  const uint32_t s = (b & 0x80u) << 8;      // sign bit -> bit 15
  if (e == 0u) return (uint16_t)s;
  const uint32_t m = b & 0x7u;
  return (uint16_t)(s | ((e + 120u) << 7) | (m << 4));
}

// Convert 8 packed e4m3 bytes into 8 bf16 (16 bytes) in two uint32 pairs.
__device__ __forceinline__ void cvt8(uint64_t src, v2u& lo4, v2u& hi4) {
  const uint32_t b0 = (uint32_t)(src & 0xFFu);
  const uint32_t b1 = (uint32_t)((src >> 8) & 0xFFu);
  const uint32_t b2 = (uint32_t)((src >> 16) & 0xFFu);
  const uint32_t b3 = (uint32_t)((src >> 24) & 0xFFu);
  const uint32_t b4 = (uint32_t)((src >> 32) & 0xFFu);
  const uint32_t b5 = (uint32_t)((src >> 40) & 0xFFu);
  const uint32_t b6 = (uint32_t)((src >> 48) & 0xFFu);
  const uint32_t b7 = (uint32_t)((src >> 56) & 0xFFu);
  lo4 = {e4m3_to_bf16(b0) | (e4m3_to_bf16(b1) << 16),
         e4m3_to_bf16(b2) | (e4m3_to_bf16(b3) << 16)};
  hi4 = {e4m3_to_bf16(b4) | (e4m3_to_bf16(b5) << 16),
         e4m3_to_bf16(b6) | (e4m3_to_bf16(b7) << 16)};
}

// WPB warps/block, KC k-elements staged per barrier (multiple of 16),
// PADB smem padding in bytes, NT n-tiles per warp.
template <int WPB, int KC, int PADB, int NT>
__global__ void gemm_v3_kernel(const uint8_t* __restrict__ A,
                               const uint8_t* __restrict__ Bn,
                               float* __restrict__ C,
                               int M, int N, int K) {
  constexpr int RS = KC * 2 + PADB;         // bf16 row stride, bytes
  constexpr int BROWS = WPB * NT * 16;
  __shared__ uint8_t sA[64 * RS];
  __shared__ uint8_t sB[BROWS * RS];

  const int tid = threadIdx.x;
  constexpr int NTHR = WPB * 64;
  const int warp = tid >> 6;
  const int lane = tid & 63;
  const int row = lane % 16;
  const int g = lane / 16;

  const int n0 = blockIdx.x * (WPB * NT * 16);
  if (n0 >= N) return;

  v4f acc[NT][4];
#pragma unroll
  for (int t = 0; t < NT; ++t)
#pragma unroll
    for (int m = 0; m < 4; ++m) acc[t][m] = v4f{0.f, 0.f, 0.f, 0.f};

  constexpr int KSTEPS = KC / 16;

  for (int k0 = 0; k0 < K; k0 += KC) {
    // ---- stage: coalesced fp8 read (8 B/thread) -> converted bf16 in smem ----
    for (int i = tid * 8; i < 64 * KC; i += NTHR * 8) {
      const int r = i / KC;
      const int c = i - r * KC;
      v2u lo, hi;
      cvt8(ld8(A + (size_t)r * K + k0 + c), lo, hi);
      uint8_t* d = sA + r * RS + c * 2;
      *(uint32_t*)(d) = lo[0];
      *(uint32_t*)(d + 4) = lo[1];
      *(uint32_t*)(d + 8) = hi[0];
      *(uint32_t*)(d + 12) = hi[1];
    }
    for (int i = tid * 8; i < BROWS * KC; i += NTHR * 8) {
      const int r = i / KC;
      const int c = i - r * KC;
      v2u lo, hi;
      cvt8(ld8(Bn + (size_t)(n0 + r) * K + k0 + c), lo, hi);
      uint8_t* d = sB + r * RS + c * 2;
      *(uint32_t*)(d) = lo[0];
      *(uint32_t*)(d + 4) = lo[1];
      *(uint32_t*)(d + 8) = hi[0];
      *(uint32_t*)(d + 12) = hi[1];
    }
    __syncthreads();

    // ---- bf16 MMA, k=16 per step ----
#pragma unroll
    for (int s = 0; s < KSTEPS; ++s) {
      const int kb = s * 32 + g * 8;        // 32 bytes per 16 k-values
      const uint8_t* a0 = sA + (0 + row) * RS;
      const uint8_t* a1 = sA + (16 + row) * RS;
      const uint8_t* a2 = sA + (32 + row) * RS;
      const uint8_t* a3 = sA + (48 + row) * RS;
      uint64_t q0 = ld8(a0 + kb), q1 = ld8(a1 + kb);
      uint64_t q2 = ld8(a2 + kb), q3 = ld8(a3 + kb);
      v2u va0 = {(uint32_t)q0, (uint32_t)(q0 >> 32)};
      v2u va1 = {(uint32_t)q1, (uint32_t)(q1 >> 32)};
      v2u va2 = {(uint32_t)q2, (uint32_t)(q2 >> 32)};
      v2u va3 = {(uint32_t)q3, (uint32_t)(q3 >> 32)};

#pragma unroll
      for (int t = 0; t < NT; ++t) {
        uint64_t qb = ld8(sB + (warp * NT * 16 + t * 16 + row) * RS + kb);
        v2u vb = {(uint32_t)qb, (uint32_t)(qb >> 32)};
        acc[t][0] = __builtin_mxc_mma_16x16x16bf16(vb, va0, acc[t][0]);
        acc[t][1] = __builtin_mxc_mma_16x16x16bf16(vb, va1, acc[t][1]);
        acc[t][2] = __builtin_mxc_mma_16x16x16bf16(vb, va2, acc[t][2]);
        acc[t][3] = __builtin_mxc_mma_16x16x16bf16(vb, va3, acc[t][3]);
      }
    }
    __syncthreads();
  }

  const int col = g * 4;
#pragma unroll
  for (int t = 0; t < NT; ++t) {
    const int no = n0 + (warp * NT + t) * 16 + col;
    if (no >= N) continue;
    float* p;
    p = C + (size_t)(0 + row) * N + no;
    p[0] = acc[t][0][0]; p[1] = acc[t][0][1];
    p[2] = acc[t][0][2]; p[3] = acc[t][0][3];
    p = C + (size_t)(16 + row) * N + no;
    p[0] = acc[t][1][0]; p[1] = acc[t][1][1];
    p[2] = acc[t][1][2]; p[3] = acc[t][1][3];
    p = C + (size_t)(32 + row) * N + no;
    p[0] = acc[t][2][0]; p[1] = acc[t][2][1];
    p[2] = acc[t][2][2]; p[3] = acc[t][2][3];
    p = C + (size_t)(48 + row) * N + no;
    p[0] = acc[t][3][0]; p[1] = acc[t][3][1];
    p[2] = acc[t][3][2]; p[3] = acc[t][3][3];
  }
}

#define LAUNCH_V3(WPB, KC, PADB, NT)                                           \
  do {                                                                         \
    const int per = (WPB) * (NT) * 16;                                         \
    const int tiles = (N + per - 1) / per;                                     \
    gemm_v3_kernel<WPB, KC, PADB, NT>                                          \
        <<<tiles, (WPB) * 64, 0, (mcStream_t)stream>>>(A, Bn, C, M, N, K);     \
  } while (0)

extern "C" int launch_gemm_v3(const uint8_t* A, const uint8_t* Bn, float* C,
                              int M, int N, int K, int cfg, void* stream) {
  switch (cfg) {
    case 0: LAUNCH_V3(4, 32, 8, 1); break;
    case 1: LAUNCH_V3(4, 64, 8, 1); break;
    case 2: LAUNCH_V3(4, 128, 8, 1); break;
    case 3: LAUNCH_V3(4, 64, 8, 2); break;
    case 4: LAUNCH_V3(4, 128, 8, 2); break;
    case 5: LAUNCH_V3(8, 64, 8, 1); break;
    case 6: LAUNCH_V3(8, 64, 8, 2); break;
    case 7: LAUNCH_V3(2, 64, 8, 1); break;
    case 8: LAUNCH_V3(4, 64, 8, 4); break;
    case 9: LAUNCH_V3(4, 64, 0, 1); break;
    default: return -1;
  }
  return 0;
}
