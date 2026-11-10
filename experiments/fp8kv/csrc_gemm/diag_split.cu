// Split the M=64 fp8 GEMM time into staging vs compute.
//
// v2 (fp8 MMA, 17.8 TFLOPS peak) and v3 (bf16 MMA, 149 TFLOPS peak) both land at
// ~91 GB/s / ~0.276 ms on gate_up (12288x2048). Since the MMA peak differs 8.4x
// and the time does not move, the MMA is not the limiter -- the global->smem
// staging plus the per-chunk barrier is. This kernel proves it by running the
// exact same staging loop with the MMA body compiled out.
//
// MODE 0: staging + barrier only (memory floor)
// MODE 1: one chunk staged, MMA loop repeated (compute only)
// MODE 2: the real thing

#include <mc_runtime.h>
#include <cstdint>

typedef __NATIVE_VECTOR__(2, uint32_t) v2u;
typedef __NATIVE_VECTOR__(4, float) v4f;

__device__ __forceinline__ uint64_t ld8(const uint8_t* p) {
  uint64_t v;
  __builtin_memcpy(&v, p, 8);
  return v;
}

__device__ __forceinline__ uint16_t e4m3_to_bf16(uint32_t b) {
  const uint32_t e = (b >> 3) & 0xFu;
  const uint32_t s = (b & 0x80u) << 8;
  if (e == 0u) return (uint16_t)s;
  const uint32_t m = b & 0x7u;
  return (uint16_t)(s | ((e + 120u) << 7) | (m << 4));
}

__device__ __forceinline__ void cvt8(uint64_t src, v2u& lo4, v2u& hi4) {
  const uint32_t b[8] = {(uint32_t)(src & 0xFFu),      (uint32_t)((src >> 8) & 0xFFu),
                         (uint32_t)((src >> 16) & 0xFFu), (uint32_t)((src >> 24) & 0xFFu),
                         (uint32_t)((src >> 32) & 0xFFu), (uint32_t)((src >> 40) & 0xFFu),
                         (uint32_t)((src >> 48) & 0xFFu), (uint32_t)((src >> 56) & 0xFFu)};
  lo4 = {e4m3_to_bf16(b[0]) | (e4m3_to_bf16(b[1]) << 16),
         e4m3_to_bf16(b[2]) | (e4m3_to_bf16(b[3]) << 16)};
  hi4 = {e4m3_to_bf16(b[4]) | (e4m3_to_bf16(b[5]) << 16),
         e4m3_to_bf16(b[6]) | (e4m3_to_bf16(b[7]) << 16)};
}

template <int WPB, int KC, int PADB, int NT, int MODE>
__global__ void diag_kernel(const uint8_t* __restrict__ A,
                            const uint8_t* __restrict__ Bn,
                            float* __restrict__ C, int M, int N, int K) {
  constexpr int RS = KC * 2 + PADB;
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
  const int kstart = (MODE == 1) ? 0 : 0;
  const int kend = (MODE == 1) ? KC : K;

  for (int k0 = kstart; k0 < kend; k0 += KC) {
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

    if (MODE != 0) {
      constexpr int REPS = (MODE == 1) ? 64 : 1;   // MODE 1: amplify compute
#pragma unroll 1
      for (int rep = 0; rep < REPS; ++rep) {
#pragma unroll
        for (int s = 0; s < KSTEPS; ++s) {
          const int kb = s * 32 + g * 8;
          uint64_t q0 = ld8(sA + (0 + row) * RS + kb);
          uint64_t q1 = ld8(sA + (16 + row) * RS + kb);
          uint64_t q2 = ld8(sA + (32 + row) * RS + kb);
          uint64_t q3 = ld8(sA + (48 + row) * RS + kb);
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
    p[0] = acc[t][0][0]; p[1] = acc[t][0][1]; p[2] = acc[t][0][2]; p[3] = acc[t][0][3];
    p = C + (size_t)(16 + row) * N + no;
    p[0] = acc[t][1][0]; p[1] = acc[t][1][1]; p[2] = acc[t][1][2]; p[3] = acc[t][1][3];
    p = C + (size_t)(32 + row) * N + no;
    p[0] = acc[t][2][0]; p[1] = acc[t][2][1]; p[2] = acc[t][2][2]; p[3] = acc[t][2][3];
    p = C + (size_t)(48 + row) * N + no;
    p[0] = acc[t][3][0]; p[1] = acc[t][3][1]; p[2] = acc[t][3][2]; p[3] = acc[t][3][3];
  }
}

extern "C" int launch_diag(const uint8_t* A, const uint8_t* Bn, float* C,
                           int M, int N, int K, int mode, void* stream) {
  const int per = 4 * 2 * 16;   // WPB=4, NT=2
  const int tiles = (N + per - 1) / per;
#define L(MD)                                                                  \
  diag_kernel<4, 128, 8, 2, MD><<<tiles, 256, 0, (mcStream_t)stream>>>(A, Bn, C, M, N, K)
  if (mode == 0) { L(0); return 0; }
  if (mode == 1) { L(1); return 0; }
  if (mode == 2) { L(2); return 0; }
#undef L
  return -1;
}
