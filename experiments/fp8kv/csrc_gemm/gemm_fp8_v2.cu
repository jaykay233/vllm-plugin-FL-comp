// fp8 (e4m3) GEMM on MetaX C500 -- v2, parameterised for tuning.
//
// What v1 got wrong (measured in experiments/fp8kv/csrc_gemm/bench_gemm.py):
//   * one __syncthreads() per 32-element k-step -> 64 barriers for K=2048
//   * smem fragments read at stride KC bytes, so consecutive lanes hit the same
//     bank (classic conflict; fixed by padding the leading dimension)
//   * only 4 independent MMA accumulator chains per warp -> the MMA latency is
//     not hidden
//
// v2 keeps the same maths and the calibrated fragment layout:
//     A : row = lane%16, k = {g*4+0..3} (reg0) and {16+g*4+0..3} (reg1)
//     B : n   = lane%16, k = same
//     C : row = lane%16, col = g*4 + reg
//   g = lane/16, warpSize = 64, and the intrinsic is called as (B, A, C).
//
// Tunables, all constexpr so the k-loop is fully unrolled:
//   WPB  warps per block
//   KC   k-elements staged per barrier (must divide K)
//   PAD  smem leading-dimension padding, in elements (0 = unpadded)
//   NT   n-tiles (16 cols each) per warp -> NT*4 independent accumulators
//
// Validated against a dequantized reference. NOTE: inputs MUST be scaled before
// quantization -- the hardware MMA flushes e4m3 subnormals to zero, so weights
// left in physical units (~0.02) lose >50% of their values. See
// dbg_subnormal.py.

#include <mc_runtime.h>
#include <maca_bfloat16.h>
#include <cstdint>

typedef __NATIVE_VECTOR__(2, uint32_t) v2u;
typedef __NATIVE_VECTOR__(4, float) v4f;

__device__ __forceinline__ uint32_t ld4(const uint8_t* p) {
  uint32_t v;
  __builtin_memcpy(&v, p, 4);
  return v;
}

template <int WPB, int KC, int PAD, int NT>
__global__ void gemm_v2_kernel(const uint8_t* __restrict__ A,
                               const uint8_t* __restrict__ Bn,
                               float* __restrict__ C,
                               int M, int N, int K) {
  constexpr int AS = KC + PAD;          // smem row stride
  constexpr int BROWS = WPB * NT * 16;  // rows of B staged per block
  __shared__ uint8_t sA[64 * AS];
  __shared__ uint8_t sB[BROWS * AS];

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

  constexpr int KSTEPS = KC / 32;

  for (int k0 = 0; k0 < K; k0 += KC) {
    // ---- coalesced staging, 8 bytes per thread per chunk ----
    for (int i = tid * 8; i < 64 * KC; i += NTHR * 8) {
      const int r = i / KC;
      const int c = i - r * KC;
      *(uint64_t*)(sA + r * AS + c) =
          *(const uint64_t*)(A + (size_t)r * K + k0 + c);
    }
    for (int i = tid * 8; i < BROWS * KC; i += NTHR * 8) {
      const int r = i / KC;
      const int c = i - r * KC;
      *(uint64_t*)(sB + r * AS + c) =
          *(const uint64_t*)(Bn + (size_t)(n0 + r) * K + k0 + c);
    }
    __syncthreads();

#pragma unroll
    for (int s = 0; s < KSTEPS; ++s) {
      const int kh = s * 32 + g * 4;
      const int kh2 = kh + 16;
      const uint8_t* a0 = sA + (0 + row) * AS;
      const uint8_t* a1 = sA + (16 + row) * AS;
      const uint8_t* a2 = sA + (32 + row) * AS;
      const uint8_t* a3 = sA + (48 + row) * AS;
      v2u va0 = {ld4(a0 + kh), ld4(a0 + kh2)};
      v2u va1 = {ld4(a1 + kh), ld4(a1 + kh2)};
      v2u va2 = {ld4(a2 + kh), ld4(a2 + kh2)};
      v2u va3 = {ld4(a3 + kh), ld4(a3 + kh2)};

#pragma unroll
      for (int t = 0; t < NT; ++t) {
        const uint8_t* bk = sB + (warp * NT * 16 + t * 16 + row) * AS;
        v2u vb = {ld4(bk + kh), ld4(bk + kh2)};
        acc[t][0] = __builtin_mxc_mma_16x16x32f8(vb, va0, acc[t][0]);
        acc[t][1] = __builtin_mxc_mma_16x16x32f8(vb, va1, acc[t][1]);
        acc[t][2] = __builtin_mxc_mma_16x16x32f8(vb, va2, acc[t][2]);
        acc[t][3] = __builtin_mxc_mma_16x16x32f8(vb, va3, acc[t][3]);
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

#define LAUNCH_V2(WPB, KC, PAD, NT)                                            \
  do {                                                                         \
    const int per = (WPB) * (NT) * 16;                                         \
    const int tiles = (N + per - 1) / per;                                     \
    gemm_v2_kernel<WPB, KC, PAD, NT>                                           \
        <<<tiles, (WPB) * 64, 0, (mcStream_t)stream>>>(A, Bn, C, M, N, K);     \
  } while (0)

extern "C" int launch_gemm_v2(const uint8_t* A, const uint8_t* Bn, float* C,
                              int M, int N, int K, int cfg, void* stream) {
  switch (cfg) {
    case 0: LAUNCH_V2(4, 32, 0, 1); break;    // v1-equivalent baseline
    case 1: LAUNCH_V2(4, 64, 0, 1); break;
    case 2: LAUNCH_V2(4, 64, 4, 1); break;    // + padding
    case 3: LAUNCH_V2(4, 128, 4, 1); break;   // + bigger k-chunk
    case 4: LAUNCH_V2(4, 256, 4, 1); break;
    case 5: LAUNCH_V2(4, 128, 4, 2); break;   // + 8 accumulators/warp
    case 6: LAUNCH_V2(8, 128, 4, 1); break;   // wider block
    case 7: LAUNCH_V2(8, 128, 4, 2); break;
    case 8: LAUNCH_V2(2, 128, 4, 1); break;
    case 9: LAUNCH_V2(4, 128, 8, 1); break;
    default: return -1;
  }
  return 0;
}
