// fp8-weight GEMM on C500 -- v8. v7 plus split-K.
//
// v7 measured, M=64, weight bytes / time, vs bf16 mcblas:
//     lm_head  N=130560       1.02x    (1020 blocks -- enough to fill 104 SMs)
//     gate_up  N=12288        0.52x
//     o_proj   N=2048         0.33x    (16 blocks -- 85% of SMs idle)
//     qkv      N=2560         0.32x
//     down     N=2048 K=6144  0.26x
// The gap tracks block count, not bandwidth, so v8 partitions the K range across
// grid.y and atomically accumulates the partials into C. Output traffic is M*N
// floats, negligible against the weights being streamed.

// fp8-weight GEMM on C500 -- v7. The staging fix, finally identified.
//
// The chain of measurements that led here:
//   mma_peak2.cu   fp8 __builtin_mxc_mma_16x16x32f8 peaks at 17.8 TFLOPS, bf16
//                  16x16x16 at 149, fp16 151. The fp8 MMA is software emulation
//                  (mctlass/arch/maca_fp8_tmp.h), so it cannot win. Route closed.
//   diag_split.cu  62% of the bf16-MMA kernel's time was staging, 8% MMA.
//   stage3.cu      Staging sat at ~168 GB/s with __builtin_memcpy, independent of
//                  occupancy and barriers. Swapping only the load for
//                  __builtin_mxc_ldg_b128 took the SAME loop to 619-832 GB/s.
//                  That was the whole problem: memcpy lowers to scalar loads.
//   gemm_fp8_v4/v6 Fragment-time dequant of B only (not A, which every warp
//                  re-reads) is the right split; v6 just inherited the slow load.
//
// v7 = v6's dataflow with the staging replaced:
//   * A stays bf16 in gmem, cp'd to smem, never dequantized. M=64 makes A tiny and
//     weight-only fp8 is exactly what decode bandwidth needs.
//   * B stays fp8 in gmem AND in smem (halves smem, bigger tiles fit), dequantized
//     at fragment read from a 256-entry exact LUT. Each B element is consumed by
//     one warp, so conversions equal unique elements.
//   * Global loads use __builtin_mxc_ldg_b128 into registers, prefetched for the
//     next k-chunk before the MMA so the copy overlaps compute, then committed to
//     the other smem buffer.
//
// e4m3 -> bf16 is exact for all 256 encodings (lut_e4m3_bf16.inc, built with
// PyTorch); relative error vs a dequantized reference is 1.9e-07.
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

// 16B global load into registers. NOT a memcpy: that lowers to scalar loads and
// costs ~4x (168 vs 619 GB/s measured).
__device__ __forceinline__ v4u ldg128(const void* p) {
  return __builtin_mxc_ldg_b128((void*)p, 0, -1, true, true, false, false);
}
__device__ __forceinline__ void st16(void* p, const v4u& v) {
  __builtin_memcpy(p, &v, 16);
}

template <int WPB, int KC, int NT, int PADA, int PADF>
__global__ void gemm_v7_kernel(const __maca_bfloat16* __restrict__ A,
                               const uint8_t* __restrict__ Bn,
                               float* __restrict__ C,
                               int M, int N, int K, int nspl) {
  constexpr int RSB = KC * 2 + PADA;        // bf16 A row stride, bytes
  constexpr int RSF = KC + PADF;            // fp8  B row stride, bytes
  constexpr int BROWS = WPB * NT * 16;
  constexpr int NTHR = WPB * 64;
  constexpr int KSTEPS = KC / 16;

  constexpr int ACL = KC * 2 / 16;          // A 16B chunks per row (8 bf16)
  constexpr int BCL = KC / 16;              // B 16B chunks per row
  constexpr int ACH = 64 * ACL;
  constexpr int BCH = BROWS * BCL;
  constexpr int NPA = ACH / NTHR;           // prefetch regs, A
  constexpr int NPB = BCH / NTHR;           // prefetch regs, B
  static_assert(ACH % NTHR == 0 && BCH % NTHR == 0, "tile must divide evenly");

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

  // split-K: this block owns K in [kbeg, kend)
  const int kspan = K / nspl;
  const int kbeg = blockIdx.y * kspan;
  const int kend = kbeg + kspan;

  for (int i = tid; i < 256; i += NTHR) lut[i] = kLut[i];

  v4f acc[NT][4];
#pragma unroll
  for (int t = 0; t < NT; ++t)
#pragma unroll
    for (int m = 0; m < 4; ++m) acc[t][m] = v4f{0.f, 0.f, 0.f, 0.f};

  // prologue: first chunk of this split straight into buffer 0
  for (int j = 0; j < NPA; ++j) {
    const int idx = tid + j * NTHR;
    const int r = idx / ACL, c = (idx - r * ACL) * 16;
    st16(sAb[0] + r * RSB + c, ldg128(A + (size_t)r * K + kbeg + c / 2));
  }
  for (int j = 0; j < NPB; ++j) {
    const int idx = tid + j * NTHR;
    const int r = idx / BCL, c = (idx - r * BCL) * 16;
    st16(sBf[0] + r * RSF + c, ldg128(Bn + (size_t)(n0 + r) * K + kbeg + c));
  }
  __syncthreads();

  const int NCH = kspan / KC;
  for (int ch = 0; ch < NCH; ++ch) {
    const int cur = ch & 1, nxt = cur ^ 1;

    // issue next chunk's global loads before the MMA so they overlap compute
    v4u pa[NPA], pb[NPB];
    if (ch + 1 < NCH) {
      const int k0 = kbeg + (ch + 1) * KC;
#pragma unroll
      for (int j = 0; j < NPA; ++j) {
        const int idx = tid + j * NTHR;
        const int r = idx / ACL, c = (idx - r * ACL) * 16;
        pa[j] = ldg128(A + (size_t)r * K + k0 + c / 2);
      }
#pragma unroll
      for (int j = 0; j < NPB; ++j) {
        const int idx = tid + j * NTHR;
        const int r = idx / BCL, c = (idx - r * BCL) * 16;
        pb[j] = ldg128(Bn + (size_t)(n0 + r) * K + k0 + c);
      }
    }

#pragma unroll
    for (int s = 0; s < KSTEPS; ++s) {
      const int kb = s * 32 + g * 8;         // A bf16: 16 k = 32 B
      uint64_t q0, q1, q2, q3;
      __builtin_memcpy(&q0, sAb[cur] + (0 + row) * RSB + kb, 8);
      __builtin_memcpy(&q1, sAb[cur] + (16 + row) * RSB + kb, 8);
      __builtin_memcpy(&q2, sAb[cur] + (32 + row) * RSB + kb, 8);
      __builtin_memcpy(&q3, sAb[cur] + (48 + row) * RSB + kb, 8);
      v2u va0 = {(uint32_t)q0, (uint32_t)(q0 >> 32)};
      v2u va1 = {(uint32_t)q1, (uint32_t)(q1 >> 32)};
      v2u va2 = {(uint32_t)q2, (uint32_t)(q2 >> 32)};
      v2u va3 = {(uint32_t)q3, (uint32_t)(q3 >> 32)};

      const int kf = s * 16 + g * 4;         // B fp8: 16 k = 16 B
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

    if (ch + 1 < NCH) {
#pragma unroll
      for (int j = 0; j < NPA; ++j) {
        const int idx = tid + j * NTHR;
        const int r = idx / ACL, c = (idx - r * ACL) * 16;
        st16(sAb[nxt] + r * RSB + c, pa[j]);
      }
#pragma unroll
      for (int j = 0; j < NPB; ++j) {
        const int idx = tid + j * NTHR;
        const int r = idx / BCL, c = (idx - r * BCL) * 16;
        st16(sBf[nxt] + r * RSF + c, pb[j]);
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
    // Single-split blocks own their output tile outright, so a plain store beats
    // atomicAdd; with real splitting the partials have to be accumulated.
    if (nspl == 1) {
      p = C + (size_t)(0 + row) * N + no;
      p[0] = acc[t][0][0]; p[1] = acc[t][0][1]; p[2] = acc[t][0][2]; p[3] = acc[t][0][3];
      p = C + (size_t)(16 + row) * N + no;
      p[0] = acc[t][1][0]; p[1] = acc[t][1][1]; p[2] = acc[t][1][2]; p[3] = acc[t][1][3];
      p = C + (size_t)(32 + row) * N + no;
      p[0] = acc[t][2][0]; p[1] = acc[t][2][1]; p[2] = acc[t][2][2]; p[3] = acc[t][2][3];
      p = C + (size_t)(48 + row) * N + no;
      p[0] = acc[t][3][0]; p[1] = acc[t][3][1]; p[2] = acc[t][3][2]; p[3] = acc[t][3][3];
    } else {
      p = C + (size_t)(0 + row) * N + no;
      atomicAdd(p + 0, acc[t][0][0]); atomicAdd(p + 1, acc[t][0][1]);
      atomicAdd(p + 2, acc[t][0][2]); atomicAdd(p + 3, acc[t][0][3]);
      p = C + (size_t)(16 + row) * N + no;
      atomicAdd(p + 0, acc[t][1][0]); atomicAdd(p + 1, acc[t][1][1]);
      atomicAdd(p + 2, acc[t][1][2]); atomicAdd(p + 3, acc[t][1][3]);
      p = C + (size_t)(32 + row) * N + no;
      atomicAdd(p + 0, acc[t][2][0]); atomicAdd(p + 1, acc[t][2][1]);
      atomicAdd(p + 2, acc[t][2][2]); atomicAdd(p + 3, acc[t][2][3]);
      p = C + (size_t)(48 + row) * N + no;
      atomicAdd(p + 0, acc[t][3][0]); atomicAdd(p + 1, acc[t][3][1]);
      atomicAdd(p + 2, acc[t][3][2]); atomicAdd(p + 3, acc[t][3][3]);
    }
  }
}

#define LAUNCH_V8(WPB, KC, NT, SPL)                                           \
  do {                                                                         \
    const int per = (WPB) * (NT) * 16;                                         \
    const int tiles = (N + per - 1) / per;                                     \
    dim3 g(tiles, (SPL));                                                      \
    gemm_v7_kernel<WPB, KC, NT, 32, 16>                                        \
        <<<g, (WPB) * 64, 0, (mcStream_t)stream>>>(                            \
            (const __maca_bfloat16*)A, Bn, C, M, N, K, (SPL));                 \
  } while (0)

// cfg = tile config * 100 + splits. Splits must divide K with K/splits a
// multiple of KC, otherwise the split range is not a whole number of chunks.
extern "C" int launch_gemm_v8(const void* A, const uint8_t* Bn, float* C,
                              int M, int N, int K, int cfg, void* stream) {
  const int tc = cfg / 100, spl = cfg % 100;
  if (spl < 1 || K % spl || (K / spl) % 64) return -2;
  switch (tc) {
    case 0: LAUNCH_V8(4, 64, 2, spl); break;
    case 1: LAUNCH_V8(2, 64, 2, spl); break;
    case 2: LAUNCH_V8(4, 64, 1, spl); break;
    case 3: LAUNCH_V8(4, 64, 4, spl); break;
    case 4: LAUNCH_V8(8, 64, 1, spl); break;
    case 5: LAUNCH_V8(2, 64, 4, spl); break;
    default: return -1;
  }
  return 0;
}
