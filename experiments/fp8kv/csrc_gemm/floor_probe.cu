// Floor probe for the small decode shapes.
//
// o_proj (M=64, N=2048, K=2048) needs to read 4 MB of fp8 weights and write
// 512 KB of fp32 output. At 1416 GB/s that is 2.8 us. The GEMM measures 38 us --
// 13.5x off. Before spending more on it, this probe establishes what is actually
// achievable by a kernel that does *nothing but the traffic*.
//
// Four rungs, each isolating one cost:
//
//   empty     grid/block configured like the GEMM, immediate return.
//             -> launch + scheduling overhead.
//   flat      every thread reads contiguous 16 B chunks over the whole 4 MB,
//             no tiling, no shared memory. The pure DRAM streaming floor.
//   tiled     reads in exactly the GEMM's pattern: one 128-column block per tile
//             of 256 rows, KC=64 chunk per row, so each row read is 64 B at
//             stride 2048. Same access shape, no shared memory, no MMA.
//             -> how much the strided pattern costs against flat.
//   tiled_w   tiled plus the epilogue write of the 64xN fp32 output.
//             -> what the store adds.
//
// Everything is summed into the output with a cheap op so nothing is dead-code
// eliminated, and every variant writes only a handful of floats.

#include <mc_runtime.h>
#include <cstdint>

typedef __NATIVE_VECTOR__(2, uint32_t) v2u;
typedef __NATIVE_VECTOR__(4, uint32_t) v4u;

__device__ __forceinline__ v4u ldg128(const void* p) {
  return __builtin_mxc_ldg_b128((void*)p, 0, -1, true, true, false, false);
}

// ---------------------------------------------------------------- empty
__global__ void k_empty(int N, int K, float* sink) {
  if (blockIdx.x < 0) sink[0] = 1.f;
}

// ---------------------------------------------------------------- flat
// Contiguous stream over the whole weight buffer.
__global__ void k_flat(const uint8_t* __restrict__ B, int NB, float* sink) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;
  uint32_t acc = 0;
  for (int i = tid; i < NB / 16; i += stride) {
    v4u v = ldg128(B + (size_t)i * 16);
    acc ^= v[0] ^ v[1] ^ v[2] ^ v[3];
  }
  if (acc == 0xDEADBEEFu) sink[blockIdx.x] = 1.f;
}

// ---------------------------------------------------------------- tiled
// Same per-row chunking as the GEMM: NTILE columns out of N, KC bytes out of each
// of the BROWS rows, walking the K axis. KC=64 matches the chosen config.
template <int KC, int NTILE>
__global__ void k_tiled(const uint8_t* __restrict__ B, int N, int K, float* sink) {
  constexpr int BCL = KC / 16;
  constexpr int NTHR = 256;
  const int n0 = blockIdx.x * NTILE;
  const int NCH = K / KC;
  uint32_t acc = 0;
  for (int ch = 0; ch < NCH; ++ch) {
    const int k0 = ch * KC;
    constexpr int CHUNKS = NTILE * BCL;
    for (int j = threadIdx.x; j < CHUNKS; j += NTHR) {
      const int r = j / BCL, c = (j - r * BCL) * 16;
      v4u v = ldg128(B + (size_t)(n0 + r) * K + k0 + c);
      acc ^= v[0] ^ v[1] ^ v[2] ^ v[3];
    }
  }
  if (acc == 0xDEADBEEFu) sink[blockIdx.x] = 1.f;
}

// ---------------------------------------------------------------- tiled_w
// tiled plus the real epilogue: M=64 rows, NTILE columns of fp32.
template <int KC, int NTILE>
__global__ void k_tiled_w(const uint8_t* __restrict__ B, int N, int K,
                          float* __restrict__ C, float* sink) {
  constexpr int BCL = KC / 16;
  constexpr int NTHR = 256;
  const int n0 = blockIdx.x * NTILE;
  const int NCH = K / KC;
  uint32_t acc = 0;
  for (int ch = 0; ch < NCH; ++ch) {
    const int k0 = ch * KC;
    constexpr int CHUNKS = NTILE * BCL;
    for (int j = threadIdx.x; j < CHUNKS; j += NTHR) {
      const int r = j / BCL, c = (j - r * BCL) * 16;
      v4u v = ldg128(B + (size_t)(n0 + r) * K + k0 + c);
      acc ^= v[0] ^ v[1] ^ v[2] ^ v[3];
    }
  }
  // store the tile's 64 x NTILE fp32 slab
  const float f = (float)(acc & 0xFFFF) * 1e-6f;
  for (int i = threadIdx.x; i < 64 * NTILE; i += NTHR) {
    const int r = i / NTILE, c = i - r * NTILE;
    C[(size_t)r * N + n0 + c] = f;
  }
  if (acc == 0xDEADBEEFu) sink[blockIdx.x] = 1.f;
}

// mode: 0 empty, 1 flat, 2 tiled, 3 tiled_w, 4 flat with 2x the blocks
extern "C" int launch_floor(const uint8_t* B, float* C, float* sink,
                            int N, int K, int mode, int nthreads, int nblk,
                            void* stream) {
  constexpr int NTILE = 128;
  const int tiles = N / NTILE;
  switch (mode) {
    case 0:
      k_empty<<<dim3(nblk), nthreads, 0, (mcStream_t)stream>>>(N, K, sink);
      break;
    case 1:
      k_flat<<<dim3(nblk), nthreads, 0, (mcStream_t)stream>>>(
          B, (int)((size_t)N * K), sink);
      break;
    case 2:
      k_tiled<64, NTILE><<<dim3(tiles), 256, 0, (mcStream_t)stream>>>(B, N, K, sink);
      break;
    case 3:
      k_tiled_w<64, NTILE><<<dim3(tiles), 256, 0, (mcStream_t)stream>>>(B, N, K, C, sink);
      break;
    case 4:
      k_flat<<<dim3(nblk), nthreads, 0, (mcStream_t)stream>>>(
          B, (int)((size_t)N * K), sink);
      break;
    default:
      return -1;
  }
  return 0;
}
