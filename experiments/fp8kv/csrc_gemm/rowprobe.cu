// Footprint probe #2: does the GEMM's strided read pattern, on its own, explain
// lm_head staging reaching only 602 GB/s when a contiguous stream gets 1505?
//
// The contiguous floor at lm_head scale (267.4 MB) measures 177.7 us / 1505 GB/s,
// while the real kernel's staging-only pass takes 444 us / 602 GB/s. That is 2.5x,
// far more than the 4 MB-shape penalty predicted, so the loss has to be localised
// before anything is rewritten.
//
// The pattern under test is exactly the GEMM's: a block owns ROWS weight rows and
// reads KC contiguous bytes out of each, walking K. KC therefore controls how long
// each row's read is, and the row stride is always K = 2048.
//
//   KC=64    -> 64 B per row   (the current config)
//   KC=128   -> 128 B per row
//   KC=256   -> 256 B per row
//   KC=512   -> 512 B per row
//   KC=1024  -> 1 KB per row
//   KC=2048  -> the entire row, i.e. contiguous
//
// If the KC=64 row tracks at 600 GB/s while KC=2048 reaches 1500, the read
// granularity is the whole story and the fix is to read longer runs per row. If
// every KC lands near 1500, the pattern is innocent and the 2.5x lives inside the
// GEMM (barriers, the register buffer, MMA issue stalling the copy).

#include <mc_runtime.h>
#include <cstdint>

typedef __NATIVE_VECTOR__(2, uint32_t) v2u;
typedef __NATIVE_VECTOR__(4, uint32_t) v4u;

__device__ __forceinline__ v4u ldg128(const void* p) {
  return __builtin_mxc_ldg_b128((void*)p, 0, -1, true, true, false, false);
}

// ROWS weight rows per block, KC bytes of each per step, NTHR threads.
template <int KC, int ROWS, int NTHR>
__global__ void k_rows(const uint8_t* __restrict__ B, int N, int K,
                       float* __restrict__ sink) {
  constexpr int BCL = KC / 16;          // 16 B chunks per row per step
  constexpr int CHUNKS = ROWS * BCL;    // chunks per step for the block
  const int n0 = blockIdx.x * ROWS;
  const int NCH = K / KC;
  uint32_t acc = 0;
  for (int ch = 0; ch < NCH; ++ch) {
    const int k0 = ch * KC;
#pragma unroll 4
    for (int j = threadIdx.x; j < CHUNKS; j += NTHR) {
      const int r = j / BCL, c = (j - r * BCL) * 16;
      v4u v = ldg128(B + (size_t)(n0 + r) * K + k0 + c);
      acc ^= v[0] ^ v[1] ^ v[2] ^ v[3];
    }
  }
  if (acc == 0xDEADBEEFu) sink[blockIdx.x] = 1.f;
}

#define CASE(KC, ROWS, NTHR)                                                   \
  case (KC / 16):                                                              \
    k_rows<KC, ROWS, NTHR><<<dim3(N / ROWS), NTHR, 0, (mcStream_t)stream>>>(    \
        B, N, K, sink);                                                        \
    break;

// kc16 selects the per-row run length. Block covers 256 rows so the grid stays
// close to the GEMM's 510 tiles at N=130560.
extern "C" int launch_rows(const uint8_t* B, float* sink, int N, int K,
                           int kc16, void* stream) {
  switch (kc16) {
    CASE(64, 256, 256)
    CASE(128, 256, 256)
    CASE(256, 256, 256)
    CASE(512, 256, 256)
    CASE(1024, 256, 256)
    CASE(2048, 256, 256)
    default: return -1;
  }
  return 0;
}

// Second axis: same KC=64 row length, but the block covers fewer rows, so each
// thread walks further along K instead of running out of chunks. Distinguishes
// "short row read" from "too few chunks per thread".
#define CASER(KC, ROWS, NTHR)                                                  \
  case ROWS:                                                                   \
    k_rows<KC, ROWS, NTHR><<<dim3(N / ROWS), NTHR, 0, (mcStream_t)stream>>>(    \
        B, N, K, sink);                                                        \
    break;

extern "C" int launch_rows_kc64(const uint8_t* B, float* sink, int N, int K,
                                int rows, void* stream) {
  switch (rows) {
    CASER(64, 256, 256)
    CASER(64, 128, 256)
    CASER(64, 64, 256)
    CASER(64, 32, 256)
    CASER(64, 16, 64)
    default: return -1;
  }
  return 0;
}
