// How fast can we stage a tile into smem with ORDINARY 16B loads?
//
// The cp.async route (__builtin_mxc_ldg_b128_bsm) measured 1019 GB/s in bw_stage.cu
// but that test never checked the bytes, and a correctness harness shows the
// copies are not reliably complete under any wait protocol I tried
// (arrive_gvmcnt(0)+barrier_inst, __builtin_mxc_arrive(64), +__syncthreads).
// So before investing more there, measure the ordinary-load ceiling properly:
//
//   MODE 0  one barrier for the whole tile (barrier cost amortised away)
//   MODE 1  one barrier per KC chunk (what the real kernel does)
//   MODE 2  no barrier at all, results unused (pure issue-rate upper bound)
//
// and sweep warps/block, since 4 warps/SM hides almost no memory latency and
// occupancy may be the real reason staging sat at ~150 GB/s.

#include <mc_runtime.h>
#include <cstdint>

typedef __NATIVE_VECTOR__(4, uint32_t) v4u;

// 16B global load into registers. The headers use this (mcflashinfer load_128b,
// mctlass maca_ldg128) rather than a memcpy, which may lower to 4 scalar loads
// and would explain staging stalling at ~160 GB/s.
__device__ __forceinline__ v4u ldg128(const void* p) {
  return __builtin_mxc_ldg_b128((void*)p, 0, -1, true, true, false, false);
}

template <int WPB, int KC, int NT, int MODE, int LD = 0>
__global__ void stage3_kernel(const uint8_t* __restrict__ A,
                              const uint8_t* __restrict__ Bn,
                              float* __restrict__ out, int N, int K) {
  constexpr int RS = KC + 16;               // 16B-aligned padded stride
  constexpr int BROWS = WPB * NT * 16;
  constexpr int NTHR = WPB * 64;
  constexpr int ACL = KC / 16;
  constexpr int ACH = 64 * ACL;
  constexpr int BCH = BROWS * ACL;

  __shared__ __align__(16) uint8_t sA[64 * RS];
  __shared__ __align__(16) uint8_t sB[BROWS * RS];

  const int tid = threadIdx.x;
  const int n0 = blockIdx.x * (WPB * NT * 16);
  if (n0 >= N) return;
  const int NCH = K / KC;

  for (int ch = 0; ch < NCH; ++ch) {
    const int k0 = ch * KC;
    for (int j = tid; j < ACH; j += NTHR) {
      const int r = j / ACL, c = (j - r * ACL) * 16;
      v4u v;
      if (LD) v = ldg128(A + (size_t)r * K + k0 + c);
      else __builtin_memcpy(&v, A + (size_t)r * K + k0 + c, 16);
      __builtin_memcpy(sA + r * RS + c, &v, 16);
    }
    for (int j = tid; j < BCH; j += NTHR) {
      const int r = j / ACL, c = (j - r * ACL) * 16;
      v4u v;
      if (LD) v = ldg128(Bn + (size_t)(n0 + r) * K + k0 + c);
      else __builtin_memcpy(&v, Bn + (size_t)(n0 + r) * K + k0 + c, 16);
      __builtin_memcpy(sB + r * RS + c, &v, 16);
    }
    if (MODE == 1) __syncthreads();
  }
  if (MODE == 0) __syncthreads();
  if (MODE == 2) return;

  if (tid == 0) out[blockIdx.x] = (float)sA[0] + (float)sB[0];
}

extern "C" int launch_stage3(const uint8_t* A, const uint8_t* Bn, float* out,
                             int N, int K, int cfg, void* stream) {
#define S3(WPB, KC, NT, MODE, LD)                                              \
  do {                                                                         \
    const int per = (WPB) * (NT) * 16;                                         \
    const int tiles = (N + per - 1) / per;                                     \
    stage3_kernel<WPB, KC, NT, MODE, LD>                                       \
        <<<tiles, (WPB) * 64, 0, (mcStream_t)stream>>>(A, Bn, out, N, K);      \
  } while (0)
  switch (cfg) {
    case 0: S3(4, 128, 2, 1, 0); break;   // memcpy 16B, per-chunk barrier
    case 1: S3(4, 128, 2, 1, 1); break;   // ldg128,   per-chunk barrier
    case 2: S3(4, 128, 2, 0, 1); break;   // ldg128,   one barrier
    case 3: S3(8, 128, 2, 0, 1); break;
    case 4: S3(16, 128, 1, 0, 1); break;
    case 5: S3(4, 128, 2, 1, 1); break;
    case 6: S3(4, 256, 2, 0, 1); break;
    case 7: S3(4, 128, 4, 0, 1); break;
    case 8: S3(2, 128, 2, 1, 1); break;
    case 9: S3(4, 128, 2, 1, 1); break;
    case 10: S3(4, 128, 2, 0, 0); break;
    case 11: S3(8, 128, 2, 0, 0); break;
    default: return -1;
  }
  return 0;
}
