// fp8-weight GEMM -- v17: the two remaining dataflow ideas, side by side.
//
// Both were proposed after v13-16 exhausted the parameter space around 0.524 ms
// (1.26x) on lm_head. They are different questions and are kept separate.
//
// ---------------------------------------------------------------------------
// (A) "Keep A resident" -> realised as a wider N tile
// ---------------------------------------------------------------------------
// A is 64 x 2048 bf16 = 256 KB, so the whole thing cannot be resident in 64 KB of
// shared memory; a block's K loop must stream A whatever happens. The only way a
// block avoids re-reading A once per N tile is to cover more N columns inside one
// K loop, i.e. a wider N tile. So the concrete question is: does a wider tile pay
// for its larger accumulator set?
//
// Correction to the plan I stated earlier: I expected to free registers by
// streaming A and spend them on accumulators. That trade does not exist. At
// KC=32 the A tile is 64*4 = 256 chunks over 256 threads, so NPA = 1 -- the A
// prefetch is already a single vector, there is nothing to free. Wide tiles are
// purely an accumulator-register question: every warp covers all 4 m-fragments
// (M=64 is exactly 4x16), so accumulators cost NPA-independent
//     NT * 4 fragments * 4 floats = 16*NT registers,
// i.e. 64 at NT=4 and 128 at NT=8, against a 154-register budget that already
// produced a 1-block limit. This is the same wall v12 hit, now measured with the
// spill counter attached instead of inferred from a slowdown.
//
// Note the wide-tile configs need KC=32 to fit shared memory: at KC=64 NT=8 the B
// tile alone is 81920 B.
//
// ---------------------------------------------------------------------------
// (B) Persistent kernel
// ---------------------------------------------------------------------------
// 510 tiles over 104 SMs is 4.90 waves, so the last wave runs 94/104 full and
// ~1.9% of the machine idles at the end; launch overhead on a 0.524 ms kernel adds
// roughly 1%. A persistent grid of exactly one block per SM, each looping over a
// strided subset of tiles, removes the wave quantisation entirely. The ceiling for
// this idea is therefore ~3%, which is small but it is the only remaining source of
// loss that is not compute or bandwidth.
//
// A persistent block cannot reuse its staged A between tiles (K restarts for each
// new N tile), so no traffic is saved -- this tests the tail and launch overhead
// only. The final barrier of the last chunk already guarantees every warp has
// finished reading shared memory, so the next tile's prologue needs no extra
// barrier.

#include <mc_runtime.h>
#include <cstdint>
#include <maca_bfloat16.h>

typedef __NATIVE_VECTOR__(2, uint32_t) v2u;
typedef __NATIVE_VECTOR__(4, uint32_t) v4u;
typedef __NATIVE_VECTOR__(4, float) v4f;

__device__ __forceinline__ v4u ldg128(const void* p) {
  return __builtin_mxc_ldg_b128((void*)p, 0, -1, true, true, false, false);
}
__device__ __forceinline__ void st16(void* p, const v4u& v) {
  __builtin_memcpy(p, &v, 16);
}

__device__ __forceinline__ v2u cvt_arith(uint32_t bq) {
  const uint32_t lo = (bq & 0x000000FFu) | ((bq & 0x0000FF00u) << 8);
  const uint32_t hi = ((bq >> 16) & 0x000000FFu) | ((bq >> 8) & 0x00FF0000u);
  const uint32_t r0 = (((lo & 0x007F007Fu) << 4) + 0x3C003C00u) | ((lo & 0x00800080u) << 8);
  const uint32_t r1 = (((hi & 0x007F007Fu) << 4) + 0x3C003C00u) | ((hi & 0x00800080u) << 8);
  return v2u{r0, r1};
}

template <int WPB, int KC, int NT, int PADA, int PADF>
__device__ __forceinline__ void gemm_tiles(const __maca_bfloat16* __restrict__ A,
                                           const uint8_t* __restrict__ Bn,
                                           float* __restrict__ C,
                                           int N, int K, int nspl,
                                           int tile_begin, int tile_end, int tile_step) {
  constexpr int RSB = KC * 2 + PADA;
  constexpr int RSF = KC + PADF;
  static_assert(RSB % 16 == 0 && RSF % 16 == 0, "16B alignment for ldg128 path");

  constexpr int NTILE = WPB * NT * 16;
  constexpr int BROWS = NTILE;
  constexpr int NTHR = WPB * 64;
  constexpr int KSTEPS = KC / 16;
  constexpr int ACL = KC * 2 / 16;
  constexpr int BCL = KC / 16;
  constexpr int ACH = 64 * ACL;
  constexpr int BCH = BROWS * BCL;
  constexpr int NPA = ACH / NTHR;
  constexpr int NPB = BCH / NTHR;
  static_assert(ACH % NTHR == 0 && BCH % NTHR == 0, "tile must divide evenly");
  static_assert(NPA >= 1, "A tile smaller than the block");

  __shared__ __align__(16) uint8_t sAb[2][64 * RSB];
  __shared__ __align__(16) uint8_t sBf[2][BROWS * RSF];

  const int tid = threadIdx.x;
  const int warp = tid >> 6;
  const int lane = tid & 63;
  const int row = lane % 16;
  const int g = lane / 16;

  // kbeg and NCH depend on the split index only, never on which N tile the loop
  // is on, so they are hoisted: computing them inside the loop cost 18 registers
  // (164 vs 146 for the same body without the tile loop) and 1.5% of runtime.
  const int kspan = K / nspl;
  const int kbeg = (nspl > 1 ? blockIdx.y : 0) * kspan;
  const int NCH = kspan / KC;

  for (int tile = tile_begin; tile < tile_end; tile += tile_step) {
    const int n0 = tile * NTILE;
    if (n0 >= N) return;

    v4f acc[NT][4];
#pragma unroll
    for (int t = 0; t < NT; ++t)
#pragma unroll
      for (int m = 0; m < 4; ++m) acc[t][m] = v4f{0.f, 0.f, 0.f, 0.f};

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

    for (int ch = 0; ch < NCH; ++ch) {
      const int cur = ch & 1, nxt = cur ^ 1;

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
        const int kb = s * 32 + g * 8;
        uint64_t q0, q1, q2, q3;
        __builtin_memcpy(&q0, sAb[cur] + (0 + row) * RSB + kb, 8);
        __builtin_memcpy(&q1, sAb[cur] + (16 + row) * RSB + kb, 8);
        __builtin_memcpy(&q2, sAb[cur] + (32 + row) * RSB + kb, 8);
        __builtin_memcpy(&q3, sAb[cur] + (48 + row) * RSB + kb, 8);
        v2u va0 = {(uint32_t)q0, (uint32_t)(q0 >> 32)};
        v2u va1 = {(uint32_t)q1, (uint32_t)(q1 >> 32)};
        v2u va2 = {(uint32_t)q2, (uint32_t)(q2 >> 32)};
        v2u va3 = {(uint32_t)q3, (uint32_t)(q3 >> 32)};

        const int kf = s * 16 + g * 4;
#pragma unroll
        for (int t = 0; t < NT; ++t) {
          uint32_t bq;
          __builtin_memcpy(&bq, sBf[cur] + (warp * NT * 16 + t * 16 + row) * RSF + kf, 4);
          v2u vb = cvt_arith(bq);
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
}

// PERSIST=0: one tile per block, grid.x tiles. PERSIST=1: one block per SM,
// strided tile loop.
template <int WPB, int KC, int NT, int PADA, int PADF, int PERSIST>
__global__ void gemm_v17_kernel(const __maca_bfloat16* __restrict__ A,
                                const uint8_t* __restrict__ Bn,
                                float* __restrict__ C,
                                int N, int K, int nspl) {
  constexpr int NTILE = WPB * NT * 16;
  const int ntiles = (N + NTILE - 1) / NTILE;
  if (PERSIST) {
    const int nb = gridDim.x;
    gemm_tiles<WPB, KC, NT, PADA, PADF>(A, Bn, C, N, K, nspl, blockIdx.x, ntiles, nb);
  } else {
    gemm_tiles<WPB, KC, NT, PADA, PADF>(A, Bn, C, N, K, nspl, blockIdx.x, blockIdx.x + 1, 1);
  }
}

template <int WPB, int KC, int NT, int PADA, int PADF, int PERSIST>
static void launch_tc_v17(const void* A, const uint8_t* Bn, float* C,
                          int M, int N, int K, int spl, void* stream) {
  const int per = WPB * NT * 16;
  const int ntiles = (N + per - 1) / per;
  int sms = 104;
  mcDeviceGetAttribute(&sms, mcDeviceAttributeMultiProcessorCount, 0);
  int gx = ntiles;
  if (PERSIST) gx = (ntiles < sms) ? ntiles : sms;
  gemm_v17_kernel<WPB, KC, NT, PADA, PADF, PERSIST>
      <<<dim3(gx, 1), WPB * 64, 0, (mcStream_t)stream>>>(
          (const __maca_bfloat16*)A, Bn, C, N, K, spl);
}

extern "C" int launch_gemm_v17(const void* A, const uint8_t* Bn, float* C,
                               int M, int N, int K, int cfg, void* stream) {
  const int spl = cfg % 100, tc = cfg / 100;
  // persistent path assumes nspl == 1 (blockIdx.y would otherwise vary the K slice)
  if (spl < 1 || K % spl || (K / spl) % 32) return -2;
  switch (tc) {
    case 20: launch_tc_v17<4, 64, 4, 16, 16, 0>(A, Bn, C, M, N, K, spl, stream); break;
    case 21: launch_tc_v17<4, 64, 4, 16, 16, 1>(A, Bn, C, M, N, K, spl, stream); break;
    case 22: launch_tc_v17<4, 32, 8, 16, 16, 0>(A, Bn, C, M, N, K, spl, stream); break;
    case 23: launch_tc_v17<4, 32, 8, 16, 16, 1>(A, Bn, C, M, N, K, spl, stream); break;
    case 24: launch_tc_v17<4, 32, 6, 16, 16, 0>(A, Bn, C, M, N, K, spl, stream); break;
    case 25: launch_tc_v17<4, 32, 4, 16, 16, 0>(A, Bn, C, M, N, K, spl, stream); break;
    case 26: launch_tc_v17<4, 32, 4, 16, 16, 1>(A, Bn, C, M, N, K, spl, stream); break;
    case 27: launch_tc_v17<2, 64, 2, 16, 16, 1>(A, Bn, C, M, N, K, spl, stream); break;
    case 28: launch_tc_v17<4, 32, 6, 16, 16, 1>(A, Bn, C, M, N, K, spl, stream); break;
    default: return -1;
  }
  return 0;
}

template <int WPB, int KC, int NT, int PADA, int PADF, int PS>
static int attr_v17(int which) {
  mcFuncAttributes fa{};
  if (mcFuncGetAttributes(&fa, (const void*)gemm_v17_kernel<WPB, KC, NT, PADA, PADF, PS>) != 0)
    return -1;
  if (which == 0) return (int)fa.sharedSizeBytes;
  if (which == 1) return fa.numRegs;
  if (which == 2) return (int)fa.localSizeBytes;
  int nb = -1;
  mcOccupancyMaxActiveBlocksPerMultiprocessor(
      &nb, (const void*)gemm_v17_kernel<WPB, KC, NT, PADA, PADF, PS>, WPB * 64, 0);
  return nb;
}

#define P17(TC, WPB, KC, NT, PA, PF, PS) case TC: return attr_v17<WPB, KC, NT, PA, PF, PS>(which);

extern "C" int v17_probe(int tc, int which) {
  switch (tc) {
    P17(20, 4, 64, 4, 16, 16, 0)
    P17(21, 4, 64, 4, 16, 16, 1)
    P17(22, 4, 32, 8, 16, 16, 0)
    P17(23, 4, 32, 8, 16, 16, 1)
    P17(24, 4, 32, 6, 16, 16, 0)
    P17(25, 4, 32, 4, 16, 16, 0)
    P17(26, 4, 32, 4, 16, 16, 1)
    P17(27, 2, 64, 2, 16, 16, 1)
    P17(28, 4, 32, 6, 16, 16, 1)
    default: return -1;
  }
}
