// fp8-weight GEMM -- v16: attack per-thread register pressure in the *stager*.
//
// v15 established two things by measurement:
//   * two resident blocks per SM does not help -- 0.794 ms vs 0.529 ms. Barrier
//     count dominates, and every route to 32 KB adds chunks or barriers.
//   * shared memory was never the binding limit anyway.
//
// What actually binds is registers. At the best config (WPB4 NT4 KC64) the A tile
// is 8192 sixteen-byte chunks over 256 threads, so NPA = 32: every thread holds 32
// v4u (128 registers) of prefetched A before storing any of it. That single buffer
// is larger than the budget for the MMAs' own operands, so it explains both the
// 1-block cap and why MMA issue is starved.
//
// The A tile is the same for every one of the 510 blocks, so it is L2-resident and
// its latency is short. Holding it in registers to hide a latency we do not have is
// a bad trade. STREAM_A interleaves load and store per chunk instead of
// load-all-then-store-all, so the live set is one or two v4u rather than 32. B keeps
// its 4-entry buffer (NPB = 4) because B is the DRAM stream whose latency does need
// covering.
//
// This keeps double buffering and therefore the single barrier per chunk that the
// 1.20x config relies on -- it changes register pressure only.

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

// STREAM_A: 1 = interleave A load/store (low register pressure), 0 = buffer all of
// A in registers first (v13 behaviour, high pressure).
template <int WPB, int KC, int NT, int PADA, int PADF, int STREAM_A>
__global__ void gemm_v16_kernel(const __maca_bfloat16* __restrict__ A,
                                const uint8_t* __restrict__ Bn,
                                float* __restrict__ C,
                                int M, int N, int K, int nspl) {
  constexpr int RSB = KC * 2 + PADA;
  constexpr int RSF = KC + PADF;
  static_assert(RSB % 16 == 0 && RSF % 16 == 0, "16B alignment for ldg128 path");

  constexpr int BROWS = WPB * NT * 16;
  constexpr int NTHR = WPB * 64;
  constexpr int KSTEPS = KC / 16;
  constexpr int ACL = KC * 2 / 16;
  constexpr int BCL = KC / 16;
  constexpr int ACH = 64 * ACL;
  constexpr int BCH = BROWS * BCL;
  constexpr int NPA = ACH / NTHR;
  constexpr int NPB = BCH / NTHR;
  static_assert(ACH % NTHR == 0 && BCH % NTHR == 0, "tile must divide evenly");

  __shared__ __align__(16) uint8_t sAb[2][64 * RSB];
  __shared__ __align__(16) uint8_t sBf[2][BROWS * RSF];

  const int tid = threadIdx.x;
  const int warp = tid >> 6;
  const int lane = tid & 63;
  const int row = lane % 16;
  const int g = lane / 16;

  const int n0 = blockIdx.x * (WPB * NT * 16);
  if (n0 >= N) return;

  const int kspan = K / nspl;
  const int kbeg = blockIdx.y * kspan;
  const int NCH = kspan / KC;

  v4f acc[NT][4];
#pragma unroll
  for (int t = 0; t < NT; ++t)
#pragma unroll
    for (int m = 0; m < 4; ++m) acc[t][m] = v4f{0.f, 0.f, 0.f, 0.f};

  // ---- prologue ----
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
    const bool has_next = (ch + 1 < NCH);
    const int k0 = kbeg + (ch + 1) * KC;

    // Each chunk issues its global loads for the next tile, runs the MMAs on the
    // current tile, then writes the next tile to shared memory.
    uint8_t* adst = sAb[nxt];
    uint8_t* bdst = sBf[nxt];

    // B: buffered in registers (NPB is small) so the DRAM latency is covered
    // across the whole compute phase.
    v4u pb[NPB];
    if (has_next) {
#pragma unroll
      for (int j = 0; j < NPB; ++j) {
        const int idx = tid + j * NTHR;
        const int r = idx / BCL, c = (idx - r * BCL) * 16;
        pb[j] = ldg128(Bn + (size_t)(n0 + r) * K + k0 + c);
      }
    }

    if (STREAM_A) {
      // A comes from L2 and every block reads the same tile, so its latency is
      // short enough to take at the store point. Interleaving keeps the live set
      // at roughly one vector instead of NPA of them.
    } else {
      v4u pa[NPA];
      if (has_next) {
#pragma unroll
        for (int j = 0; j < NPA; ++j) {
          const int idx = tid + j * NTHR;
          const int r = idx / ACL, c = (idx - r * ACL) * 16;
          pa[j] = ldg128(A + (size_t)r * K + k0 + c / 2);
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

      if (has_next) {
#pragma unroll
        for (int j = 0; j < NPA; ++j) {
          const int idx = tid + j * NTHR;
          const int r = idx / ACL, c = (idx - r * ACL) * 16;
          st16(adst + r * RSB + c, pa[j]);
        }
      }
    }

    if (STREAM_A) {
      // Issue the A loads and their stores back to back. The tile is small, shared
      // by every block, and L2-resident.
      if (has_next) {
#pragma unroll
        for (int j = 0; j < NPA; ++j) {
          const int idx = tid + j * NTHR;
          const int r = idx / ACL, c = (idx - r * ACL) * 16;
          st16(adst + r * RSB + c, ldg128(A + (size_t)r * K + k0 + c / 2));
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
    }

    if (has_next) {
#pragma unroll
      for (int j = 0; j < NPB; ++j) {
        const int idx = tid + j * NTHR;
        const int r = idx / BCL, c = (idx - r * BCL) * 16;
        st16(bdst + r * RSF + c, pb[j]);
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

template <int WPB, int KC, int NT, int PADA, int PADF, int SA>
static void launch_tc_v16(const void* A, const uint8_t* Bn, float* C,
                          int M, int N, int K, int spl, void* stream) {
  const int per = WPB * NT * 16;
  const int tiles = (N + per - 1) / per;
  dim3 g(tiles, spl);
  gemm_v16_kernel<WPB, KC, NT, PADA, PADF, SA>
      <<<g, WPB * 64, 0, (mcStream_t)stream>>>(
          (const __maca_bfloat16*)A, Bn, C, M, N, K, spl);
}

extern "C" int launch_gemm_v16(const void* A, const uint8_t* Bn, float* C,
                               int M, int N, int K, int cfg, void* stream) {
  const int spl = cfg % 100, tc = cfg / 100;
  if (spl < 1 || K % spl || (K / spl) % 32) return -2;
  switch (tc) {
    case 20: launch_tc_v16<4, 64, 4, 16, 16, 0>(A, Bn, C, M, N, K, spl, stream); break;
    case 21: launch_tc_v16<4, 64, 4, 16, 16, 1>(A, Bn, C, M, N, K, spl, stream); break;
    case 22: launch_tc_v16<2, 64, 2, 16, 16, 1>(A, Bn, C, M, N, K, spl, stream); break;
    case 23: launch_tc_v16<2, 64, 2, 16, 16, 0>(A, Bn, C, M, N, K, spl, stream); break;
    case 24: launch_tc_v16<4, 64, 2, 16, 16, 1>(A, Bn, C, M, N, K, spl, stream); break;
    case 25: launch_tc_v16<8, 64, 2, 16, 16, 1>(A, Bn, C, M, N, K, spl, stream); break;
    case 26: launch_tc_v16<4, 64, 4, 48, 16, 1>(A, Bn, C, M, N, K, spl, stream); break;
    case 27: launch_tc_v16<2, 64, 4, 16, 16, 1>(A, Bn, C, M, N, K, spl, stream); break;
    // KC=128 so each B row's read is a full 128B line instead of half of one.
    // Shared memory only allows it with the N tile halved: A (64 rows, bf16, two
    // stages) is 34816 B on its own, leaving room for 64 B rows of B.
    case 28: launch_tc_v16<2, 128, 2, 16, 16, 1>(A, Bn, C, M, N, K, spl, stream); break;
    case 29: launch_tc_v16<2, 128, 2, 16, 16, 0>(A, Bn, C, M, N, K, spl, stream); break;
    case 30: launch_tc_v16<2, 64, 2, 16, 16, 1>(A, Bn, C, M, N, K, spl, stream); break;
    default: return -1;
  }
  return 0;
}

template <int WPB, int KC, int NT, int PADA, int PADF, int SA>
static int smem_v16() {
  mcFuncAttributes fa{};
  if (mcFuncGetAttributes(&fa, (const void*)gemm_v16_kernel<WPB, KC, NT, PADA, PADF, SA>) != 0)
    return -1;
  return (int)fa.sharedSizeBytes;
}
template <int WPB, int KC, int NT, int PADA, int PADF, int SA>
static int regs_v16() {
  mcFuncAttributes fa{};
  if (mcFuncGetAttributes(&fa, (const void*)gemm_v16_kernel<WPB, KC, NT, PADA, PADF, SA>) != 0)
    return -1;
  return fa.numRegs;
}
template <int WPB, int KC, int NT, int PADA, int PADF, int SA>
static int spill_v16() {
  mcFuncAttributes fa{};
  if (mcFuncGetAttributes(&fa, (const void*)gemm_v16_kernel<WPB, KC, NT, PADA, PADF, SA>) != 0)
    return -1;
  return (int)fa.localSizeBytes;
}
template <int WPB, int KC, int NT, int PADA, int PADF, int SA>
static int occ_v16() {
  int nb = -1;
  mcOccupancyMaxActiveBlocksPerMultiprocessor(
      &nb, (const void*)gemm_v16_kernel<WPB, KC, NT, PADA, PADF, SA>, WPB * 64, 0);
  return nb;
}

// which: 0 smem, 1 regs, 2 spill bytes, 3 blocks/SM
#define PROBE16(TC, WPB, KC, NT, PA, PF, SA)                                   \
  case TC: return which == 0 ? smem_v16<WPB, KC, NT, PA, PF, SA>()              \
          : which == 1 ? regs_v16<WPB, KC, NT, PA, PF, SA>()                    \
          : which == 2 ? spill_v16<WPB, KC, NT, PA, PF, SA>()                   \
                       : occ_v16<WPB, KC, NT, PA, PF, SA>();

extern "C" int v16_probe(int tc, int which) {
  switch (tc) {
    PROBE16(20, 4, 64, 4, 16, 16, 0)
    PROBE16(21, 4, 64, 4, 16, 16, 1)
    PROBE16(22, 2, 64, 2, 16, 16, 1)
    PROBE16(23, 2, 64, 2, 16, 16, 0)
    PROBE16(24, 4, 64, 2, 16, 16, 1)
    PROBE16(25, 8, 64, 2, 16, 16, 1)
    PROBE16(26, 4, 64, 4, 48, 16, 1)
    PROBE16(27, 2, 64, 4, 16, 16, 1)
    PROBE16(28, 2, 128, 2, 16, 16, 1)
    PROBE16(29, 2, 128, 2, 16, 16, 0)
    PROBE16(30, 2, 64, 2, 16, 16, 1)
    default: return -1;
  }
}
