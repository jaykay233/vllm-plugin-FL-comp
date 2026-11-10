// fp8-weight GEMM -- v15: try to get two blocks resident per SM on lm_head.
//
// v13's lm_head config (WPB4 NT4 KC64 PADA16) uses 59 KB of shared memory, so
// exactly one block fits per SM and there is nothing to hide the
// load-fragment -> MMA dependency behind when a barrier drains the block. The
// staging/compute decomposition says staging alone is 0.444 ms and full is
// 0.529 ms; a second resident block is the textbook way to close that.
//
// Shared memory at KC=32 with PADF=16 (the bank-friendly B stride) is
//     2*64*(64+16)  +  2*(4*4*16)*(32+16) = 10240 + 24576 = 34816 B
// which misses the 32 KB line by 2 KB. Two independent ways to get under it:
//
//   AS=1  single-buffer the A tile (A is only 64 rows; B stays double buffered).
//         Costs one extra barrier per chunk. 29696 B.
//   PADF=0  drop B's pad. 26624 B, but B's row stride becomes 32 B = 8 words and
//         gcd(8,32)=8, so a lane group hits only 4 bank sets -- a 4-way conflict
//         on every B fragment read. This is the same defect PADA=32 had on the A
//         side in v13, so it is expected to lose; measured anyway to quantify
//         what the pad is worth.
//
// Both are selectable, and occupancy is queried at runtime through
// mcOccupancyMaxActiveBlocksPerMultiprocessor so the claim "2 blocks/SM" is
// verified rather than assumed.

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

template <int WPB, int KC, int NT, int PADA, int PADF, int AS>
__global__ void gemm_v15_kernel(const __maca_bfloat16* __restrict__ A,
                                const uint8_t* __restrict__ Bn,
                                float* __restrict__ C,
                                int M, int N, int K, int nspl) {
  constexpr int RSB = KC * 2 + PADA;
  constexpr int RSF = KC + PADF;
  static_assert(RSB % 16 == 0 && RSF % 16 == 0, "16B alignment for ldg128 path");
  static_assert(AS == 1 || AS == 2, "A is single or double buffered");

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

  __shared__ __align__(16) uint8_t sAb[AS][64 * RSB];
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
#pragma unroll
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
  if (AS == 2 && NCH > 1) {
#pragma unroll
    for (int j = 0; j < NPA; ++j) {
      const int idx = tid + j * NTHR;
      const int r = idx / ACL, c = (idx - r * ACL) * 16;
      st16(sAb[1] + r * RSB + c, ldg128(A + (size_t)r * K + kbeg + KC + c / 2));
    }
  }
  __syncthreads();

  for (int ch = 0; ch < NCH; ++ch) {
    const int cur = ch & 1, nxt = cur ^ 1;

    v4u pa[NPA], pb[NPB];
    const int nc = ch + 1;
    if (nc < NCH) {
      const int k0 = kbeg + nc * KC;
      // with AS=2 the next A chunk is already in smem; only B is fetched here
      if (AS == 1) {
#pragma unroll
        for (int j = 0; j < NPA; ++j) {
          const int idx = tid + j * NTHR;
          const int r = idx / ACL, c = (idx - r * ACL) * 16;
          pa[j] = ldg128(A + (size_t)r * K + k0 + c / 2);
        }
      }
#pragma unroll
      for (int j = 0; j < NPB; ++j) {
        const int idx = tid + j * NTHR;
        const int r = idx / BCL, c = (idx - r * BCL) * 16;
        pb[j] = ldg128(Bn + (size_t)(n0 + r) * K + k0 + c);
      }
    }

    // AS=2: A chunk ch sits in sAb[ch&1]. AS=1: everything lives in sAb[0].
    const uint8_t* aptr = (AS == 1) ? sAb[0] : sAb[cur];

#pragma unroll
    for (int s = 0; s < KSTEPS; ++s) {
      const int kb = s * 32 + g * 8;
      uint64_t q0, q1, q2, q3;
      __builtin_memcpy(&q0, aptr + (0 + row) * RSB + kb, 8);
      __builtin_memcpy(&q1, aptr + (16 + row) * RSB + kb, 8);
      __builtin_memcpy(&q2, aptr + (32 + row) * RSB + kb, 8);
      __builtin_memcpy(&q3, aptr + (48 + row) * RSB + kb, 8);
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

    if (nc < NCH) {
      if (AS == 1) {
        // A's only buffer is still being read by the warps above, so the reuse
        // needs a barrier before it, and a second one after so the write is
        // visible before the next chunk reads it.
        __syncthreads();
#pragma unroll
        for (int j = 0; j < NPA; ++j) {
          const int idx = tid + j * NTHR;
          const int r = idx / ACL, c = (idx - r * ACL) * 16;
          st16(sAb[0] + r * RSB + c, pa[j]);
        }
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

#define LAUNCH_V15(WPB, KC, NT, PADA, PADF, AS, SPL)                           \
  do {                                                                         \
    const int per = (WPB) * (NT) * 16;                                         \
    const int tiles = (N + per - 1) / per;                                     \
    dim3 g(tiles, (SPL));                                                      \
    gemm_v15_kernel<WPB, KC, NT, PADA, PADF, AS>                               \
        <<<g, (WPB) * 64, 0, (mcStream_t)stream>>>(                            \
            (const __maca_bfloat16*)A, Bn, C, M, N, K, (SPL));                 \
  } while (0)

template <int WPB, int KC, int NT, int PADA, int PADF, int AS>
static void launch_tc_v15(const void* A, const uint8_t* Bn, float* C,
                          int M, int N, int K, int spl, void* stream) {
  LAUNCH_V15(WPB, KC, NT, PADA, PADF, AS, spl);
}

extern "C" int launch_gemm_v15(const void* A, const uint8_t* Bn, float* C,
                               int M, int N, int K, int cfg, void* stream) {
  const int spl = cfg % 100, tc = cfg / 100;
  if (spl < 1 || K % spl || (K / spl) % 32) return -2;
  switch (tc) {
    case 20: launch_tc_v15<4, 64, 4, 16, 16, 2>(A, Bn, C, M, N, K, spl, stream); break;
    case 30: launch_tc_v15<4, 32, 4, 16, 16, 2>(A, Bn, C, M, N, K, spl, stream); break;
    case 31: launch_tc_v15<4, 32, 4, 16, 16, 1>(A, Bn, C, M, N, K, spl, stream); break;
    case 32: launch_tc_v15<4, 32, 4, 16,  0, 2>(A, Bn, C, M, N, K, spl, stream); break;
    case 33: launch_tc_v15<4, 32, 4, 16,  0, 1>(A, Bn, C, M, N, K, spl, stream); break;
    case 34: launch_tc_v15<4, 32, 2, 16, 16, 1>(A, Bn, C, M, N, K, spl, stream); break;
    case 35: launch_tc_v15<2, 64, 2, 16, 16, 1>(A, Bn, C, M, N, K, spl, stream); break;
    case 36: launch_tc_v15<4, 32, 4, 16, 16, 1>(A, Bn, C, M, N, K, spl, stream); break;
    case 37: launch_tc_v15<2, 32, 4, 16, 16, 1>(A, Bn, C, M, N, K, spl, stream); break;
    default: return -1;
  }
  return 0;
}

// Verify the "two blocks per SM" premise instead of trusting the byte count.
template <int WPB, int KC, int NT, int PADA, int PADF, int AS>
static int occ_probe() {
  int nb = -1;
  mcOccupancyMaxActiveBlocksPerMultiprocessor(
      &nb, (const void*)gemm_v15_kernel<WPB, KC, NT, PADA, PADF, AS>, WPB * 64, 0);
  return nb;
}

template <int WPB, int KC, int NT, int PADA, int PADF, int AS>
static int smem_probe() {
  mcFuncAttributes fa{};
  if (mcFuncGetAttributes(&fa, (const void*)gemm_v15_kernel<WPB, KC, NT, PADA, PADF, AS>) != 0)
    return -1;
  return (int)fa.sharedSizeBytes;
}

extern "C" int v15_smem_bytes(int tc) {
  switch (tc) {
    case 20: return smem_probe<4, 64, 4, 16, 16, 2>();
    case 30: return smem_probe<4, 32, 4, 16, 16, 2>();
    case 31: return smem_probe<4, 32, 4, 16, 16, 1>();
    case 32: return smem_probe<4, 32, 4, 16,  0, 2>();
    case 33: return smem_probe<4, 32, 4, 16,  0, 1>();
    case 34: return smem_probe<4, 32, 2, 16, 16, 1>();
    case 35: return smem_probe<2, 64, 2, 16, 16, 1>();
    case 36: return smem_probe<4, 32, 4, 16, 16, 1>();
    case 37: return smem_probe<2, 32, 4, 16, 16, 1>();
    default: return -1;
  }
}

extern "C" int v15_occ_blocks(int tc) {
  switch (tc) {
    case 20: return occ_probe<4, 64, 4, 16, 16, 2>();
    case 30: return occ_probe<4, 32, 4, 16, 16, 2>();
    case 31: return occ_probe<4, 32, 4, 16, 16, 1>();
    case 32: return occ_probe<4, 32, 4, 16,  0, 2>();
    case 33: return occ_probe<4, 32, 4, 16,  0, 1>();
    case 34: return occ_probe<4, 32, 2, 16, 16, 1>();
    case 35: return occ_probe<2, 64, 2, 16, 16, 1>();
    case 36: return occ_probe<4, 32, 4, 16, 16, 1>();
    case 37: return occ_probe<2, 32, 4, 16, 16, 1>();
    default: return -1;
  }
}
