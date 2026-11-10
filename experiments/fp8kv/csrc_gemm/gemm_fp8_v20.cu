// fp8-weight GEMM -- v20: A read straight from global, B staged in shared memory.
//
// v19 took the row probe at face value and moved to KC=128, which made things
// worse (0.660 vs 0.527 ms). The reason is traffic, not pattern: the achieved
// *raw* bandwidth was actually higher at the smaller tiles, but they move far more
// data because every block re-reads the whole A tile.
//
//     config      N tile  blocks    A      B      total   time    raw GB/s
//     KC64 NT4     256      510   130MB  267MB   397MB   523us     759
//     KC128 NT2    128     1020   261MB  267MB   528MB   660us     800
//     KC64 NT2      64     2040   522MB  267MB   789MB   660us    1195
//
// A is only 256 KB and is read by every block, so it lives in L2 and does not need
// DRAM. Staging it into shared memory is what forces the small N tiles: at KC=128
// with 256 columns the A stages cost 34816 B, which is 43 KB of the budget spent
// on data that is already cached.
//
// So drop it. A fragments are read directly from global (hitting L2) at the MMA
// site, and shared memory holds only the B stream. That frees the N tile to grow:
// at KC=128 the B tile alone is 2*NCOLS*(128+16), which allows 192 columns in
// 55296 B instead of 128, and 384 columns at KC=64.
//
// A's read is 4 x 8 B per thread per k-step at row stride K. Uncoalesced, but L2
// resident would be 267 MB per pass, so 174 MB extra for the 192-column config is
// L2 traffic rather than DRAM.

// fp8-weight GEMM -- v19: KC=128, the fix the row probe pointed at.
//
// rowprobe.cu measured the GEMM's exact staging pattern in isolation, at lm_head
// scale (267.4 MB), with nothing else running:
//
//     per-row run length        time      GB/s
//     KC=64   (current)        371.9 us    719
//     KC=128                   181.8 us   1471
//     KC=256                   171.4 us   1560
//     KC=2048 (contiguous)     169.6 us   1577
//
// The cliff sits exactly at 64 B -> 128 B: at KC=64 each row contributes only half
// a 128 B line per step, and the achieved bandwidth is 46% of peak. At KC=128 it is
// 1471 GB/s, i.e. 93% of the contiguous floor. Same block count (510), same threads,
// same stride -- only the run length changes.
//
// This overturns the earlier "cache-line granularity" dead end recorded in
// RESULTS.md. That experiment tested KC=128 through the GEMM and lost (1.018 ms),
// but it had to halve the N tile to fit shared memory, so the tile change was
// confounded with the run length. The isolated probe removes the confound and shows
// the run length is worth 2x on its own. The earlier entry is corrected there.
//
// The obstacle is shared memory: at KC=128 with 256 output columns,
//     2*64*(256+16) + 2*256*(128+16) = 34816 + 73728 = 108544 B
// which is 43 KB over budget. A single-buffered A tile costs one extra barrier per
// chunk but saves 17408 B, which brings 128-column blocks (1020 tiles) inside the
// limit:
//     64*(256+16) + 2*128*(128+16) = 17408 + 36864 = 54272 B
//
// A can afford a single buffer here: it is 256 KB of activations shared by every
// block, hence L2-resident, and at KC=128 there are only 16 chunks to synchronise
// instead of 32. B, the DRAM stream, stays double buffered.

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

// ASTG: 1 = single-buffered A (extra barrier before overwrite), 2 = double.
template <int WPB, int KC, int NT, int PADA, int PADF, int ASTG>
__global__ void gemm_v19_kernel(const __maca_bfloat16* __restrict__ A,
                                const uint8_t* __restrict__ Bn,
                                float* __restrict__ C,
                                int M, int N, int K, int nspl) {
  constexpr int RSB = KC * 2 + PADA;
  constexpr int RSF = KC + PADF;
  static_assert(RSB % 16 == 0 && RSF % 16 == 0, "16B alignment for ldg128 path");
  static_assert(ASTG == 1 || ASTG == 2, "A is single or double buffered");

  constexpr int BROWS = WPB * NT * 16;
  constexpr int NTHR = WPB * 64;
  constexpr int KSTEPS = KC / 16;
  constexpr int ACL = KC * 2 / 16;
  constexpr int BCL = KC / 16;
  constexpr int ACH = 64 * ACL;
  constexpr int BCH = BROWS * BCL;
  constexpr int NPB = BCH / NTHR;
  static_assert(BCH % NTHR == 0, "tile must divide evenly");
  (void)ACH; (void)ACL;

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

    // With ASTG=2 the next A chunk is staged in lockstep with B; with ASTG=1 it is
    // fetched at the end, after the barrier that frees the single buffer.
    v4u pb[NPB];
    if (has_next) {
#pragma unroll
      for (int j = 0; j < NPB; ++j) {
        const int idx = tid + j * NTHR;
        const int r = idx / BCL, c = (idx - r * BCL) * 16;
        pb[j] = ldg128(Bn + (size_t)(n0 + r) * K + k0 + c);
      }
    }

    const __maca_bfloat16* abase = A + (size_t)kbeg + (size_t)(ch * KC);

#pragma unroll
    for (int s = 0; s < KSTEPS; ++s) {
      // 4 x 8 B per thread directly from L2, no shared-memory round trip
      const int ko = s * 16 + g * 4;
      const __maca_bfloat16* ap0 = abase + (size_t)(0 + row) * K + ko;
      const __maca_bfloat16* ap1 = abase + (size_t)(16 + row) * K + ko;
      const __maca_bfloat16* ap2 = abase + (size_t)(32 + row) * K + ko;
      const __maca_bfloat16* ap3 = abase + (size_t)(48 + row) * K + ko;
      v2u va0, va1, va2, va3;
      __builtin_memcpy(&va0, ap0, 8);
      __builtin_memcpy(&va1, ap1, 8);
      __builtin_memcpy(&va2, ap2, 8);
      __builtin_memcpy(&va3, ap3, 8);

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

template <int WPB, int KC, int NT, int PADA, int PADF, int ASTG>
static void launch_tc_v20(const void* A, const uint8_t* Bn, float* C,
                          int M, int N, int K, int spl, void* stream) {
  const int per = WPB * NT * 16;
  const int tiles = (N + per - 1) / per;
  dim3 g(tiles, spl);
  gemm_v19_kernel<WPB, KC, NT, PADA, PADF, ASTG>
      <<<g, WPB * 64, 0, (mcStream_t)stream>>>(
          (const __maca_bfloat16*)A, Bn, C, M, N, K, spl);
}

extern "C" int launch_gemm_v20(const void* A, const uint8_t* Bn, float* C,
                               int M, int N, int K, int cfg, void* stream) {
  const int spl = cfg % 100, tc = cfg / 100;
  if (spl < 1 || K % spl || (K / spl) % 32) return -2;
  switch (tc) {
    case 20: launch_tc_v20<4, 64, 4, 16, 16, 2>(A, Bn, C, M, N, K, spl, stream); break;
    // KC=128, 192 columns: 55296 B of B only, grid 680
    case 60: launch_tc_v20<4, 128, 3, 16, 16, 2>(A, Bn, C, M, N, K, spl, stream); break;
    // KC=128, 128 columns: 36864 B, grid 1020
    case 61: launch_tc_v20<4, 128, 2, 16, 16, 2>(A, Bn, C, M, N, K, spl, stream); break;
    // KC=64, 384 columns: 61440 B, grid 340 -- fewest A re-reads
    case 62: launch_tc_v20<4, 64, 6, 16, 16, 2>(A, Bn, C, M, N, K, spl, stream); break;
    // KC=64, 256 columns
    case 63: launch_tc_v20<4, 64, 4, 16, 16, 2>(A, Bn, C, M, N, K, spl, stream); break;
    // KC=256, 64 columns: B tile 2*64*272 = 34816 B
    case 64: launch_tc_v20<4, 256, 1, 16, 16, 2>(A, Bn, C, M, N, K, spl, stream); break;
    // KC=128, 256 columns does not fit (73728 B); 192 is the largest
    default: return -1;
  }
  return 0;
}

template <int WPB, int KC, int NT, int PADA, int PADF, int ASTG>
static int attr_v19(int which) {
  mcFuncAttributes fa{};
  if (mcFuncGetAttributes(&fa, (const void*)gemm_v19_kernel<WPB, KC, NT, PADA, PADF, ASTG>) != 0)
    return -1;
  if (which == 0) return (int)fa.sharedSizeBytes;
  if (which == 1) return fa.numRegs;
  if (which == 2) return (int)fa.localSizeBytes;
  int nb = -1;
  mcOccupancyMaxActiveBlocksPerMultiprocessor(
      &nb, (const void*)gemm_v19_kernel<WPB, KC, NT, PADA, PADF, ASTG>, WPB * 64, 0);
  return nb;
}

#define P20(TC, WPB, KC, NT, PA, PF, AS_) case TC: return attr_v19<WPB, KC, NT, PA, PF, AS_>(which);

extern "C" int v20_probe(int tc, int which) {
  switch (tc) {
    P20(20, 4, 64, 4, 16, 16, 2)
    P20(60, 4, 128, 3, 16, 16, 2)
    P20(61, 4, 128, 2, 16, 16, 2)
    P20(62, 4, 64, 6, 16, 16, 2)
    P20(63, 4, 64, 4, 16, 16, 2)
    P20(64, 4, 256, 1, 16, 16, 2)
    default: return -1;
  }
}
