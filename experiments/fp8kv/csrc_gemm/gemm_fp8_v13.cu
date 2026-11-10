// fp8-weight GEMM, multi-stage pipeline. Target: recover the overlap headroom.
//
// v12's decomposition on lm_head (N=130560, K=2048) showed:
//     staging only      0.444 ms
//     compute only      0.436 ms
//     full              0.522 ms   -> 1.26x of bf16 mcblas
// Perfect overlap would give max(0.444, 0.436) = 0.444 ms, i.e. 1.49x. The gap is
// the per-chunk barrier: with only 2 smem stages, the store into the other buffer
// sits immediately before __syncthreads(), so the copy has no room to land and the
// barrier drains whatever is still in flight.
//
// v13 generalises the buffer count. With STG stages we prefetch STG-1 chunks ahead
// and always store into the stage that is STG-1 iterations away from being read,
// which gives the copy a full STG-1 compute phases to complete instead of zero.
//
// smem is the constraint, so STG and KC trade off against each other:
//     KC=64 STG=2  WPB4 NT4   61 KB   1 block/SM   (v12, 1.26x)
//     KC=32 STG=3  WPB4 NT4   55 KB   1 block/SM
//     KC=64 STG=3  WPB4 NT2   61 KB   1 block/SM
//     KC=32 STG=3  WPB4 NT2   37 KB   1 block/SM
//     KC=32 STG=2  WPB2 NT2   19 KB   2 blocks/SM
//
// Dataflow and conversion are unchanged from v12 (verified there): A stays bf16
// and is never converted; B stays fp8 in gmem and smem and is dequantized at
// fragment read by pure arithmetic (no LUT, zero extra smem reads).

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

// e4m3 -> bf16 by arithmetic. Widening the byte into a 16-bit lane and shifting
// the (exp|mant) field left by 4 aligns the bits, then the exponent is rebased
// with +0x3C00. Two lanes per word. Exact for normals; subnormals are not
// special-cased (they decode as if e==0), which measures out at 2.2e-06.
__device__ __forceinline__ v2u cvt_arith(uint32_t bq) {
  const uint32_t lo = (bq & 0x000000FFu) | ((bq & 0x0000FF00u) << 8);
  const uint32_t hi = ((bq >> 16) & 0x000000FFu) | ((bq >> 8) & 0x00FF0000u);
  const uint32_t r0 = (((lo & 0x007F007Fu) << 4) + 0x3C003C00u) | ((lo & 0x00800080u) << 8);
  const uint32_t r1 = (((hi & 0x007F007Fu) << 4) + 0x3C003C00u) | ((hi & 0x00800080u) << 8);
  return v2u{r0, r1};
}

template <int WPB, int KC, int NT, int PADA, int PADF, int STG>
__global__ void gemm_v13_kernel(const __maca_bfloat16* __restrict__ A,
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
  static_assert(STG >= 2, "need at least double buffering");

  __shared__ __align__(16) uint8_t sAb[STG][64 * RSB];
  __shared__ __align__(16) uint8_t sBf[STG][BROWS * RSF];

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

  // ---- prologue: fill the first STG-1 stages ----
#pragma unroll
  for (int p = 0; p < STG - 1; ++p) {
    if (p >= NCH) break;
    const int k0 = kbeg + p * KC;
    for (int j = 0; j < NPA; ++j) {
      const int idx = tid + j * NTHR;
      const int r = idx / ACL, c = (idx - r * ACL) * 16;
      st16(sAb[p] + r * RSB + c, ldg128(A + (size_t)r * K + k0 + c / 2));
    }
    for (int j = 0; j < NPB; ++j) {
      const int idx = tid + j * NTHR;
      const int r = idx / BCL, c = (idx - r * BCL) * 16;
      st16(sBf[p] + r * RSF + c, ldg128(Bn + (size_t)(n0 + r) * K + k0 + c));
    }
  }
  __syncthreads();

  for (int ch = 0; ch < NCH; ++ch) {
    const int cur = ch % STG;
    const int fut = (ch + STG - 1) % STG;

    // issue the load for the stage that will be needed STG-1 iterations from now,
    // so the copy overlaps the next STG-1 compute phases instead of 0
    v4u pa[NPA], pb[NPB];
    const int nload = ch + STG - 1;
    if (nload < NCH) {
      const int k0 = kbeg + nload * KC;
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

    if (nload < NCH) {
#pragma unroll
      for (int j = 0; j < NPA; ++j) {
        const int idx = tid + j * NTHR;
        const int r = idx / ACL, c = (idx - r * ACL) * 16;
        st16(sAb[fut] + r * RSB + c, pa[j]);
      }
#pragma unroll
      for (int j = 0; j < NPB; ++j) {
        const int idx = tid + j * NTHR;
        const int r = idx / BCL, c = (idx - r * BCL) * 16;
        st16(sBf[fut] + r * RSF + c, pb[j]);
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

#define LAUNCH_V13(WPB, KC, NT, STG, SPL)                                      \
  LAUNCH_V13P(WPB, KC, NT, 32, STG, SPL)

#define LAUNCH_V13P(WPB, KC, NT, PADA, STG, SPL)                               \
  do {                                                                         \
    const int per = (WPB) * (NT) * 16;                                         \
    const int tiles = (N + per - 1) / per;                                     \
    dim3 g(tiles, (SPL));                                                      \
    gemm_v13_kernel<WPB, KC, NT, PADA, 16, STG>                                \
        <<<g, (WPB) * 64, 0, (mcStream_t)stream>>>(                            \
            (const __maca_bfloat16*)A, Bn, C, M, N, K, (SPL));                 \
  } while (0)

// cfg = tc * 100 + splits
extern "C" int launch_gemm_v13(const void* A, const uint8_t* Bn, float* C,
                               int M, int N, int K, int cfg, void* stream) {
  const int spl = cfg % 100, tc = cfg / 100;
  if (spl < 1 || K % spl || (K / spl) % 64) return -2;
  switch (tc) {
    case 0: LAUNCH_V13(4, 64, 4, 2, spl); break;   // v12 baseline
    case 1: LAUNCH_V13(4, 32, 4, 3, spl); break;
    case 2: LAUNCH_V13(4, 64, 2, 3, spl); break;
    case 3: LAUNCH_V13(2, 64, 2, 3, spl); break;
    case 4: LAUNCH_V13(4, 32, 2, 3, spl); break;
    case 5: LAUNCH_V13(4, 32, 2, 4, spl); break;
    case 6: LAUNCH_V13(2, 32, 2, 2, spl); break;
    case 7: LAUNCH_V13(2, 64, 2, 2, spl); break;
    case 8: LAUNCH_V13(2, 32, 4, 2, spl); break;
    case 9: LAUNCH_V13(8, 64, 2, 2, spl); break;
    case 10: LAUNCH_V13(4, 32, 8, 2, spl); break;   // 512 cols/block, halves A re-reads
    case 11: LAUNCH_V13(2, 32, 8, 2, spl); break;
    case 13: LAUNCH_V13(2, 16, 8, 2, spl); break;
    // A-side padding sweep. PADA controls the A row stride, and the bank a lane
    // hits is (row*RSB + kb)/4 mod 32. With KC=64 and PADA=32 the stride is 160B
    // = 40 words and gcd(40,32)=8, so the 16 rows of a lane group land on only 4
    // bank sets -- a 4-way conflict on every A fragment read. PADA=16 (36 words)
    // or 48 (44 words) gives gcd 4 and 8 distinct bank sets.
    case 20: LAUNCH_V13P(4, 64, 4, 16, 2, spl); break;
    case 21: LAUNCH_V13P(4, 64, 4, 32, 2, spl); break;
    case 22: LAUNCH_V13P(4, 64, 4, 48, 2, spl); break;
    case 23: LAUNCH_V13P(2, 64, 2, 16, 2, spl); break;
    case 24: LAUNCH_V13P(2, 64, 2, 32, 2, spl); break;
    case 25: LAUNCH_V13P(2, 64, 2, 48, 2, spl); break;
    case 26: LAUNCH_V13P(4, 32, 4, 16, 2, spl); break;
    case 27: LAUNCH_V13P(4, 32, 4, 48, 2, spl); break;
    default: return -1;
  }
  return 0;
}
