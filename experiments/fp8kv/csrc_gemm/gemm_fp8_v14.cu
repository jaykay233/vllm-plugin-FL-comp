// fp8-weight GEMM -- v14: dequantize during staging instead of during compute.
//
// Reasoning from v12's measurements on lm_head:
//   staging only   0.444 ms -> 602 GB/s
//   compute only   0.436 ms
//   full           0.522 ms
// Compute alone is 436 us for 34 GFLOP, i.e. 78 TFLOPS of an achievable 149 TFLOPS
// peak -- only 52%. The cause is instruction issue, not the MMA:
//
//   per k-step per warp (NT=4): 16 MMAs, 4 B fragment loads, 4 dequantizations.
//   Each dequant is ~18 ALU ops, so 72 ALU ops ride along with 16 MMAs. Roughly
//   72/(16+72) = 82% of issue slots go to dequantization, which is what halves
//   the MMA rate.
//
// The dequant has to happen somewhere, but not here. Staging is memory bound, so
// ALU work done there is free. v14 converts fp8 to bf16 on the way into shared
// memory and the inner loop becomes pure load+MMA, like the bf16 reference kernel.
//
// The cost is shared memory: a bf16 B tile is 2x an fp8 one, so the tile shrinks
// (KC=32 instead of 64) to stay inside 64 KB. That was exactly v3's structure --
// v3 only lost because it used memcpy for the global load (168 GB/s). With ldg128
// the staging path that made v3 slow is fixed.
//
// Note this also removes cvt_arith from the critical path but keeps it in the
// kernel for staging, so accuracy is unchanged (2.2e-06).

#include <mc_runtime.h>
#include <cstdint>
#include <maca_bfloat16.h>

typedef __NATIVE_VECTOR__(2, uint32_t) v2u;
typedef __NATIVE_VECTOR__(4, uint32_t) v4u;
typedef __NATIVE_VECTOR__(4, float) v4f;

__device__ __forceinline__ v4u ldg128(const void* p) {
  return __builtin_mxc_ldg_b128((void*)p, 0, -1, true, true, false, false);
}

__device__ __forceinline__ v2u cvt_arith(uint32_t bq) {
  const uint32_t lo = (bq & 0x000000FFu) | ((bq & 0x0000FF00u) << 8);
  const uint32_t hi = ((bq >> 16) & 0x000000FFu) | ((bq >> 8) & 0x00FF0000u);
  const uint32_t r0 = (((lo & 0x007F007Fu) << 4) + 0x3C003C00u) | ((lo & 0x00800080u) << 8);
  const uint32_t r1 = (((hi & 0x007F007Fu) << 4) + 0x3C003C00u) | ((hi & 0x00800080u) << 8);
  return v2u{r0, r1};
}

// Convert one 16-byte fp8 chunk (16 values) into 32 bytes of bf16 and store it at
// dst, whose row stride is RSF2. dst_off is the byte offset within the row; the
// 32 output bytes are contiguous only if they do not straddle the row padding, so
// callers pass chunks that stay inside a row (KC is a multiple of 16).
__device__ __forceinline__ void stage_cvt16(uint8_t* dst, const v4u& src) {
  const uint32_t w[4] = {src[0], src[1], src[2], src[3]};
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    v2u r = cvt_arith(w[i]);
    __builtin_memcpy(dst + i * 8, &r, 8);
  }
}

template <int WPB, int KC, int NT, int PADB>
__global__ void gemm_v14_kernel(const __maca_bfloat16* __restrict__ A,
                                const uint8_t* __restrict__ Bn,
                                float* __restrict__ C,
                                int M, int N, int K, int nspl) {
  // Both tiles are bf16 in smem now. KC is in *elements* of the k dimension.
  constexpr int RS = KC * 2 + PADB;         // bf16 row stride, bytes
  static_assert(RS % 16 == 0, "16B alignment");
  constexpr int BROWS = WPB * NT * 16;
  constexpr int NTHR = WPB * 64;
  constexpr int KSTEPS = KC / 16;

  constexpr int ACL = KC * 2 / 16;          // 16B chunks per A row (8 bf16 each)
  constexpr int BCL = KC / 16;              // 16B fp8 chunks per B row
  constexpr int ACH = 64 * ACL;
  constexpr int BCH = BROWS * BCL;
  constexpr int NPA = ACH / NTHR;
  constexpr int NPB = BCH / NTHR;
  static_assert(ACH % NTHR == 0 && BCH % NTHR == 0, "tile must divide evenly");

  __shared__ __align__(16) uint8_t sAb[2][64 * RS];
  __shared__ __align__(16) uint8_t sBf[2][BROWS * RS];

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

  // A: copy 16B chunks unchanged (already bf16). B: load fp8 and convert.
  for (int j = 0; j < NPA; ++j) {
    const int idx = tid + j * NTHR;
    const int r = idx / ACL, c = (idx - r * ACL) * 16;
    v4u v = ldg128(A + (size_t)r * K + kbeg + c / 2);
    __builtin_memcpy(sAb[0] + r * RS + c, &v, 16);
  }
  for (int j = 0; j < NPB; ++j) {
    const int idx = tid + j * NTHR;
    const int r = idx / BCL, c = (idx - r * BCL) * 16;
    v4u v = ldg128(Bn + (size_t)(n0 + r) * K + kbeg + c);
    stage_cvt16(sBf[0] + r * RS + c * 2, v);
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

    // inner loop: pure shared-memory load + MMA, no arithmetic
#pragma unroll
    for (int s = 0; s < KSTEPS; ++s) {
      const int kb = s * 32 + g * 8;
      uint64_t q0, q1, q2, q3;
      __builtin_memcpy(&q0, sAb[cur] + (0 + row) * RS + kb, 8);
      __builtin_memcpy(&q1, sAb[cur] + (16 + row) * RS + kb, 8);
      __builtin_memcpy(&q2, sAb[cur] + (32 + row) * RS + kb, 8);
      __builtin_memcpy(&q3, sAb[cur] + (48 + row) * RS + kb, 8);
      v2u va0 = {(uint32_t)q0, (uint32_t)(q0 >> 32)};
      v2u va1 = {(uint32_t)q1, (uint32_t)(q1 >> 32)};
      v2u va2 = {(uint32_t)q2, (uint32_t)(q2 >> 32)};
      v2u va3 = {(uint32_t)q3, (uint32_t)(q3 >> 32)};

      const int kbb = s * 32 + g * 8;
#pragma unroll
      for (int t = 0; t < NT; ++t) {
        uint64_t qb;
        __builtin_memcpy(&qb, sBf[cur] + (warp * NT * 16 + t * 16 + row) * RS + kbb, 8);
        v2u vb = {(uint32_t)qb, (uint32_t)(qb >> 32)};
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
        __builtin_memcpy(sAb[nxt] + r * RS + c, &pa[j], 16);
      }
#pragma unroll
      for (int j = 0; j < NPB; ++j) {
        const int idx = tid + j * NTHR;
        const int r = idx / BCL, c = (idx - r * BCL) * 16;
        stage_cvt16(sBf[nxt] + r * RS + c * 2, pb[j]);
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

#define LAUNCH_V14(WPB, KC, NT, PADB, SPL)                                     \
  do {                                                                         \
    const int per = (WPB) * (NT) * 16;                                         \
    const int tiles = (N + per - 1) / per;                                     \
    dim3 g(tiles, (SPL));                                                      \
    gemm_v14_kernel<WPB, KC, NT, PADB>                                         \
        <<<g, (WPB) * 64, 0, (mcStream_t)stream>>>(                            \
            (const __maca_bfloat16*)A, Bn, C, M, N, K, (SPL));                 \
  } while (0)

extern "C" int launch_gemm_v14(const void* A, const uint8_t* Bn, float* C,
                               int M, int N, int K, int cfg, void* stream) {
  const int spl = cfg % 100, tc = cfg / 100;
  if (spl < 1 || K % spl || (K / spl) % 32) return -2;
  switch (tc) {
    case 0: LAUNCH_V14(4, 32, 4, 32, spl); break;
    case 1: LAUNCH_V14(4, 32, 2, 32, spl); break;
    case 2: LAUNCH_V14(2, 32, 4, 32, spl); break;
    case 3: LAUNCH_V14(4, 64, 2, 32, spl); break;
    case 4: LAUNCH_V14(2, 64, 2, 32, spl); break;
    case 5: LAUNCH_V14(2, 32, 2, 32, spl); break;
    case 6: LAUNCH_V14(2, 16, 4, 16, spl); break;
    case 7: LAUNCH_V14(2, 64, 1, 32, spl); break;
    default: return -1;
  }
  return 0;
}
