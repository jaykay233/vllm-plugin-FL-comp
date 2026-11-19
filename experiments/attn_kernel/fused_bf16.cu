// Fused bf16 causal attention (QK^T + online softmax + PV) on the native MMA.
//
// Structure chosen from the measured constraints:
//   * MMA issue is linear in warps/SM up to 16 warps (92.3 of 98.3 TFLOPS), so the
//     design wants few registers rather than large tiles.
//   * O for a BM x 128 block is BM*128/256 floats per lane, which dominates the
//     register budget, so BM must stay modest.  Taking BM=128 with 8 warps gives
//     each warp 16 query rows: Q is 8 chunks x 4 = 32 registers, O is the same 32,
//     and nothing is replicated across warps.
//   * S is 16x16, so one BN=16 KV block per step; head_dim=128 is walked as eight
//     16-wide k-chunks, each one MMA (the verified mapping is exact for k=16).
//
// Two layout facts make this work, both verified separately:
//   * an A fragment indexes row = 2*(L/16) + 8*(j/2) + (j%2), col = L%16
//     (pairs along M); a B fragment indexes k = 2*(L/16) + 8*(j/2) + (j%2),
//     n = L%16 (pairs along K); an accumulator holds row = L/16 + 4q, col = L%16.
//   * so a row of S lives entirely in 16 consecutive lanes (those sharing L/16),
//     which makes both the max and the sum a 16-lane xor reduction.
//
// P has to move from accumulator layout to A-fragment layout.  That mapping is
//     (row r, col c) at (lane L, reg q) -> (lane 16*((r>>1)&3) + c, reg 2*((r>>3)&1) + (r&1))
// which is a 4x4 transpose across lane groups, so it goes through a per-warp
// shared-memory scratch rather than a shuffle network.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

typedef __ix_bfloat16 bf_t;
typedef __bf16 v4y __attribute__((vector_size(8)));
typedef float v4f __attribute__((vector_size(16)));

#define NWARP 8
#define NTHREAD (NWARP * 64)
#define BM 128        // 16 query rows per warp
#define D 128
#define NCH (D / 16)  // 8 head_dim chunks

__device__ __forceinline__ v4y pack4(const bf_t *h) {
  v4y r;
  __builtin_memcpy(&r, h, 8);
  return r;
}

// K and V are staged into plain row-major smem by a fully coalesced 16-byte copy,
// and the fragments read straight out of it.  The previous version scattered each
// element into a pre-swizzled layout with 2-byte stores, which measured as 47% of
// the kernel time (11.52 -> 6.09 ms when the staging loop was removed; removing a
// barrier changed nothing, so it was the scattered stores, not the barriers).
//
// The row stride is padded from 128 to D+8 = 136 bf16 = 272 B, which is still
// 16-byte aligned (17 x 16) so the vectorised copy stays legal, and 272 mod 128 =
// 16 spreads the fragment reads across bank groups instead of colliding.
//
//   QK^T B fragment: k = head_dim, n = kv position
//        pos = L%16,  dim = 16*ch + (2*(L/16) + 8*(j/2) + (j%2))
//   PV   B fragment: k = kv position, n = head_dim
//        pos = 2*(L/16) + 8*(j/2) + (j%2),  dim = 16*ch + (L%16)

// Staging and the fragment read have opposite layout preferences, and both sit in
// the hot loop, so the layout has to satisfy both:
//
//   the fragment read wants a lane's four values contiguous  -> 8-byte load
//   the staging wants wide stores                           -> 8-byte store
//
// A pre-swizzled [chunk][lane][4] array gives the read.  For the store, note which
// elements share one lane's four registers:
//
//   K (lanes carry pos):      pos fixed, dims {2a, 2a+1, 8+2a, 9+2a}  -> regs 0..3
//   V (lanes carry dim%16):   dim fixed, pos  {2a, 2a+1, 8+2a, 9+2a}  -> regs 0..3
//
// so one 8-byte store covers four elements in both cases.  Threads are mapped
// a = u&3 (lane within group), so consecutive threads write consecutive lanes and
// the stores do not collide.
//
// An earlier version dropped the swizzle and read fragments out of a padded
// row-major buffer instead.  That made staging a clean 16-byte copy but turned each
// fragment read into four 2-byte loads, 32 per iteration instead of 8, and the
// kernel went from 11.5 to 19.7 ms.  The read is in the inner loop; the staging is
// not.  Both now are wide.

// Fused bf16 causal attention (QK^T + online softmax + PV) on the native MMA.
//
// Structure chosen from the measured constraints:
//   * MMA issue is linear in warps/SM up to 16 warps (92.3 of 98.3 TFLOPS), so the
//     design wants few registers rather than large tiles.
//   * O for a BM x 128 block is BM*128/256 floats per lane, which dominates the
//     register budget, so BM stays at 128 with 8 warps, i.e. 16 query rows per warp:
//     Q is 8 chunks x 4 = 32 registers, O is the same 32, nothing replicated.
//   * S is 16x16, so one BN=16 KV block per step; head_dim=128 is walked as eight
//     16-wide k-chunks, each one MMA (the verified mapping is exact for k=16).
//
// Two layout facts make this work, both verified separately:
//   * an A fragment indexes row = 2*(L/16) + 8*(j/2) + (j%2), col = L%16
//     (pairs along M); a B fragment indexes k = 2*(L/16) + 8*(j/2) + (j%2),
//     n = L%16 (pairs along K); an accumulator holds row = L/16 + 4q, col = L%16.
//   * so a row of S lives entirely in 16 consecutive lanes (those sharing L/16),
//     which makes both the max and the sum a 16-lane reduction, four xor steps.
//
// Why NT tiles per iteration:
//
// Measured on the 4k causal shape the kernel issued only ~1.34 instructions per
// cycle per SM (2400 warp-instructions per CTA-iteration over 1793 cycles), with 56
// scoreboard waits in a 300-instruction loop body.  It is stalled, not short of
// issue slots.  Splitting just the QK accumulator chain (NS=2/4/8) moved nothing,
// which locates the stall in the serialisation of a whole iteration -- load, MMA
// chain, reduce, exp, transpose, MMA chain -- rather than in one stage of it.
//
// So the loop processes NT independent KV tiles: NT separate score accumulators,
// NT sets of exponentials, and only the row statistics merged.  The per-iteration
// fixed cost (the O rescale, the row reductions) is then amortised over NT times
// the MMA work, and the scheduler has NT chains to interleave.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

typedef __ix_bfloat16 bf_t;
typedef __bf16 v4y __attribute__((vector_size(8)));
typedef float v4f __attribute__((vector_size(16)));

#define NWARP 8
#define NTHREAD (NWARP * 64)
#define BM (NWARP * 16)  // 16 query rows per warp
#define D 128
#define NCH (D / 16)     // 8 head_dim chunks
#define BNT 16           // KV positions per tile
#define TF (NCH * 64 * 4)

// K and V are staged into a pre-swizzled [chunk][lane][4] array: a lane's four
// fragment values are then one 8-byte load, and one 8-byte store still covers four
// elements because
//   K (lanes carry pos):     pos fixed, dims {2a, 2a+1, 8+2a, 9+2a}
//   V (lanes carry dim%16):  dim fixed, pos  {2a, 2a+1, 8+2a, 9+2a}
// Threads are mapped a = tid&3 so consecutive threads hit consecutive lanes.
template <int CAUSAL, int NT>
__global__ __launch_bounds__(NTHREAD) void fused(const bf_t *Q, const bf_t *K, const bf_t *V,
                                                 float *O, int kv, int NH, int NKVH,
                                                 float log2scale) {
  __shared__ bf_t Ksw[NT * TF];
  __shared__ bf_t Vsw[NT * TF];

  const int tid = threadIdx.x;
  const int lane = tid & 63;
  const int warp = tid >> 6;
  const int qbase = blockIdx.x * BM;
  const int head = blockIdx.y;
  // GQA: 16 query heads share 2 KV heads in the real model, so K/V reads are 8x
  // fewer than a per-head test would suggest.
  const int kvhead = NKVH == NH ? head : head * NKVH / NH;
  const int ldKV = NKVH * D;

  // ---- Q fragments, once ----
  v4y qf[NCH];
#pragma unroll
  for (int ch = 0; ch < NCH; ++ch) {
    bf_t h[4];
    const int row = qbase + warp * 16 + 2 * (lane / 16);
    const int col = 16 * ch + (lane % 16);
#pragma unroll
    for (int j = 0; j < 4; ++j)
      h[j] = Q[(size_t)(row + 8 * (j / 2) + (j % 2)) * NH * D + (size_t)head * D + col];
    qf[ch] = pack4(h);
  }

  v4f acc[NCH];
#pragma unroll
  for (int ch = 0; ch < NCH; ++ch) acc[ch] = (v4f){0.f, 0.f, 0.f, 0.f};
  float mrun[4] = {-1e30f, -1e30f, -1e30f, -1e30f};
  float lrun[4] = {0.f, 0.f, 0.f, 0.f};

  // Staging addresses resolve once and then advance by a constant stride.  The
  // thread -> (lane group, position, chunk) mapping is fixed, so only the source
  // pointer moves.  V's four values sit in four different kv rows, so they cannot
  // be paired into wider loads; K's can.
  const int sa = tid & 3, sc16 = (tid >> 2) & 15, sch = tid >> 6;
  const int sld = ldKV;
  const bf_t *pK = K + (long)kvhead * D + (long)sc16 * sld + 16 * sch + 2 * sa;
  const bf_t *pV = V + (long)kvhead * D + 16 * sch + sc16;
  const int sstep = BNT * sld;

  const int nblk = CAUSAL ? (qbase + BM < kv ? (qbase + BM + BNT * NT - 1) / (BNT * NT)
                                            : (kv + BNT * NT - 1) / (BNT * NT))
                          : (kv + BNT * NT - 1) / (BNT * NT);

  for (int blk = 0; blk < nblk; ++blk) {
    // ---- stage NT tiles of K/V ----
    bf_t kv2[NT][4], vv2[NT][4];
    const bf_t *qK = pK, *qV = pV;
#pragma unroll
    for (int t = 0; t < NT; ++t) {
      unsigned int p0, p1;
      __builtin_memcpy(&p0, &qK[0], 4);
      __builtin_memcpy(&p1, &qK[8], 4);
      __builtin_memcpy(&kv2[t][0], &p0, 2);
      __builtin_memcpy(&kv2[t][1], (const char *)&p0 + 2, 2);
      __builtin_memcpy(&kv2[t][2], &p1, 2);
      __builtin_memcpy(&kv2[t][3], (const char *)&p1 + 2, 2);
      vv2[t][0] = qV[2 * sa * sld];
      vv2[t][1] = qV[(2 * sa + 1) * sld];
      vv2[t][2] = qV[(8 + 2 * sa) * sld];
      vv2[t][3] = qV[(9 + 2 * sa) * sld];
      qK += sstep;
      qV += sstep;
    }
    __syncthreads();
#pragma unroll
    for (int t = 0; t < NT; ++t) {
      const int o = ((sch * 64) + 16 * sa + sc16) * 4;
      __builtin_memcpy(&Ksw[t * TF + o], kv2[t], 8);
      __builtin_memcpy(&Vsw[t * TF + o], vv2[t], 8);
    }
    __syncthreads();
    pK = qK;
    pV = qV;

    // ---- S = Q @ K^T, one independent accumulator per tile ----
    v4f s[NT];
#pragma unroll
    for (int t = 0; t < NT; ++t) {
      v4y kf[NCH];
#pragma unroll
      for (int ch = 0; ch < NCH; ++ch) kf[ch] = pack4(&Ksw[t * TF + ((ch * 64) + lane) * 4]);
      v4f a = (v4f){0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int ch = 0; ch < NCH; ++ch) a = __ivcorex_matrix_mad_f32x4_bf16x4(qf[ch], kf[ch], a);
      s[t] = a;
    }

    // ---- masked log2-domain scores ----
    const int n0 = blk * (BNT * NT);
    float x[NT][4];
#pragma unroll
    for (int t = 0; t < NT; ++t)
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const int col = n0 + t * BNT + (lane % 16);
        float v = s[t][q] * log2scale;
        if (CAUSAL && col > qbase + warp * 16 + (lane / 16 + 4 * q)) v = -1e30f;
        x[t][q] = v;
      }

    // ---- row max: merge across tiles first, then one 16-lane reduction ----
    float alpha[4];
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      float mx = x[0][q];
#pragma unroll
      for (int t = 1; t < NT; ++t) mx = fmaxf(mx, x[t][q]);
#pragma unroll
      for (int off = 1; off < 16; off <<= 1) mx = fmaxf(mx, __shfl_xor_sync(~0ULL, mx, off));
      const float mnew = fmaxf(mrun[q], mx);
      // x is log2-domain, so the rescale factor is exactly 2^d.
      alpha[q] = exp2f(mrun[q] - mnew);
      mrun[q] = mnew;
    }

    // ---- rescale O once per iteration, not per tile ----
#pragma unroll
    for (int ch = 0; ch < NCH; ++ch)
#pragma unroll
      for (int q = 0; q < 4; ++q) acc[ch][q] *= alpha[q];

    // ---- P to A-fragment order by shuffle ----
    //   accumulator (group g, reg q) -> row = g + 4q
    //   A fragment  (group g', reg j) -> row = 2g' + 8*(j>>1) + (j&1)
    // Inverting: g = 2*(g'&1) + (j&1), q = 2*(j>>1) + (g'>>1); the column is
    // untouched.  For fixed j the candidates are 2*(j>>1) and 2*(j>>1)+1, so two
    // shuffles plus a select per register.  This replaced a store/syncwarp/load
    // round trip that measured 47% of the kernel time.
    v4y pa[NT];
    float pv[NT][4];
#pragma unroll
    for (int t = 0; t < NT; ++t) {
#pragma unroll
      for (int q = 0; q < 4; ++q) pv[t][q] = exp2f(x[t][q] - mrun[q]);
      const int gp = (lane >> 4) & 3;
      const int qsel = (lane >> 5) & 1;
      bf_t h[4];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int src = 16 * (2 * (gp & 1) + (j & 1)) + (lane & 15);
        const int q0 = 2 * (j >> 1);
        const float v0 = __shfl_sync(~0ULL, pv[t][q0], src);
        const float v1 = __shfl_sync(~0ULL, pv[t][q0 + 1], src);
        h[j] = __float2bfloat16(qsel ? v1 : v0);
      }
      pa[t] = pack4(h);
    }

    // ---- row sum: merge across tiles first, then one 16-lane reduction ----
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      float sum = pv[0][q];
#pragma unroll
      for (int t = 1; t < NT; ++t) sum += pv[t][q];
#pragma unroll
      for (int off = 1; off < 16; off <<= 1) sum += __shfl_xor_sync(~0ULL, sum, off);
      lrun[q] = lrun[q] * alpha[q] + sum;
    }

    // ---- O += P @ V ----
#pragma unroll
    for (int t = 0; t < NT; ++t) {
      v4y vf[NCH];
#pragma unroll
      for (int ch = 0; ch < NCH; ++ch) vf[ch] = pack4(&Vsw[t * TF + ((ch * 64) + lane) * 4]);
#pragma unroll
      for (int ch = 0; ch < NCH; ++ch) acc[ch] = __ivcorex_matrix_mad_f32x4_bf16x4(pa[t], vf[ch], acc[ch]);
    }
  }

  // ---- epilogue ----
  float inv[4];
#pragma unroll
  for (int q = 0; q < 4; ++q) inv[q] = lrun[q] > 0.f ? 1.f / lrun[q] : 0.f;
#pragma unroll
  for (int ch = 0; ch < NCH; ++ch)
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const int row = lane / 16 + 4 * q;
      O[(size_t)(qbase + warp * 16 + row) * NH * D + (size_t)head * D + 16 * ch + (lane % 16)] =
          acc[ch][q] * inv[q];
    }
}

// ---- host reference, double precision ----
static void ref_attn(const bf_t *hQ, const bf_t *hK, const bf_t *hV, float *hO, int tok, int kv,
                     int NH, int NKVH, int causal) {
  const double scale = 1.0 / std::sqrt((double)D);
  for (int t = 0; t < tok; ++t)
    for (int h = 0; h < NH; ++h) {
      const int kh = NKVH == NH ? h : h * NKVH / NH;
      double mx = -1e30;
      double s[4096];
      // Causal attention over a KV cache shorter than the query block still only
      // sees kv columns: min(t+1, kv).  Without the clamp the reference reads past
      // the end of hK when kv < tok, which reported a false failure at
      // tok=512/kv=256 while the kernel was correct.
      const int nk = causal ? (t + 1 < kv ? t + 1 : kv) : kv;
      for (int j = 0; j < nk; ++j) {
        double d = 0;
        for (int x = 0; x < D; ++x) {
          unsigned short qa, kb;
          std::memcpy(&qa, &hQ[(size_t)t * NH * D + (size_t)h * D + x], 2);
          std::memcpy(&kb, &hK[(size_t)j * NKVH * D + (size_t)kh * D + x], 2);
          unsigned ua = (unsigned)qa << 16, ub = (unsigned)kb << 16;
          float fa, fb;
          std::memcpy(&fa, &ua, 4);
          std::memcpy(&fb, &ub, 4);
          d += (double)fa * fb;
        }
        s[j] = d * scale;
        if (s[j] > mx) mx = s[j];
      }
      double sum = 0;
      for (int j = 0; j < nk; ++j) { s[j] = std::exp(s[j] - mx); sum += s[j]; }
      for (int x = 0; x < D; ++x) {
        double o = 0;
        for (int j = 0; j < nk; ++j) {
          unsigned short vb;
          std::memcpy(&vb, &hV[(size_t)j * NKVH * D + (size_t)kh * D + x], 2);
          unsigned ub = (unsigned)vb << 16;
          float fb;
          std::memcpy(&fb, &ub, 4);
          o += s[j] / sum * (double)fb;
        }
        hO[(size_t)t * NH * D + (size_t)h * D + x] = (float)o;
      }
    }
}

static void set_bf(bf_t *p, float f) {
  unsigned u;
  std::memcpy(&u, &f, 4);
  u += 0x7fffu + ((u >> 16) & 1u);
  const unsigned short h = (unsigned short)(u >> 16);
  std::memcpy(p, &h, 2);
}


// ---- benchmark driver ----
// usage: fused_bf16 <tokens> <kv> [ref=1] [nkv_heads] [nt]
static const char *arg(int argc, char **argv, int i, const char *dflt) {
  return i < argc ? argv[i] : dflt;
}

int main(int argc, char **argv) {
  const int tok = atoi(arg(argc, argv, 1, "1024"));
  const int kv = atoi(arg(argc, argv, 2, "1024"));
  const bool want_ref = atoi(arg(argc, argv, 3, "1")) != 0;
  const int NH = 16;
  const int NKVH = atoi(arg(argc, argv, 4, "16")) > 0 ? atoi(arg(argc, argv, 4, "16")) : NH;
  const int nt_sel = atoi(arg(argc, argv, 5, "4"));

  using kern = void (*)(const bf_t *, const bf_t *, const bf_t *, float *, int, int, int, float);
  auto pick = [&](int causal, int nt) -> kern {
    if (causal) {
      switch (nt) {
        case 1: return (kern)fused<1, 1>;
        case 2: return (kern)fused<1, 2>;
        case 4: return (kern)fused<1, 4>;
        default: return (kern)fused<1, 8>;
      }
    }
    switch (nt) {
      case 1: return (kern)fused<0, 1>;
      case 2: return (kern)fused<0, 2>;
      case 4: return (kern)fused<0, 4>;
      default: return (kern)fused<0, 8>;
    }
  };

  const size_t qsz = (size_t)tok * NH * D, ksz = (size_t)kv * NKVH * D;
  bf_t *hQ = (bf_t *)malloc(qsz * 2), *hK = (bf_t *)malloc(ksz * 2), *hV = (bf_t *)malloc(ksz * 2);
  float *hO = (float *)malloc(qsz * 4), *hRef = (float *)malloc(qsz * 4);
  srand(3);
  for (size_t i = 0; i < qsz; ++i) set_bf(&hQ[i], (rand() % 200 - 100) / 64.0f);
  for (size_t i = 0; i < ksz; ++i) {
    set_bf(&hK[i], (rand() % 200 - 100) / 64.0f);
    set_bf(&hV[i], (rand() % 200 - 100) / 64.0f);
  }

  bf_t *dQ, *dK, *dV;
  float *dO;
  cudaMalloc(&dQ, qsz * 2); cudaMalloc(&dK, ksz * 2);
  cudaMalloc(&dV, ksz * 2); cudaMalloc(&dO, qsz * 4);
  cudaMemcpy(dQ, hQ, qsz * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(dK, hK, ksz * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(dV, hV, ksz * 2, cudaMemcpyHostToDevice);

  const float log2scale = (float)(1.4426950408889634 / std::sqrt((double)D));
  const int nt = tok / BM;
  const dim3 grid(nt, NH);

  // Error metric matters here.  A per-element relative error divides by |ref|, and
  // attention outputs pass through near zero, so it reports huge values for elements
  // that are numerically fine.  The meaningful measure is per-row normalised error,
  // ||diff||_2 / ||ref||_2 over each (token, head) output vector.
  auto row_err = [&](int *nbad, double *worst) {
    *worst = 0; *nbad = 0;
    for (int t = 0; t < tok; ++t)
      for (int h = 0; h < NH; ++h) {
        double dn = 0, rn = 0;
        for (int d = 0; d < D; ++d) {
          const size_t i = (size_t)t * NH * D + (size_t)h * D + d;
          const double diff = (double)hO[i] - hRef[i];
          dn += diff * diff;
          rn += (double)hRef[i] * hRef[i];
        }
        const double e = std::sqrt(dn) / (std::sqrt(rn) + 1e-12);
        if (e > *worst) *worst = e;
        if (e > 1e-2) ++*nbad;
      }
  };

  auto check = [&](int causal) {
    pick(causal, nt_sel)<<<grid, NTHREAD>>>(dQ, dK, dV, dO, kv, NH, NKVH, log2scale);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) { printf("ERR %s\n", cudaGetErrorString(e)); exit(1); }
    if (!want_ref) return;
    cudaMemcpy(hO, dO, qsz * 4, cudaMemcpyDeviceToHost);
    ref_attn(hQ, hK, hV, hRef, tok, kv, NH, NKVH, causal);
    int nbad = 0; double worst = 0;
    row_err(&nbad, &worst);
    printf("fused %-6s tok=%d kv=%d heads=%d kvh=%d NT=%d: worst row err %.2e  bad %d/%d  %s\n",
           causal ? "causal" : "dense", tok, kv, NH, NKVH, nt_sel, worst, nbad, tok * NH,
           nbad == 0 ? "OK" : "FAIL");
  };

  check(0);
  check(1);

  // ---- timing ----
  auto timeit = [&](const char *tag, kern k, double flops) {
    for (int i = 0; i < 3; ++i) k<<<grid, NTHREAD>>>(dQ, dK, dV, dO, kv, NH, NKVH, log2scale);
    cudaEvent_t st, en;
    cudaEventCreate(&st); cudaEventCreate(&en);
    cudaEventRecord(st);
    for (int i = 0; i < 10; ++i) k<<<grid, NTHREAD>>>(dQ, dK, dV, dO, kv, NH, NKVH, log2scale);
    cudaEventRecord(en); cudaEventSynchronize(en);
    float ms; cudaEventElapsedTime(&ms, st, en); ms /= 10;
    printf("  %-22s %7.3f ms   %6.1f TFLOPS (%5.1f%% of 98.3)\n", tag, ms,
           flops / (ms * 1e-3) / 1e12, 100 * flops / (ms * 1e-3) / 1e12 / 98.3);
  };

  // causal touches min(qbase+BM, kv) columns per query block
  double causal_flops = 0;
  for (int qb = 0; qb < tok; qb += BM)
    causal_flops += 4.0 * BM * (double)(qb + BM < kv ? qb + BM : kv) * D * NH;
  const double full_flops = 4.0 * tok * kv * D * NH;

  printf("\n=== tok=%d kv=%d heads=%d kvh=%d (BM=%d, %d warps) ===\n", tok, kv, NH, NKVH, BM, NWARP);
  char tag[64];
  for (int t = 1; t <= 8; t *= 2) {
    snprintf(tag, sizeof(tag), "causal NT=%d", t);
    timeit(tag, pick(1, t), causal_flops);
    int nb = 0; cudaFuncAttributes a;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, pick(1, t), NTHREAD, 0);
    cudaFuncGetAttributes(&a, pick(1, t));
    printf("      NT=%d -> %d CTA/SM (%d warps), smem %zu B, %d regs\n", t, nb, nb * NWARP,
           (size_t)a.sharedSizeBytes, a.numRegs);
  }
  timeit("full square", pick(0, 2), full_flops);
  printf("  vendor FA (4k prefill causal, per layer): 1.304 ms / 52.7 TFLOPS\n");
  return 0;
}
