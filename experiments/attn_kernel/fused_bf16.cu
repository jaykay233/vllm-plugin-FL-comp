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
#define BN 16
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

template <int CAUSAL>
__global__ __launch_bounds__(NTHREAD) void fused(const bf_t *Q, const bf_t *K, const bf_t *V,
                                                 float *O, int kv, int NH, int NKVH,
                                                 float log2scale) {
  __shared__ bf_t Ksw[NCH * 64 * 4];
  __shared__ bf_t Vsw[NCH * 64 * 4];
  __shared__ bf_t Psw[NWARP * 64 * 4];

  const int tid = threadIdx.x;
  const int lane = tid & 63;
  const int warp = tid >> 6;
  const int qbase = blockIdx.x * BM;
  const int head = blockIdx.y;
  // GQA: 16 query heads share 2 KV heads in the real model, so K/V reads are 8x
  // fewer than a per-head test would suggest.  Measuring without this overstates
  // the staging cost by 8x.
  const int kvhead = NKVH == NH ? head : head * NKVH / NH;
  const int ldKV = NKVH * D;

  // ---- Q fragments, once ----
  v4y qf[NCH];
#pragma unroll
  for (int ch = 0; ch < NCH; ++ch) {
    bf_t h[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int m = 2 * (lane / 16) + 8 * (j / 2) + (j % 2);
      h[j] = Q[(size_t)(qbase + warp * 16 + m) * NH * D + (size_t)head * D + 16 * ch + (lane % 16)];
    }
    qf[ch] = pack4(h);
  }

  v4f acc[NCH];
#pragma unroll
  for (int ch = 0; ch < NCH; ++ch) acc[ch] = (v4f){0.f, 0.f, 0.f, 0.f};
  float mrun[4] = {-1e30f, -1e30f, -1e30f, -1e30f};
  float lrun[4] = {0.f, 0.f, 0.f, 0.f};

  // Staging addresses are resolved once and then advanced by a constant stride each
  // iteration.  Recomputing them per iteration emitted 172 64-bit addo/addc pairs
  // per KV block, which was the single largest instruction group in the kernel.
  const int sa = tid & 3, sc16 = (tid >> 2) & 15, sch = tid >> 6;
  const int sld = ldKV;                       // K/V row stride in elements
  const bf_t *pK = K + (long)kvhead * D + (long)sc16 * sld + 16 * sch + 2 * sa;
  const bf_t *pV = V + (long)kvhead * D + 16 * sch + sc16;
  const int sstep = BN * sld;

  const int kmax = CAUSAL ? (qbase + BM < kv ? qbase + BM : kv) : kv;

  // stage one KV block into buffer b, advancing the loop-carried source pointers
  auto stage = [&]() {
    bf_t *Ks = Ksw;
    bf_t *Vs = Vsw;
    bf_t kv2[4], vv2[4];
    // K: pos = sc16 (lane group from sa), dims 2sa, 2sa+1, 8+2sa, 9+2sa
    kv2[0] = pK[0];
    kv2[1] = pK[1];
    kv2[2] = pK[8];
    kv2[3] = pK[9];
    __builtin_memcpy(&Ks[((sch * 64) + 16 * sa + sc16) * 4], kv2, 8);
    // V: dim = sc16 + 16*sch, pos = 2sa, 2sa+1, 8+2sa, 9+2sa
    vv2[0] = pV[2 * sa * sld];
    vv2[1] = pV[(2 * sa + 1) * sld];
    vv2[2] = pV[(8 + 2 * sa) * sld];
    vv2[3] = pV[(9 + 2 * sa) * sld];
    __builtin_memcpy(&Vs[((sch * 64) + 16 * sa + sc16) * 4], vv2, 8);
    pK += sstep;
    pV += sstep;
  };

  // Double buffering was tried here and measured NEUTRAL (5.96 vs 5.86 ms) while
  // costing 16 registers, so it was reverted.  Together with the earlier null result
  // for removing a barrier, that rules out barrier cost as the limiter.
  for (int n0 = 0; n0 < kmax; n0 += BN) {
    __syncthreads();
    stage();
    __syncthreads();

    // Hoist both fragment sets into registers up front.  Loaded inline, every MMA
    // waited on its own smem load (71 sl_wait per iteration, against 16 MMAs).
    v4y kf[NCH], vf[NCH];
#pragma unroll
    for (int ch = 0; ch < NCH; ++ch) {
      kf[ch] = pack4(&Ksw[((ch * 64) + lane) * 4]);
      vf[ch] = pack4(&Vsw[((ch * 64) + lane) * 4]);
    }

    // ---- S = Q @ K^T ----
    v4f s = (v4f){0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int ch = 0; ch < NCH; ++ch) s = __ivcorex_matrix_mad_f32x4_bf16x4(qf[ch], kf[ch], s);

    // ---- online softmax ----
    float p[4], alpha[4];
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const int row = lane / 16 + 4 * q;  // warp-local query row
      const int qrow = warp * 16 + row;
      float x = s[q] * log2scale;
      if (CAUSAL) {
        // mask columns past the diagonal; col = lane % 16
        if (n0 + (lane % 16) > qbase + qrow) x = -1e30f;
      }
      p[q] = x;
    }
#pragma unroll
    for (int q = 0; q < 4; ++q) {
#pragma unroll
      for (int off = 1; off < 16; off <<= 1)
        p[q] = fmaxf(p[q], __shfl_xor_sync(0xffffffffffffffffULL, p[q], off));
      const float mnew = fmaxf(mrun[q], p[q]);
      // x is in log2 domain (log2scale folds in log2(e)), so the rescale is 2^d.
      // __expf(d * ln2) is the same thing but exp2f says it directly.
      alpha[q] = exp2f(mrun[q] - mnew);
      mrun[q] = mnew;
    }
    // exponentiate, write P in A-fragment order, and accumulate the row sum
    {
      float pv[4];
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const int row = lane / 16 + 4 * q;
        float x = s[q] * log2scale;
        if (CAUSAL && n0 + (lane % 16) > qbase + warp * 16 + row) x = -1e30f;
        // BUG FIX: this was __expf(x - m), i.e. e^(x-m), but x is log2-domain so
        // it must be 2^(x-m).  alpha happened to be right because exp(ln2*z)=2^z,
        // which is why only P was wrong.
        pv[q] = exp2f(x - mrun[q]);
        const int Lp = 16 * ((row >> 1) & 3) + (lane % 16);
        const int jp = 2 * ((row >> 3) & 1) + (row & 1);
        Psw[((warp * 64) + Lp) * 4 + jp] = __float2bfloat16(pv[q]);
      }
#pragma unroll
      for (int q = 0; q < 4; ++q) {
#pragma unroll
        for (int off = 1; off < 16; off <<= 1)
          pv[q] += __shfl_xor_sync(0xffffffffffffffffULL, pv[q], off);
        lrun[q] = lrun[q] * alpha[q] + pv[q];
      }
    }
    // rescale O and read P back as an A fragment
#pragma unroll
    for (int ch = 0; ch < NCH; ++ch)
#pragma unroll
      for (int q = 0; q < 4; ++q) acc[ch][q] *= alpha[q];

    // Psw is warp-private (indexed by warp), so this only needs a warp-level sync,
    // unlike the buffer flip below which is CTA-wide.
    __syncwarp(0xffffffffffffffffULL);
    const v4y pa = pack4(&Psw[((warp * 64) + lane) * 4]);

    // ---- O += P @ V ----
#pragma unroll
    for (int ch = 0; ch < NCH; ++ch) acc[ch] = __ivcorex_matrix_mad_f32x4_bf16x4(pa, vf[ch], acc[ch]);
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
      const int nk = causal ? t + 1 : kv;
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

int main(int argc, char **argv) {
  const int tok = argc > 1 ? atoi(argv[1]) : BM;
  const int kv = argc > 2 ? atoi(argv[2]) : 256;
  const int NH = 16;
  const int NKVH = (argc > 4 && atoi(argv[4]) > 0) ? atoi(argv[4]) : NH;

  size_t qsz = (size_t)tok * NH * D;
  size_t ksz = (size_t)kv * NKVH * D;
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
  cudaMalloc(&dQ, qsz * 2);
  cudaMalloc(&dK, ksz * 2);
  cudaMalloc(&dV, ksz * 2);
  cudaMalloc(&dO, qsz * 4);
  cudaMemcpy(dQ, hQ, qsz * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(dK, hK, ksz * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(dV, hV, ksz * 2, cudaMemcpyHostToDevice);

  const float log2scale = (float)(1.4426950408889634 / std::sqrt((double)D));

  const int nt = tok / BM;
  const bool noref = (argc > 3 && atoi(argv[3]) == 0);
  fused<0><<<dim3(nt, NH), NTHREAD>>>(dQ, dK, dV, dO, kv, NH, NKVH, log2scale);
  cudaError_t e = cudaDeviceSynchronize();
  if (e != cudaSuccess) { printf("ERR %s\n", cudaGetErrorString(e)); return 1; }
  cudaMemcpy(hO, dO, qsz * 4, cudaMemcpyDeviceToHost);
  if (!noref) ref_attn(hQ, hK, hV, hRef, tok, kv, NH, NKVH, 0);

  // Error metric matters here.  A per-element relative error divides by |ref|, and
  // attention outputs pass through near zero, so it reports huge values for
  // elements that are numerically fine.  The meaningful measure is per-row
  // normalised error: ||diff||_2 / ||ref||_2 over each (token, head) output vector.
  auto row_err = [&](float *got, float *ref, int *nbad, double *worst) {
    *worst = 0; *nbad = 0;
    for (int t = 0; t < tok; ++t)
      for (int h = 0; h < NH; ++h) {
        double dn = 0, rn = 0, mx = 0;
        for (int d = 0; d < D; ++d) {
          const size_t i = (size_t)t * NH * D + (size_t)h * D + d;
          const double diff = (double)got[i] - ref[i];
          dn += diff * diff;
          rn += (double)ref[i] * ref[i];
          if (fabs(diff) > mx) mx = fabs(diff);
        }
        const double e = std::sqrt(dn) / (std::sqrt(rn) + 1e-12);
        if (e > *worst) *worst = e;
        if (e > 1e-2) ++*nbad;
      }
  };

  int nbad = 0;
  double worst = 0;
  if (!noref) {
    row_err(hO, hRef, &nbad, &worst);
    printf("fused dense  tok=%d kv=%d heads=%d: worst row-normalised err %.2e  rows over 1e-2: %d/%d  %s\n",
           tok, kv, NH, worst, nbad, tok * NH, nbad == 0 ? "OK" : "FAIL");
  }

  // causal check
  fused<1><<<dim3(nt, NH), NTHREAD>>>(dQ, dK, dV, dO, kv, NH, NKVH, log2scale);
  e = cudaDeviceSynchronize();
  if (e != cudaSuccess) { printf("ERR %s\n", cudaGetErrorString(e)); return 1; }
  cudaMemcpy(hO, dO, qsz * 4, cudaMemcpyDeviceToHost);
  if (!noref) {
    ref_attn(hQ, hK, hV, hRef, tok, kv, NH, NKVH, 1);
    row_err(hO, hRef, &nbad, &worst);
    printf("fused causal tok=%d kv=%d heads=%d: worst row-normalised err %.2e  rows over 1e-2: %d/%d  %s\n",
           tok, kv, NH, worst, nbad, tok * NH, nbad == 0 ? "OK" : "FAIL");
  }

  // ---- timing ----
  auto timeit = [&](const char *tag, auto fn, double flops) {
    for (int i = 0; i < 3; ++i) fn();
    cudaEvent_t st, en;
    cudaEventCreate(&st); cudaEventCreate(&en);
    cudaEventRecord(st);
    for (int i = 0; i < 10; ++i) fn();
    cudaEventRecord(en); cudaEventSynchronize(en);
    float ms; cudaEventElapsedTime(&ms, st, en); ms /= 10;
    printf("  %-22s %7.3f ms   %6.1f TFLOPS (%5.1f%% of 98.3)\n", tag, ms, flops / (ms * 1e-3) / 1e12,
           100 * flops / (ms * 1e-3) / 1e12 / 98.3);
    return ms;
  };
  // causal does ~half the KV blocks
  double full_flops = 2.0 * 2.0 * tok * kv * D * NH;
  double causal_flops = 0;
  for (int qb = 0; qb < tok; qb += BM)
    causal_flops += 2.0 * 2.0 * BM * (double)((qb + BM < kv ? qb + BM : kv)) * D * NH;
  printf("\n=== timing tok=%d kv=%d heads=%d (BM=%d BN=%d) ===\n", tok, kv, NH, BM, BN);
  float msf = timeit("full square", [&]() { fused<0><<<dim3(nt, NH), NTHREAD>>>(dQ, dK, dV, dO, kv, NH, NKVH, log2scale); }, full_flops);
  float msc = timeit("causal", [&]() { fused<1><<<dim3(nt, NH), NTHREAD>>>(dQ, dK, dV, dO, kv, NH, NKVH, log2scale); }, causal_flops);
  int nb = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, fused<0>, NTHREAD, 0);
  cudaFuncAttributes at;
  cudaFuncGetAttributes(&at, fused<0>);
  printf("  occupancy: %d CTA/SM = %d warps | smem %zu B | regs %d\n", nb, nb * NWARP,
         (size_t)at.sharedSizeBytes, at.numRegs);
  printf("  vendor FA (4k prefill causal, per layer): 1.304 ms / 52.7 TFLOPS\n");
  (void)msf; (void)msc;
  return 0;
}
