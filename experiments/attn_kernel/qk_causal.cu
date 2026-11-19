// Causal QK^T: the winning tile geometry with whole-block skipping.
//
// This is the timing that actually matters, because prefill attention is causal.
// With BM=512 and BN=16 the causal boundary falls on a block edge, so no partial
// block is ever needed: a KV block can be dropped whenever n0 > qbase+BM-1, which
// is exact rather than approximate.
//
// Geometry is the best from the sweep (all warps on M, one N tile each):
//   WMUL=4 WNUL=1 MT=8 NT=1  ->  BM=512 BN=16, 174 registers, 16 warps/SM
//
// K stays in the pre-swizzled shared layout so a B fragment is one 16-byte load;
// the next block is prefetched into registers to cover global latency.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

using namespace nvcuda;

#define DK 128
#define KCH (DK / 32)
#define WMUL 4
#define WNUL 1
#define MT 8
#define NT 1
#define NWARP (WMUL * WNUL)
#define NTHREAD (NWARP * 64)
#define BM (WMUL * MT * 16)
#define BN (WNUL * NT * 16)
#define NCHUNK ((BN * DK / 8 + NTHREAD - 1) / NTHREAD)


__global__ void poison(float *p, size_t n) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) p[i] = -12345.0f;
}

template <int CAUSAL, int STORE>
__global__ __launch_bounds__(NTHREAD) void qk(const __half *Q, const __half *K, float *S,
                                              int kv, int NH) {
  __shared__ unsigned int Ksw[BN * DK / 2];

  const int tid = threadIdx.x;
  const int lane = tid & 63;
  const int wm = tid >> 6;
  const int qbase = blockIdx.x * BM, head = blockIdx.y;

  unsigned int qreg[MT][KCH][4];
#pragma unroll
  for (int i = 0; i < MT; ++i)
#pragma unroll
    for (int c = 0; c < KCH; ++c)
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const int m = wm * (MT * 16) + i * 16 + (lane / 16) * 2 + 8 * (q % 2);
        const int k = c * 32 + (lane % 16) + 16 * (q / 2);
        __half2 h2;
        h2.x = Q[(size_t)(qbase + m) * NH * DK + (size_t)head * DK + k];
        h2.y = Q[(size_t)(qbase + m + 1) * NH * DK + (size_t)head * DK + k];
        qreg[i][c][q] = *(unsigned int *)&h2;
      }

  // causal: rows are qbase..qbase+BM-1, so nothing past qbase+BM-1 is needed
  const int kmax = CAUSAL ? qbase + BM : kv;
  float sink = 0.f;
  const uint4 *kbase = (const uint4 *)Ksw;
  uint4 pf[NCHUNK];

  auto fetch = [&](int n0) {
#pragma unroll
    for (int t = 0; t < NCHUNK; ++t) {
      const int v = tid + t * NTHREAD;
      if (v < BN * DK / 8) pf[t] = *(const uint4 *)&K[(size_t)(n0 + (v * 8) / DK) * DK + (v * 8) % DK];
    }
  };
  auto publish = [&]() {
#pragma unroll
    for (int t = 0; t < NCHUNK; ++t) {
      const int v = tid + t * NTHREAD;
      if (v < BN * DK / 8) {
        const int n = (v * 8) / DK, k0 = (v * 8) % DK;
        const int b = (((n >> 4) * KCH + (k0 >> 5)) * 256) + ((k0 & 31) >> 3) + ((n & 15) * 4);
        Ksw[b] = pf[t].x;
        Ksw[b + 64] = pf[t].y;
        Ksw[b + 128] = pf[t].z;
        Ksw[b + 192] = pf[t].w;
      }
    }
  };

  fetch(0);

  for (int n0 = 0; n0 < kmax; n0 += BN) {
    __syncthreads();
    publish();
    __syncthreads();
    fetch(n0 + BN);

    wmma::fragment<wmma::accumulator, 16, 16, 32, float> acc[MT][NT];
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
      for (int j = 0; j < NT; ++j) wmma::fill_fragment(acc[i][j], 0.f);

#pragma unroll
    for (int c = 0; c < KCH; ++c) {
      wmma::fragment<wmma::matrix_b, 16, 16, 32, __half, wmma::col_major> fb[NT];
#pragma unroll
      for (int j = 0; j < NT; ++j) {
        const uint4 f = kbase[((j) * KCH + c) * 64 + lane];
        fb[j].x[0] = f.x; fb[j].x[1] = f.y; fb[j].x[2] = f.z; fb[j].x[3] = f.w;
      }
#pragma unroll
      for (int i = 0; i < MT; ++i) {
        wmma::fragment<wmma::matrix_a, 16, 16, 32, __half, wmma::row_major> fa;
#pragma unroll
        for (int q = 0; q < 4; ++q) fa.x[q] = qreg[i][c][q];
#pragma unroll
        for (int j = 0; j < NT; ++j) wmma::mma_sync(acc[i][j], fa, fb[j], acc[i][j]);
      }
    }

    const int ccol = lane & 15, crow = lane >> 4;
    if (STORE) {
#pragma unroll
      for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          const int row = wm * (MT * 16) + i * 16 + crow + 4 * q;
          if (!CAUSAL || n0 + ccol <= qbase + row)
            S[(size_t)(qbase + row) * kv + n0 + ccol] = acc[i][0].x[q];
        }
    } else {
#pragma unroll
      for (int i = 0; i < MT; ++i) sink += acc[i][0].x[0];
    }
  }
  if (sink == 1234.5678f) S[0] = sink;
}

int main(int argc, char **argv) {
  const int tok = argc > 1 ? atoi(argv[1]) : 4096;
  const int kv = argc > 2 ? atoi(argv[2]) : 4096;
  const int NH = 16;

  size_t qsz = (size_t)tok * NH * DK;
  __half *hQ = (__half *)malloc(qsz * 2), *hK = (__half *)malloc((size_t)kv * DK * 2);
  srand(11);
  for (size_t i = 0; i < qsz; ++i) hQ[i] = __float2half((rand() % 200 - 100) / 128.0f);
  for (size_t i = 0; i < (size_t)kv * DK; ++i) hK[i] = __float2half((rand() % 200 - 100) / 128.0f);

  __half *dQ, *dK;
  float *dS;
  cudaMalloc(&dQ, qsz * 2);
  cudaMalloc(&dK, (size_t)kv * DK * 2);
  cudaMalloc(&dS, (size_t)tok * kv * 4);
  cudaMemcpy(dQ, hQ, qsz * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(dK, hK, (size_t)kv * DK * 2, cudaMemcpyHostToDevice);

  // causal correctness: rows at or below the diagonal must be exact, and every
  // element above the diagonal must remain untouched (-1 poison).
  {
    const int k2 = 1024;
    { size_t np = (size_t)BM * k2; poison<<<(np + 255) / 256, 256>>>(dS, np); }
    qk<1, 1><<<dim3(1, 1), NTHREAD>>>(dQ, dK, dS, k2, NH);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) { printf("ERR %s\n", cudaGetErrorString(e)); return 1; }
    float *hS = (float *)malloc((size_t)BM * k2 * 4);
    cudaMemcpy(hS, dS, (size_t)BM * k2 * 4, cudaMemcpyDeviceToHost);
    int bad = 0, leak = 0, missing = 0;
    double worst = 0;
    for (int m = 0; m < BM; ++m)
      for (int j = 0; j < k2; ++j) {
        const float got = hS[(size_t)m * k2 + j];
        if (j <= m) {
          if (got == -12345.0f) { ++missing; continue; }
          double ref = 0;
          for (int d = 0; d < DK; ++d)
            ref += (double)__half2float(hQ[(size_t)m * NH * DK + d]) *
                   __half2float(hK[(size_t)j * DK + d]);
          const double er = fabs(got - ref) / (fabs(ref) + 1e-6);
          if (er > worst) worst = er;
          if (er > 3e-2) ++bad;
        } else if (got != -12345.0f) {
          ++leak;  // wrote above the diagonal: causal mask broken
        }
      }
    printf("causal correctness %dx%d: worst %.2e  bad %d  mask-leak %d  missing %d  %s\n", BM, k2,
           worst, bad, leak, missing,
           (bad == 0 && leak == 0 && missing == 0) ? "OK" : "FAIL");
    free(hS);
  }

  dim3 grid(tok / BM, NH);
  auto timeit = [&](const char *tag, auto fn) {
    for (int i = 0; i < 3; ++i) fn();
    cudaEvent_t s, e;
    cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s);
    for (int i = 0; i < 10; ++i) fn();
    cudaEventRecord(e); cudaEventSynchronize(e);
    float ms; cudaEventElapsedTime(&ms, s, e); ms /= 10;
    printf("  %-18s %7.3f ms\n", tag, ms);
    return ms;
  };

  printf("tok=%d kv=%d heads=%d  BM=%d BN=%d\n", tok, kv, NH, BM, BN);
  float full = timeit("full square", [&]() { qk<0, 0><<<grid, NTHREAD>>>(dQ, dK, dS, kv, NH); });
  float caus = timeit("causal", [&]() { qk<1, 0><<<grid, NTHREAD>>>(dQ, dK, dS, kv, NH); });

  // work actually done in the causal case: blocks with n0 < min(kv, qbase+BM)
  double blocks = 0;
  for (int qb = 0; qb < tok; qb += BM) blocks += (double)((qb + BM < kv ? qb + BM : kv) / BN);
  const double causal_flops = 2.0 * blocks * BM * BN * DK * NH;
  const double full_flops = 2.0 * tok * kv * DK * NH;
  printf("  causal does %.1f%% of the full-square work -> %.1f TFLOPS effective\n",
         100 * causal_flops / full_flops, causal_flops / (caus * 1e-3) / 1e12);
  printf("  vendor FA for the same causal 4k layer: 1.304 ms (52.7 TFLOPS incl. softmax+PV)\n");
  return 0;
}
