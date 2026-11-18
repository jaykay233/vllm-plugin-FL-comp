// QK^T with Q resident in registers.
//
// Q does not change across the KV loop, so its fragments can be built once and
// kept.  That removes every in-loop A load (half of all ld.shared), deletes the
// Q shared-memory buffer entirely (33 KB freed at BM=128), and leaves shared
// memory holding only the K block.
//
// The A fragment uint packs (m, k) and (m+1, k), so building it straight from
// global memory costs two 2-byte reads per uint -- paid once, not per KV block.
//
//   BM=128 BN=64 DK=128, 8 warps as 4(M) x 2(N), warp tile 32x32
//   Q: 2 M-tiles x 4 k-chunks x 4 uints = 32 registers per lane
//   smem: K only, BN x (DK+2) x 2 B = 16.6 KB
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

using namespace nvcuda;

#define BM 128
#define BN 64
#define DK 128
#define NWARP 8
#define NTHREAD (NWARP * 64)
#define KS (DK + 2)
#define KCH (DK / 32)  // 4 k-chunks of 32

__global__ __launch_bounds__(NTHREAD) void qk(const __half *Q, const __half *K, float *S,
                                              int kv, int NH, int store_s) {
  __shared__ __half Ks[BN][KS];

  const int tid = threadIdx.x;
  const int lane = tid & 63;
  const int warp = tid >> 6;
  const int wm = warp >> 1;
  const int wn = warp & 1;
  const int tile = blockIdx.x;
  const int head = blockIdx.y;
  const int qbase = tile * BM;

  // ---- build Q fragments once, straight from global ----
  // fragment index within a tile: m = (lane/16)*2 + 8*(q%2), k = (lane%16) + 16*(q/2)
  unsigned int qreg[2][KCH][4];
#pragma unroll
  for (int i = 0; i < 2; ++i)
#pragma unroll
    for (int c = 0; c < KCH; ++c)
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const int m = wm * 32 + i * 16 + (lane / 16) * 2 + 8 * (q % 2);
        const int k = c * 32 + (lane % 16) + 16 * (q / 2);
        const __half lo = Q[(size_t)(qbase + m) * NH * DK + (size_t)head * DK + k];
        const __half hi = Q[(size_t)(qbase + m + 1) * NH * DK + (size_t)head * DK + k];
        __half2 h2;
        h2.x = lo;
        h2.y = hi;
        qreg[i][c][q] = *(unsigned int *)&h2;
      }
  __syncthreads();

  float sink = 0.f;

  // Register prefetch: one K block is BN*DK/8 = 1024 vec8 units, spread over
  // NTHREAD=512 threads => 2 ulonglong2 per thread.  Loading the next block into
  // registers while the current one feeds the MMAs keeps global latency off the
  // critical path without needing async copy or a second smem stage.
  constexpr int PV = BN * DK / 8 / NTHREAD;  // 2
  ulonglong2 pf[PV];
  auto fetch = [&](int n0) {
#pragma unroll
    for (int t = 0; t < PV; ++t) {
      const int v = tid + t * NTHREAD;
      pf[t] = *(const ulonglong2 *)&K[(size_t)(n0 + (v * 8) / DK) * DK + (v * 8) % DK];
    }
  };
  fetch(0);

  for (int n0 = 0; n0 < kv; n0 += BN) {
#pragma unroll
    for (int t = 0; t < PV; ++t) {
      const int v = tid + t * NTHREAD;
      *(ulonglong2 *)&Ks[(v * 8) / DK][(v * 8) % DK] = pf[t];
    }
    __syncthreads();
    if (n0 + BN < kv) fetch(n0 + BN);

    wmma::fragment<wmma::accumulator, 16, 16, 32, float> acc[2][2];
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
      for (int j = 0; j < 2; ++j) wmma::fill_fragment(acc[i][j], 0.f);

#pragma unroll
    for (int c = 0; c < KCH; ++c) {
      wmma::fragment<wmma::matrix_b, 16, 16, 32, __half, wmma::col_major> fb[2];
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        unsigned int *pb = (unsigned int *)&fb[j].x[0];
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          const int k = c * 32 + (lane / 16) * 2 + 8 * q;
          const int n = wn * 32 + j * 16 + (lane % 16);
          pb[q] = *(const unsigned int *)&Ks[n][k];
        }
      }
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        wmma::fragment<wmma::matrix_a, 16, 16, 32, __half, wmma::row_major> fa;
#pragma unroll
        for (int q = 0; q < 4; ++q) fa.x[q] = qreg[i][c][q];
#pragma unroll
        for (int j = 0; j < 2; ++j) wmma::mma_sync(acc[i][j], fa, fb[j], acc[i][j]);
      }
    }

    const int ccol = lane & 15, crow = lane >> 4;
    if (store_s) {
#pragma unroll
      for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j)
#pragma unroll
          for (int q = 0; q < 4; ++q)
            S[(size_t)(qbase + wm * 32 + i * 16 + crow + 4 * q) * kv + n0 + wn * 32 + j * 16 +
              ccol] = acc[i][j].x[q];
    } else {
#pragma unroll
      for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j) sink += acc[i][j].x[0];
    }
    __syncthreads();
  }
  if (sink == 1234.5678f) S[0] = sink;
}

int main(int argc, char **argv) {
  const int tok = argc > 1 ? atoi(argv[1]) : 4096;
  const int kv = argc > 2 ? atoi(argv[2]) : 4096;
  const int iters = argc > 3 ? atoi(argv[3]) : 10;
  const int store_s = argc > 4 ? atoi(argv[4]) : 0;
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

  {
    const int k2 = 256;
    qk<<<dim3(tok / BM, 1), NTHREAD>>>(dQ, dK, dS, k2, NH, 1);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) { printf("ERR %s\n", cudaGetErrorString(e)); return 1; }
    int t2 = BM;
    float *hS = (float *)malloc((size_t)t2 * k2 * 4);
    cudaMemcpy(hS, dS, (size_t)t2 * k2 * 4, cudaMemcpyDeviceToHost);
    double worst = 0; int bad = 0;
    for (int m = 0; m < t2; ++m)
      for (int n = 0; n < k2; ++n) {
        double ref = 0;
        for (int d = 0; d < DK; ++d)
          ref += (double)__half2float(hQ[(size_t)m * NH * DK + d]) *
                 __half2float(hK[(size_t)n * DK + d]);
        double er = fabs(hS[(size_t)m * k2 + n] - ref) / (fabs(ref) + 1e-6);
        if (er > worst) worst = er;
        if (er > 3e-2) ++bad;
      }
    printf("correctness (head 0, %dx%d): worst rel err %.2e bad %d %s\n", t2, k2, worst, bad,
           bad == 0 ? "CORRECT" : "MISMATCH");
    free(hS);
  }

  dim3 grid(tok / BM, NH);
  auto run = [&]() { qk<<<grid, NTHREAD>>>(dQ, dK, dS, kv, NH, store_s); };
  for (int i = 0; i < 3; ++i) run();
  cudaEvent_t s, ev;
  cudaEventCreate(&s); cudaEventCreate(&ev);
  cudaEventRecord(s);
  for (int i = 0; i < iters; ++i) run();
  cudaEventRecord(ev);
  cudaEventSynchronize(ev);
  float ms; cudaEventElapsedTime(&ms, s, ev); ms /= iters;
  double fl = 2.0 * tok * kv * DK * NH;
  double tf = fl / (ms * 1e-3) / 1e12;
  int nb = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, qk, NTHREAD, 0);
  cudaFuncAttributes at;
  cudaFuncGetAttributes(&at, qk);
  printf("tok=%d kv=%d store_s=%d: %.3f ms  %.1f TFLOPS (%.0f%%) | %d CTA/SM=%d warps | smem %zuB regs %d\n",
         tok, kv, store_s, ms, tf, 100 * tf / 96.6, nb, nb * NWARP,
         (size_t)at.sharedSizeBytes, at.numRegs);
  return 0;
}
