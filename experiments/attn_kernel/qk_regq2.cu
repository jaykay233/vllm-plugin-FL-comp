// QK^T, final shape of the design: Q in registers, K in shared, warp tile 64x16.
//
// Reasoning from the previous measurement: with Q register-resident the only
// in-loop loads are the B fragments, so ld.shared per MMA is (N tiles)/(M*N
// tiles) = 1/M_tiles.  A 16-row warp tile therefore costs 2 loads per MMA and
// MMA occupies only ~26% of issue slots, which is exactly the efficiency seen.
// Widening to 64 rows drops it to 1 load per MMA at no shared-memory cost.
//
//   BM=128 BN=64 DK=128, 8 warps as 2(M) x 4(N), warp tile 64M x 16N
//   qreg = 4 M-tiles x 4 k-chunks x 4 uints = 64 registers per lane
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
#define WMUL 2              // M groups (2 x 64 rows = 128)
#define WNUL 4              // N groups (4 x 16 cols = 64)
#define NWARP (WMUL * WNUL) // 8
#define NTHREAD (NWARP * 64)
#define KS (DK + 2)
#define KCH (DK / 32)
#define MT 4  // M tiles per warp = 64 rows

__global__ __launch_bounds__(NTHREAD) void qk(const __half *Q, const __half *K, float *S,
                                              int kv, int NH, int store_s) {
  __shared__ __half Ks[BN][KS];

  const int tid = threadIdx.x;
  const int lane = tid & 63;
  const int warp = tid >> 6;
  const int wm = warp / WNUL;  // 0..1
  const int wn = warp % WNUL;  // 0..3
  const int tile = blockIdx.x;
  const int head = blockIdx.y;
  const int qbase = tile * BM;

  unsigned int qreg[MT][KCH][4];
#pragma unroll
  for (int i = 0; i < MT; ++i)
#pragma unroll
    for (int c = 0; c < KCH; ++c)
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const int m = wm * 64 + i * 16 + (lane / 16) * 2 + 8 * (q % 2);
        const int k = c * 32 + (lane % 16) + 16 * (q / 2);
        __half2 h2;
        h2.x = Q[(size_t)(qbase + m) * NH * DK + (size_t)head * DK + k];
        h2.y = Q[(size_t)(qbase + m + 1) * NH * DK + (size_t)head * DK + k];
        qreg[i][c][q] = *(unsigned int *)&h2;
      }
  __syncthreads();

  constexpr int PV = BN * DK / 8 / NTHREAD;
  ulonglong2 pf[PV];
  auto fetch = [&](int n0) {
#pragma unroll
    for (int t = 0; t < PV; ++t) {
      const int v = tid + t * NTHREAD;
      pf[t] = *(const ulonglong2 *)&K[(size_t)(n0 + (v * 8) / DK) * DK + (v * 8) % DK];
    }
  };
  float sink = 0.f;
  fetch(0);

  for (int n0 = 0; n0 < kv; n0 += BN) {
#pragma unroll
    for (int t = 0; t < PV; ++t) {
      const int v = tid + t * NTHREAD;
      *(ulonglong2 *)&Ks[(v * 8) / DK][(v * 8) % DK] = pf[t];
    }
    __syncthreads();
    if (n0 + BN < kv) fetch(n0 + BN);

    wmma::fragment<wmma::accumulator, 16, 16, 32, float> acc[MT];
#pragma unroll
    for (int i = 0; i < MT; ++i) wmma::fill_fragment(acc[i], 0.f);

#pragma unroll
    for (int c = 0; c < KCH; ++c) {
      wmma::fragment<wmma::matrix_b, 16, 16, 32, __half, wmma::col_major> fb;
      unsigned int *pb = (unsigned int *)&fb.x[0];
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const int k = c * 32 + (lane / 16) * 2 + 8 * q;
        const int n = wn * 16 + (lane % 16);
        pb[q] = *(const unsigned int *)&Ks[n][k];
      }
#pragma unroll
      for (int i = 0; i < MT; ++i) {
        wmma::fragment<wmma::matrix_a, 16, 16, 32, __half, wmma::row_major> fa;
#pragma unroll
        for (int q = 0; q < 4; ++q) fa.x[q] = qreg[i][c][q];
        wmma::mma_sync(acc[i], fa, fb, acc[i]);
      }
    }

    const int ccol = lane & 15, crow = lane >> 4;
    if (store_s) {
#pragma unroll
      for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int q = 0; q < 4; ++q)
          S[(size_t)(qbase + wm * 64 + i * 16 + crow + 4 * q) * kv + n0 + wn * 16 + ccol] =
              acc[i].x[q];
    } else {
#pragma unroll
      for (int i = 0; i < MT; ++i) sink += acc[i].x[0];
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
    const int k2 = 256, t2 = BM;
    qk<<<dim3(tok / BM, 1), NTHREAD>>>(dQ, dK, dS, k2, NH, 1);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) { printf("ERR %s\n", cudaGetErrorString(e)); return 1; }
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
