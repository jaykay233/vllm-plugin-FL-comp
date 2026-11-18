// QK^T with a 32x32 warp tile so each shared-memory fragment feeds four MMAs
// instead of one.  Fragment loads are 256 B per warp each (64 lanes x 4 B), so
// 16x16 tiles are smem-bound; the register blocking is what fixes it.
//
//   BM=128 BN=64 DK=128, 8 warps as 4(M) x 2(N), warp tile 32M x 32N
//   Qs[k][m] stride BM+2, Ks[n][k] stride DK+2  (odd word stride => no conflicts)
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
#define QS (BM + 2)
#define KS (DK + 2)

__global__ __launch_bounds__(NTHREAD) void qk(const __half *Q, const __half *K, float *S,
                                              int kv, int NH, int store_s) {
  __shared__ __half Qs[DK][QS];
  __shared__ __half Ks[BN][KS];

  const int tid = threadIdx.x;
  const int lane = tid & 63;
  const int warp = tid >> 6;
  const int wm = warp >> 1;   // 0..3 -> 32 rows
  const int wn = warp & 1;    // 0..1  -> 32 cols
  const int tile = blockIdx.x;
  const int head = blockIdx.y;
  const int qbase = tile * BM;

  for (int v = tid; v < BM * DK; v += NTHREAD) {
    const int m = v / DK, d = v % DK;
    Qs[d][m] = Q[(size_t)(qbase + m) * NH * DK + (size_t)head * DK + d];
  }
  __syncthreads();

  float sink = 0.f;

  for (int n0 = 0; n0 < kv; n0 += BN) {
    for (int v = tid; v < BN * DK / 8; v += NTHREAD) {
      *(ulonglong2 *)&Ks[(v * 8) / DK][(v * 8) % DK] =
          *(const ulonglong2 *)&K[(size_t)(n0 + (v * 8) / DK) * DK + (v * 8) % DK];
    }
    __syncthreads();

    wmma::fragment<wmma::accumulator, 16, 16, 32, float> acc[2][2];
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
      for (int j = 0; j < 2; ++j) wmma::fill_fragment(acc[i][j], 0.f);

    wmma::fragment<wmma::matrix_a, 16, 16, 32, __half, wmma::row_major> fa[2];
    wmma::fragment<wmma::matrix_b, 16, 16, 32, __half, wmma::col_major> fb[2];

    for (int kc = 0; kc < DK; kc += 32) {
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        unsigned int *pa = (unsigned int *)&fa[i].x[0];
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          const int m = wm * 32 + i * 16 + (lane / 16) * 2 + 8 * (q % 2);
          const int k = kc + (lane % 16) + 16 * (q / 2);
          pa[q] = *(const unsigned int *)&Qs[k][m];
        }
      }
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        unsigned int *pb = (unsigned int *)&fb[j].x[0];
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          const int k = kc + (lane / 16) * 2 + 8 * q;
          const int n = wn * 32 + j * 16 + (lane % 16);
          pb[q] = *(const unsigned int *)&Ks[n][k];
        }
      }
#pragma unroll
      for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j) wmma::mma_sync(acc[i][j], fa[i], fb[j], acc[i][j]);
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
    const int t2 = 128, k2 = 256;
    dim3 g(t2 / BM > 0 ? t2 / BM : 1, 1);
    qk<<<g, NTHREAD>>>(dQ, dK, dS, k2, NH, 1);
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
