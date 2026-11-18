// QK^T with a cp.async double-buffered pipeline.
//
// The previous version stalled at ~27 TFLOPS because occupancy and instruction
// efficiency were pinned against each other by the register budget.  With
// __ivcorex_cp_async_ca_shared_global_16 the K staging can leave the critical
// path entirely, so latency hiding stops depending on warp count.
//
// Pipeline (single __syncthreads per iteration, one group in flight):
//   sync            <- everyone finished reading buf[(s+1)%2] last iteration
//   issue(s+1)      <- lands in buf[(s+1)%2], overlaps the compute below
//   wait_group(1)   <- group s complete, group s+1 may still be running
//   compute(buf[s%2])
//
// Ks row stride is DK+8 halves = 272 B = 17 x 16 B, which keeps every cp.async
// destination 16-byte aligned.  That costs a 2-way bank conflict on the B
// fragment loads versus the +2 padding, which is the right trade here.
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
#define WMUL 2
#define WNUL 4
#define NWARP (WMUL * WNUL)
#define NTHREAD (NWARP * 64)
#define KS (DK + 8)  // 16 B-aligned row stride
#define KCH (DK / 32)
#define MT 4
#define NSTAGE 2

typedef unsigned int v4u __attribute__((vector_size(16)));

__global__ __launch_bounds__(NTHREAD) void qk(const __half *Q, const __half *K, float *S,
                                              int kv, int NH, int store_s) {
  __shared__ __half Ks[NSTAGE][BN][KS];

  const int tid = threadIdx.x;
  const int lane = tid & 63;
  const int warp = tid >> 6;
  const int wm = warp / WNUL;
  const int wn = warp % WNUL;
  const int tile = blockIdx.x;
  const int head = blockIdx.y;
  const int qbase = tile * BM;
  const int nb = kv / BN;

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

  constexpr int PV = BN * DK / 8 / NTHREAD;  // 16 B units per thread per block
  float sink = 0.f;

  auto issue = [&](int blk, int st) {
#pragma unroll
    for (int t = 0; t < PV; ++t) {
      const int v = tid + t * NTHREAD;
      const int n = (v * 8) / DK, k = (v * 8) % DK;
      __ivcorex_cp_async_ca_shared_global_16(
          (v4u *)&Ks[st][n][k],
          (const v4u *)&K[(size_t)(blk * BN + n) * DK + k]);
    }
    __ivcorex_cp_async_commit_group();
  };

  issue(0, 0);

  for (int s = 0; s < nb; ++s) {
    __syncthreads();  // buf[(s+1)%2] is free: last read was iteration s-1
    if (s + 1 < nb) issue(s + 1, (s + 1) % NSTAGE);
    __ivcorex_cp_async_wait_group(1);
    __syncthreads();

    const int st = s % NSTAGE;

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
        pb[q] = *(const unsigned int *)&Ks[st][n][k];
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
          S[(size_t)(qbase + wm * 64 + i * 16 + crow + 4 * q) * kv + s * BN + wn * 16 + ccol] =
              acc[i].x[q];
    } else {
#pragma unroll
      for (int i = 0; i < MT; ++i) sink += acc[i].x[0];
    }
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
