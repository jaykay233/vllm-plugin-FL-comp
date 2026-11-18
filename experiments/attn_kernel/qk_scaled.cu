// Scaled QK^T: full prefill shape (tokens x 16 heads x kv) across all SMs, so
// the number can be compared directly against the vendor FA kernel.
// Same fragment assembly as qk_proto.cu, which is verified to 0.00e+00.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

using namespace nvcuda;

#define BM 64
#define BN 64
#define DK 128
#define NWARP 4
#define NTHREAD (NWARP * 64)

// Q: [tokens][NH][DK]   K: [kv][DK]  (one kv head; GQA group)   S: [tokens][kv]
__global__ __launch_bounds__(NTHREAD) void qk(const __half *Q, const __half *K, float *S,
                                              int kv, int NH) {
  __shared__ __half Qs[DK][BM];
  __shared__ __half Ks[BN][DK];

  const int tid = threadIdx.x;
  const int lane = tid & 63;
  const int warp = tid >> 6;
  const int tile = blockIdx.x;   // query tile
  const int head = blockIdx.y;
  const int mbase = tile * BM + warp * 16;
  const int qbase = tile * BM;

  for (int v = tid; v < BM * DK; v += NTHREAD) {
    const int m = v / DK, d = v % DK;
    Qs[d][m] = Q[(size_t)(qbase + m) * NH * DK + (size_t)head * DK + d];
  }
  __syncthreads();

  for (int n0 = 0; n0 < kv; n0 += BN) {
    for (int v = tid; v < BN * DK; v += NTHREAD) {
      const int n = v / DK, d = v % DK;
      Ks[n][d] = K[(size_t)(n0 + n) * DK + d];
    }
    __syncthreads();

    wmma::fragment<wmma::accumulator, 16, 16, 32, float> acc[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) wmma::fill_fragment(acc[j], 0.f);

    for (int kc = 0; kc < DK; kc += 32) {
      wmma::fragment<wmma::matrix_a, 16, 16, 32, __half, wmma::row_major> fa;
      unsigned int *pa = (unsigned int *)&fa.x[0];
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const int m = (lane / 16) * 2 + 8 * (q % 2);
        const int k = kc + (lane % 16) + 16 * (q / 2);
        pa[q] = *(const unsigned int *)&Qs[k][warp * 16 + m];
      }
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        wmma::fragment<wmma::matrix_b, 16, 16, 32, __half, wmma::col_major> fb;
        unsigned int *pb = (unsigned int *)&fb.x[0];
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          const int k = kc + (lane / 16) * 2 + 8 * q;
          const int n = j * 16 + (lane % 16);
          pb[q] = *(const unsigned int *)&Ks[n][k];
        }
        wmma::mma_sync(acc[j], fa, fb, acc[j]);
      }
    }

    const int ccol = lane & 15, crow = lane >> 4;
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
      for (int q = 0; q < 4; ++q)
        S[(size_t)(mbase + crow + 4 * q) * kv + n0 + j * 16 + ccol] = acc[j].x[q];
    __syncthreads();
  }
}

int main(int argc, char **argv) {
  const int tok = argc > 1 ? atoi(argv[1]) : 4096;
  const int kv = argc > 2 ? atoi(argv[2]) : 4096;
  const int NH = 16;
  const int iters = argc > 3 ? atoi(argv[3]) : 10;

  __half *hQ = (__half *)malloc((size_t)tok * NH * DK * 2);
  __half *hK = (__half *)malloc((size_t)kv * DK * 2);
  srand(11);
  for (size_t i = 0; i < (size_t)tok * NH * DK; ++i) hQ[i] = __float2half((rand() % 200 - 100) / 128.0f);
  for (size_t i = 0; i < (size_t)kv * DK; ++i) hK[i] = __float2half((rand() % 200 - 100) / 128.0f);

  __half *dQ, *dK;
  float *dS;
  cudaMalloc(&dQ, (size_t)tok * NH * DK * 2);
  cudaMalloc(&dK, (size_t)kv * DK * 2);
  cudaMalloc(&dS, (size_t)tok * kv * 4);
  cudaMemcpy(dQ, hQ, (size_t)tok * NH * DK * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(dK, hK, (size_t)kv * DK * 2, cudaMemcpyHostToDevice);

  dim3 grid(tok / BM, NH);
  auto run = [&]() { qk<<<grid, NTHREAD>>>(dQ, dK, dS, kv, NH); };
  run();
  cudaError_t e = cudaDeviceSynchronize();
  if (e != cudaSuccess) { printf("ERR %s\n", cudaGetErrorString(e)); return 1; }

  // spot check
  float *hS = (float *)malloc((size_t)tok * kv * 4);
  cudaMemcpy(hS, dS, (size_t)tok * kv * 4, cudaMemcpyDeviceToHost);
  double worst = 0;
  int bad = 0;
  for (int m = 0; m < tok; m += 97)
    for (int n = 0; n < kv; n += 131) {
      int h = m % NH;
      double ref = 0;
      for (int d = 0; d < DK; ++d)
        ref += (double)__half2float(hQ[((size_t)m * NH + h) * DK + d]) *
               __half2float(hK[(size_t)n * DK + d]);
      double er = fabs(hS[(size_t)m * kv + n] - ref) / (fabs(ref) + 1e-6);
      if (er > worst) worst = er;
      if (er > 3e-2) ++bad;
    }
  printf("scaled QK^T tok=%d kv=%d heads=%d: worst rel err %.2e bad %d %s\n", tok, kv, NH, worst,
         bad, bad == 0 ? "CORRECT" : "MISMATCH");
  free(hS);

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
  printf("  %.3f ms  %.1f TFLOPS  (%.0f%% of 96.6)\n", ms, tf, 100 * tf / 96.6);
  printf("  note: this includes the S write to HBM (%d MB), which a fused FA avoids\n",
         (int)((double)tok * kv * 4 / 1e6));
  return 0;
}
