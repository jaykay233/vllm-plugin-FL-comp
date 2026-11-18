// Prototype: fused QK^T for one GQA head at a real prefill shape, on the
// BI-V150 native MMA.  Purpose is (a) to prove the recovered fragment mappings
// assemble a correct S matrix, and (b) to measure what the MMA path actually
// delivers once shared memory uses a plain layout instead of the vendor swizzle.
//
// Fragment mappings (derived from __hmma_ld_row_b16 / __hmma_ld_col_b16 by
// dropping the swizzle).  Each fragment uint holds a pair of halves.
//
//   matrix_a (16,16,32) row_major: uint q covers rows (m, m+1) at column k
//       m = (L/16)*2 + 8*(q%2)      k = (L%16) + 16*(q/2)
//     -> pair runs along M, so A must be staged k-major: As[k][m]
//
//   matrix_b (16,16,32) col_major: uint q covers rows (k, k+1) at column n
//       k = (L/16)*2 + 8*q          n = (L%16)
//     -> pair runs along K, so B must be staged n-major: Bs[n][k]
//     and for Q@K^T the reduction index is head_dim, which is already the
//     contiguous axis of the KV cache, so Bs is a straight copy.
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

__global__ __launch_bounds__(NTHREAD) void qk(const __half *Q, const __half *K, float *S,
                                              int kv) {
  __shared__ __half Qs[DK][BM];  // k-major for the A fragment
  __shared__ __half Ks[BN][DK];  // n-major for the B fragment

  const int tid = threadIdx.x;
  const int lane = tid & 63;
  const int warp = tid >> 6;
  const int mbase = warp * 16;  // 16 rows of S per warp

  // stage all of Q (reused across every KV block)
  for (int v = tid; v < BM * DK; v += NTHREAD) {
    const int m = v / DK, d = v % DK;
    Qs[d][m] = Q[m * DK + d];
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
        pa[q] = *(const unsigned int *)&Qs[k][mbase + m];
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

    // epilogue: lane L, reg q -> row L/16 + 4q, col L%16
    const int ccol = lane & 15;
    const int crow = lane >> 4;
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
      for (int q = 0; q < 4; ++q)
        S[(size_t)(mbase + crow + 4 * q) * kv + n0 + j * 16 + ccol] = acc[j].x[q];

    __syncthreads();
  }
}

int main(int argc, char **argv) {
  const int kv = argc > 1 ? atoi(argv[1]) : 512;
  const int iters = argc > 2 ? atoi(argv[2]) : 20;

  __half *hQ = (__half *)malloc(BM * DK * 2), *hK = (__half *)malloc((size_t)kv * DK * 2);
  float *hS = (float *)malloc((size_t)BM * kv * 4);
  srand(11);
  for (int i = 0; i < BM * DK; ++i) hQ[i] = __float2half((rand() % 200 - 100) / 128.0f);
  for (size_t i = 0; i < (size_t)kv * DK; ++i) hK[i] = __float2half((rand() % 200 - 100) / 128.0f);

  __half *dQ, *dK;
  float *dS;
  cudaMalloc(&dQ, BM * DK * 2);
  cudaMalloc(&dK, (size_t)kv * DK * 2);
  cudaMalloc(&dS, (size_t)BM * kv * 4);
  cudaMemcpy(dQ, hQ, BM * DK * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(dK, hK, (size_t)kv * DK * 2, cudaMemcpyHostToDevice);

  auto run = [&]() { qk<<<1, NTHREAD>>>(dQ, dK, dS, kv); };
  run();
  cudaError_t e = cudaDeviceSynchronize();
  if (e != cudaSuccess) { printf("ERR %s\n", cudaGetErrorString(e)); return 1; }
  cudaMemcpy(hS, dS, (size_t)BM * kv * 4, cudaMemcpyDeviceToHost);

  double worst = 0;
  int bad = 0;
  for (int m = 0; m < BM; ++m)
    for (int n = 0; n < kv; n += 13) {
      double ref = 0;
      for (int d = 0; d < DK; ++d)
        ref += (double)__half2float(hQ[m * DK + d]) * __half2float(hK[(size_t)n * DK + d]);
      double er = fabs(hS[(size_t)m * kv + n] - ref) / (fabs(ref) + 1e-6);
      if (er > worst) worst = er;
      if (er > 3e-2) ++bad;
    }
  printf("QK^T BM=%d BN=%d D=%d kv=%d : worst rel err %.2e  bad %d\n", BM, BN, DK, kv, worst, bad);
  printf("  %s\n", bad == 0 ? "CORRECT" : "MISMATCH");

  for (int i = 0; i < 3; ++i) run();
  cudaEvent_t s, ev;
  cudaEventCreate(&s); cudaEventCreate(&ev);
  cudaEventRecord(s);
  for (int i = 0; i < iters; ++i) run();
  cudaEventRecord(ev);
  cudaEventSynchronize(ev);
  float ms; cudaEventElapsedTime(&ms, s, ev); ms /= iters;
  double fl = 2.0 * BM * kv * DK;
  printf("  %.4f ms  %.1f TFLOPS  (%.0f%% of 96.6)  [single CTA, 4 warps: occupancy-limited]\n",
         ms, fl / (ms * 1e-3) / 1e12, 100 * fl / (ms * 1e-3) / 1e12 / 96.6);
  return 0;
}
