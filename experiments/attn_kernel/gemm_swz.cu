// Full-size NT GEMM on the vendor MMA with the recovered swizzles.
// Block = 8 warps x 64 lanes = 512 threads; warp grid 2(M) x 4(N); each warp 32x32.
// BM=64 BN=128 BK=32.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

using namespace nvcuda;

#define BM 64
#define BN 128
#define BK 32
#define WM 2
#define WN 4
#define NTHREAD 512

__device__ __forceinline__ int sw_row(int m, int k) {
  const int i = k / 16, c = k % 16, r = m / 2;
  return i * 256 + (r / 4) * 128 + ((r * 32 + c * 2 + (m & 1) + i * 64) & 127);
}
__device__ __forceinline__ int sw_col(int k, int n) {
  const int r = k >> 1;
  return (n >> 2) * 128 + (r & 3) * 32 + ((r >> 3) ^ (n >> 3)) * 16 +
         ((r >> 2 & 1) ^ (n >> 2 & 1)) * 8 + (n & 3) * 2 + (k & 1);
}

__global__ __launch_bounds__(NTHREAD) void gemm(const __half *A, const __half *Bm, float *C,
                                                int M, int N, int K) {
  // A: (BM/16)=4 tiles of 16x32 ; B: (BN/16)=8 tiles of 32x16
  __shared__ __half As[BM * BK];
  __shared__ __half Bs[BN * BK];

  const int bm = blockIdx.y * BM, bn = blockIdx.x * BN;
  const int tid = threadIdx.x;
  const int warp = tid >> 6;
  const int wm = warp / WN, wn = warp % WN;

  wmma::fragment<wmma::accumulator, 16, 16, 32, float> acc[2][2];
#pragma unroll
  for (int i = 0; i < 2; ++i)
#pragma unroll
    for (int j = 0; j < 2; ++j) wmma::fill_fragment(acc[i][j], 0.f);

  wmma::fragment<wmma::matrix_a, 16, 16, 32, __half, wmma::row_major> fa[2];
  wmma::fragment<wmma::matrix_b, 16, 16, 32, __half, wmma::col_major> fb[2];

  const int a_vec = BM * BK / 8;  // 256
  const int b_vec = BN * BK / 8;  // 512

  for (int k0 = 0; k0 < K; k0 += BK) {
    for (int v = tid; v < a_vec; v += NTHREAD) {
      const int m = v / (BK / 8), k = (v % (BK / 8)) * 8;
      ulonglong2 raw = *(const ulonglong2 *)(A + (size_t)(bm + m) * K + k0 + k);
      const __half *h = (const __half *)&raw;
      const int base = (m / 16) * 512;
#pragma unroll
      for (int t = 0; t < 8; ++t) As[base + sw_row(m % 16, k + t)] = h[t];
    }
    for (int v = tid; v < b_vec; v += NTHREAD) {
      const int n = v / (BK / 8), k = (v % (BK / 8)) * 8;
      ulonglong2 raw = *(const ulonglong2 *)(Bm + (size_t)(bn + n) * K + k0 + k);
      const __half *h = (const __half *)&raw;
      const int base = (n / 16) * 512;
#pragma unroll
      for (int t = 0; t < 8; ++t) Bs[base + sw_col(k + t, n % 16)] = h[t];
    }
    __syncthreads();

#pragma unroll
    for (int i = 0; i < 2; ++i)
      wmma::load_matrix_sync(fa[i], &As[(wm * 2 + i) * 512], 0);
#pragma unroll
    for (int j = 0; j < 2; ++j)
      wmma::load_matrix_sync(fb[j], &Bs[(wn * 2 + j) * 512], 0);
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
      for (int j = 0; j < 2; ++j) wmma::mma_sync(acc[i][j], fa[i], fb[j], acc[i][j]);
    __syncthreads();
  }

  // Epilogue: store_matrix_sync ignores ldm and always writes a contiguous
  // 16x16 block, so write from registers using the mapping read out of
  // __imma_st_row_b32:  lane L, reg q  ->  row = L/16 + 4q, col = L%16.
  const int lane = tid & 63;
  const int ccol = lane & 15;
  const int crow = lane >> 4;
#pragma unroll
  for (int i = 0; i < 2; ++i)
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      const int r0 = bm + wm * 32 + i * 16;
      const int c0 = bn + wn * 32 + j * 16;
#pragma unroll
      for (int q = 0; q < 4; ++q)
        C[(size_t)(r0 + crow + 4 * q) * N + c0 + ccol] = acc[i][j].x[q];
    }
}

int main(int argc, char **argv) {
  int n = argc > 1 ? atoi(argv[1]) : 2048;
  int iters = argc > 2 ? atoi(argv[2]) : 20;
  size_t sz = (size_t)n * n;
  __half *hA = (__half *)malloc(sz * 2), *hB = (__half *)malloc(sz * 2);
  srand(5);
  for (size_t i = 0; i < sz; ++i) {
    hA[i] = __float2half((rand() % 200 - 100) / 128.0f);
    hB[i] = __float2half((rand() % 200 - 100) / 128.0f);
  }
  __half *dA, *dB;
  float *dC;
  cudaMalloc(&dA, sz * 2);
  cudaMalloc(&dB, sz * 2);
  cudaMalloc(&dC, sz * 4);
  cudaMemcpy(dA, hA, sz * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(dB, hB, sz * 2, cudaMemcpyHostToDevice);

  dim3 grid(n / BN, n / BM);
  auto run = [&]() { gemm<<<grid, NTHREAD>>>(dA, dB, dC, n, n, n); };
  run();
  cudaError_t e = cudaDeviceSynchronize();
  if (e != cudaSuccess) { printf("ERR %s\n", cudaGetErrorString(e)); return 1; }

  if (n <= 2048) {
    float *hC = (float *)malloc(sz * 4);
    cudaMemcpy(hC, dC, sz * 4, cudaMemcpyDeviceToHost);
    // report per 16x16 tile so a bad warp / tile index is visible
    printf("bad-tile map (row tile m/16 x col tile n/16), '.'=ok 'X'=bad:\n");
    int nt = n / 16;
    for (int ti = 0; ti < nt && ti < 8; ++ti) {
      printf("  mt%02d:", ti);
      for (int tj = 0; tj < nt && tj < 16; ++tj) {
        int bad = 0;
        for (int a = 0; a < 16; ++a)
          for (int b = 0; b < 16; ++b) {
            int i = ti * 16 + a, j = tj * 16 + b;
            double ref = 0;
            for (int k = 0; k < n; ++k)
              ref += (double)__half2float(hA[(size_t)i * n + k]) *
                     __half2float(hB[(size_t)j * n + k]);
            if (fabs(hC[(size_t)i * n + j] - ref) / (fabs(ref) + 1e-6) > 3e-2) ++bad;
          }
        printf(" %s", bad ? "X" : ".");
      }
      printf("\n");
    }
    free(hC);
  }

  for (int i = 0; i < 3; ++i) run();
  cudaEvent_t s, ev;
  cudaEventCreate(&s); cudaEventCreate(&ev);
  cudaEventRecord(s);
  for (int i = 0; i < iters; ++i) run();
  cudaEventRecord(ev);
  cudaEventSynchronize(ev);
  float ms; cudaEventElapsedTime(&ms, s, ev); ms /= iters;
  double tf = 2.0 * (double)n * n * n / (ms * 1e-3) / 1e12;
  printf("swizzled NT gemm %dx%dx%d: %.3f ms  %.1f TFLOPS  (%.0f%% of ixBLAS)\n", n, n, n, ms, tf,
         100.0 * tf / 96.6);
  return 0;
}
