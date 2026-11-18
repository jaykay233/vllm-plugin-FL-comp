// Verify that the vendor MMA is usable once shared memory is laid out with the
// vendor's own swizzle (CoordToOffset<16, mem_row_major/col_major>), rather than
// plain row-major.  One warp (64 lanes), one 16x16x32 tile, checked on CPU.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

using namespace nvcuda;

__device__ __forceinline__ int sw_row(int m, int k) {
  const int i = k / 16, c = k % 16, r = m / 2;
  return i * 256 + (r / 4) * 128 + ((r * 32 + c * 2 + (m & 1) + i * 64) & 127);
}

__device__ __forceinline__ int sw_col(int k, int n) {  // row=k, column=n
  const int r = k >> 1;
  return (n >> 2) * 128 + (r & 3) * 32 + ((r >> 3) ^ (n >> 3)) * 16 +
         ((r >> 2 & 1) ^ (n >> 2 & 1)) * 8 + (n & 3) * 2 + (k & 1);
}

__global__ void tile(const __half *Qg, const __half *Kg, float *C, int K) {
  __shared__ __half As[512];  // 16 x 32, vendor swizzle
  __shared__ __half Bs[512];  // 32 x 16 (k, n), vendor swizzle
  const int tid = threadIdx.x;

  // A[m][k] = Qg[m*K + k]
  for (int v = tid; v < 16 * 32; v += 64) {
    const int m = v / 32, k = v % 32;
    As[sw_row(m, k)] = Qg[m * K + k];
  }
  // B[k][n] = Kg[n*K + k]   (transpose: B is K x N)
  for (int v = tid; v < 32 * 16; v += 64) {
    const int k = v / 16, n = v % 16;
    Bs[sw_col(k, n)] = Kg[n * K + k];
  }
  __syncthreads();

  wmma::fragment<wmma::matrix_a, 16, 16, 32, __half, wmma::row_major> fa;
  wmma::fragment<wmma::matrix_b, 16, 16, 32, __half, wmma::col_major> fb;
  wmma::fragment<wmma::accumulator, 16, 16, 32, float> acc;
  wmma::fill_fragment(acc, 0.f);
  wmma::load_matrix_sync(fa, &As[0], 0);
  wmma::load_matrix_sync(fb, &Bs[0], 0);
  wmma::mma_sync(acc, fa, fb, acc);
  wmma::store_matrix_sync(C, acc, 16, wmma::mem_row_major);
}

int main() {
  const int K = 32;
  __half *hQ = (__half *)malloc(16 * K * 2), *hK = (__half *)malloc(16 * K * 2);
  srand(3);
  for (int i = 0; i < 16 * K; ++i) {
    hQ[i] = __float2half((rand() % 200 - 100) / 64.0f);
    hK[i] = __float2half((rand() % 200 - 100) / 64.0f);
  }
  __half *dQ, *dK;
  float *dC, hC[256];
  cudaMalloc(&dQ, 16 * K * 2);
  cudaMalloc(&dK, 16 * K * 2);
  cudaMalloc(&dC, 256 * 4);
  cudaMemcpy(dQ, hQ, 16 * K * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(dK, hK, 16 * K * 2, cudaMemcpyHostToDevice);
  tile<<<1, 64>>>(dQ, dK, dC, K);
  cudaError_t e = cudaDeviceSynchronize();
  if (e != cudaSuccess) { printf("ERR %s\n", cudaGetErrorString(e)); return 1; }
  cudaMemcpy(hC, dC, 256 * 4, cudaMemcpyDeviceToHost);

  double worst = 0;
  int bad = 0;
  for (int m = 0; m < 16; ++m)
    for (int n = 0; n < 16; ++n) {
      double ref = 0;
      for (int k = 0; k < K; ++k)
        ref += (double)__half2float(hQ[m * K + k]) * __half2float(hK[n * K + k]);
      double err = fabs(hC[m * 16 + n] - ref) / (fabs(ref) + 1e-6);
      if (err > worst) worst = err;
      if (err > 2e-2) ++bad;
    }
  printf("16x16x32 tile: worst rel err %.2e   bad %d/256   %s\n", worst, bad,
         bad == 0 ? "OK -- vendor MMA usable with swizzled smem" : "MISMATCH");
  printf("sample C[0][0]=%.3f  C[7][9]=%.3f  C[15][15]=%.3f\n", hC[0], hC[7 * 16 + 9], hC[255]);
  return 0;
}
