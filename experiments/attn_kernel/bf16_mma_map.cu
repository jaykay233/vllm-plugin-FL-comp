// bf16 16x16x16 fragment mapping for the raw intrinsic, verified against CPU.
//
// The vendor wmma header only wraps __half, and the model is bf16, so the kernel
// uses __ivcorex_matrix_mad_f32x4_bf16x4 directly.  That needs the element mapping
// stated explicitly rather than hidden behind a fragment class.
//
// Derived by dropping the k=32 fragment's second MMA pass (the wmma
// fragment<16,16,32> lowers to exactly two MMA, so each covers k=0..15 then
// k=16..31):
//
//   operand (both A and B), lane L, element j:
//       row = 2*(L/16) + 8*(j/2) + (j%2)      col = L%16
//   accumulator, lane L, register q:
//       row = L/16 + 4*q                       col = L%16
//
// A is (m,k) and B is (k,n) with the same "k-major pair" pattern, which is why
// staging both k-major works for Q@K^T and for P@V once V is transposed.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

typedef __bf16 v4y __attribute__((vector_size(8)));
typedef float v4f __attribute__((vector_size(16)));

// __bf16 and __ix_bfloat16 (what __float2bfloat16 returns) are distinct types of
// the same size, and the implicit conversion between them is ambiguous.  Pack and
// unpack explicitly instead of relying on conversion rules.
typedef __ix_bfloat16 bf_t;

__device__ __forceinline__ v4y pack4(const bf_t *p) {
  v4y r;
  __builtin_memcpy(&r, p, 8);
  return r;
}
// Host-side conversions: the device intrinsics are not available in main(), and
// bf16 is just the top 16 bits of a float with round-to-nearest-even, so doing it
// by hand keeps the reference path independent of the library under test.
static unsigned short f2bf(float f) {
  unsigned u;
  __builtin_memcpy(&u, &f, 4);
  const unsigned lsb = (u >> 16) & 1u;
  u += 0x7fffu + lsb;
  return (unsigned short)(u >> 16);
}
static float bf2f(unsigned short b) {
  unsigned u = (unsigned)b << 16;
  float f;
  __builtin_memcpy(&f, &u, 4);
  return f;
}
static void set_bf(bf_t *p, float f) {
  const unsigned short h = f2bf(f);
  __builtin_memcpy(p, &h, 2);
}
static float get_bf(const bf_t *p) {
  unsigned short h;
  __builtin_memcpy(&h, p, 2);
  return bf2f(h);
}

__global__ void one_mma(const bf_t *A, const bf_t *B, float *C) {
  const int L = threadIdx.x & 63;
  bf_t av[4], bv[4];
  v4f c = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int row = 2 * (L / 16) + 8 * (j / 2) + (j % 2);
    const int col = L % 16;
    av[j] = A[row * 16 + col];
    bv[j] = B[row * 16 + col];
  }
  v4y a = pack4(av), b = pack4(bv);
  c = __ivcorex_matrix_mad_f32x4_bf16x4(a, b, c);
#pragma unroll
  for (int q = 0; q < 4; ++q) C[(L / 16 + 4 * q) * 16 + (L % 16)] = c[q];
}

// A k=32 reduction done as two 16x16x16 calls, feeding each one a DIFFERENT slice
// of the source rather than trying to split the vendor's k=32 fragment layout.
// Which elements a k=32 fragment hands to each of its two MMA passes is not
// documented and two guesses at it were wrong, so the kernel sidesteps the
// question: the verified 16x16x16 mapping is used and simply called once per
// 16-wide k chunk.  head_dim=128 therefore costs eight calls.
__global__ void chunked_mma(const bf_t *A, const bf_t *B, float *C, int nchunk) {
  const int L = threadIdx.x & 63;
  v4f c = {0.f, 0.f, 0.f, 0.f};
  for (int ch = 0; ch < nchunk; ++ch) {
    bf_t av[4], bv[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int m = 2 * (L / 16) + 8 * (j / 2) + (j % 2);
      av[j] = A[m * (16 * nchunk) + 16 * ch + (L % 16)];
      bv[j] = B[(16 * ch + m) * 16 + (L % 16)];
    }
    v4y a = pack4(av), b = pack4(bv);
    c = __ivcorex_matrix_mad_f32x4_bf16x4(a, b, c);
  }
#pragma unroll
  for (int q = 0; q < 4; ++q) C[(L / 16 + 4 * q) * 16 + (L % 16)] = c[q];
}

int main() {
  // ---- 16x16x16 ----
  bf_t hA[256], hB[256];
  float hC[256];
  srand(7);
  for (int i = 0; i < 256; ++i) {
    set_bf(&hA[i], (rand() % 200 - 100) / 64.0f);
    set_bf(&hB[i], (rand() % 200 - 100) / 64.0f);
  }
  bf_t *dA, *dB;
  float *dC;
  cudaMalloc(&dA, sizeof hA);
  cudaMalloc(&dB, sizeof hB);
  cudaMalloc(&dC, sizeof hC);
  cudaMemcpy(dA, hA, sizeof hA, cudaMemcpyHostToDevice);
  cudaMemcpy(dB, hB, sizeof hB, cudaMemcpyHostToDevice);

  one_mma<<<1, 64>>>(dA, dB, dC);
  cudaError_t e = cudaDeviceSynchronize();
  if (e != cudaSuccess) { printf("ERR %s\n", cudaGetErrorString(e)); return 1; }
  cudaMemcpy(hC, dC, sizeof hC, cudaMemcpyDeviceToHost);

  double worst = 0;
  int bad = 0;
  for (int m = 0; m < 16; ++m)
    for (int n = 0; n < 16; ++n) {
      double ref = 0;
      for (int k = 0; k < 16; ++k)
        ref += (double)get_bf(&hA[m * 16 + k]) * get_bf(&hB[k * 16 + n]);
      const double er = fabs(hC[m * 16 + n] - ref) / (fabs(ref) + 1e-6);
      if (er > worst) worst = er;
      if (er > 3e-2) ++bad;
    }
  printf("bf16 16x16x16 (A[m][k], B[k][n]): worst rel err %.2e  bad %d  %s\n", worst, bad,
         bad == 0 ? "OK" : "FAIL");

  // ---- two chained MMA over k=32 ----
  static bf_t hA2[512], hB2[512];
  for (int i = 0; i < 512; ++i) {
    set_bf(&hA2[i], (rand() % 200 - 100) / 64.0f);
    set_bf(&hB2[i], (rand() % 200 - 100) / 64.0f);
  }
  bf_t *dA2, *dB2;
  cudaMalloc(&dA2, sizeof hA2);
  cudaMalloc(&dB2, sizeof hB2);
  cudaMemcpy(dA2, hA2, sizeof hA2, cudaMemcpyHostToDevice);
  cudaMemcpy(dB2, hB2, sizeof hB2, cudaMemcpyHostToDevice);
  chunked_mma<<<1, 64>>>(dA2, dB2, dC, 2);
  e = cudaDeviceSynchronize();
  if (e != cudaSuccess) { printf("ERR %s\n", cudaGetErrorString(e)); return 1; }
  cudaMemcpy(hC, dC, sizeof hC, cudaMemcpyDeviceToHost);
  worst = 0; bad = 0;
  for (int m = 0; m < 16; ++m)
    for (int n = 0; n < 16; ++n) {
      double ref = 0;
      for (int k = 0; k < 32; ++k)
        ref += (double)get_bf(&hA2[m * 32 + k]) * get_bf(&hB2[k * 16 + n]);
      const double er = fabs(hC[m * 16 + n] - ref) / (fabs(ref) + 1e-6);
      if (er > worst) worst = er;
      if (er > 3e-2) ++bad;
    }
  printf("bf16 k=32 as two chunked MMA:     worst rel err %.2e  bad %d  %s\n", worst, bad,
         bad == 0 ? "OK" : "FAIL");

  // ---- how fast is one bf16 MMA vs one fp16 MMA, per warp? ----
  printf("\npeak reference: 98.3 TFLOPS = 16 SM x 0.5 MMA/cycle x 8192 flops x 1.5 GHz\n");
  return 0;
}
