// What is the real ceiling of the BI-V150 MMA instruction?
// __ivcorex_matrix_mad_f32x4_f16x4 does an M*N*K = 16*16*16 matrix mad, but the
// warp result is 32 lanes * 4 floats = 128 floats = 16x8, so one instruction
// retires 128*16 MACs = 2048 MACs = 4096 flops.
// Pure register loop, 4 independent chains, no memory -> measures issue rate.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>

#define FLOPS_PER_MMA 4096.0  // 16*8*16*2

typedef __fp16 v4h __attribute__((vector_size(8)));
typedef float v4f __attribute__((vector_size(16)));

__global__ __launch_bounds__(256) void mma_rate(float *sink, int iters) {
  // 4 independent accumulator chains (hide the ~multi-cycle latency)
  v4h a0, a1, b0, b1;
  v4f c0, c1, c2, c3;
  const unsigned t = threadIdx.x + blockIdx.x * blockDim.x;
  a0 = (v4h){1.f, 1.f, 1.f, 1.f};
  a1 = (v4h){1.f, 1.f, 1.f, 1.f};
  b0 = (v4h){1.f, 1.f, 1.f, 1.f};
  b1 = (v4h){1.f, 1.f, 1.f, 1.f};
  c0 = (v4f){0.f, 0.f, 0.f, 0.f};
  c1 = c0;
  c2 = c0;
  c3 = c0;

  for (int i = 0; i < iters; ++i) {
    c0 = __ivcorex_matrix_mad_f32x4_f16x4(a0, b0, c0);
    c1 = __ivcorex_matrix_mad_f32x4_f16x4(a1, b1, c1);
    c2 = __ivcorex_matrix_mad_f32x4_f16x4(a0, b1, c2);
    c3 = __ivcorex_matrix_mad_f32x4_f16x4(a1, b0, c3);
  }
  float s = c0[0] + c1[1] + c2[2] + c3[3];
  if (t == 0xFFFFFFFFu) sink[0] = s;
}

int main(int argc, char **argv) {
  int blocks = argc > 1 ? atoi(argv[1]) : 16 * 8;
  int iters = argc > 2 ? atoi(argv[2]) : 20000;
  float *d;
  cudaMalloc(&d, 4);
  for (int i = 0; i < 3; ++i) mma_rate<<<blocks, 256>>>(d, iters);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) { printf("ERR %s\n", cudaGetErrorString(err)); return 1; }

  cudaEvent_t s, e;
  cudaEventCreate(&s); cudaEventCreate(&e);
  cudaEventRecord(s);
  mma_rate<<<blocks, 256>>>(d, iters);
  cudaEventRecord(e);
  cudaEventSynchronize(e);
  float ms; cudaEventElapsedTime(&ms, s, e);

  // 4 MMA per loop iteration per thread; warps = blocks*256/32
  double warps = (double)blocks * 256 / 32;
  double mma_inst = warps * (double)iters * 4.0;
  double tf = mma_inst * FLOPS_PER_MMA / (ms * 1e-3) / 1e12;
  printf("blocks=%d warps=%.0f iters=%d  %.3f ms  %.1f TFLOPS issue ceiling (%.0f%% of 96.6)\n",
         blocks, warps, iters, ms, tf, 100 * tf / 96.6);
  return 0;
}
