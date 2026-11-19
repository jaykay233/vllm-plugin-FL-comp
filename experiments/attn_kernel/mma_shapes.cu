// Which MMA instruction shape actually delivers the most flops per cycle on
// BI-V150, and does native bf16 run at the same rate as fp16?
//
// This matters because the model is bf16.  Using __ivcorex_matrix_mad_f32x4_bf16x4
// avoids any conversion, so results are bit-identical in dtype to the vendor path
// instead of going through fp16 (which has a 5-bit exponent and can overflow where
// bf16 would not).
//
// The instruction set offers much larger shapes than the 16x16x16 used so far:
//   m16n16k16   V4f  V4y  V4y  V4f    8192 flops   (what the kernel uses today)
//   m64n16k16   V16f V16y V4y  V16f  32768 flops   -- needs +long-mtx, whose
//   m16n64k16   V16f V4y  V16y V16f  32768 flops      lowering segfaults llc
//                                                     (see cp_async_repro.cu
//                                                     for the same failure)
// so the large shapes are compiled only under -DUSE_LONG_MTX and are unusable
// today.  What this file does measure is whether native bf16 matches fp16 at the
// 16x16x16 shape, which is what the attention kernel actually uses.
//
// Pure register loop, four independent chains, no memory: measures issue rate.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>

typedef __fp16 v4h __attribute__((vector_size(8)));
typedef __bf16 v4y __attribute__((vector_size(8)));
typedef __bf16 v16y __attribute__((vector_size(32)));
typedef float v4f __attribute__((vector_size(16)));
typedef float v16f __attribute__((vector_size(64)));

#define PEAK 196.6  // 16 SM x 1 MMA/cycle x 8192 flops x 1.5 GHz

// ---------- fp16 16x16x16 ----------
__global__ __launch_bounds__(256) void k_f16_16(float *sink, int iters) {
  v4h a0 = (v4h){1.f, 1.f, 1.f, 1.f}, a1 = a0, b0 = a0, b1 = a0;
  v4f c0 = {0}, c1 = {0}, c2 = {0}, c3 = {0};
  for (int i = 0; i < iters; ++i) {
    c0 = __ivcorex_matrix_mad_f32x4_f16x4(a0, b0, c0);
    c1 = __ivcorex_matrix_mad_f32x4_f16x4(a1, b1, c1);
    c2 = __ivcorex_matrix_mad_f32x4_f16x4(a0, b1, c2);
    c3 = __ivcorex_matrix_mad_f32x4_f16x4(a1, b0, c3);
  }
  float s = c0[0] + c1[1] + c2[2] + c3[3];
  if (threadIdx.x + blockIdx.x * blockDim.x == 0xFFFFFFFFu) sink[0] = s;
}

// ---------- bf16 16x16x16 ----------
__global__ __launch_bounds__(256) void k_bf16_16(float *sink, int iters) {
  v4y a0 = (v4y){1.f, 1.f, 1.f, 1.f}, a1 = a0, b0 = a0, b1 = a0;
  v4f c0 = {0}, c1 = {0}, c2 = {0}, c3 = {0};
  for (int i = 0; i < iters; ++i) {
    c0 = __ivcorex_matrix_mad_f32x4_bf16x4(a0, b0, c0);
    c1 = __ivcorex_matrix_mad_f32x4_bf16x4(a1, b1, c1);
    c2 = __ivcorex_matrix_mad_f32x4_bf16x4(a0, b1, c2);
    c3 = __ivcorex_matrix_mad_f32x4_bf16x4(a1, b0, c3);
  }
  float s = c0[0] + c1[1] + c2[2] + c3[3];
  if (threadIdx.x + blockIdx.x * blockDim.x == 0xFFFFFFFFu) sink[0] = s;
}

#ifdef USE_LONG_MTX
// ---------- bf16 64x16x16 ----------
__global__ __launch_bounds__(256) void k_bf16_64(float *sink, int iters) {
  v16y a0, a1;
  v4y b;
  v16f c0, c1, c2, c3;
  a0 = (v16y){1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f};
  a1 = a0;
  b = (v4y){1.f, 1.f, 1.f, 1.f};
  c0 = (v16f){0}; c1 = c0; c2 = c0; c3 = c0;
  for (int i = 0; i < iters; ++i) {
    c0 = __ivcorex_matrix_mad_f32_bf16_m64n16k16(a0, b, c0);
    c1 = __ivcorex_matrix_mad_f32_bf16_m64n16k16(a1, b, c1);
    c2 = __ivcorex_matrix_mad_f32_bf16_m64n16k16(a0, b, c2);
    c3 = __ivcorex_matrix_mad_f32_bf16_m64n16k16(a1, b, c3);
  }
  float s = c0[0] + c1[1] + c2[2] + c3[3];
  if (threadIdx.x + blockIdx.x * blockDim.x == 0xFFFFFFFFu) sink[0] = s;
}

// ---------- bf16 16x64x16 ----------
__global__ __launch_bounds__(256) void k_bf16_n64(float *sink, int iters) {
  v4y a;
  v16y b0, b1;
  v16f c0, c1, c2, c3;
  a = (v4y){1.f, 1.f, 1.f, 1.f};
  b0 = (v16y){1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f};
  b1 = b0;
  c0 = (v16f){0}; c1 = c0; c2 = c0; c3 = c0;
  for (int i = 0; i < iters; ++i) {
    c0 = __ivcorex_matrix_mad_f32_bf16_m16n64k16(a, b0, c0);
    c1 = __ivcorex_matrix_mad_f32_bf16_m16n64k16(a, b1, c1);
    c2 = __ivcorex_matrix_mad_f32_bf16_m16n64k16(a, b0, c2);
    c3 = __ivcorex_matrix_mad_f32_bf16_m16n64k16(a, b1, c3);
  }
  float s = c0[0] + c1[1] + c2[2] + c3[3];
  if (threadIdx.x + blockIdx.x * blockDim.x == 0xFFFFFFFFu) sink[0] = s;
}

#endif

template <typename F>
static void run(const char *tag, F kernel, int blocks, int iters, double flops_per_mma) {
  float *sink;
  cudaMalloc(&sink, 4);
  auto go = [&]() { kernel<<<blocks, 256>>>(sink, iters); };
  go();
  cudaError_t e = cudaDeviceSynchronize();
  if (e != cudaSuccess) { printf("%-16s ERR %s\n", tag, cudaGetErrorString(e)); return; }
  go();
  cudaEvent_t s, ev;
  cudaEventCreate(&s); cudaEventCreate(&ev);
  cudaEventRecord(s);
  for (int i = 0; i < 5; ++i) go();
  cudaEventRecord(ev);
  cudaEventSynchronize(ev);
  float ms; cudaEventElapsedTime(&ms, s, ev); ms /= 5;
  // MMA is a warp-level op: 256 threads is 4 warps of 64, not 256 independent
  // lanes, so the instruction count is per warp.  Charging per thread inflates by
  // the warp size; the first run of this file did exactly that and reported
  // impossible numbers (1477 TFLOPS at 8 warps/SM).
  const double warps = (double)blocks * (256.0 / 64.0);
  const double mma = warps * iters * 4.0;  // 4 MMA per iteration per warp
  const double tf = mma * flops_per_mma / (ms * 1e-3) / 1e12;
  printf("%-16s %5.1f flops/instr   %7.3f ms   %7.1f TFLOPS  (%5.1f%% of %.0f)\n", tag,
         flops_per_mma, ms, tf, 100 * tf / PEAK, PEAK);
  cudaFree(sink);
}

int main(int argc, char **argv) {
  const int iters = argc > 1 ? atoi(argv[1]) : 20000;
  printf("issue-rate ceiling per instruction shape (%d iters, 128 blocks = 64 warps/SM)\n\n", iters);
  for (int b : {16, 32, 64, 128}) {
    printf("--- blocks=%d (%d warps/SM) ---\n", b, b * 8 / 16);
    run("f16  16x16x16", k_f16_16, b, iters, 8192.0);
    run("bf16 16x16x16", k_bf16_16, b, iters, 8192.0);
#ifdef USE_LONG_MTX
    run("bf16 64x16x16", k_bf16_64, b, iters, 32768.0);
    run("bf16 16x64x16", k_bf16_n64, b, iters, 32768.0);
#endif
    printf("\n");
  }
  return 0;
}
