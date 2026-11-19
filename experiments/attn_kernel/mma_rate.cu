// What is the real ceiling of the BI-V150 MMA instruction?
//
// A 16x16x16 MMA computes 8192 flops and is executed by a 64-lane warp.  The
// machine retires one such instruction per TWO cycles per SM, so
//     peak = 16 SM x 0.5 MMA/cycle x 8192 flops x 1.5 GHz = 98.3 TFLOPS.
//
// Three independent confirmations, after two rounds of getting this wrong:
//   * this file, at 32 warps/SM: 98.2 TFLOPS;
//   * ixBLAS bf16 GEMM saturates at 97.3 TFLOPS even at 16384^3, where a tuned
//     library is at ~99% of peak -- so peak cannot be much above 98;
//   * tile_ok.cu builds a verified 16x16x32 tile with exactly two
//     ml_matrix_mad_f32_f16_m16n16k16 instructions, i.e. k=16 per instruction, and
//     output_registers = 64 lanes x 4 floats = 256 = a full 16x16 tile.
//
// HISTORY, because this number has been wrong twice and the corrections matter:
//   * the original version charged 4096 flops/MMA on the assumption of a 32-lane
//     warp AND divided by 32 to get the warp count.  Those two errors cancel, so
//     its flops total was right; only its "warps" label (blocks*8) was wrong.
//   * a later edit fixed the flop constant to 8192 but left warps = blocks*256/32,
//     which double-counted warps and reported 196.6 TFLOPS -- impossible, since
//     ixBLAS would then be at 49% of peak.  Reverted.
// The warp count below is now blocks*4 (256 threads / 64 lanes), and each warp
// issues 4 MMA per iteration, so nothing is scaled by a hidden factor.
//
// Pure register loop, 4 independent chains, no memory -> measures issue rate.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>

#define FLOPS_PER_MMA 8192.0  // 16*16*16*2, 64-lane warp

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

  // 4 MMA per loop iteration per warp.  A 256-thread block is 4 warps of 64 lanes,
  // not 8 -- using 256/32 here was the bug that doubled an earlier result.
  double warps = (double)blocks * 256 / 64;
  double mma_inst = warps * (double)iters * 4.0;
  double tf = mma_inst * FLOPS_PER_MMA / (ms * 1e-3) / 1e12;
  printf("blocks=%d warps=%.0f iters=%d  %.3f ms  %.1f TFLOPS issue ceiling (%.0f%% of 98.3)\n",
         blocks, warps, iters, ms, tf, 100 * tf / 98.3);
  return 0;
}
