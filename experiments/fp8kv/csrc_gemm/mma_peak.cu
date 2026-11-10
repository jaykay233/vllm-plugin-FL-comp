// What is the actual issue rate of __builtin_mxc_mma_16x16x32f8 on C500?
//
// Every GEMM variant so far lands at 65-93 GB/s, ~11x below the bf16 mcblas
// reference (1080 GB/s). Before spending more effort on pipelining, measure the
// ceiling: issue MMAs back-to-back out of registers, with NCH independent
// accumulator chains so dependency stalls are not the limiter, and time it.
//
// Implied bandwidth for a GEMM at M=64 is  BW[GB/s] = TFLOPS / (2*M) * 1000 / 1e3
// i.e. TFLOPS ~= 0.128 * BW, so reaching 1080 GB/s would require ~138 TFLOPS of
// fp8 MMA. Whatever this kernel reports is the hard cap on that plan.

#include <mc_runtime.h>
#include <cstdint>

typedef __NATIVE_VECTOR__(2, uint32_t) v2u;
typedef __NATIVE_VECTOR__(4, float) v4f;

__device__ __forceinline__ uint32_t ld4(const uint8_t* p) {
  uint32_t v;
  __builtin_memcpy(&v, p, 4);
  return v;
}

template <int NCH, int ITER>
__global__ void mma_peak_kernel(const uint8_t* __restrict__ src,
                                float* __restrict__ out) {
  const int lane = threadIdx.x & 63;
  const int g = lane / 16;
  const int row = lane % 16;

  // NCH independent chains: chain i gets its own accumulator and its own v2u
  // operands, so the scheduler can overlap them.
  v4f acc[NCH];
  v2u va[NCH];
  v2u vb[NCH];
#pragma unroll
  for (int i = 0; i < NCH; ++i) {
    acc[i] = v4f{0.f, 0.f, 0.f, 0.f};
    va[i] = {ld4(src + (row * 16 + i) * 32 + g * 4),
             ld4(src + (row * 16 + i) * 32 + 16 + g * 4)};
    vb[i] = {ld4(src + (row * 16 + i) * 32 + g * 4),
             ld4(src + (row * 16 + i) * 32 + 16 + g * 4)};
  }

#pragma unroll 1
  for (int it = 0; it < ITER; ++it) {
#pragma unroll
    for (int i = 0; i < NCH; ++i) {
      acc[i] = __builtin_mxc_mma_16x16x32f8(vb[i], va[i], acc[i]);
    }
  }

  float s = 0.f;
#pragma unroll
  for (int i = 0; i < NCH; ++i) s += acc[i][0] + acc[i][1] + acc[i][2] + acc[i][3];
  if (threadIdx.x == 0) out[blockIdx.x] = s;
}

template <int NCH, int ITER>
static void run(const uint8_t* src, float* out, int blocks, void* stream) {
  mma_peak_kernel<NCH, ITER><<<blocks, 256, 0, (mcStream_t)stream>>>(src, out);
}

extern "C" int launch_mma_peak(const uint8_t* src, float* out, int blocks,
                               int nch, int iter, void* stream) {
  // 256 threads = 4 warps per block
#define CASE(N)                                                                 \
  if (nch == N && iter == 256) { run<N, 256>(src, out, blocks, stream); return 0; }
  CASE(1) CASE(2) CASE(4) CASE(8) CASE(16)
#undef CASE
  return -1;
}
