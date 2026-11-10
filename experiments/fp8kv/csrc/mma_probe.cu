// Measures the real cost of __builtin_mxc_mma_16x16x32f8 on xcore1000
// (MetaX C500): cycles per instruction for an independent chain (throughput)
// and a dependent chain (latency).  Also counts launched lanes so we can tell
// hardware MMA from a software decomposition.

#include <mc_runtime.h>
#include <cstdint>

typedef __NATIVE_VECTOR__(2, uint32_t) v2u;
typedef __NATIVE_VECTOR__(4, float) v4f;

namespace {

__device__ __forceinline__ v4f mma(v2u b, v2u a, v4f c) {
  return __builtin_mxc_mma_16x16x32f8(b, a, c);
}

}  // namespace

// 4 independent accumulator chains -> throughput
extern "C" __global__ void mma_thru(float* out, long long* cyc, int iters) {
  v2u a = {0x38383838u, 0x38383838u};
  v2u b = {0x38383838u, 0x38383838u};
  v4f c0 = {0.f, 0.f, 0.f, 0.f}, c1 = c0, c2 = c0, c3 = c0;
  const long long t0 = clock64();
  for (int i = 0; i < iters; ++i) {
    c0 = mma(b, a, c0);
    c1 = mma(b, a, c1);
    c2 = mma(b, a, c2);
    c3 = mma(b, a, c3);
  }
  const long long t1 = clock64();
  if (threadIdx.x == 0) {
    out[blockIdx.x] = c0[0] + c1[0] + c2[0] + c3[0];
    cyc[blockIdx.x] = t1 - t0;
  }
}

// single dependent chain -> latency
extern "C" __global__ void mma_lat(float* out, long long* cyc, int iters) {
  v2u a = {0x38383838u, 0x38383838u};
  v2u b = {0x38383838u, 0x38383838u};
  v4f c0 = {0.f, 0.f, 0.f, 0.f};
  const long long t0 = clock64();
  for (int i = 0; i < iters; ++i) c0 = mma(b, a, c0);
  const long long t1 = clock64();
  if (threadIdx.x == 0) {
    out[blockIdx.x] = c0[0];
    cyc[blockIdx.x] = t1 - t0;
  }
}

// fp32 FMA throughput on the same grid for a sanity reference.
extern "C" __global__ void fma_thru(float* out, long long* cyc, int iters) {
  float a = 1.000001f, b = 0.999999f;
  float c0 = 0.f, c1 = 1.f, c2 = 2.f, c3 = 3.f;
  const long long t0 = clock64();
  for (int i = 0; i < iters; ++i) {
    c0 = c0 * a + b;
    c1 = c1 * a + b;
    c2 = c2 * a + b;
    c3 = c3 * a + b;
  }
  const long long t1 = clock64();
  if (threadIdx.x == 0) {
    out[blockIdx.x] = c0 + c1 + c2 + c3;
    cyc[blockIdx.x] = t1 - t0;
  }
}

extern "C" int launch_mma_probe(int which, void* out, void* cyc, int iters,
                                int warps, int ctas, void* stream) {
  if (which == 0)
    mma_thru<<<ctas, 64 * warps, 0, (mcStream_t)stream>>>(
        (float*)out, (long long*)cyc, iters);
  else if (which == 1)
    mma_lat<<<ctas, 64 * warps, 0, (mcStream_t)stream>>>(
        (float*)out, (long long*)cyc, iters);
  else
    fma_thru<<<ctas, 64 * warps, 0, (mcStream_t)stream>>>(
        (float*)out, (long long*)cyc, iters);
  return 0;
}
