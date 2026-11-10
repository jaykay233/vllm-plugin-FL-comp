// Why is __builtin_mxc_mma_16x16x32f8 slow inside the decode kernel when the
// isolated microbenchmark says it is fast?
//
// The only structural difference is operand construction: the microbenchmark
// feeds loop-invariant operands (they can live in the MMA operand registers
// forever), while the decode kernel assembles a fresh {lo,hi} pair from
// general-purpose registers for every MMA.  This probe measures exactly that.

#include <mc_runtime.h>
#include <cstdint>

typedef __NATIVE_VECTOR__(2, uint32_t) v2u;
typedef __NATIVE_VECTOR__(4, float) v4f;

namespace {

__device__ __forceinline__ v4f mma(v2u b, v2u a, v4f c) {
  return __builtin_mxc_mma_16x16x32f8(b, a, c);
}

}  // namespace

// Baseline: operands are loop invariant.
extern "C" __global__ void mma_const(float* out, long long* cyc, int iters) {
  v2u a = {0x38383838u, 0x38383838u};
  v2u b = {0x38383838u, 0x38383838u};
  v4f c[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) c[i] = {0.f, 0.f, 0.f, 0.f};
  const long long t0 = clock64();
  for (int i = 0; i < iters; ++i) {
#pragma unroll
    for (int j = 0; j < 8; ++j) c[j] = mma(b, a, c[j]);
  }
  const long long t1 = clock64();
  if (threadIdx.x == 0) {
    float s = 0.f;
#pragma unroll
    for (int j = 0; j < 8; ++j) s += c[j][0];
    out[blockIdx.x] = s;
    cyc[blockIdx.x] = t1 - t0;
  }
}

// Same, but the operand pair is assembled from two scalars every MMA, with a
// loop-carried dependency so the compiler cannot hoist it.
extern "C" __global__ void mma_dyn(float* out, long long* cyc, int iters) {
  const uint32_t seed = 0x38383838u;
  v2u a = {seed, seed};
  v4f c = {0.f, 0.f, 0.f, 0.f};
  const long long t0 = clock64();
  for (int i = 0; i < iters; ++i) {
    const uint32_t lo = seed + (__float_as_uint(c[0]) & 0x100u);
    const uint32_t hi = seed + (__float_as_uint(c[1]) & 0x100u);
    v2u b = {lo, hi};
    c = mma(b, a, c);
  }
  const long long t1 = clock64();
  if (threadIdx.x == 0) {
    out[blockIdx.x] = c[0];
    cyc[blockIdx.x] = t1 - t0;
  }
}

// Operand assembled from shared memory (mimics reading a staged fragment).
extern "C" __global__ void mma_smem_dyn(float* out, long long* cyc, int iters) {
  __shared__ uint32_t frag[64 * 2];
  const int lane = threadIdx.x & 63;
  frag[lane * 2 + 0] = 0x38383838u;
  frag[lane * 2 + 1] = 0x38383838u;
  __syncthreads();
  v2u a = {0x38383838u, 0x38383838u};
  v4f c = {0.f, 0.f, 0.f, 0.f};
  const long long t0 = clock64();
  for (int i = 0; i < iters; ++i) {
    const uint32_t idx = (uint32_t)(__float_as_uint(c[0]) & 0x20u);
    v2u b = {frag[lane * 2 + idx], frag[lane * 2 + (idx ^ 0u)]};
    c = mma(b, a, c);
  }
  const long long t1 = clock64();
  if (threadIdx.x == 0) {
    out[blockIdx.x] = c[0];
    cyc[blockIdx.x] = t1 - t0;
  }
}

extern "C" int launch_mma_dyn(int which, void* out, void* cyc, int iters,
                              int warps, int ctas, void* stream) {
  if (which == 0)
    mma_const<<<ctas, 64 * warps, 0, (mcStream_t)stream>>>(
        (float*)out, (long long*)cyc, iters);
  else if (which == 1)
    mma_dyn<<<ctas, 64 * warps, 0, (mcStream_t)stream>>>(
        (float*)out, (long long*)cyc, iters);
  else
    mma_smem_dyn<<<ctas, 64 * warps, 0, (mcStream_t)stream>>>(
        (float*)out, (long long*)cyc, iters);
  return 0;
}
