// What does the C500 async global->shared copy engine actually deliver, and
// what controls it?
//
// In the decode kernel the K pipeline alone (no MMA, no V) runs at 583 GB/s
// while the vendor FlashAttention kernel sustains 1252 GB/s of K+V.  This
// probe strips the question down to the engine: every warp streams a private
// contiguous region through __builtin_mxc_ldg_b128_bsm and we sweep
//
//   * warps per CTA   (1 .. 8)
//   * requests in flight before the wait  (1 .. 16)
//
// so we can see whether the limit is per-warp outstanding capacity, total
// warps, or simply bytes per request.

#include <mc_runtime.h>
#include <cstdint>

namespace {

__device__ __forceinline__ void g2s16(void* dst, const void* src) {
  __builtin_mxc_ldg_b128_bsm(dst, src, 0, -1, false, true, false, false);
}

// __builtin_mxc_arrive requires a compile-time constant, so the wait levels
// have to be spelled out; these are the only ones the sweep needs.
template <int kN>
__device__ __forceinline__ void arrive() {
  __builtin_mxc_arrive(64 + kN);
}

}  // namespace

// depth == 0 -> wait only once, at the end (max in flight)
extern "C" __global__ void copy_bw(const uint8_t* __restrict__ src,
                                   long long src_stride, int calls, int depth,
                                   float* __restrict__ sink,
                                   long long* __restrict__ cyc) {
  __shared__ uint8_t smem[8][1024];
  const int wid = threadIdx.x >> 6;
  const int lane = threadIdx.x & 63;
  uint8_t* dst = smem[wid];

  const uint8_t* p =
      src + (long long)(blockIdx.x * (blockDim.x >> 6) + wid) * src_stride;

  const long long t0 = clock64();
  for (int i = 0; i < calls; ++i) {
    g2s16(dst + lane * 16, p + (long long)i * 1024 + lane * 16);
    if (depth > 0 && (i % depth) == depth - 1) {
      switch (depth) {
        case 1: arrive<0>(); break;
        case 2: arrive<0>(); break;
        case 4: arrive<0>(); break;
        default: arrive<0>(); break;
      }
    }
  }
  arrive<0>();
  const long long t1 = clock64();

  if (threadIdx.x == 0) {
    cyc[blockIdx.x] = t1 - t0;
    sink[blockIdx.x] = (float)smem[0][0];
  }
}

extern "C" int launch_copy_bw(const void* src, long long src_stride, int calls,
                              int depth, int warps, int ctas, void* sink,
                              void* cyc, void* stream) {
  copy_bw<<<ctas, 64 * warps, 0, (mcStream_t)stream>>>(
      (const uint8_t*)src, src_stride, calls, depth, (float*)sink,
      (long long*)cyc);
  return 0;
}
