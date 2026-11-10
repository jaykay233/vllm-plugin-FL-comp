// Layout-agnostic probe for __builtin_mxc_mma_16x16x16bf16.
//
// The fragment -> lane mapping is unknown, but it does not matter for this
// test: fill every element of both operands with the same encoding of 1.0.
// Then every C element must be K = 16, and the sum over all C elements must be
// 16 * 16 * 16 = 4096 -- independent of how elements are distributed.
//
// If the sum comes out as 4096 the encoding AND the builtin's semantics are
// right, and any remaining wrongness is a fragment-indexing bug.  If it comes
// out as 0 (or 0.0000x) the bits are being read as some other format, and if it
// comes out as something else the argument order is swapped.

#include <mc_runtime.h>
#include <cstdint>

typedef __NATIVE_VECTOR__(4, __fp16) v4h;
typedef __NATIVE_VECTOR__(4, float) v4f;

namespace {

__device__ __forceinline__ v4f mma(v4h b, v4h a, v4f c) {
  return __builtin_mxc_mma_16x16x16bf16(b, a, c);
}

__device__ __forceinline__ v4h bcast(uint32_t bits) {
  v4h r;
  const uint32_t w = bits | (bits << 16);
  __builtin_memcpy(&r, &w, 4);
  __builtin_memcpy(((char*)&r) + 4, &w, 4);
  return r;
}

}  // namespace

// mode 0: A and B both all-ones  -> C should be all 16, total 4096.
// mode 1: A all-ones, B all-zero -> C should be all 0,   total 0.
extern "C" __global__ void probe_ones(uint32_t bits_a, uint32_t bits_b,
                                      float* out) {
  v4h a = bcast(bits_a);
  v4h b = bcast(bits_b);
  v4f c = {0.f, 0.f, 0.f, 0.f};
  c = mma(b, a, c);
  const int lane = threadIdx.x & 63;
  float* o = out + blockIdx.x * 64 * 4 + lane * 4;
  o[0] = c[0];
  o[1] = c[1];
  o[2] = c[2];
  o[3] = c[3];
}

extern "C" int launch_probe_ones(uint32_t ba, uint32_t bb, void* out,
                                 void* stream) {
  probe_ones<<<1, 64, 0, (mcStream_t)stream>>>(ba, bb, (float*)out);
  return 0;
}
