// Dump the fragment maps of __builtin_mxc_mma_16x16x16bf16.
//
// Strategy: fill A and B with 1.0, then bump exactly one element (lane L, slot
// S) to 2.0.  Everything else stays put, so C[m][n] = 16 + delta, and the set
// of C positions that became 17 is exactly:
//
//   A-element bump  ->  every C position in row m
//   B-element bump  ->  every C position in column n
//
// which identifies the row/column without ever needing to know the C layout in
// advance.  Scanning all 64 lanes x 4 slots for A and for B yields all three
// maps (A, B and, from the union of the bumps, C).

#include <mc_runtime.h>
#include <cstdint>

typedef __NATIVE_VECTOR__(4, __fp16) v4h;
typedef __NATIVE_VECTOR__(4, float) v4f;

namespace {

__device__ __forceinline__ v4f mma(v4h b, v4h a, v4f c) {
  return __builtin_mxc_mma_16x16x16bf16(b, a, c);
}

constexpr uint32_t kOne = 0x3F80u;  // bf16 1.0
constexpr uint32_t kTwo = 0x4000u;  // bf16 2.0

__device__ __forceinline__ v4h fill(uint32_t bits) {
  v4h r;
  const uint32_t w = bits | (bits << 16);
  __builtin_memcpy(&r, &w, 4);
  __builtin_memcpy(((char*)&r) + 4, &w, 4);
  return r;
}

__device__ __forceinline__ v4h setslot(v4h v, int slot, uint32_t bits) {
  uint32_t w[2];
  __builtin_memcpy(w, &v, 8);
  const int which = slot >> 1;
  const int half = slot & 1;
  w[which] = half ? ((w[which] & 0xFFFFu) | (bits << 16))
                  : ((w[which] & 0xFFFF0000u) | bits);
  v4h r;
  __builtin_memcpy(&r, w, 8);
  return r;
}

}  // namespace

// which == 0 -> bump A, 1 -> bump B, 2 -> bump neither (baseline).
extern "C" __global__ void probe_bump(int which, int blane, int bslot,
                                      float* out) {
  const int lane = threadIdx.x & 63;
  v4h a = fill(kOne);
  v4h b = fill(kOne);
  if (lane == blane) {
    if (which == 0) a = setslot(a, bslot, kTwo);
    if (which == 1) b = setslot(b, bslot, kTwo);
  }
  v4f c = {0.f, 0.f, 0.f, 0.f};
  c = mma(b, a, c);
  float* o = out + blockIdx.x * 256 + lane * 4;
  o[0] = c[0];
  o[1] = c[1];
  o[2] = c[2];
  o[3] = c[3];
}

extern "C" int launch_probe_bump(int which, int blane, int bslot, void* out,
                                 void* stream) {
  probe_bump<<<1, 64, 0, (mcStream_t)stream>>>(which, blane, bslot,
                                               (float*)out);
  return 0;
}
