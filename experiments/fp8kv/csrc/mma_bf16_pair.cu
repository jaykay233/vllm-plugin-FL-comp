// Determine the k-pairing between A and B fragments of
// __builtin_mxc_mma_16x16x16bf16.
//
// The row/column maps are already known:
//   A(lane,slot) belongs to C row (lane & 15)
//   B(lane,slot) belongs to C column (lane & 15)
//   C(lane,slot) is (row = lane & 15, col = 4*(lane>>4) + slot)
// so each row owns 16 A elements but the k order among them is unknown, and
// likewise for B.  Those two orders must agree or the GEMM reduces the wrong
// pairs of products.
//
// Probe: set A(lane,slot) = 2.0 at exactly one position and B(lane,slot) = 0.0
// at exactly one position, everything else 1.0.  Then
//
//   C[0][0] = 16 + 1 - (1 if the two positions share a k else 0)
//
// i.e. 16 means "pairs up", 17 means "different k".  A slot-order mismatch
// (e.g. little- vs big-endian element packing) shows up as a reversed pairing.

#include <mc_runtime.h>
#include <cstdint>

typedef __NATIVE_VECTOR__(4, __fp16) v4h;
typedef __NATIVE_VECTOR__(4, float) v4f;

namespace {

__device__ __forceinline__ v4f mma(v4h b, v4h a, v4f c) {
  return __builtin_mxc_mma_16x16x16bf16(b, a, c);
}

constexpr uint32_t kOne = 0x3F80u;
constexpr uint32_t kTwo = 0x4000u;

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

extern "C" __global__ void probe_pair(int alane, int aslot, int blane,
                                      int bslot, float* out) {
  const int lane = threadIdx.x & 63;
  v4h a = fill(kOne);
  v4h b = fill(kOne);
  if (lane == alane) a = setslot(a, aslot, kTwo);
  if (lane == blane) b = setslot(b, bslot, 0u);
  v4f c = {0.f, 0.f, 0.f, 0.f};
  c = mma(b, a, c);
  if (blockIdx.x == 0) {
    float* o = out + lane * 4;
    o[0] = c[0];
    o[1] = c[1];
    o[2] = c[2];
    o[3] = c[3];
  }
}

extern "C" int launch_probe_pair(int alane, int aslot, int blane, int bslot,
                                 void* out, void* stream) {
  probe_pair<<<1, 64, 0, (mcStream_t)stream>>>(alane, aslot, blane, bslot,
                                               (float*)out);
  return 0;
}
