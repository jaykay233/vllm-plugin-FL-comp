// End-to-end validation of the bf16 fragment convention used by the decode
// kernel, decoupled from any KV-cache layout.
//
// Loads a 16x16 A and a 16x16 B from global memory using exactly the mapping
// derived from the bump probe, runs one mma, and writes C back with the
// derived C mapping.  Comparing against torch A @ B (16x16) then validates
// encoding + argument order + all three fragment maps in one shot.
//
//   A(lane,slot) = A[row = lane & 15][k = 4*(lane>>4) + slot]
//   B(lane,slot) = B[k = 4*(lane>>4) + slot][col = lane & 15]
//   C(lane,slot) = C[row = lane & 15][col = 4*(lane>>4) + slot]

#include <mc_runtime.h>
#include <maca_bfloat16.h>
#include <cstdint>

typedef __NATIVE_VECTOR__(4, __fp16) v4h;
typedef __NATIVE_VECTOR__(4, float) v4f;

namespace {

__device__ __forceinline__ v4f mma(v4h a, v4h b, v4f c) {
  return __builtin_mxc_mma_16x16x16bf16(a, b, c);
}

}  // namespace

// swap == 0 -> mma(A, B, C);  swap == 1 -> mma(B, A, C)
extern "C" __global__ void gemm16(const __maca_bfloat16* A,
                                  const __maca_bfloat16* B, float* C,
                                  int swap) {
  const int lane = threadIdx.x & 63;
  const int m = lane & 15;
  const int k0 = 4 * (lane >> 4);
  v4h a, b;
  __builtin_memcpy(&a, A + m * 16 + k0, 8);
  // B fragment: slot s holds B[k0 + s][m]
  {
    uint32_t w[2];
    for (int half = 0; half < 2; ++half) {
      uint32_t lo = 0, hi = 0;
      for (int s = 0; s < 2; ++s) {
        uint16_t v;
        __builtin_memcpy(&v, B + (k0 + half * 2 + s) * 16 + m, 2);
        if (s == 0) lo = v;
        else hi = v;
      }
      w[half] = lo | (hi << 16);
    }
    __builtin_memcpy(&b, w, 8);
  }
  v4f c = {0.f, 0.f, 0.f, 0.f};
  c = (swap == 0) ? mma(a, b, c) : mma(b, a, c);
  // C: row = m, col = k0 + slot
  float* o = C + m * 16 + k0;
  o[0] = c[0];
  o[1] = c[1];
  o[2] = c[2];
  o[3] = c[3];
}

extern "C" int launch_gemm16(const void* A, const void* B, void* C, int swap,
                             void* stream) {
  gemm16<<<1, 64, 0, (mcStream_t)stream>>>(
      (const __maca_bfloat16*)A, (const __maca_bfloat16*)B, (float*)C, swap);
  return 0;
}
