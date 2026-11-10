// Peak issue rates of the actual hardware MMA primitive on C500.
//
// Reading the headers settles the shape question: MACA's only real MMA is
// __builtin_mxc_mma_16x16x16{d,f16,bf16,i8}. Everything wider is built from it:
//   * cute MACA_16x16x32_F32BF16BF16F32  = two 16x16x16bf16 calls
//   * mctlass __builtin_mxc_mma_f32_16x16x32f8_e4m3
//                                      = fp8->f16 software convert, then two
//                                        16x16x16f16 calls
// So fp8 costs 2x the MMA calls of bf16 plus a conversion pass, and the earlier
// 17.8 TFLOPS for __builtin_mxc_mma_16x16x32f8 is consistent with that.
//
// This kernel measures 16x16x16 bf16, f16 and i8 in the identical harness so the
// numbers are directly comparable. If bf16 is much faster than the fp8 wrapper,
// the right design is: keep weights fp8 in DRAM (2x fewer bytes), convert to
// bf16 in registers, and run bf16 MMAs.

#include <mc_runtime.h>
#include <cstdint>

typedef __NATIVE_VECTOR__(2, uint32_t) v2u;
typedef __NATIVE_VECTOR__(4, float) v4f;

__device__ __forceinline__ uint32_t ld4(const uint8_t* p) {
  uint32_t v;
  __builtin_memcpy(&v, p, 4);
  return v;
}

// KIND: 0 = bf16, 1 = f16, 2 = i8, 3 = fp8 (32-wide wrapper), 4 = bf16x32 (2 calls)
template <int NCH, int ITER, int KIND>
__global__ void peak_kernel(const uint8_t* __restrict__ src,
                            float* __restrict__ out) {
  const int lane = threadIdx.x & 63;
  const int g = lane / 16;
  const int row = lane % 16;

  v4f acc[NCH];
  v2u va[NCH];
  v2u vb[NCH];
#pragma unroll
  for (int i = 0; i < NCH; ++i) {
    acc[i] = v4f{0.f, 0.f, 0.f, 0.f};
    const int base = (row * 16 + i) * 32;
    va[i] = {ld4(src + base + g * 4), ld4(src + base + 16 + g * 4)};
    vb[i] = {ld4(src + base + g * 4), ld4(src + base + 16 + g * 4)};
  }

#pragma unroll 1
  for (int it = 0; it < ITER; ++it) {
#pragma unroll
    for (int i = 0; i < NCH; ++i) {
      if constexpr (KIND == 0) {
        acc[i] = __builtin_mxc_mma_16x16x16bf16(vb[i], va[i], acc[i]);
      } else if constexpr (KIND == 1) {
        acc[i] = __builtin_mxc_mma_16x16x16f16(vb[i], va[i], acc[i]);
      } else if constexpr (KIND == 3) {
        acc[i] = __builtin_mxc_mma_16x16x32f8(vb[i], va[i], acc[i]);
      } else {
        acc[i] = __builtin_mxc_mma_16x16x16bf16(vb[i], va[i], acc[i]);
        acc[i] = __builtin_mxc_mma_16x16x16bf16(vb[i], va[i], acc[i]);
      }
    }
  }

  float s = 0.f;
#pragma unroll
  for (int i = 0; i < NCH; ++i) s += acc[i][0] + acc[i][1] + acc[i][2] + acc[i][3];
  if (threadIdx.x == 0) out[blockIdx.x] = s;
}

template <int NCH, int KIND>
static void run(const uint8_t* src, float* out, int blocks, void* stream) {
  peak_kernel<NCH, 256, KIND><<<blocks, 256, 0, (mcStream_t)stream>>>(src, out);
}

extern "C" int launch_peak(const uint8_t* src, float* out, int blocks, int nch,
                           int kind, void* stream) {
#define CASES(K)                                                              \
  if (kind == K) {                                                            \
    if (nch == 1) { run<1, K>(src, out, blocks, stream); return 0; }          \
    if (nch == 4) { run<4, K>(src, out, blocks, stream); return 0; }          \
    if (nch == 8) { run<8, K>(src, out, blocks, stream); return 0; }          \
    if (nch == 16) { run<16, K>(src, out, blocks, stream); return 0; }        \
  }
  CASES(0) CASES(1) CASES(3) CASES(4)
#undef CASES
  return -1;
}
