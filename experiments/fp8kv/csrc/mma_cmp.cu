// Decisive comparison for the C500 decode kernel.
//
// The decode kernel is now bound by MMA operand assembly: the isolated
// `16x16x32f8` MMA costs ~2.5 cycles when its operands are loop invariant, but
// ~150 ns/inst inside the kernel where a fresh {lo,hi} pair has to be moved out
// of general-purpose registers for every instruction.
//
// Vendor FlashAttention on C500 has no FP8 MMA path at all: it dequantises FP8
// to BF16 with byte permutes and uses the BF16 MMA.  This probe measures
// whether that path really has a cheaper GPR -> operand-register move, which is
// what decides if matching vendor speed is possible with FP8 KV.
//
//   f8   m16n16k32 = 16384 flop/inst
//   bf16 m16n16k16 =  8192 flop/inst   (2x the instructions for the same work)

#include <mc_runtime.h>
#include <cstdint>

typedef __NATIVE_VECTOR__(2, uint32_t) v2u;
typedef __NATIVE_VECTOR__(4, __fp16) v4h;
typedef __NATIVE_VECTOR__(4, float) v4f;

namespace {

__device__ __forceinline__ v4f mma_f8(v2u b, v2u a, v4f c) {
  return __builtin_mxc_mma_16x16x32f8(b, a, c);
}
__device__ __forceinline__ v4f mma_bf16(v4h b, v4h a, v4f c) {
  return __builtin_mxc_mma_16x16x16bf16(b, a, c);
}

// mode 2: fp8 -> bf16 dequantised from a 256-entry shared-memory LUT, then the
// bf16 MMA.  Two bf16 instructions cover the same K=32 work as one f8
// instruction, so the comparison below is per equal work.

// 8 independent accumulator chains, operands rebuilt from GPRs every MMA.
// This is exactly the shape of the decode kernel's PV loop.
// Returns the bf16 bit pattern; MACA cannot return a bare __fp16 by value.
__device__ __forceinline__ uint32_t f8_to_bf16_bits(const uint16_t* lut,
                                                    uint32_t b) {
  return lut[b & 0xffu];
}

// Branchless e4m3 -> bf16, no table:
//   bf16 = (u<<4) + 0x3C00 + (sign ? 0x7800 : 0),  with e==0 forced to zero.
// (u<<4) already places m in bits 4..6 and e in bits 7..10; 120<<7 = 0x3C00
// rebases the exponent, and the sign has to travel from bit 11 to bit 15.
__device__ __forceinline__ uint32_t f8_to_bf16_alu(uint32_t u) {
  const uint32_t mag = u & 0x7fu;
  const uint32_t v = (mag << 4) + 0x3C00u;
  const uint32_t zero = ((mag >> 3) == 0u) ? 0u : 0xffffffffu;  // e == 0
  return ((((u & 0x80u) << 8) | v) & zero);
}

__device__ __forceinline__ v4h pack4_alu(uint32_t b) {
  uint32_t w[2];
  w[0] = f8_to_bf16_alu(b) | (f8_to_bf16_alu(b + 1) << 16);
  w[1] = f8_to_bf16_alu(b + 2) | (f8_to_bf16_alu(b + 3) << 16);
  v4h r;
  __builtin_memcpy(&r, w, 8);
  return r;
}

__device__ __forceinline__ v4h pack4(const uint16_t* lut, uint32_t b) {
  uint32_t w[2];
  w[0] = f8_to_bf16_bits(lut, b) | (f8_to_bf16_bits(lut, b + 1) << 16);
  w[1] = f8_to_bf16_bits(lut, b + 2) | (f8_to_bf16_bits(lut, b + 3) << 16);
  v4h r;
  __builtin_memcpy(&r, w, 8);
  return r;
}

template <int kWhich>
__device__ __forceinline__ void run(float* out, long long* cyc, int iters) {
  const uint32_t seed = 0x38383838u;
  v2u a8 = {seed, seed};
  v4h ah = {(__fp16)1.f, (__fp16)1.f, (__fp16)1.f, (__fp16)1.f};
  v4f c[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) c[i] = {0.f, 0.f, 0.f, 0.f};

  __shared__ uint16_t lut[256];
  if (kWhich == 2) {
    for (int i = threadIdx.x; i < 256; i += blockDim.x) {
      // e4m3 -> bf16 bits: s[15] E=e+120 [14:7] M=m<<4 [6:0]
      const uint32_t u = (uint32_t)i;
      const uint32_t e = (u >> 3) & 0xfu;
      const uint32_t bits = ((u & 0x80u) << 8) |
                            (e ? (((e + 120u) << 7) | ((u & 7u) << 4)) : 0u);
      lut[i] = (uint16_t)bits;
    }
    __syncthreads();
  }

  const long long t0 = clock64();
  for (int i = 0; i < iters; ++i) {
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      // Carry a dependency on the previous result so nothing can be hoisted,
      // but keep the 8 chains mutually independent.
      const uint32_t v = seed + (__float_as_uint(c[j][0]) & 0x100u);
      if (kWhich == 0) {
        v2u b = {v, v ^ 0x01010101u};
        c[j] = mma_f8(b, a8, c[j]);
      } else if (kWhich == 1) {
        v4h b = {(__fp16)(v & 0x7), (__fp16)((v + 1) & 0x7),
                 (__fp16)((v + 2) & 0x7), (__fp16)((v + 3) & 0x7)};
        c[j] = mma_bf16(b, ah, c[j]);
      } else if (kWhich == 2) {
        // 8 fp8 bytes -> 8 bf16, then two K=16 MMAs == one K=32 f8 MMA.
        v4h b0 = pack4(lut, v);
        v4h b1 = pack4(lut, v + 4);
        c[j] = mma_bf16(b0, ah, c[j]);
        c[j] = mma_bf16(b1, ah, c[j]);
      } else {
        // Same, but the e4m3 -> bf16 conversion is pure ALU (no shared memory),
        // which is what the decode kernel wants: it keeps the K tile's banks
        // free for the async copy.
        v4h b0 = pack4_alu(v);
        v4h b1 = pack4_alu(v + 4);
        c[j] = mma_bf16(b0, ah, c[j]);
        c[j] = mma_bf16(b1, ah, c[j]);
      }
    }
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

}  // namespace

extern "C" __global__ void cmp_f8(float* o, long long* c, int it) { run<0>(o, c, it); }
extern "C" __global__ void cmp_bf16(float* o, long long* c, int it) { run<1>(o, c, it); }
extern "C" __global__ void cmp_lut(float* o, long long* c, int it) { run<2>(o, c, it); }
extern "C" __global__ void cmp_alu(float* o, long long* c, int it) { run<3>(o, c, it); }

// Host-side launcher: device symbols are not visible through the .so.
extern "C" int launch_mma_cmp(int which, void* out, void* cyc, int iters,
                              int warps, int ctas, void* stream) {
  if (which == 0)
    cmp_f8<<<ctas, 64 * warps, 0, (mcStream_t)stream>>>(
        (float*)out, (long long*)cyc, iters);
  else if (which == 1)
    cmp_bf16<<<ctas, 64 * warps, 0, (mcStream_t)stream>>>(
        (float*)out, (long long*)cyc, iters);
  else if (which == 2)
    cmp_lut<<<ctas, 64 * warps, 0, (mcStream_t)stream>>>(
        (float*)out, (long long*)cyc, iters);
  else
    cmp_alu<<<ctas, 64 * warps, 0, (mcStream_t)stream>>>(
        (float*)out, (long long*)cyc, iters);
  return 0;
}
