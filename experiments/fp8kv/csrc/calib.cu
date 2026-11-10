// Calibrate the register-fragment layout of __builtin_mxc_mma_16x16x32f8 (fp8 e4m3).
// A: [16][32] fp8 (row-major, m x k).  Bn: [16][32] fp8 (n-major, n x k), i.e. B[k][n].
// C: [16][16] f32, mapping taken from mcflashinfer f16 reference:
//    row = lane%16, col = (lane/16)*4 + reg.
#include <mc_runtime.h>
#include <cstdint>

typedef __NATIVE_VECTOR__(2, uint32_t) v2u;
typedef __NATIVE_VECTOR__(4, float) v4f;

__device__ __forceinline__ uint32_t ld4(const uint8_t* p) {
  uint32_t v;
  __builtin_memcpy(&v, p, 4);
  return v;
}

template <int VA, int VB>
__global__ void calib_kernel(const uint8_t* __restrict__ A,
                             const uint8_t* __restrict__ Bn,
                             float* __restrict__ Cout) {
  const int lane = threadIdx.x;  // 0..63
  const int row = lane % 16;     // C / A row
  const int g = lane / 16;       // 0..3

  // ---- A fragment (m = row, k split per VA) ----
  uint32_t a0, a1;
  if constexpr (VA == 0) {
    const int k0 = g * 4;
    a0 = ld4(A + row * 32 + k0);
    a1 = ld4(A + row * 32 + 16 + k0);
  } else {
    const int k0 = g * 8;
    a0 = ld4(A + row * 32 + k0);
    a1 = ld4(A + row * 32 + k0 + 4);
  }
  // ---- B fragment (n, k split per VB) ----
  uint32_t b0, b1;
  int nA, kg;
  if constexpr (VB == 0) { nA = lane % 16; kg = lane / 16; }
  else                  { nA = lane / 4;  kg = lane % 4;  }
  if constexpr (VB == 0 || VB == 1) {
    b0 = ld4(Bn + nA * 32 + kg * 4);
    b1 = ld4(Bn + nA * 32 + 16 + kg * 4);
  } else {
    b0 = ld4(Bn + nA * 32 + kg * 8);
    b1 = ld4(Bn + nA * 32 + kg * 8 + 4);
  }

  v2u va = {a0, a1};
  v2u vb = {b0, b1};
  v4f c = {0.f, 0.f, 0.f, 0.f};
  // Mirror mcflashinfer wrapper argument order: (B, A, C)
  v4f d = __builtin_mxc_mma_16x16x32f8(vb, va, c);

  const int col = g * 4;
  Cout[0 * 256 + row * 16 + col + 0] = d[0];
  Cout[0 * 256 + row * 16 + col + 1] = d[1];
  Cout[0 * 256 + row * 16 + col + 2] = d[2];
  Cout[0 * 256 + row * 16 + col + 3] = d[3];
}

extern "C" int launch_calib(const uint8_t* A, const uint8_t* Bn, float* Cout,
                            int VA, int VB, void* stream) {
  dim3 grid(1), block(64);
#define L(va, vb)                                                             \
  if (VA == va && VB == vb)                                                   \
    calib_kernel<va, vb><<<grid, block, 0, (mcStream_t)stream>>>(A, Bn, Cout);
  L(0, 0) L(0, 1) L(0, 2) L(0, 3)
  L(1, 0) L(1, 1) L(1, 2) L(1, 3)
#undef L
  return 0;
}
