// Dump the staged smem tiles so staging can be diffed byte-for-byte in Python.
// gemm_fp8_v6 produces NaN even for a single k-chunk, so the fault is either in
// the cp.async staging (indexing / stride / alignment) or in the fragment decode.
// This kernel reuses the exact same staging code and then copies smem back out.

#include <mc_runtime.h>
#include <cstdint>
#include <maca_bfloat16.h>

typedef __NATIVE_VECTOR__(4, uint32_t) v4u;

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
  __builtin_mxc_ldg_b128_bsm((v4u*)smem, (const v4u*)gmem, 0, -1, true, true, false, true);
}

template <int WPB, int KC, int NT, int PADA, int PADF>
__global__ void dump_kernel(const __maca_bfloat16* __restrict__ A,
                            const uint8_t* __restrict__ Bn,
                            uint8_t* __restrict__ dA,   // [64][RSB]
                            uint8_t* __restrict__ dB,   // [BROWS][RSF]
                            int N, int K) {
  constexpr int RSB = KC * 2 + PADA;
  constexpr int RSF = KC + PADF;
  constexpr int BROWS = WPB * NT * 16;
  constexpr int NTHR = WPB * 64;
  constexpr int ACL = KC * 2 / 16;
  constexpr int BCL = KC / 16;
  constexpr int ACH = 64 * ACL;
  constexpr int BCH = BROWS * BCL;

  __shared__ __align__(16) uint8_t sAb[64 * RSB];
  __shared__ __align__(16) uint8_t sBf[BROWS * RSF];

  const int tid = threadIdx.x;
  const int n0 = blockIdx.x * (WPB * NT * 16);
  if (n0 >= N) return;

  for (int j = tid; j < ACH; j += NTHR) {
    const int r = j / ACL, c = (j - r * ACL) * 16;
    cp_async16(sAb + r * RSB + c, A + (size_t)r * K + c / 2);
  }
  for (int j = tid; j < BCH; j += NTHR) {
    const int r = j / BCL, c = (j - r * BCL) * 16;
    cp_async16(sBf + r * RSF + c, Bn + (size_t)(n0 + r) * K + c);
  }
  __builtin_mxc_arrive_gvmcnt(0);
  __builtin_mxc_barrier_inst();

  for (int i = tid; i < 64 * RSB; i += NTHR) dA[i] = sAb[i];
  for (int i = tid; i < BROWS * RSF; i += NTHR) dB[i] = sBf[i];
}

extern "C" int launch_dump(const void* A, const uint8_t* Bn, uint8_t* dA,
                           uint8_t* dB, int N, int K, void* stream) {
  constexpr int WPB = 4, KC = 64, NT = 2, PADA = 32, PADF = 16;
  // one block only: every block would otherwise race on the same output buffers
  dump_kernel<WPB, KC, NT, PADA, PADF><<<1, WPB * 64, 0,
                                         (mcStream_t)stream>>>(
      (const __maca_bfloat16*)A, Bn, dA, dB, N, K);
  return 0;
}
