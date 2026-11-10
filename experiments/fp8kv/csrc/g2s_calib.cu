// Calibration for the async global->shared copy path on C500 (xcore1000).
//
// Verifies, in isolation, that
//    __builtin_mxc_ldg_b128_bsm(dst_smem, src_global, 0, -1, false, true, false, false)
// followed by __builtin_mxc_arrive(<val>) actually lands 16 bytes per lane in
// shared memory where expected.  `wait` selects the arrive immediate so we can
// find which one really waits for the copies (mctlass uses 64 = arrive_gvmcnt(0),
// maca_async.h's __pipeline_wait_prior uses 0x3040).

#include <mc_runtime.h>
#include <cstdint>

namespace {

__device__ __forceinline__ void g2s16(void* dst, const void* src) {
  __builtin_mxc_ldg_b128_bsm(dst, src, 0, -1, false, true, false, false);
}

__device__ __forceinline__ void g2s16_ret(void* dst, const void* src) {
  __builtin_mxc_ldg_b128_bsm(dst, src, 0, -1, false, true, false, false);
}

__device__ __forceinline__ void wait_sel(int which) {
  if (which == 0)
    __builtin_mxc_arrive(64);      // arrive_gvmcnt(0)
  else if (which == 1)
    __builtin_mxc_arrive(0x3040);  // __pipeline_wait_prior
  else if (which == 2)
    __builtin_mxc_arrive(64 + 0x40);
  else
    __builtin_mxc_arrive(4096);    // arrive_bsmcnt(0)
}

}  // namespace

// Copy `rows` x 128 bytes with the async engine, wait, read back with ld4 and
// write the recovered bytes to out so the host can compare.
extern "C" __global__ void g2s_test(const uint8_t* __restrict__ src, int rows,
                                    int stride, int dst_stride, int which,
                                    uint8_t* __restrict__ out) {
  __shared__ uint8_t sm[32 * 160];
  const int lane = threadIdx.x & 63;
  const int t = lane >> 3;
  const int soff = (lane & 7) * 16;

#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int t2 = j * 8 + t;
    if (t2 < rows)
      g2s16(sm + t2 * dst_stride + soff, src + (size_t)t2 * stride + soff);
  }
  wait_sel(which);
  __syncwarp();

  // read back as 4-byte words
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int t2 = j * 8 + t;
    if (t2 < rows) {
      const uint8_t* p = sm + t2 * dst_stride + soff;
      uint8_t* o = out + (size_t)t2 * stride + soff;
#pragma unroll
      for (int i = 0; i < 16; ++i) o[i] = p[i];
    }
  }
}

extern "C" int launch_g2s_test(const void* src, int rows, int stride,
                               int dst_stride, int which, void* out,
                               int warps, int ctas, void* stream) {
  g2s_test<<<ctas, 64 * warps, 0, (mcStream_t)stream>>>(
      (const uint8_t*)src, rows, stride, dst_stride, which, (uint8_t*)out);
  return 0;
}
