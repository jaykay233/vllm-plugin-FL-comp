// Does __builtin_mxc_ldg_b128_bsm actually deliver the bytes, and what is the
// correct completion protocol? bw_stage.cu showed it is fast, but the tile it
// produced was never checked for content, and the GEMM built on it produces NaN.
//
// Each variant copies a 16 KB tile global->smem, then smem->global, so the
// round-trip can be diffed against the source in Python.
//   V0  cp.async, then __syncthreads()
//   V1  cp.async, arrive_gvmcnt(0), barrier_inst()
//   V2  cp.async, arrive_gvmcnt(0), __syncthreads(), barrier_inst()
//   V3  cp.async, __syncthreads(), arrive_gvmcnt(0), barrier_inst()
//   V4  plain 16B memcpy (control: must always be right)

#include <mc_runtime.h>
#include <cstdint>

typedef __NATIVE_VECTOR__(4, uint32_t) v4u;

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
  __builtin_mxc_ldg_b128_bsm((v4u*)smem, (const v4u*)gmem, 0, -1, true, true, false, true);
}

// Strided-destination test: does cp.async require a particular alignment on the
// smem side? gemm_fp8_v6 only staged row 0 correctly, and row 0 is the only
// 128-byte-aligned row for the strides used there (160 for A, 80 for B).
//   DST  smem row stride in bytes (64 data bytes per row, 4 chunks of 16B)
//   SRC  gmem row stride in bytes
// Same copy, but indexed per-row with an inner chunk loop, so the address is
// r*DST + c with r and c as independent loop variables and no div/mod.
// The flattened form (j/4, j%4) only worked for DST == 64, where it simplifies
// to the linear j*16 -- hence this variant to tell indexing from layout.
template <int DST, int NC>
__global__ void rowmajor_test(const uint8_t* __restrict__ in,
                             uint8_t* __restrict__ out, int nrows) {
  __shared__ __align__(16) uint8_t s[64 * 256];
  const int tid = threadIdx.x;
  for (int r = tid; r < nrows; r += blockDim.x) {
    for (int q = 0; q < NC; ++q) {
      cp_async16(s + r * DST + q * 16, in + (size_t)r * 64 + q * 16);
    }
  }
  __builtin_mxc_arrive_gvmcnt(0);
  __builtin_mxc_barrier_inst();
  for (int r = 0; r < nrows; ++r)
    for (int c = tid * 16; c < 64; c += blockDim.x * 16)
      __builtin_memcpy(out + (size_t)r * 64 + c, s + r * DST + c, 16);
}

template <int V, int DST = 64, int SRC = 64>
__global__ void strided_test(const uint8_t* __restrict__ in,
                             uint8_t* __restrict__ out, int nrows) {
  __shared__ __align__(16) uint8_t s[64 * 256];
  const int tid = threadIdx.x;
  const int nch = nrows * 4;

  for (int j = tid; j < nch; j += blockDim.x) {
    const int r = j / 4, c = (j - r * 4) * 16;
    cp_async16(s + r * DST + c, in + (size_t)r * SRC + c);
  }
  // V0 arrive_gvmcnt(0), V1 __builtin_mxc_arrive(64) (mctlass uses this),
  // V2 arrive_gvmcnt(0) + syncthreads + barrier
  if constexpr (V == 1) {
    __builtin_mxc_arrive(64);
    __builtin_mxc_barrier_inst();
  } else if constexpr (V == 2) {
    __builtin_mxc_arrive_gvmcnt(0);
    __syncthreads();
    __builtin_mxc_barrier_inst();
  } else {
    __builtin_mxc_arrive_gvmcnt(0);
    __builtin_mxc_barrier_inst();
  }

  for (int r = 0; r < nrows; ++r)
    for (int c = tid * 16; c < 64; c += blockDim.x * 16)
      __builtin_memcpy(out + (size_t)r * 64 + c, s + r * DST + c, 16);
}

template <int V>
__global__ void cpasync_test(const uint8_t* __restrict__ in,
                             uint8_t* __restrict__ out, int nchunks) {
  __shared__ __align__(16) uint8_t s[16384];
  const int tid = threadIdx.x;

  for (int j = tid; j < nchunks; j += blockDim.x) {
    if (V == 4) {
      v4u v;
      __builtin_memcpy(&v, in + j * 16, 16);
      __builtin_memcpy(s + j * 16, &v, 16);
    } else {
      cp_async16(s + j * 16, in + j * 16);
    }
  }

  if (V == 0) {
    __syncthreads();
  } else if (V == 1) {
    __builtin_mxc_arrive_gvmcnt(0);
    __builtin_mxc_barrier_inst();
  } else if (V == 2) {
    __builtin_mxc_arrive_gvmcnt(0);
    __syncthreads();
    __builtin_mxc_barrier_inst();
  } else if (V == 3) {
    __syncthreads();
    __builtin_mxc_arrive_gvmcnt(0);
    __builtin_mxc_barrier_inst();
  } else {
    __syncthreads();
  }

  for (int j = tid; j < nchunks; j += blockDim.x) {
    v4u v;
    __builtin_memcpy(&v, s + j * 16, 16);
    __builtin_memcpy(out + j * 16, &v, 16);
  }
}

extern "C" int launch_strided(const uint8_t* in, uint8_t* out, int nrows,
                              int cfg, void* stream) {
  if (cfg >= 100) {   // row-major indexing variants
    const int d = cfg - 100;
    if (d == 0) rowmajor_test<64, 4><<<1, 64, 0, (mcStream_t)stream>>>(in, out, nrows);
    else if (d == 1) rowmajor_test<80, 4><<<1, 64, 0, (mcStream_t)stream>>>(in, out, nrows);
    else if (d == 2) rowmajor_test<160, 4><<<1, 64, 0, (mcStream_t)stream>>>(in, out, nrows);
    else if (d == 3) rowmajor_test<128, 4><<<1, 64, 0, (mcStream_t)stream>>>(in, out, nrows);
    else return -1;
    return 0;
  }

  // cfg = wait_variant * 10 + dst_stride/16
  const int v = cfg / 10, d = cfg % 10;
#define D(V, S) strided_test<V, S, 64><<<1, 64, 0, (mcStream_t)stream>>>(in, out, nrows)
  if (v == 0) { if (d == 4) D(0, 64); else if (d == 5) D(0, 80); else if (d == 8) D(0, 128);
                else if (d == 0) D(0, 160); else return -1; }
  else if (v == 1) { if (d == 4) D(1, 64); else if (d == 5) D(1, 80); else if (d == 8) D(1, 128);
                     else if (d == 0) D(1, 160); else return -1; }
  else if (v == 2) { if (d == 4) D(2, 64); else if (d == 5) D(2, 80); else if (d == 8) D(2, 128);
                     else if (d == 0) D(2, 160); else return -1; }
  else return -1;
#undef D
  return 0;
}

extern "C" int launch_cpasync_test(const uint8_t* in, uint8_t* out, int nchunks,
                                   int variant, void* stream) {
#define L(V) cpasync_test<V><<<1, 256, 0, (mcStream_t)stream>>>(in, out, nchunks)
  switch (variant) {
    case 0: L(0); return 0;
    case 1: L(1); return 0;
    case 2: L(2); return 0;
    case 3: L(3); return 0;
    case 4: L(4); return 0;
    default: return -1;
  }
}
