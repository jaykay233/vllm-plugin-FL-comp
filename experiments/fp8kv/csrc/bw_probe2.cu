// Bandwidth probes for the fp8 decode kernel.
//
//   stream_pages : reads the *exact* paged K/V address stream the decode kernel
//                  consumes, with fully coalesced 16B/lane loads, no compute.
//                  -> upper bound reachable by the decode grid/pipeline.
//   stream_flat  : reads a flat buffer to find the machine's real peak.
//
// Both take a warps-per-CTA knob so we can separate "warp count" from
// "access pattern".

#include <mc_runtime.h>
#include <cstdint>

namespace {

__device__ __forceinline__ float wsum(float x) {
  const uint64_t m = ~0ull;
  x += __shfl_xor_sync(m, x, 16);
  x += __shfl_xor_sync(m, x, 32);
  return x;
}

__device__ __forceinline__ float lane_sum(uint4 v) {
  return (float)(v.x + v.y + v.z + v.w);
}

}  // namespace

// k/v: [num_blocks, P, HKV, D] uint8.  One warp per (b, kvh, split).
extern "C" __global__ void stream_pages(const uint8_t* __restrict__ k,
                                        const uint8_t* __restrict__ v,
                                        const int* __restrict__ bt,
                                        const int* __restrict__ lens,
                                        float* __restrict__ out, int B, int HKV,
                                        int P, int stride_bt, int splits,
                                        int warps_per_cta, int tstep) {
  const int lane = threadIdx.x & 63;
  const int wid = (blockIdx.x * warps_per_cta) + (threadIdx.x >> 6);
  const int split = wid % splits;
  const int rest = wid / splits;
  const int kvh = rest % HKV;
  const int b = rest / HKV;

  const int seqlen = lens[b];
  const int npages = (seqlen + P - 1) / P;
  const int chunk = (npages + splits - 1) / splits;
  int p0 = split * chunk;
  int p1 = p0 + chunk;
  if (p1 > npages) p1 = npages;

  const size_t tok_stride = (size_t)HKV * 128;
  const int t = lane >> 3;
  const int soff = (lane & 7) * 16;

  float acc = 0.f;
  for (int p = p0; p < p1; ++p) {
    const int page_id = bt[(size_t)b * stride_bt + p];
    const size_t base = (size_t)page_id * P * tok_stride + (size_t)kvh * 128;
    const int page_valid = seqlen - p * P;
    for (int tl = 0; tl < P; tl += tstep) {
      const int nvalid = min(tstep, page_valid - tl);
      if (nvalid <= 0) break;
      if (t < nvalid) {
        const uint8_t* kp = k + base + (size_t)(tl + t) * tok_stride + soff;
        const uint8_t* vp = v + base + (size_t)(tl + t) * tok_stride + soff;
        acc += lane_sum(*(const uint4*)kp);
        acc += lane_sum(*(const uint4*)vp);
      }
    }
  }
  acc = wsum(acc);
  if (lane == 0) out[wid] = acc;
}

// Flat contiguous read.
extern "C" __global__ void stream_flat(const uint4* __restrict__ src,
                                       float* __restrict__ out, int n4,
                                       int warps_per_cta) {
  const int lane = threadIdx.x & 63;
  const int wid = (blockIdx.x * warps_per_cta) + (threadIdx.x >> 6);
  const int nw = gridDim.x * warps_per_cta;
  const int stride = 64 * nw;
  float acc = 0.f;
  for (int i = wid * 64 + lane; i < n4; i += stride) acc += lane_sum(src[i]);
  acc = wsum(acc);
  if (lane == 0) out[wid] = acc;
}

extern "C" int launch_stream_pages(const void* k, const void* v,
                                   const void* bt, const void* lens, void* out,
                                   int B, int HKV, int P, int stride_bt,
                                   int splits, int warps_per_cta, int tstep,
                                   void* stream) {
  const int nw = B * HKV * splits;
  const int ctas = (nw + warps_per_cta - 1) / warps_per_cta;
  stream_pages<<<ctas, 64 * warps_per_cta, 0, (mcStream_t)stream>>>(
      (const uint8_t*)k, (const uint8_t*)v, (const int*)bt, (const int*)lens,
      (float*)out, B, HKV, P, stride_bt, splits, warps_per_cta, tstep);
  return 0;
}

extern "C" int launch_stream_flat(const void* src, void* out, int n4,
                                  int warps_per_cta, void* stream) {
  const int nw = 2048 * warps_per_cta;
  stream_flat<<<2048, 64 * warps_per_cta, 0, (mcStream_t)stream>>>(
      (const uint4*)src, (float*)out, n4, warps_per_cta);
  return 0;
}
