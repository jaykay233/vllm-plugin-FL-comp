// Minimal MACA toolchain probe: vector add + warp-size discovery.
#include <mc_runtime.h>
#include <cstdint>

__global__ void add_kernel(const float* __restrict__ a,
                           const float* __restrict__ b,
                           float* __restrict__ c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}

extern "C" int launch_add(const float* a, const float* b, float* c, int n,
                          void* stream) {
  int th = 64;
  int bl = (n + th - 1) / th;
  add_kernel<<<bl, th, 0, (mcStream_t)stream>>>(a, b, c, n);
  return 0;
}

__global__ void warp_probe_kernel(int* out) { out[0] = (int)warpSize; }

extern "C" int warp_probe(int* out, void* stream) {
  warp_probe_kernel<<<1, 64, 0, (mcStream_t)stream>>>(out);
  return 0;
}
