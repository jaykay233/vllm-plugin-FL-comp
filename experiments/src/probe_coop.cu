// Does ivcore11 / corex support cooperative launch + grid-wide sync?
// A persistent megakernel (the only lever that could remove all 336 graph
// boundaries) needs this.  If unsupported, that lever is dead on this stack.
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstdio>

namespace cg = cooperative_groups;

__global__ void sync_kernel(int *out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  out[i] = i;
  cg::this_grid().sync();
  out[i] += 1;
}

int main() {
  int dev = 0;
  cudaSetDevice(dev);
  int coop = -1;
  cudaError_t ae = cudaDeviceGetAttribute(
      &coop, cudaDevAttrCooperativeLaunch, dev);
  printf("cooperativeLaunch=%d (attr_err=%s)\n", coop, cudaGetErrorString(ae));

  int *d = nullptr;
  cudaMalloc(&d, 2048 * sizeof(int));
  void *args[] = {&d};
  cudaError_t e = cudaLaunchCooperativeKernel(
      (void *)sync_kernel, dim3(32), dim3(64), args, 0, 0);
  printf("launch=%s\n", cudaGetErrorString(e));
  if (e == cudaSuccess) {
    e = cudaDeviceSynchronize();
    printf("sync=%s\n", cudaGetErrorString(e));
    int h[4] = {0, 0, 0, 0};
    cudaMemcpy(h, d, sizeof(h), cudaMemcpyDeviceToHost);
    printf("out=%d %d (expected 1 4)\n", h[0], h[3]);
  }
  return 0;
}
