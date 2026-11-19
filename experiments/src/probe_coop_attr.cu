#include <cuda_runtime.h>
#include <cstdio>

int main() {
  int cooperative = -1;
  cudaError_t e = cudaDeviceGetAttribute(
      &cooperative, cudaDevAttrCooperativeLaunch, 0);
  printf("cooperativeLaunch=%d err=%s\n", cooperative,
         cudaGetErrorString(e));
  return 0;
}
