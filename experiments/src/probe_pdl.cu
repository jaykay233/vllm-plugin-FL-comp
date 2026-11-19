// Does PDL (programmatic dependent launch) work on corex/ivcore11, and how
// much inter-kernel drain does it remove?  The decode step has ~336 dependent
// kernels; if PDL removes a few us per boundary it is the one remaining lever
// for the ~7.8 ms structural gap.
#include <cuda_runtime.h>
#include <cstdio>

__global__ void k1(float *x, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  for (int k = i; k < n; k += stride) x[k] += 1.0f;
  __ivcorex_griddepcontrol_launch_dependents();
  for (int k = i; k < n; k += stride) x[k] += 1.0f;
}

__global__ void k2(float *x, int n) {
  __ivcorex_griddepcontrol_wait();
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  for (int k = i; k < n; k += stride) x[k] *= 1.000001f;
}

int main() {
  int n = 1 << 24;
  float *x = nullptr;
  cudaMalloc(&x, (size_t)n * sizeof(float));
  cudaMemset(x, 0, (size_t)n * sizeof(float));
  void *args[] = {&x, &n};

  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(256);
  cfg.blockDim = dim3(256);
  cfg.dynamicSmemBytes = 0;
  cfg.stream = nullptr;
  cudaLaunchAttribute attr;
  attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr.val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = &attr;
  cfg.numAttrs = 1;

  for (int r = 0; r < 5; r++) {
    k1<<<256, 256>>>(x, n);
    k2<<<256, 256>>>(x, n);
  }
  cudaError_t we = cudaDeviceSynchronize();
  printf("warmup sync=%s\n", cudaGetErrorString(we));

  cudaEvent_t s, e;
  cudaEventCreate(&s);
  cudaEventCreate(&e);
  cudaEventRecord(s);
  for (int r = 0; r < 100; r++) {
    k1<<<256, 256>>>(x, n);
    k2<<<256, 256>>>(x, n);
  }
  cudaEventRecord(e);
  cudaEventSynchronize(e);
  float t_serial = 0;
  cudaEventElapsedTime(&t_serial, s, e);

  cudaEventRecord(s);
  for (int r = 0; r < 100; r++) {
    k1<<<256, 256>>>(x, n);
    cudaError_t le = cudaLaunchKernelExC(&cfg, (const void *)k2, args);
    if (r == 0) printf("pdl launch=%s\n", cudaGetErrorString(le));
  }
  cudaEventRecord(e);
  cudaError_t se = cudaEventSynchronize(e);
  float t_pdl = 0;
  cudaEventElapsedTime(&t_pdl, s, e);
  printf("pdl sync=%s\n", cudaGetErrorString(se));
  printf("serial=%.3f ms  pdl=%.3f ms  ratio=%.4f  saved_per_pair=%.2f us\n",
         t_serial / 100, t_pdl / 100, t_pdl / t_serial,
         (t_serial - t_pdl) / 100 * 1000.0);
  return 0;
}
