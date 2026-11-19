// PDL with a partial second wave: does overlapping the dependent kernel's
// prologue with the primary's idle SMs recover wave-quantization loss?
#include <cuda_runtime.h>
#include <cstdio>

__global__ void k1(float *x, int iters) {
  float acc = 0.f;
  int tid = threadIdx.x;
  for (int i = 0; i < iters; i++) acc += x[(tid + i) & 1023];
  x[tid] = acc;
  // let the dependent launch now; the dependent still waits before reading
  __ivcorex_griddepcontrol_launch_dependents();
  // epilogue: the part that overlaps with the dependent's prologue
  for (int i = 0; i < iters / 2; i++) acc = fmaf(acc, 1.0000001f, 1.f);
  x[tid] = acc;
}

__global__ void k2(float *x, int iters) {
  __ivcorex_griddepcontrol_wait();
  float acc = 0.f;
  int tid = threadIdx.x;
  for (int i = 0; i < iters; i++) acc += x[(tid + i) & 1023];
  x[tid] = acc;
}

int main() {
  int iters = 200000;
  float *x = nullptr;
  cudaMalloc(&x, 1024 * sizeof(float));
  cudaMemset(x, 0, 1024 * sizeof(float));
  void *args1[] = {&x, &iters};

  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(20);   // 16 SMs -> 1 full wave + 4 CTAs
  cfg.blockDim = dim3(256);
  cfg.stream = nullptr;
  cudaLaunchAttribute attr;
  attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr.val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = &attr;
  cfg.numAttrs = 1;

  for (int r = 0; r < 5; r++) {
    k1<<<20, 256>>>(x, iters);
    k2<<<20, 256>>>(x, iters);
  }
  cudaDeviceSynchronize();

  cudaEvent_t s, e;
  cudaEventCreate(&s);
  cudaEventCreate(&e);
  cudaEventRecord(s);
  for (int r = 0; r < 50; r++) {
    k1<<<20, 256>>>(x, iters);
    k2<<<20, 256>>>(x, iters);
  }
  cudaEventRecord(e);
  cudaEventSynchronize(e);
  float t_serial = 0;
  cudaEventElapsedTime(&t_serial, s, e);

  cudaEventRecord(s);
  for (int r = 0; r < 50; r++) {
    cudaLaunchKernelExC(&cfg, (const void *)k1, args1);
    cudaLaunchKernelExC(&cfg, (const void *)k2, args1);
  }
  cudaEventRecord(e);
  cudaError_t se = cudaEventSynchronize(e);
  float t_pdl = 0;
  cudaEventElapsedTime(&t_pdl, s, e);
  printf("sync=%s serial=%.3fms pdl=%.3fms ratio=%.4f saved/pair=%.2fus\n",
         cudaGetErrorString(se), t_serial / 50, t_pdl / 50, t_pdl / t_serial,
         (t_serial - t_pdl) / 50 * 1000.0);
  return 0;
}
