#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>
#include <cmath>

__device__ __forceinline__ float bf16_to_f32(uint16_t x) {
  uint32_t bits = static_cast<uint32_t>(x) << 16;
  float f;
  memcpy(&f, &bits, sizeof(f));
  return f;
}

__device__ __forceinline__ uint16_t f32_to_bf16(float f) {
  uint32_t bits;
  memcpy(&bits, &f, sizeof(bits));
  return static_cast<uint16_t>(bits >> 16);
}

__global__ void quant_a_static_kernel(
    const uint16_t* __restrict__ x,
    const float* __restrict__ q_scale,
    int8_t* __restrict__ x8,
    int M,
    int K) {
  const int n = M * K;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int step = blockDim.x * gridDim.x;
  for (; i < n; i += step) {
    const int k = i % K;
    float v = bf16_to_f32(x[i]) * q_scale[k];
    int q = __float2int_rn(v);
    q = max(-128, min(127, q));
    x8[i] = static_cast<int8_t>(q);
  }
}

__global__ void epilogue_i32_bf16_kernel(
    const int32_t* __restrict__ acc,
    const float* __restrict__ out_scale,
    const uint16_t* __restrict__ bias,
    uint16_t* __restrict__ out,
    int M,
    int N,
    int has_bias) {
  const int n = M * N;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int step = blockDim.x * gridDim.x;
  for (; i < n; i += step) {
    const int col = i % N;
    float v = static_cast<float>(acc[i]) * out_scale[col];
    if (has_bias) {
      v += bf16_to_f32(bias[col]);
    }
    out[i] = f32_to_bf16(v);
  }
}

extern "C" void launch_quant_a_static(
    const uint16_t* x,
    const float* q_scale,
    int8_t* x8,
    int M,
    int K) {
  const int n = M * K;
  const int t = 256;
  int blocks = (n + t - 1) / t;
  if (blocks > 4096) {
    blocks = 4096;
  }
  quant_a_static_kernel<<<blocks, t>>>(x, q_scale, x8, M, K);
}

extern "C" void launch_epilogue_i32_bf16(
    const int32_t* acc,
    const float* out_scale,
    const uint16_t* bias,
    uint16_t* out,
    int M,
    int N,
    int has_bias) {
  const int n = M * N;
  const int t = 256;
  int blocks = (n + t - 1) / t;
  if (blocks > 4096) {
    blocks = 4096;
  }
  epilogue_i32_bf16_kernel<<<blocks, t>>>(acc, out_scale, bias, out, M, N, has_bias);
}
