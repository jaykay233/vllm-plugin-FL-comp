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

// Vectorized epi (int4 acc + __ldg scales) — same math as scalar path.
__global__ void epilogue_i32_bf16_vec4_kernel(
    const int32_t* __restrict__ acc,
    const float* __restrict__ out_scale,
    const uint16_t* __restrict__ bias,
    uint16_t* __restrict__ out,
    int M,
    int N,
    int has_bias) {
  const int row = blockIdx.x;
  if (row >= M) return;
  const int32_t* __restrict__ acc_row = acc + row * (size_t)N;
  uint16_t* __restrict__ out_row = out + row * (size_t)N;
  const int n4 = N >> 2;
  for (int c4 = threadIdx.x; c4 < n4; c4 += blockDim.x) {
    const int c = c4 << 2;
    int4 a = *reinterpret_cast<const int4*>(acc_row + c);
    float s0 = __ldg(out_scale + c + 0);
    float s1 = __ldg(out_scale + c + 1);
    float s2 = __ldg(out_scale + c + 2);
    float s3 = __ldg(out_scale + c + 3);
    float v0 = (float)a.x * s0;
    float v1 = (float)a.y * s1;
    float v2 = (float)a.z * s2;
    float v3 = (float)a.w * s3;
    if (has_bias) {
      v0 += bf16_to_f32(__ldg(bias + c + 0));
      v1 += bf16_to_f32(__ldg(bias + c + 1));
      v2 += bf16_to_f32(__ldg(bias + c + 2));
      v3 += bf16_to_f32(__ldg(bias + c + 3));
    }
    uint2 packed;
    uint16_t* p = reinterpret_cast<uint16_t*>(&packed);
    p[0] = f32_to_bf16(v0);
    p[1] = f32_to_bf16(v1);
    p[2] = f32_to_bf16(v2);
    p[3] = f32_to_bf16(v3);
    *reinterpret_cast<uint2*>(out_row + c) = packed;
  }
  for (int c = (n4 << 2) + threadIdx.x; c < N; c += blockDim.x) {
    float v = (float)acc_row[c] * __ldg(out_scale + c);
    if (has_bias) v += bf16_to_f32(__ldg(bias + c));
    out_row[c] = f32_to_bf16(v);
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
    int K,
    cudaStream_t stream) {
  const int n = M * K;
  const int t = 256;
  int blocks = (n + t - 1) / t;
  if (blocks > 4096) {
    blocks = 4096;
  }
  quant_a_static_kernel<<<blocks, t, 0, stream>>>(x, q_scale, x8, M, K);
}

extern "C" void launch_epilogue_i32_bf16(
    const int32_t* acc,
    const float* out_scale,
    const uint16_t* bias,
    uint16_t* out,
    int M,
    int N,
    int has_bias,
    cudaStream_t stream) {
  if (N % 4 == 0 && M > 0 && M <= 4096) {
    const int threads = (N >= 4096) ? 512 : 256;
    epilogue_i32_bf16_vec4_kernel<<<M, threads, 0, stream>>>(
        acc, out_scale, bias, out, M, N, has_bias);
    return;
  }
  const int n = M * N;
  const int t = 256;
  int blocks = (n + t - 1) / t;
  if (blocks > 4096) {
    blocks = 4096;
  }
  epilogue_i32_bf16_kernel<<<blocks, t, 0, stream>>>(
      acc, out_scale, bias, out, M, N, has_bias);
}
