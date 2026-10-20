#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

extern "C" void launch_quant_a_static(
    const uint16_t* x,
    const float* q_scale,
    int8_t* x8,
    int M,
    int K,
    cudaStream_t stream);

extern "C" void launch_epilogue_i32_bf16(
    const int32_t* acc,
    const float* out_scale,
    const uint16_t* bias,
    uint16_t* out,
    int M,
    int N,
    int has_bias,
    cudaStream_t stream);

void static_w8a8_linear(
    torch::Tensor x,
    torch::Tensor w8,
    torch::Tensor q_scale,
    torch::Tensor out_scale,
    torch::Tensor x8,
    torch::Tensor acc,
    torch::Tensor out,
    torch::Tensor bias) {
  TORCH_CHECK(x.scalar_type() == at::kBFloat16);
  TORCH_CHECK(w8.scalar_type() == at::kChar);
  TORCH_CHECK(x8.scalar_type() == at::kChar);
  TORCH_CHECK(acc.scalar_type() == at::kInt);
  TORCH_CHECK(out.scalar_type() == at::kBFloat16);
  TORCH_CHECK(x.is_contiguous() && w8.is_contiguous());
  TORCH_CHECK(x8.is_contiguous() && acc.is_contiguous() && out.is_contiguous());
  const int M = static_cast<int>(x.size(0));
  const int K = static_cast<int>(x.size(1));
  const int N = static_cast<int>(w8.size(1));
  // at::_int_mm on BI-V150 rejects M <= 16.  Pad to 32 so FULL_DECODE_ONLY
  // capture sizes {1,2,4,8,16} can record a legal INT8 GEMM; replay keeps the
  // padded launch while only the first M rows of `out` are consumed.
  constexpr int kIntMmMinM = 32;
  const int M_eff = M < kIntMmMinM ? kIntMmMinM : M;
  TORCH_CHECK(w8.size(0) == K);
  TORCH_CHECK(x8.size(0) >= M_eff && x8.size(1) == K);
  TORCH_CHECK(acc.size(0) >= M_eff && acc.size(1) == N);
  TORCH_CHECK(out.size(0) >= M_eff && out.size(1) == N);
  TORCH_CHECK(q_scale.numel() == K);
  TORCH_CHECK(out_scale.numel() == N);

  // Must launch on the current PyTorch stream so CUDAGraph capture records us
  // (bare <<<>>> defaults to stream 0 → empty graph / stale replay on BI-V150).
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  if (M_eff > M) {
    x8.narrow(0, M, M_eff - M).zero_();
  }
  launch_quant_a_static(
      reinterpret_cast<const uint16_t*>(x.data_ptr()),
      q_scale.data_ptr<float>(),
      x8.data_ptr<int8_t>(),
      M,
      K,
      stream);
  auto x8_mm = x8.narrow(0, 0, M_eff);
  auto acc_mm = acc.narrow(0, 0, M_eff);
  at::_int_mm_out(acc_mm, x8_mm, w8);
  const bool has_bias = bias.defined() && bias.numel() > 0;
  launch_epilogue_i32_bf16(
      acc_mm.data_ptr<int32_t>(),
      out_scale.data_ptr<float>(),
      has_bias ? reinterpret_cast<const uint16_t*>(bias.data_ptr()) : nullptr,
      reinterpret_cast<uint16_t*>(out.data_ptr()),
      M_eff,
      N,
      has_bias ? 1 : 0,
      stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("static_w8a8_linear", &static_w8a8_linear);
}
