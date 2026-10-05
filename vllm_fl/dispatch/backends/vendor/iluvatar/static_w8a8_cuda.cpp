#include <torch/extension.h>

extern "C" void launch_quant_a_static(
    const uint16_t* x,
    const float* q_scale,
    int8_t* x8,
    int M,
    int K);

extern "C" void launch_epilogue_i32_bf16(
    const int32_t* acc,
    const float* out_scale,
    const uint16_t* bias,
    uint16_t* out,
    int M,
    int N,
    int has_bias);

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
  TORCH_CHECK(w8.size(0) == K);
  TORCH_CHECK(x8.size(0) == M && x8.size(1) == K);
  TORCH_CHECK(acc.size(0) == M && acc.size(1) == N);
  TORCH_CHECK(out.size(0) == M && out.size(1) == N);
  TORCH_CHECK(q_scale.numel() == K);
  TORCH_CHECK(out_scale.numel() == N);

  launch_quant_a_static(
      reinterpret_cast<const uint16_t*>(x.data_ptr()),
      q_scale.data_ptr<float>(),
      x8.data_ptr<int8_t>(),
      M,
      K);
  at::_int_mm_out(acc, x8, w8);
  const bool has_bias = bias.defined() && bias.numel() > 0;
  launch_epilogue_i32_bf16(
      acc.data_ptr<int32_t>(),
      out_scale.data_ptr<float>(),
      has_bias ? reinterpret_cast<const uint16_t*>(bias.data_ptr()) : nullptr,
      reinterpret_cast<uint16_t*>(out.data_ptr()),
      M,
      N,
      has_bias ? 1 : 0);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("static_w8a8_linear", &static_w8a8_linear);
}
