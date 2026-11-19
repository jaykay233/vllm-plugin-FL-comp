#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cstdint>

namespace {
constexpr int kHd = 128;
constexpr int kPage = 16;
}  // namespace

extern "C" void launch_native_decode(const uint16_t *q, const uint16_t *k,
                                     const uint16_t *v, uint16_t *out, float *Op,
                                     float *Mp, float *Lp, const int32_t *block_table,
                                     const int32_t *seqused, const int32_t *qrow,
                                     int nunit, int S, int maxblk, int NH, int NKVH,
                                     int W, float log2scale, cudaStream_t stream);

// q/k/v in the layout the vendor decode kernel already consumes:
//   q   [rows, NH, 128] bf16 contiguous, one token per sequence
//   k,v [num_blocks, NKVH, PAGE, 128] bf16 contiguous (HND, the layout the
//       Iluvatar KV-cache patch installs)
// Op/Mp/Lp are the split partials, sized and cached by the caller so that no
// allocation happens inside a CUDA graph.
void native_decode(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor out,
                   torch::Tensor block_table, torch::Tensor seqused,
                   torch::Tensor qrow, torch::Tensor Op, torch::Tensor Mp,
                   torch::Tensor Lp, double softmax_scale, int64_t S, int64_t W) {
  TORCH_CHECK(q.scalar_type() == at::kBFloat16 && out.scalar_type() == at::kBFloat16);
  TORCH_CHECK(k.scalar_type() == at::kBFloat16 && v.scalar_type() == at::kBFloat16);
  TORCH_CHECK(block_table.scalar_type() == at::kInt, "block_table must be int32");
  TORCH_CHECK(seqused.scalar_type() == at::kInt, "seqused must be int32");
  TORCH_CHECK(qrow.scalar_type() == at::kInt, "qrow must be int32");
  TORCH_CHECK(q.is_contiguous() && out.is_contiguous());
  TORCH_CHECK(k.is_contiguous() && v.is_contiguous());
  TORCH_CHECK(block_table.is_contiguous() && seqused.is_contiguous());
  TORCH_CHECK(qrow.is_contiguous());
  TORCH_CHECK(Op.is_contiguous() && Mp.is_contiguous() && Lp.is_contiguous());

  const int rows = static_cast<int>(q.size(0));
  const int NH = static_cast<int>(q.size(1));
  const int NKVH = static_cast<int>(k.size(1));
  const int maxblk = static_cast<int>(block_table.size(1));
  TORCH_CHECK(q.size(2) == kHd);
  TORCH_CHECK(k.size(2) == kPage && k.size(3) == kHd);
  TORCH_CHECK(NH % NKVH == 0 && NH / NKVH <= 8);
  TORCH_CHECK(block_table.size(0) == rows && seqused.numel() == rows);
  TORCH_CHECK(S >= 1 && S <= 16);
  const int nunit = rows * NKVH;
  TORCH_CHECK(Mp.numel() >= (int64_t)nunit * S * (NH / NKVH));

  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  launch_native_decode(
      reinterpret_cast<const uint16_t *>(q.data_ptr()),
      reinterpret_cast<const uint16_t *>(k.data_ptr()),
      reinterpret_cast<const uint16_t *>(v.data_ptr()),
      reinterpret_cast<uint16_t *>(out.data_ptr()), Op.data_ptr<float>(),
      Mp.data_ptr<float>(), Lp.data_ptr<float>(), block_table.data_ptr<int32_t>(),
      seqused.data_ptr<int32_t>(), qrow.data_ptr<int32_t>(), nunit,
      static_cast<int>(S), maxblk, NH, NKVH, static_cast<int>(W),
      static_cast<float>(softmax_scale * 1.4426950408889634), stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("native_decode", &native_decode, "native split-KV decode attention");
}
