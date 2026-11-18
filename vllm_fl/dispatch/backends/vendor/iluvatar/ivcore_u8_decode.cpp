// Torch binding for ivcore u8 decode (v42: INT8-K + INT8-V with per-token scales).

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAGraphsC10Utils.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

extern "C" void launch_ivcore_u8_decode(
    const uint16_t* q, const uint8_t* k_u8, const uint8_t* v_u8,
    const float* k_scales, const float* v_scales, int v_mode,
    const int32_t* block_table, const int32_t* seqused,
    uint16_t* out_partial, float* lse_partial,
    int batch, int n_qo, int n_kv, int max_pages, int stride_bt,
    float scale_log2, int split, int group, int use_sme,
    cudaStream_t stream);

extern "C" void launch_ivcore_combine(
    const uint16_t* out_partial, const float* lse_partial, uint16_t* out,
    int batch, int n_qo, int split, cudaStream_t stream);

extern "C" void launch_ivcore_pack_bt(
    const uint16_t* k_phys, const uint16_t* v_phys,
    uint8_t* ku8, uint8_t* vu8, float* ksc, float* vsc,
    const int32_t* block_table, const int32_t* seqused,
    int batch, int max_pages, int stride_bt, int n_kv, int n_blocks,
    cudaStream_t stream);

extern "C" void launch_ivcore_pack_bt_last(
    const uint16_t* k_phys, const uint16_t* v_phys,
    uint8_t* ku8, uint8_t* vu8, float* ksc, float* vsc,
    const int32_t* block_table, const int32_t* seqused,
    int batch, int max_pages, int stride_bt, int n_kv, int n_blocks,
    cudaStream_t stream);

static constexpr int64_t kPage = 16;
static constexpr int64_t kHd = 128;
static constexpr int64_t kGroup = 8;

torch::Tensor ivcore_u8_paged_decode(
    torch::Tensor q,           // [B, N_QO, 128] bf16
    torch::Tensor k_u8,        // [blocks, N_KV, 128, 16] u8  HD-major int8
    torch::Tensor v_u8,        // [blocks, N_KV, 16, 128] u8  page-major int8
    torch::Tensor block_table, // [B, max_pages] i32
    torch::Tensor seqused,     // [B] i32
    double softmax_scale,
    int64_t split,
    int64_t v_mode,            // 0=hi-byte, 1=fp8e4m3, 2=int8*scale
    c10::optional<torch::Tensor> k_scales_opt,  // [blocks,N_KV,16] f32
    c10::optional<torch::Tensor> v_scales_opt,  // [blocks,N_KV,16] f32 if mode=2
    c10::optional<torch::Tensor> out_opt,       // optional dst [B,N_QO,HD] bf16
    int64_t use_sme = 0) {                      // SME G2S double-buffer (trial)
  TORCH_CHECK(q.is_cuda() && k_u8.is_cuda() && v_u8.is_cuda());
  TORCH_CHECK(q.scalar_type() == at::kBFloat16);
  TORCH_CHECK(k_u8.scalar_type() == at::kByte);
  TORCH_CHECK(v_u8.scalar_type() == at::kByte);
  TORCH_CHECK(q.dim() == 3 && k_u8.dim() == 4 && v_u8.dim() == 4);
  TORCH_CHECK(q.size(2) == kHd && k_u8.size(2) == kHd && k_u8.size(3) == kPage);
  TORCH_CHECK(v_u8.size(2) == kPage && v_u8.size(3) == kHd);
  TORCH_CHECK(split >= 1 && split <= 8);
  TORCH_CHECK(v_mode >= 0 && v_mode <= 2);

  const int64_t B = q.size(0);
  const int64_t N_QO = q.size(1);
  const int64_t N_KV = k_u8.size(1);
  TORCH_CHECK(N_QO == N_KV * kGroup);
  TORCH_CHECK(v_u8.size(1) == N_KV);

  TORCH_CHECK(k_scales_opt.has_value(), "k_scales [blocks,N_KV,16] f32 required");
  auto k_scales = k_scales_opt.value();
  TORCH_CHECK(k_scales.is_contiguous() && k_scales.scalar_type() == at::kFloat);
  TORCH_CHECK(k_scales.sizes() ==
              at::IntArrayRef({k_u8.size(0), N_KV, kPage}));

  const float* v_scales_ptr = nullptr;
  torch::Tensor v_scales;
  if (v_mode == 2) {
    TORCH_CHECK(v_scales_opt.has_value(), "v_mode=2 requires v_scales [blocks,N_KV,16] f32");
    v_scales = v_scales_opt.value();
    TORCH_CHECK(v_scales.is_contiguous() && v_scales.scalar_type() == at::kFloat);
    TORCH_CHECK(v_scales.sizes() ==
                at::IntArrayRef({k_u8.size(0), N_KV, kPage}));
    v_scales_ptr = v_scales.data_ptr<float>();
  }

  // Cudagraph: never allocate. Callers must warm workspaces / pass contig views.
  const bool capturing = c10::cuda::currentStreamCaptureStatusMayInitCtx() !=
                         c10::cuda::CaptureStatus::None;
  TORCH_CHECK(q.is_contiguous() && k_u8.is_contiguous() && v_u8.is_contiguous(),
              "ivcore decode requires contiguous q/k_u8/v_u8");
  TORCH_CHECK(block_table.scalar_type() == at::kInt && block_table.is_contiguous(),
              "ivcore decode requires contiguous int32 block_table");
  TORCH_CHECK(seqused.scalar_type() == at::kInt && seqused.is_contiguous(),
              "ivcore decode requires contiguous int32 seqused");
  auto q_c = q;
  auto k_c = k_u8;
  auto v_c = v_u8;
  auto bt = block_table;
  auto su = seqused;

  const int64_t max_pages = bt.size(1);
  auto opts_f = q.options().dtype(at::kFloat);

  // Persistent decode workspace. Partials are bf16 (half combine BW). Idle
  // CTAs zero their own slots in-kernel — no host zero_() per call.
  // Grow-only; allocation forbidden while capturing (warm before CG).
  struct DecWs {
    torch::Tensor out_partial;
    torch::Tensor lse_partial;
    torch::Tensor out;
    int64_t split = 0, B = 0, N_QO = 0;
  };
  static thread_local DecWs ws;
  const bool need_ws = ws.split < split || ws.B < B || ws.N_QO < N_QO ||
                       !ws.out.defined() || ws.out.device() != q.device();
  if (need_ws) {
    TORCH_CHECK(!capturing,
                "ivcore decode workspace not warmed before cudagraph capture "
                "(call warm_for_cudagraph)");
    const int64_t Bs = std::max(ws.B, B);
    const int64_t Ss = std::max(ws.split, split);
    const int64_t Hs = std::max(ws.N_QO, N_QO);
    ws.out_partial = torch::empty({Ss, Bs, Hs, kHd}, q.options());  // bf16
    ws.lse_partial = torch::empty({Bs, Hs, Ss}, opts_f);
    ws.out = torch::empty({Bs, Hs, kHd}, q.options());
    ws.split = Ss;
    ws.B = Bs;
    ws.N_QO = Hs;
  }

  torch::Tensor out_dst;
  if (out_opt.has_value()) {
    out_dst = out_opt.value();
    TORCH_CHECK(out_dst.is_cuda() && out_dst.is_contiguous());
    TORCH_CHECK(out_dst.scalar_type() == at::kBFloat16);
    TORCH_CHECK(out_dst.sizes() == at::IntArrayRef({B, N_QO, kHd}));
  } else {
    out_dst = ws.out.narrow(0, 0, B).narrow(1, 0, N_QO);
  }

  // Narrow growable workspace to this call's (split,B,N_QO).
  auto out_partial =
      ws.out_partial.narrow(0, 0, split).narrow(1, 0, B).narrow(2, 0, N_QO);
  auto lse_partial =
      ws.lse_partial.narrow(0, 0, B).narrow(1, 0, N_QO).narrow(2, 0, split);

  const float scale_log2 =
      static_cast<float>(softmax_scale) * 1.4426950408889634f;  // 1/ln2

  at::cuda::CUDAGuard guard(q.device());
  auto stream = at::cuda::getCurrentCUDAStream();

  launch_ivcore_u8_decode(
      reinterpret_cast<const uint16_t*>(q_c.data_ptr()),
      k_c.data_ptr<uint8_t>(),
      v_c.data_ptr<uint8_t>(),
      k_scales.data_ptr<float>(),
      v_scales_ptr,
      static_cast<int>(v_mode),
      bt.data_ptr<int32_t>(),
      su.data_ptr<int32_t>(),
      reinterpret_cast<uint16_t*>(out_partial.data_ptr()),
      lse_partial.data_ptr<float>(),
      static_cast<int>(B), static_cast<int>(N_QO), static_cast<int>(N_KV),
      static_cast<int>(max_pages), static_cast<int>(bt.stride(0)),
      scale_log2, static_cast<int>(split), static_cast<int>(kGroup),
      static_cast<int>(use_sme != 0),
      stream.stream());

  launch_ivcore_combine(
      reinterpret_cast<const uint16_t*>(out_partial.data_ptr()),
      lse_partial.data_ptr<float>(),
      reinterpret_cast<uint16_t*>(out_dst.data_ptr()),
      static_cast<int>(B), static_cast<int>(N_QO), static_cast<int>(split),
      stream.stream());

  return out_dst;
}

void ivcore_pack_bt(
    torch::Tensor k_phys,      // [blocks, n_kv, PAGE, HD] bf16 contiguous
    torch::Tensor v_phys,
    torch::Tensor ku8,         // [blocks, n_kv, HD, PAGE] u8 int8
    torch::Tensor vu8,         // [blocks, n_kv, PAGE, HD] u8 int8
    torch::Tensor ksc,         // [blocks, n_kv, PAGE] f32
    torch::Tensor vsc,         // [blocks, n_kv, PAGE] f32
    torch::Tensor block_table, // [B, max_pages] i32
    torch::Tensor seqused) {   // [B] i32
  TORCH_CHECK(k_phys.is_cuda() && v_phys.is_cuda());
  TORCH_CHECK(k_phys.scalar_type() == at::kBFloat16);
  TORCH_CHECK(v_phys.scalar_type() == at::kBFloat16);
  TORCH_CHECK(k_phys.is_contiguous() && v_phys.is_contiguous());
  TORCH_CHECK(ku8.is_contiguous() && vu8.is_contiguous());
  TORCH_CHECK(ksc.is_contiguous() && vsc.is_contiguous());
  TORCH_CHECK(k_phys.sizes() == v_phys.sizes());
  TORCH_CHECK(k_phys.dim() == 4);
  TORCH_CHECK(k_phys.size(2) == kPage && k_phys.size(3) == kHd);
  const int64_t blocks = k_phys.size(0);
  const int64_t n_kv = k_phys.size(1);
  TORCH_CHECK(ku8.sizes() == at::IntArrayRef({blocks, n_kv, kHd, kPage}));
  TORCH_CHECK(vu8.sizes() == at::IntArrayRef({blocks, n_kv, kPage, kHd}));
  TORCH_CHECK(ksc.sizes() == at::IntArrayRef({blocks, n_kv, kPage}));
  TORCH_CHECK(vsc.sizes() == at::IntArrayRef({blocks, n_kv, kPage}));
  TORCH_CHECK(ku8.scalar_type() == at::kByte && vu8.scalar_type() == at::kByte);
  TORCH_CHECK(ksc.scalar_type() == at::kFloat && vsc.scalar_type() == at::kFloat);

  auto bt = block_table.to(at::kInt).contiguous();
  auto su = seqused.to(at::kInt).contiguous();
  TORCH_CHECK(bt.dim() == 2 && su.dim() == 1);
  TORCH_CHECK(bt.size(0) == su.size(0));
  const int64_t batch = bt.size(0);
  const int64_t max_pages = bt.size(1);

  at::cuda::CUDAGuard guard(k_phys.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  launch_ivcore_pack_bt(
      reinterpret_cast<const uint16_t*>(k_phys.data_ptr()),
      reinterpret_cast<const uint16_t*>(v_phys.data_ptr()),
      ku8.data_ptr<uint8_t>(),
      vu8.data_ptr<uint8_t>(),
      ksc.data_ptr<float>(),
      vsc.data_ptr<float>(),
      bt.data_ptr<int32_t>(),
      su.data_ptr<int32_t>(),
      static_cast<int>(batch), static_cast<int>(max_pages),
      static_cast<int>(bt.stride(0)), static_cast<int>(n_kv),
      static_cast<int>(blocks),
      stream.stream());
}

void ivcore_pack_bt_last(
    torch::Tensor k_phys, torch::Tensor v_phys,
    torch::Tensor ku8, torch::Tensor vu8,
    torch::Tensor ksc, torch::Tensor vsc,
    torch::Tensor block_table, torch::Tensor seqused) {
  // Same layout checks as full BNDH/phys pack; last-page-only for steady decode.
  TORCH_CHECK(k_phys.is_cuda() && v_phys.is_cuda());
  TORCH_CHECK(k_phys.scalar_type() == at::kBFloat16);
  TORCH_CHECK(v_phys.scalar_type() == at::kBFloat16);
  TORCH_CHECK(k_phys.is_contiguous() && v_phys.is_contiguous());
  TORCH_CHECK(ku8.is_contiguous() && vu8.is_contiguous());
  TORCH_CHECK(ksc.is_contiguous() && vsc.is_contiguous());
  TORCH_CHECK(k_phys.sizes() == v_phys.sizes());
  TORCH_CHECK(k_phys.dim() == 4);
  TORCH_CHECK(k_phys.size(2) == kPage && k_phys.size(3) == kHd);
  const int64_t blocks = k_phys.size(0);
  const int64_t n_kv = k_phys.size(1);
  TORCH_CHECK(ku8.sizes() == at::IntArrayRef({blocks, n_kv, kHd, kPage}));
  TORCH_CHECK(vu8.sizes() == at::IntArrayRef({blocks, n_kv, kPage, kHd}));
  auto bt = block_table.to(at::kInt).contiguous();
  auto su = seqused.to(at::kInt).contiguous();
  at::cuda::CUDAGuard guard(k_phys.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  launch_ivcore_pack_bt_last(
      reinterpret_cast<const uint16_t*>(k_phys.data_ptr()),
      reinterpret_cast<const uint16_t*>(v_phys.data_ptr()),
      ku8.data_ptr<uint8_t>(),
      vu8.data_ptr<uint8_t>(),
      ksc.data_ptr<float>(),
      vsc.data_ptr<float>(),
      bt.data_ptr<int32_t>(),
      su.data_ptr<int32_t>(),
      static_cast<int>(bt.size(0)), static_cast<int>(bt.size(1)),
      static_cast<int>(bt.stride(0)), static_cast<int>(n_kv),
      static_cast<int>(blocks),
      stream.stream());
}

extern "C" void launch_ivcore_pack_bt_asym(
    const uint16_t* k_hd, const uint16_t* v_bndh,
    uint8_t* ku8, uint8_t* vu8, float* ksc, float* vsc,
    const int32_t* block_table, const int32_t* seqused,
    int batch, int max_pages, int stride_bt, int n_kv, int n_blocks,
    cudaStream_t stream);

extern "C" void launch_ivcore_pack_bt_asym_last(
    const uint16_t* k_hd, const uint16_t* v_bndh,
    uint8_t* ku8, uint8_t* vu8, float* ksc, float* vsc,
    const int32_t* block_table, const int32_t* seqused,
    int batch, int max_pages, int stride_bt, int n_kv, int n_blocks,
    cudaStream_t stream);

void ivcore_pack_bt_asym(
    torch::Tensor k_hd,        // [blocks, n_kv, HD, PAGE] bf16 contiguous
    torch::Tensor v_bndh,      // [blocks, PAGE, n_kv, HD] bf16 contiguous
    torch::Tensor ku8,         // [blocks, n_kv, HD, PAGE] u8 int8
    torch::Tensor vu8,         // [blocks, n_kv, PAGE, HD] u8 int8
    torch::Tensor ksc,         // [blocks, n_kv, PAGE] f32
    torch::Tensor vsc,         // [blocks, n_kv, PAGE] f32
    torch::Tensor block_table, // [B, max_pages] i32
    torch::Tensor seqused) {   // [B] i32
  TORCH_CHECK(k_hd.is_cuda() && v_bndh.is_cuda());
  TORCH_CHECK(k_hd.scalar_type() == at::kBFloat16);
  TORCH_CHECK(v_bndh.scalar_type() == at::kBFloat16);
  TORCH_CHECK(k_hd.is_contiguous() && v_bndh.is_contiguous());
  TORCH_CHECK(ku8.is_contiguous() && vu8.is_contiguous());
  TORCH_CHECK(ksc.is_contiguous() && vsc.is_contiguous());
  TORCH_CHECK(k_hd.dim() == 4 && v_bndh.dim() == 4);
  TORCH_CHECK(k_hd.size(2) == kHd && k_hd.size(3) == kPage);
  TORCH_CHECK(v_bndh.size(1) == kPage && v_bndh.size(3) == kHd);
  const int64_t blocks = k_hd.size(0);
  const int64_t n_kv = k_hd.size(1);
  TORCH_CHECK(v_bndh.size(0) == blocks && v_bndh.size(2) == n_kv);
  TORCH_CHECK(ku8.sizes() == at::IntArrayRef({blocks, n_kv, kHd, kPage}));
  TORCH_CHECK(vu8.sizes() == at::IntArrayRef({blocks, n_kv, kPage, kHd}));
  TORCH_CHECK(ksc.sizes() == at::IntArrayRef({blocks, n_kv, kPage}));
  TORCH_CHECK(vsc.sizes() == at::IntArrayRef({blocks, n_kv, kPage}));
  TORCH_CHECK(ku8.scalar_type() == at::kByte && vu8.scalar_type() == at::kByte);
  TORCH_CHECK(ksc.scalar_type() == at::kFloat && vsc.scalar_type() == at::kFloat);

  auto bt = block_table.to(at::kInt).contiguous();
  auto su = seqused.to(at::kInt).contiguous();
  TORCH_CHECK(bt.dim() == 2 && su.dim() == 1);
  TORCH_CHECK(bt.size(0) == su.size(0));
  const int64_t batch = bt.size(0);
  const int64_t max_pages = bt.size(1);

  at::cuda::CUDAGuard guard(k_hd.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  launch_ivcore_pack_bt_asym(
      reinterpret_cast<const uint16_t*>(k_hd.data_ptr()),
      reinterpret_cast<const uint16_t*>(v_bndh.data_ptr()),
      ku8.data_ptr<uint8_t>(),
      vu8.data_ptr<uint8_t>(),
      ksc.data_ptr<float>(),
      vsc.data_ptr<float>(),
      bt.data_ptr<int32_t>(),
      su.data_ptr<int32_t>(),
      static_cast<int>(batch), static_cast<int>(max_pages),
      static_cast<int>(bt.stride(0)), static_cast<int>(n_kv),
      static_cast<int>(blocks),
      stream.stream());
}

void ivcore_pack_bt_asym_last(
    torch::Tensor k_hd, torch::Tensor v_bndh,
    torch::Tensor ku8, torch::Tensor vu8,
    torch::Tensor ksc, torch::Tensor vsc,
    torch::Tensor block_table, torch::Tensor seqused) {
  // Same checks as full asym pack; launches last-page-only kernel.
  TORCH_CHECK(k_hd.is_cuda() && v_bndh.is_cuda());
  TORCH_CHECK(k_hd.scalar_type() == at::kBFloat16);
  TORCH_CHECK(v_bndh.scalar_type() == at::kBFloat16);
  TORCH_CHECK(k_hd.is_contiguous() && v_bndh.is_contiguous());
  TORCH_CHECK(ku8.is_contiguous() && vu8.is_contiguous());
  TORCH_CHECK(ksc.is_contiguous() && vsc.is_contiguous());
  TORCH_CHECK(k_hd.dim() == 4 && v_bndh.dim() == 4);
  TORCH_CHECK(k_hd.size(2) == kHd && k_hd.size(3) == kPage);
  TORCH_CHECK(v_bndh.size(1) == kPage && v_bndh.size(3) == kHd);
  const int64_t blocks = k_hd.size(0);
  const int64_t n_kv = k_hd.size(1);
  TORCH_CHECK(v_bndh.size(0) == blocks && v_bndh.size(2) == n_kv);
  TORCH_CHECK(ku8.sizes() == at::IntArrayRef({blocks, n_kv, kHd, kPage}));
  TORCH_CHECK(vu8.sizes() == at::IntArrayRef({blocks, n_kv, kPage, kHd}));
  auto bt = block_table.to(at::kInt).contiguous();
  auto su = seqused.to(at::kInt).contiguous();
  at::cuda::CUDAGuard guard(k_hd.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  launch_ivcore_pack_bt_asym_last(
      reinterpret_cast<const uint16_t*>(k_hd.data_ptr()),
      reinterpret_cast<const uint16_t*>(v_bndh.data_ptr()),
      ku8.data_ptr<uint8_t>(),
      vu8.data_ptr<uint8_t>(),
      ksc.data_ptr<float>(),
      vsc.data_ptr<float>(),
      bt.data_ptr<int32_t>(),
      su.data_ptr<int32_t>(),
      static_cast<int>(bt.size(0)), static_cast<int>(bt.size(1)),
      static_cast<int>(bt.stride(0)), static_cast<int>(n_kv),
      static_cast<int>(blocks),
      stream.stream());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ivcore_u8_paged_decode", &ivcore_u8_paged_decode,
        "ivcore u8 paged decode (v52: optional SME G2S)",
        py::arg("q"), py::arg("k_u8"), py::arg("v_u8"),
        py::arg("block_table"), py::arg("seqused"),
        py::arg("softmax_scale"), py::arg("split") = 1,
        py::arg("v_mode") = 2, py::arg("k_scales") = c10::nullopt,
        py::arg("v_scales") = c10::nullopt,
        py::arg("out") = c10::nullopt,
        py::arg("use_sme") = 0);
  m.def("ivcore_pack_bt", &ivcore_pack_bt,
        "graph-safe bf16→u8 pack via block_table (int8-K + int8-V)",
        py::arg("k_phys"), py::arg("v_phys"),
        py::arg("ku8"), py::arg("vu8"), py::arg("ksc"), py::arg("vsc"),
        py::arg("block_table"), py::arg("seqused"));
  m.def("ivcore_pack_bt_last", &ivcore_pack_bt_last,
        "BNDH/phys pack last page only (decode incremental)",
        py::arg("k_phys"), py::arg("v_phys"),
        py::arg("ku8"), py::arg("vu8"), py::arg("ksc"), py::arg("vsc"),
        py::arg("block_table"), py::arg("seqused"));
  m.def("ivcore_pack_bt_asym", &ivcore_pack_bt_asym,
        "graph-safe asym pack: K HD-major + V BNDH → u8",
        py::arg("k_hd"), py::arg("v_bndh"),
        py::arg("ku8"), py::arg("vu8"), py::arg("ksc"), py::arg("vsc"),
        py::arg("block_table"), py::arg("seqused"));
  m.def("ivcore_pack_bt_asym_last", &ivcore_pack_bt_asym_last,
        "asym pack last page only (decode incremental)",
        py::arg("k_hd"), py::arg("v_bndh"),
        py::arg("ku8"), py::arg("vu8"), py::arg("ksc"), py::arg("vsc"),
        py::arg("block_table"), py::arg("seqused"));
}
