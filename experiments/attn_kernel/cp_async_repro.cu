// Minimal reproducer: cp.async lowering crashes llc on corex 4.5.0 / ivcore11.
//
//   clang -O3 -x ivcore --cuda-gpu-arch=ivcore11 --cuda-path=/usr/local/corex \
//         -Xclang -target-feature -Xclang +async-copy -c -o /dev/null cp_async_repro.cu
//
//   clang: error: unable to execute command: Segmentation fault (core dumped)
//   clang: error: llc command failed due to signal
//
// Same at -O0.  Without the target feature the builtins are rejected:
//   error: '__ivcorex_cp_async_wait_group' needs target feature async-copy
//
// The family is declared in
//   /usr/local/corex/include/clang/Basic/BuiltinsIluvatar.def
//   __ivcorex_cp_async_ca_shared_global_4 / _8 / _16   "vV4Ui*V4UiC*"
//   __ivcorex_cp_async_commit_group                     "v"
//   __ivcorex_cp_async_wait_group                       "vUi"
// and the IR intrinsics (llvm.bi.cp.async.ca.shared.global.{4,8,16}) are present
// in the clang binary, so only the lowering path is broken.
//
// This blocks the only route to vendor-class MMA throughput from a self-hosted
// kernel: without async copy the staging latency has to be hidden by warp count,
// and warp count is capped by the register budget.  See README section 10.
#include <cuda_runtime.h>

typedef unsigned int v4u __attribute__((vector_size(16)));

__global__ void k(const float *src, float *dst) {
  __shared__ float s[1024];
  __ivcorex_cp_async_ca_shared_global_16((v4u *)&s[threadIdx.x * 4],
                                         (const v4u *)&src[threadIdx.x * 4]);
  __ivcorex_cp_async_commit_group();
  __ivcorex_cp_async_wait_group(0);
  __syncthreads();
  dst[threadIdx.x] = s[threadIdx.x];
}

int main() { return 0; }
