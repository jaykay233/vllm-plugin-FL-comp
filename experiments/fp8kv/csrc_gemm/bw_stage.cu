// Global -> shared-memory staging bandwidth probe for the M=64 decode GEMM.
//
// diag_split.cu showed staging is 62% of the kernel and only reaches 147 GB/s on a
// box whose HBM peak is ~1416 GB/s. This probe isolates that path: same block
// count, same tile shape, same barrier structure as the real kernel, but the MMA
// is removed. Four variants tell me what actually limits it.
//
//   PLAIN  : __builtin_memcpy 16B into smem (goes through registers, so the load
//            latency is only hidden by however many loads a thread has in flight)
//   ASYNC  : __builtin_mxc_ldg_b128_bsm, i.e. cp.async 16B straight to smem with
//            commit_group/wait_group -- no register round trip, real DMA
//   WIDE   : 32B per thread per step (two cp.asyncs), fewer, fatter transfers
//
// Both fp8 (1 byte/elem) and bf16 (2 bytes/elem) source widths are measured,
// because the fp8 plan is only worth it if the fp8 path is genuinely faster at the
// same tile geometry.

#include <mc_runtime.h>
#include <cstdint>

typedef __NATIVE_VECTOR__(4, uint32_t) v4u;

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
  auto d = (v4u*)smem;
  auto s = (const v4u*)gmem;
  __builtin_mxc_ldg_b128_bsm(d, s, 0, -1, true, true, false, true);
}
// MACA has no cp.async.* PTX; async global->smem loads are tracked by a counter.
// Pattern taken from mcflashinfer/utils.cuh: arrive_gvmcnt(N) then barrier_inst().
__device__ __forceinline__ void cp_commit() { __builtin_mxc_arrive_gvmcnt(0); }
__device__ __forceinline__ void cp_wait_all() { __builtin_mxc_barrier_inst(); }

// MODE 0 plain, 1 async, 2 async-wide
template <int WPB, int KC, int NT, int ESCALE, int MODE>
__global__ void stage_kernel(const uint8_t* __restrict__ A,
                             const uint8_t* __restrict__ Bn,
                             float* __restrict__ out, int N, int K) {
  constexpr int RS = KC * ESCALE + 8;
  constexpr int BROWS = WPB * NT * 16;
  constexpr int NTHR = WPB * 64;
  constexpr int ACHUNKS = 64 * KC * ESCALE / 16;   // 16B chunks in the A tile
  constexpr int BCHUNKS = BROWS * KC * ESCALE / 16;
  constexpr int AROWS = KC * ESCALE / 16;          // chunks per row

  __shared__ uint8_t sA[2][64 * RS];
  __shared__ uint8_t sB[2][BROWS * RS];

  const int tid = threadIdx.x;
  const int n0 = blockIdx.x * (WPB * NT * 16);
  if (n0 >= N) return;

  const int NCH = K / KC;

  for (int ch = 0; ch < NCH; ++ch) {
    const int cur = ch & 1;
    const int k0 = ch * KC;

    if (MODE == 0) {
      for (int j = tid; j < ACHUNKS; j += NTHR) {
        const int r = j / AROWS;
        const int c = (j - r * AROWS) * 16;
        v4u v;
        __builtin_memcpy(&v, A + (size_t)r * K * ESCALE + (size_t)k0 * ESCALE + c, 16);
        __builtin_memcpy(sA[cur] + r * RS + c, &v, 16);
      }
      for (int j = tid; j < BCHUNKS; j += NTHR) {
        const int r = j / AROWS;
        const int c = (j - r * AROWS) * 16;
        v4u v;
        __builtin_memcpy(&v, Bn + (size_t)(n0 + r) * K * ESCALE + (size_t)k0 * ESCALE + c, 16);
        __builtin_memcpy(sB[cur] + r * RS + c, &v, 16);
      }
    } else {
      for (int j = tid; j < ACHUNKS; j += NTHR) {
        const int r = j / AROWS;
        const int c = (j - r * AROWS) * 16;
        cp_async16(sA[cur] + r * RS + c,
                   A + (size_t)r * K * ESCALE + (size_t)k0 * ESCALE + c);
      }
      for (int j = tid; j < BCHUNKS; j += NTHR) {
        const int r = j / AROWS;
        const int c = (j - r * AROWS) * 16;
        cp_async16(sB[cur] + r * RS + c,
                   Bn + (size_t)(n0 + r) * K * ESCALE + (size_t)k0 * ESCALE + c);
      }
      cp_commit();
      cp_wait_all();
    }
    __syncthreads();
  }

  // keep the compiler from removing the staging
  if (tid == 0) out[blockIdx.x] = (float)sA[0][0] + (float)sB[1][0];
}

extern "C" int launch_stage(const uint8_t* A, const uint8_t* Bn, float* out,
                            int N, int K, int cfg, void* stream) {
#define S(WPB, KC, NT, ESC, MODE)                                              \
  do {                                                                         \
    const int per = (WPB) * (NT) * 16;                                         \
    const int tiles = (N + per - 1) / per;                                     \
    stage_kernel<WPB, KC, NT, ESC, MODE>                                       \
        <<<tiles, (WPB) * 64, 0, (mcStream_t)stream>>>(A, Bn, out, N, K);      \
  } while (0)
  switch (cfg) {
    case 0: S(4, 128, 2, 1, 0); break;   // fp8, plain 16B
    case 1: S(4, 128, 2, 1, 1); break;   // fp8, cp.async 16B
    case 2: S(4, 64, 2, 2, 0); break;    // bf16, plain 16B
    case 3: S(4, 64, 2, 2, 1); break;    // bf16, cp.async 16B
    case 4: S(4, 64, 2, 1, 1); break;    // fp8, small chunk
    case 5: S(8, 128, 1, 1, 1); break;   // fp8, wide block
    case 6: S(4, 64, 4, 1, 1); break;    // fp8, more n-tiles
    case 7: S(2, 128, 2, 1, 1); break;   // fp8, narrow block
    default: return -1;
  }
  return 0;
}
