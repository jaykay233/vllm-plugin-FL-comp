// fp8 (e4m3) GEMM on MetaX C500 using the hardware MMA intrinsic.
//
// Why this file exists
// --------------------
// Every software path for fp8 GEMM on this stack is broken (measured):
//   torch._scaled_mm            hard capability gate, device reports 8.0
//   _C::cutlass_scaled_mm       silently writes garbage (inf/nan, or 147% err)
//   triton tl.dot(fp8, fp8)     "PassManager::run failed" -- cannot lower
//   mctlassEx fp8_blockwise     silent no-op, output buffer never written
//
// But the ISA does have an fp8 MMA: __builtin_mxc_mma_16x16x32f8
// (m16n16k32, e4m3 x e4m3 -> f32, one 64-lane warp). This file drives it
// directly, which is the only remaining route.
//
// Fragment layout is the one calibrated in ../csrc/calib.cu (VA=0, VB=0):
//     A : row = lane%16, k = {g*4+0..3} (a0) and {16+g*4+0..3} (a1)
//     B : n   = lane%16, k = {g*4+0..3} (b0) and {16+g*4+0..3} (b1)
//     C : row = lane%16, col = g*4 + reg
//   with g = lane/16 and warpSize = 64.
// Argument order is (B, A, C), mirroring the mcflashinfer wrapper.
//
// Two entry points:
//   launch_mma16x16x32   single tile, for numerical validation against a
//                        dequantized torch reference (isolates the intrinsic)
//   launch_fp8_gemm      tiled GEMM with a K loop, for throughput

#include <mc_runtime.h>
#include <maca_bfloat16.h>
#include <cstdint>

typedef __NATIVE_VECTOR__(2, uint32_t) v2u;
typedef __NATIVE_VECTOR__(4, float) v4f;

__device__ __forceinline__ uint32_t ld4(const uint8_t* p) {
  uint32_t v;
  __builtin_memcpy(&v, p, 4);
  return v;
}

// ---------------------------------------------------------------------------
// 1. single 16x16x32 tile
// ---------------------------------------------------------------------------
__global__ void mma_tile_kernel(const uint8_t* __restrict__ A,
                                const uint8_t* __restrict__ Bn,
                                float* __restrict__ Cout) {
  const int lane = threadIdx.x;
  const int row = lane % 16;
  const int g = lane / 16;

  const uint32_t a0 = ld4(A + row * 32 + g * 4);
  const uint32_t a1 = ld4(A + row * 32 + 16 + g * 4);
  const uint32_t b0 = ld4(Bn + (lane % 16) * 32 + g * 4);
  const uint32_t b1 = ld4(Bn + (lane % 16) * 32 + 16 + g * 4);

  v2u va = {a0, a1};
  v2u vb = {b0, b1};
  v4f c = {0.f, 0.f, 0.f, 0.f};
  v4f d = __builtin_mxc_mma_16x16x32f8(vb, va, c);

  float* o = Cout + row * 16 + g * 4;
  o[0] = d[0];
  o[1] = d[1];
  o[2] = d[2];
  o[3] = d[3];
}

extern "C" int launch_mma16x16x32(const uint8_t* A, const uint8_t* Bn,
                                  float* Cout, void* stream) {
  mma_tile_kernel<<<1, 64, 0, (mcStream_t)stream>>>(A, Bn, Cout);
  return 0;
}

// ---------------------------------------------------------------------------
// 2. tiled GEMM:  C[M,N] = sum_k A[m,k] * B[n,k]      (A,B fp8 k-major)
// ---------------------------------------------------------------------------
template <int WPB>
__global__ void gemm_fp8_kernel(const uint8_t* __restrict__ A,
                                const uint8_t* __restrict__ Bn,
                                float* __restrict__ C,
                                int M, int N, int K) {
  const int warp = threadIdx.x >> 6;   // 0..WPB-1  (warpSize == 64)
  const int lane = threadIdx.x & 63;
  const int row = lane % 16;
  const int g = lane / 16;

  const int tiles_n = N >> 4;
  const int tile = blockIdx.x * WPB + warp;
  const int tm = tile / tiles_n;
  const int tn = tile % tiles_n;
  const int m0 = tm << 4;
  const int n0 = tn << 4;
  if (m0 >= M) return;

  const uint8_t* a_row = A + (size_t)(m0 + row) * K;
  const uint8_t* b_row = Bn + (size_t)(n0 + (lane % 16)) * K;

  v4f c = {0.f, 0.f, 0.f, 0.f};
  for (int k0 = 0; k0 < K; k0 += 32) {
    v2u va = {ld4(a_row + k0 + g * 4), ld4(a_row + k0 + 16 + g * 4)};
    v2u vb = {ld4(b_row + k0 + g * 4), ld4(b_row + k0 + 16 + g * 4)};
    c = __builtin_mxc_mma_16x16x32f8(vb, va, c);
  }

  float* o = C + (size_t)(m0 + row) * N + n0 + g * 4;
  o[0] = c[0];
  o[1] = c[1];
  o[2] = c[2];
  o[3] = c[3];
}

extern "C" int launch_fp8_gemm(const uint8_t* A, const uint8_t* Bn, float* C,
                               int M, int N, int K, void* stream) {
  constexpr int WPB = 4;
  const int tiles = (M >> 4) * (N >> 4);
  const int blocks = (tiles + WPB - 1) / WPB;
  gemm_fp8_kernel<WPB><<<blocks, WPB * 64, 0, (mcStream_t)stream>>>(A, Bn, C, M, N, K);
  return 0;
}

// ---------------------------------------------------------------------------
// 3. decode-specialised: M == 64, one warp owns a 64x16 output tile
// ---------------------------------------------------------------------------
// The generic kernel above re-reads each B tile once per 16-row m-tile, i.e. 4x
// redundant traffic when M=64. Here one warp holds all four m-tiles for a given
// n-tile, so B is loaded once and reused across four MMAs. A is small enough to
// stay resident in L2 across n-tiles, so total DRAM traffic ~= the weight bytes,
// which is the bound that matters for decode (batch = 64).
template <int WPB>
__global__ void gemm64_kernel(const uint8_t* __restrict__ A,
                              const uint8_t* __restrict__ Bn,
                              float* __restrict__ C,
                              int M, int N, int K) {
  const int warp = threadIdx.x >> 6;
  const int lane = threadIdx.x & 63;
  const int row = lane % 16;
  const int g = lane / 16;

  const int tn = blockIdx.x * WPB + warp;
  const int n0 = tn << 4;
  if (n0 >= N) return;

  const int nrow = lane % 16;
  const uint8_t* b_row = Bn + (size_t)(n0 + nrow) * K;
  const uint8_t* a0_row = A + (size_t)(0 + row) * K;
  const uint8_t* a1_row = A + (size_t)(16 + row) * K;
  const uint8_t* a2_row = A + (size_t)(32 + row) * K;
  const uint8_t* a3_row = A + (size_t)(48 + row) * K;

  v4f c0 = {0.f, 0.f, 0.f, 0.f};
  v4f c1 = {0.f, 0.f, 0.f, 0.f};
  v4f c2 = {0.f, 0.f, 0.f, 0.f};
  v4f c3 = {0.f, 0.f, 0.f, 0.f};

  for (int k0 = 0; k0 < K; k0 += 32) {
    const int kh = k0 + g * 4;
    const int kh2 = kh + 16;
    v2u vb = {ld4(b_row + kh), ld4(b_row + kh2)};
    v2u va0 = {ld4(a0_row + kh), ld4(a0_row + kh2)};
    v2u va1 = {ld4(a1_row + kh), ld4(a1_row + kh2)};
    v2u va2 = {ld4(a2_row + kh), ld4(a2_row + kh2)};
    v2u va3 = {ld4(a3_row + kh), ld4(a3_row + kh2)};
    c0 = __builtin_mxc_mma_16x16x32f8(vb, va0, c0);
    c1 = __builtin_mxc_mma_16x16x32f8(vb, va1, c1);
    c2 = __builtin_mxc_mma_16x16x32f8(vb, va2, c2);
    c3 = __builtin_mxc_mma_16x16x32f8(vb, va3, c3);
  }

  const int col = g * 4;
  float* p;
  p = C + (size_t)(0 + row) * N + n0 + col;
  p[0] = c0[0]; p[1] = c0[1]; p[2] = c0[2]; p[3] = c0[3];
  p = C + (size_t)(16 + row) * N + n0 + col;
  p[0] = c1[0]; p[1] = c1[1]; p[2] = c1[2]; p[3] = c1[3];
  p = C + (size_t)(32 + row) * N + n0 + col;
  p[0] = c2[0]; p[1] = c2[1]; p[2] = c2[2]; p[3] = c2[3];
  p = C + (size_t)(48 + row) * N + n0 + col;
  p[0] = c3[0]; p[1] = c3[1]; p[2] = c3[2]; p[3] = c3[3];
}

extern "C" int launch_gemm64(const uint8_t* A, const uint8_t* Bn, float* C,
                             int M, int N, int K, void* stream) {
  constexpr int WPB = 4;
  const int tiles = N >> 4;
  const int blocks = (tiles + WPB - 1) / WPB;
  gemm64_kernel<WPB><<<blocks, WPB * 64, 0, (mcStream_t)stream>>>(A, Bn, C, M, N, K);
  return 0;
}

extern "C" int launch_gemm64_w1(const uint8_t* A, const uint8_t* Bn, float* C,
                                int M, int N, int K, void* stream) {
  constexpr int WPB = 1;
  const int tiles = N >> 4;
  gemm64_kernel<WPB><<<tiles, 64, 0, (mcStream_t)stream>>>(A, Bn, C, M, N, K);
  return 0;
}

// ---------------------------------------------------------------------------
// 4. same math, but K/V tiles are staged through shared memory
// ---------------------------------------------------------------------------
// The direct-from-global version above reads every fragment with 4-byte `ld4`
// where consecutive lanes touch DIFFERENT rows (stride = K). That is one memory
// transaction per lane, so effective bandwidth collapses to 4-11 GB/s.
// Here each k-step is first copied into smem with coalesced 8-byte loads, then
// fragments are pulled from smem. 4 warps/block, one 16-col n-tile per warp,
// all four m-tiles per warp (M=64 in one pass, so B is read exactly once).
template <int WPB>
__global__ void gemm64_smem_kernel(const uint8_t* __restrict__ A,
                                   const uint8_t* __restrict__ Bn,
                                   float* __restrict__ C,
                                   int M, int N, int K) {
  __shared__ uint8_t sA[64 * 32];
  __shared__ uint8_t sB[WPB * 16 * 32];

  const int tid = threadIdx.x;
  const int nthr = WPB * 64;
  const int warp = tid >> 6;
  const int lane = tid & 63;
  const int row = lane % 16;
  const int g = lane / 16;

  const int n0 = blockIdx.x * (WPB << 4);
  if (n0 >= N) return;

  v4f c0 = {0.f, 0.f, 0.f, 0.f};
  v4f c1 = {0.f, 0.f, 0.f, 0.f};
  v4f c2 = {0.f, 0.f, 0.f, 0.f};
  v4f c3 = {0.f, 0.f, 0.f, 0.f};

  for (int k0 = 0; k0 < K; k0 += 32) {
    // coalesced 8-byte staging: 2048 B of A, WPB*512 B of B
    for (int i = tid * 8; i < 64 * 32; i += nthr * 8) {
      const int r = i >> 5;
      const int cc = i & 31;
      *(uint64_t*)(sA + i) = *(const uint64_t*)(A + (size_t)r * K + k0 + cc);
    }
    for (int i = tid * 8; i < WPB * 16 * 32; i += nthr * 8) {
      const int r = i >> 5;
      const int cc = i & 31;
      *(uint64_t*)(sB + i) =
          *(const uint64_t*)(Bn + (size_t)(n0 + r) * K + k0 + cc);
    }
    __syncthreads();

    const int kh = g * 4;
    const int kh2 = kh + 16;
    v2u vb = {ld4(sB + (warp * 16 + row) * 32 + kh),
              ld4(sB + (warp * 16 + row) * 32 + kh2)};
    v2u va0 = {ld4(sA + (0 + row) * 32 + kh), ld4(sA + (0 + row) * 32 + kh2)};
    v2u va1 = {ld4(sA + (16 + row) * 32 + kh), ld4(sA + (16 + row) * 32 + kh2)};
    v2u va2 = {ld4(sA + (32 + row) * 32 + kh), ld4(sA + (32 + row) * 32 + kh2)};
    v2u va3 = {ld4(sA + (48 + row) * 32 + kh), ld4(sA + (48 + row) * 32 + kh2)};
    c0 = __builtin_mxc_mma_16x16x32f8(vb, va0, c0);
    c1 = __builtin_mxc_mma_16x16x32f8(vb, va1, c1);
    c2 = __builtin_mxc_mma_16x16x32f8(vb, va2, c2);
    c3 = __builtin_mxc_mma_16x16x32f8(vb, va3, c3);
    __syncthreads();
  }

  const int col = g * 4;
  const int no = n0 + (warp << 4) + col;
  float* p;
  p = C + (size_t)(0 + row) * N + no;
  p[0] = c0[0]; p[1] = c0[1]; p[2] = c0[2]; p[3] = c0[3];
  p = C + (size_t)(16 + row) * N + no;
  p[0] = c1[0]; p[1] = c1[1]; p[2] = c1[2]; p[3] = c1[3];
  p = C + (size_t)(32 + row) * N + no;
  p[0] = c2[0]; p[1] = c2[1]; p[2] = c2[2]; p[3] = c2[3];
  p = C + (size_t)(48 + row) * N + no;
  p[0] = c3[0]; p[1] = c3[1]; p[2] = c3[2]; p[3] = c3[3];
}

extern "C" int launch_gemm64_smem(const uint8_t* A, const uint8_t* Bn, float* C,
                                  int M, int N, int K, void* stream) {
  constexpr int WPB = 4;
  const int tiles = (N + (WPB << 4) - 1) / (WPB << 4);
  gemm64_smem_kernel<WPB><<<tiles, WPB * 64, 0, (mcStream_t)stream>>>(A, Bn, C, M, N, K);
  return 0;
}

// ---------------------------------------------------------------------------
// 5. double-buffered: prefetch k-step s+1 while the MMAs run on step s
// ---------------------------------------------------------------------------
// The version above stalls every k-step on a global load (sync -> load ->
// sync -> math, no overlap). With 256 threads and 2048 B per buffer, each
// thread owns exactly one 8-byte chunk per buffer, so the prefetch is a single
// uint64 load per buffer that the compiler can hoist above the MMA block.
template <int WPB>
__global__ void gemm64_pipe_kernel(const uint8_t* __restrict__ A,
                                   const uint8_t* __restrict__ Bn,
                                   float* __restrict__ C,
                                   int M, int N, int K) {
  __shared__ uint8_t sA[2][64 * 32];
  __shared__ uint8_t sB[2][WPB * 16 * 32];

  const int tid = threadIdx.x;
  const int warp = tid >> 6;
  const int lane = tid & 63;
  const int row = lane % 16;
  const int g = lane / 16;

  const int n0 = blockIdx.x * (WPB << 4);
  if (n0 >= N) return;

  const int slot = tid * 8;
  const int ar = slot >> 5;
  const int ac = slot & 31;
  const uint8_t* ap = A + (size_t)ar * K;
  const uint8_t* bp = Bn + (size_t)(n0 + ar) * K;

  *(uint64_t*)(sA[0] + slot) = *(const uint64_t*)(ap + ac);
  *(uint64_t*)(sB[0] + slot) = *(const uint64_t*)(bp + ac);
  __syncthreads();

  v4f c0 = {0.f, 0.f, 0.f, 0.f};
  v4f c1 = {0.f, 0.f, 0.f, 0.f};
  v4f c2 = {0.f, 0.f, 0.f, 0.f};
  v4f c3 = {0.f, 0.f, 0.f, 0.f};

  const int nsteps = K / 32;
  for (int s = 0; s < nsteps; ++s) {
    const int cur = s & 1;
    const int nxt = cur ^ 1;

    uint64_t pa = 0, pb = 0;
    if (s + 1 < nsteps) {
      pa = *(const uint64_t*)(ap + (s + 1) * 32 + ac);
      pb = *(const uint64_t*)(bp + (s + 1) * 32 + ac);
    }

    const uint8_t* ka = sA[cur];
    const uint8_t* kb = sB[cur];
    const int kh = g * 4;
    const int kh2 = kh + 16;
    const uint8_t* bk = kb + (warp * 16 + row) * 32;
    v2u vb = {ld4(bk + kh), ld4(bk + kh2)};
    const uint8_t* a0 = ka + (0 + row) * 32;
    const uint8_t* a1 = ka + (16 + row) * 32;
    const uint8_t* a2 = ka + (32 + row) * 32;
    const uint8_t* a3 = ka + (48 + row) * 32;
    v2u va0 = {ld4(a0 + kh), ld4(a0 + kh2)};
    v2u va1 = {ld4(a1 + kh), ld4(a1 + kh2)};
    v2u va2 = {ld4(a2 + kh), ld4(a2 + kh2)};
    v2u va3 = {ld4(a3 + kh), ld4(a3 + kh2)};
    c0 = __builtin_mxc_mma_16x16x32f8(vb, va0, c0);
    c1 = __builtin_mxc_mma_16x16x32f8(vb, va1, c1);
    c2 = __builtin_mxc_mma_16x16x32f8(vb, va2, c2);
    c3 = __builtin_mxc_mma_16x16x32f8(vb, va3, c3);

    if (s + 1 < nsteps) {
      *(uint64_t*)(sA[nxt] + slot) = pa;
      *(uint64_t*)(sB[nxt] + slot) = pb;
    }
    __syncthreads();
  }

  const int col = g * 4;
  const int no = n0 + (warp << 4) + col;
  float* p;
  p = C + (size_t)(0 + row) * N + no;
  p[0] = c0[0]; p[1] = c0[1]; p[2] = c0[2]; p[3] = c0[3];
  p = C + (size_t)(16 + row) * N + no;
  p[0] = c1[0]; p[1] = c1[1]; p[2] = c1[2]; p[3] = c1[3];
  p = C + (size_t)(32 + row) * N + no;
  p[0] = c2[0]; p[1] = c2[1]; p[2] = c2[2]; p[3] = c2[3];
  p = C + (size_t)(48 + row) * N + no;
  p[0] = c3[0]; p[1] = c3[1]; p[2] = c3[2]; p[3] = c3[3];
}

extern "C" int launch_gemm64_pipe(const uint8_t* A, const uint8_t* Bn, float* C,
                                  int M, int N, int K, void* stream) {
  constexpr int WPB = 4;
  const int tiles = (N + (WPB << 4) - 1) / (WPB << 4);
  gemm64_pipe_kernel<WPB><<<tiles, WPB * 64, 0, (mcStream_t)stream>>>(A, Bn, C, M, N, K);
  return 0;
}
