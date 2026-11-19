// QK^T with honest warp geometry, templated so the tile shape can be swept.
//
// An earlier version put all 8 warps along M with a single 16-wide N tile while
// staging BN columns, so it computed 16 of every BN columns and the reported
// throughput was inflated by BN/16.  This file keeps the invariant
//     WMUL * WNUL == 8        (warps)
//     BM = WMUL * MT * 16     (M tiles per warp, MT)
//     BN = WNUL * NT * 16     (N tiles per warp, NT)
// so every element of the BM x BN tile is produced exactly once.
//
// Flops accounting, verified against the instruction stream: tile_ok.cu builds a
// 16x16x32 tile with exactly two ml_matrix_mad_f32_f16_m16n16k16, so one is
// 16x16x16 = 8192 flops.  Peak is 16 SM x 1 MMA/cycle x 8192 x 1.5 GHz = 196.6
// TFLOPS; measured pure-register ceilings are 46 at 8 warps/SM, 92 at 16, 138 at
// 24, 184 at 32.
//
// B fragments live in shared memory pre-swizzled so each lane's four uints are
// contiguous and a whole fragment is one 16-byte load.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

using namespace nvcuda;

#define DK 128
#define KCH (DK / 32)

__device__ __forceinline__ void swz_stage(unsigned int *Ksw, const __half *K, int n0, int BNP,
                                          int tid, int nthread) {
  const int tot = BNP * DK / 8;
#pragma unroll 4
  for (int v = tid; v < tot; v += nthread) {
    const int n = (v * 8) / DK, k0 = (v * 8) % DK;
    const uint4 raw = *(const uint4 *)&K[(size_t)(n0 + n) * DK + k0];
    const int b = (((n >> 4) * KCH + (k0 >> 5)) * 256) + ((k0 & 31) >> 3) + ((n & 15) * 4);
    Ksw[b] = raw.x;
    Ksw[b + 64] = raw.y;
    Ksw[b + 128] = raw.z;
    Ksw[b + 192] = raw.w;
  }
}

template <int WMUL, int WNUL, int MT, int NT, int BNP, int STORE>
__global__ __launch_bounds__((WMUL * WNUL) * 64) void qk(const __half *Q, const __half *K,
                                                         float *S, int kv, int NH) {
  constexpr int NWARP = WMUL * WNUL;
  constexpr int NTHREAD = NWARP * 64;
  constexpr int BM = WMUL * MT * 16;

  __shared__ unsigned int Ksw[BNP * DK / 2];

  const int tid = threadIdx.x;
  const int lane = tid & 63;
  const int warp = tid >> 6;
  const int wm = warp / WNUL;
  const int wn = warp % WNUL;
  const int qbase = blockIdx.x * BM, head = blockIdx.y;

  unsigned int qreg[MT][KCH][4];
#pragma unroll
  for (int i = 0; i < MT; ++i)
#pragma unroll
    for (int c = 0; c < KCH; ++c)
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const int m = wm * (MT * 16) + i * 16 + (lane / 16) * 2 + 8 * (q % 2);
        const int k = c * 32 + (lane % 16) + 16 * (q / 2);
        __half2 h2;
        h2.x = Q[(size_t)(qbase + m) * NH * DK + (size_t)head * DK + k];
        h2.y = Q[(size_t)(qbase + m + 1) * NH * DK + (size_t)head * DK + k];
        qreg[i][c][q] = *(unsigned int *)&h2;
      }

  float sink = 0.f;
  const uint4 *kbase = (const uint4 *)Ksw;

  for (int n0 = 0; n0 < kv; n0 += BNP) {
    __syncthreads();
    swz_stage(Ksw, K, n0, BNP, tid, NTHREAD);
    __syncthreads();

    wmma::fragment<wmma::accumulator, 16, 16, 32, float> acc[MT][NT];
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
      for (int j = 0; j < NT; ++j) wmma::fill_fragment(acc[i][j], 0.f);

#pragma unroll
    for (int c = 0; c < KCH; ++c) {
      wmma::fragment<wmma::matrix_b, 16, 16, 32, __half, wmma::col_major> fb[NT];
#pragma unroll
      for (int j = 0; j < NT; ++j) {
        const uint4 f = kbase[((wn * NT + j) * KCH + c) * 64 + lane];
        fb[j].x[0] = f.x; fb[j].x[1] = f.y; fb[j].x[2] = f.z; fb[j].x[3] = f.w;
      }
#pragma unroll
      for (int i = 0; i < MT; ++i) {
        wmma::fragment<wmma::matrix_a, 16, 16, 32, __half, wmma::row_major> fa;
#pragma unroll
        for (int q = 0; q < 4; ++q) fa.x[q] = qreg[i][c][q];
#pragma unroll
        for (int j = 0; j < NT; ++j) wmma::mma_sync(acc[i][j], fa, fb[j], acc[i][j]);
      }
    }

    const int ccol = lane & 15, crow = lane >> 4;
    if (STORE) {
#pragma unroll
      for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < NT; ++j)
#pragma unroll
          for (int q = 0; q < 4; ++q)
            S[(size_t)(qbase + wm * (MT * 16) + i * 16 + crow + 4 * q) * kv + n0 +
              wn * (NT * 16) + j * 16 + ccol] = acc[i][j].x[q];
    } else {
#pragma unroll
      for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < NT; ++j) sink += acc[i][j].x[0];
    }
  }
  if (sink == 1234.5678f) S[0] = sink;
}

template <int WMUL, int WNUL, int MT, int NT, int BNP>
static void bench(const __half *dQ, const __half *dK, float *dS, int tok, int kv, int NH) {
  constexpr int BM = WMUL * MT * 16;
  char tag[96];
  snprintf(tag, sizeof tag, "W%dx%d MT%d NT%d BM%d BN%d", WMUL, WNUL, MT, NT, BM, BNP);
  if (tok % BM) { printf("%-30s skip (BM does not divide tok)\n", tag); return; }
  dim3 grid(tok / BM, NH);
  auto run = [&]() { qk<WMUL, WNUL, MT, NT, BNP, 0><<<grid, (WMUL * WNUL) * 64>>>(dQ, dK, dS, kv, NH); };
  for (int i = 0; i < 3; ++i) run();
  cudaEvent_t s, e;
  cudaEventCreate(&s); cudaEventCreate(&e);
  cudaEventRecord(s);
  for (int i = 0; i < 10; ++i) run();
  cudaEventRecord(e); cudaEventSynchronize(e);
  float ms; cudaEventElapsedTime(&ms, s, e); ms /= 10;
  const double fl = 2.0 * tok * kv * DK * NH;
  const double tf = fl / (ms * 1e-3) / 1e12;
  int nb = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, qk<WMUL, WNUL, MT, NT, BNP, 0>,
                                                (WMUL * WNUL) * 64, 0);
  cudaFuncAttributes at;
  cudaFuncGetAttributes(&at, qk<WMUL, WNUL, MT, NT, BNP, 0>);
  printf("%-30s %7.3f ms  %6.1f TFLOPS (%5.1f%% peak) | %d CTA = %2d warps | regs %d\n",
         tag, ms, tf, 100 * tf / 196.6, nb, nb * (WMUL * WNUL), at.numRegs);
}

int main(int argc, char **argv) {
  const int tok = argc > 1 ? atoi(argv[1]) : 4096;
  const int kv = argc > 2 ? atoi(argv[2]) : 4096;
  const int NH = 16;

  size_t qsz = (size_t)tok * NH * DK;
  __half *hQ = (__half *)malloc(qsz * 2), *hK = (__half *)malloc((size_t)kv * DK * 2);
  srand(11);
  for (size_t i = 0; i < qsz; ++i) hQ[i] = __float2half((rand() % 200 - 100) / 128.0f);
  for (size_t i = 0; i < (size_t)kv * DK; ++i) hK[i] = __float2half((rand() % 200 - 100) / 128.0f);

  __half *dQ, *dK;
  float *dS;
  cudaMalloc(&dQ, qsz * 2);
  cudaMalloc(&dK, (size_t)kv * DK * 2);
  cudaMalloc(&dS, (size_t)tok * kv * 4);
  cudaMemcpy(dQ, hQ, qsz * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(dK, hK, (size_t)kv * DK * 2, cudaMemcpyHostToDevice);

  // Full-coverage correctness: every element of the BM x kv region must be right,
  // and the fraction of elements that are non-zero is reported so a coverage bug
  // cannot hide behind uninitialised memory.
  {
    constexpr int WMUL = 2, WNUL = 4, MT = 4, NT = 1, BNP = 64;
    const int k2 = 256, t2 = WMUL * MT * 16;
    cudaMemset(dS, 0xFF, (size_t)t2 * k2 * 4);  // poison
    qk<WMUL, WNUL, MT, NT, BNP, 1><<<dim3(1, 1), (WMUL * WNUL) * 64>>>(dQ, dK, dS, k2, NH);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) { printf("ERR %s\n", cudaGetErrorString(e)); return 1; }
    float *hS = (float *)malloc((size_t)t2 * k2 * 4);
    cudaMemcpy(hS, dS, (size_t)t2 * k2 * 4, cudaMemcpyDeviceToHost);
    double worst = 0;
    int bad = 0, nonzero = 0;
    const double poisoned = -1.0f;
    for (int m = 0; m < t2; ++m)
      for (int n = 0; n < k2; ++n) {
        if (hS[(size_t)m * k2 + n] != poisoned) ++nonzero;
        double ref = 0;
        for (int d = 0; d < DK; ++d)
          ref += (double)__half2float(hQ[(size_t)m * NH * DK + d]) *
                 __half2float(hK[(size_t)n * DK + d]);
        double er = fabs(hS[(size_t)m * k2 + n] - ref) / (fabs(ref) + 1e-6);
        if (er > worst) worst = er;
        if (er > 3e-2) ++bad;
      }
    printf("correctness %dx%d: worst rel err %.2e  bad %d  written %d/%d (must be all)\n", t2, k2,
           worst, bad, nonzero, t2 * k2);
    printf("  %s\n", (bad == 0 && nonzero == t2 * k2) ? "CORRECT + FULL COVERAGE"
                                                      : "PROBLEM");
    free(hS);
  }

  bench<2, 4, 4, 1, 64>(dQ, dK, dS, tok, kv, NH);
  bench<2, 4, 2, 1, 64>(dQ, dK, dS, tok, kv, NH);
  bench<4, 2, 2, 2, 64>(dQ, dK, dS, tok, kv, NH);
  bench<2, 4, 4, 1, 128>(dQ, dK, dS, tok, kv, NH);
  bench<2, 4, 8, 1, 128>(dQ, dK, dS, tok, kv, NH);
  bench<1, 8, 8, 1, 128>(dQ, dK, dS, tok, kv, NH);
  bench<1, 8, 4, 2, 128>(dQ, dK, dS, tok, kv, NH);
  bench<1, 8, 8, 2, 256>(dQ, dK, dS, tok, kv, NH);
  return 0;
}
