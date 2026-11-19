// QK^T: push BM up by putting all warps on M, which also removes the register
// duplication of the A tile.
//
// A fragments are replicated once per warp in the N direction, so with WNUL warps
// across N the Q tile costs WNUL times its natural register size.  Measured on
// the previous sweep: W2x4 held qreg=128 for BM=256 (4x replication) and W1x8 held
// qreg=128 for BM=128 (8x).  Flipping to W8x1 makes the replication 1x, so BM can
// be eight times larger for the same registers.
//
// The trade is BN: with one N tile per warp BN is only 16 columns.  That is fine
// for arithmetic intensity because staging cost scales with BN while MMA work
// scales with BM*BN, so the ratio depends only on BM.  It does mean more loop
// iterations and therefore more barriers, which the sweep measures.
//
// Geometry is derived, never independently parameterised:
//   BM = WMUL*MT*16   BN = WNUL*NT*16   NWARP = WMUL*WNUL
// Every configuration is verified against a host reference with the output buffer
// poisoned, so a coverage hole cannot pass.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

using namespace nvcuda;

#define DK 128
#define KCH (DK / 32)

template <int WMUL, int WNUL, int MT, int NT, int STORE, int MINCTA = 0>
__global__ __launch_bounds__((WMUL * WNUL) * 64, MINCTA) void qk(const __half *Q, const __half *K,
                                                         float *S, int kv, int NH) {
  constexpr int NWARP = WMUL * WNUL;
  constexpr int NTHREAD = NWARP * 64;
  constexpr int BM = WMUL * MT * 16;
  constexpr int BN = WNUL * NT * 16;
  constexpr int TOT = BN * DK / 8;                 // 16-byte chunks per KV block
  constexpr int NCHUNK = (TOT + NTHREAD - 1) / NTHREAD;

  __shared__ unsigned int Ksw[BN * DK / 2];

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
  uint4 pf[NCHUNK];

  auto fetch = [&](int n0) {
#pragma unroll
    for (int t = 0; t < NCHUNK; ++t) {
      const int v = tid + t * NTHREAD;
      if (v < TOT) pf[t] = *(const uint4 *)&K[(size_t)(n0 + (v * 8) / DK) * DK + (v * 8) % DK];
    }
  };
  auto publish = [&]() {
#pragma unroll
    for (int t = 0; t < NCHUNK; ++t) {
      const int v = tid + t * NTHREAD;
      if (v < TOT) {
        const int n = (v * 8) / DK, k0 = (v * 8) % DK;
        const int b = (((n >> 4) * KCH + (k0 >> 5)) * 256) + ((k0 & 31) >> 3) + ((n & 15) * 4);
        Ksw[b] = pf[t].x;
        Ksw[b + 64] = pf[t].y;
        Ksw[b + 128] = pf[t].z;
        Ksw[b + 192] = pf[t].w;
      }
    }
  };

  fetch(0);

  for (int n0 = 0; n0 < kv; n0 += BN) {
    __syncthreads();
    publish();
    __syncthreads();
    fetch(n0 + BN);

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

template <int WMUL, int WNUL, int MT, int NT>
static void verify(const __half *hQ, const __half *hK, __half *dQ, __half *dK, float *dS, int NH,
                   const char *tag) {
  constexpr int BM = WMUL * MT * 16;
  constexpr int BN = WNUL * NT * 16;
  const int kvv = 2 * BN;
  const size_t n = (size_t)BM * kvv;
  cudaMemset(dS, 0xFF, n * 4);
  qk<WMUL, WNUL, MT, NT, 1><<<dim3(1, 1), (WMUL * WNUL) * 64>>>(dQ, dK, dS, kvv, NH);
  cudaError_t e = cudaDeviceSynchronize();
  if (e != cudaSuccess) { printf("  %-22s ERR %s\n", tag, cudaGetErrorString(e)); return; }
  float *hS = (float *)malloc(n * 4);
  cudaMemcpy(hS, dS, n * 4, cudaMemcpyDeviceToHost);
  int bad = 0, tox = 0;
  double worst = 0;
  for (int m = 0; m < BM; ++m)
    for (int j = 0; j < kvv; ++j) {
      const float got = hS[(size_t)m * kvv + j];
      if (got == -1.0f) ++tox;
      double ref = 0;
      for (int d = 0; d < DK; ++d)
        ref += (double)__half2float(hQ[(size_t)m * NH * DK + d]) *
               __half2float(hK[(size_t)j * DK + d]);
      const double er = fabs(got - ref) / (fabs(ref) + 1e-6);
      if (er > worst) worst = er;
      if (er > 3e-2) ++bad;
    }
  free(hS);
  printf("  %-22s BM=%4d BN=%3d  worst %.2e  bad %d  untouched %d  %s\n", tag, BM, BN, worst, bad,
         tox, (bad == 0 && tox == 0) ? "OK" : "FAIL");
}

template <int WMUL, int WNUL, int MT, int NT, int MINCTA = 0>
static void bench(const __half *dQ, const __half *dK, float *dS, int tok, int kv, int NH) {
  constexpr int BM = WMUL * MT * 16;
  constexpr int BN = WNUL * NT * 16;
  char tag[64];
  snprintf(tag, sizeof tag, "W%dx%d_MT%d_NT%d_%dcta", WMUL, WNUL, MT, NT, MINCTA);
  if (tok % BM) { printf("%-22s skip (BM=%d > tok or not divisor)\n", tag, BM); return; }
  dim3 grid(tok / BM, NH);
  auto run = [&]() { qk<WMUL, WNUL, MT, NT, 0, MINCTA><<<grid, (WMUL * WNUL) * 64>>>(dQ, dK, dS, kv, NH); };
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
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, qk<WMUL, WNUL, MT, NT, 0, MINCTA>,
                                                (WMUL * WNUL) * 64, 0);
  cudaFuncAttributes at;
  cudaFuncGetAttributes(&at, qk<WMUL, WNUL, MT, NT, 0, MINCTA>);
  printf("%-22s BM=%4d BN=%3d  %7.3f ms  %6.1f TFLOPS (%5.1f%% peak) | %d CTA = %2d warps | regs %d\n",
         tag, BM, BN, ms, tf, 100 * tf / 196.6, nb, nb * (WMUL * WNUL), at.numRegs);
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

  printf("=== correctness (poisoned, full coverage required) ===\n");
  verify<8, 1, 4, 1>(hQ, hK, dQ, dK, dS, NH, "W8x1_MT4_NT1");
  verify<8, 1, 8, 1>(hQ, hK, dQ, dK, dS, NH, "W8x1_MT8_NT1");
  verify<4, 2, 8, 1>(hQ, hK, dQ, dK, dS, NH, "W4x2_MT8_NT1");
  verify<2, 4, 8, 1>(hQ, hK, dQ, dK, dS, NH, "W2x4_MT8_NT1");
  printf("\n=== throughput, tok=%d kv=%d heads=%d ===\n", tok, kv, NH);
  bench<4, 1, 8, 1>(dQ, dK, dS, tok, kv, NH);
  bench<4, 1, 8, 1, 5>(dQ, dK, dS, tok, kv, NH);
  bench<4, 1, 8, 1, 6>(dQ, dK, dS, tok, kv, NH);
  bench<4, 1, 8, 1, 8>(dQ, dK, dS, tok, kv, NH);
  bench<2, 1, 8, 1>(dQ, dK, dS, tok, kv, NH);
  bench<2, 1, 16, 1>(dQ, dK, dS, tok, kv, NH);
  bench<4, 1, 16, 1>(dQ, dK, dS, tok, kv, NH);
  bench<8, 1, 8, 1, 3>(dQ, dK, dS, tok, kv, NH);
  return 0;
}
