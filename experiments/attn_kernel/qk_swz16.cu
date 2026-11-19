// QK^T with the B fragment laid out in shared memory as one 16-byte quantity per
// lane.
//
// Why: the MMA issue rate on this part is strictly linear in warps/SM --
// measured 23.1 TFLOPS at 8 warps/SM, 46.1 at 16, 69.2 at 24, 92.2 at 32, with
// 262144 registers and 128 KB of shared memory per SM.  So the way to throughput
// is more resident warps, and the way to warps is fewer registers and fewer
// issued instructions per MMA.
//
// In the natural [n][k] layout a B fragment needs four ld.shared.b32 because the
// four uints a lane owns sit 16 bytes apart.  Storing K pre-swizzled so that each
// lane's four uints are contiguous turns that into a single ld.shared.128.
//
// Swizzle, for half (n, k):
//     nblk = n/16   nl = n%16   c = k/32   kl = k%32
//     q    = kl/8               j = (kl%8)/2
//     uint = (nblk*KCH + c)*256 + (nl + 16*j)*4 + q
// and the fragment for lane L is uint4 at ((nblk*KCH + c)*64 + L), so the read is
// a direct index by lane with no arithmetic and no conflicts (64 lanes x 16 B is
// one contiguous 1024 B span).
//
// tile: BM=128 BN=64 DK=128, 8 warps as 4(M) x 2(N), warp tile 32M x 32N
// regs: qreg 32 + acc 16 + fb 8 -- roughly half of the 64x16 variant
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

using namespace nvcuda;

#define BM 128
#define BN 64
#define DK 128
#define MT 2                // M tiles per warp (32 rows)
#define WNUL 2              // N tiles per warp (32 cols)
#define WMUL (BM / 16 / MT) // 4
#define NWARP (WMUL * WNUL) // 8
#define NTHREAD (NWARP * 64)
#define KCH (DK / 32)       // 4

__global__ __launch_bounds__(NTHREAD) void qk(const __half *Q, const __half *K, float *S,
                                              int kv, int NH) {
  __shared__ unsigned int Ksw[BN * DK / 2];  // 16 KB, swizzled

  const int tid = threadIdx.x;
  const int lane = tid & 63;
  const int warp = tid >> 6;
  const int wm = warp / WNUL;
  const int wn = warp % WNUL;
  const int tile = blockIdx.x;
  const int head = blockIdx.y;
  const int qbase = tile * BM;

  // ---- Q fragments, built once, straight from global ----
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

  constexpr int CH = BN * DK / 8 / NTHREAD;  // 2 chunks of 8 halves per thread
  float sink = 0.f;

  for (int n0 = 0; n0 < kv; n0 += BN) {
    __syncthreads();  // previous block fully consumed
    // stage K: 16 B global read, then four 4 B writes into the swizzle
#pragma unroll
    for (int t = 0; t < CH; ++t) {
      const int v = tid + t * NTHREAD;
      const int n = (v * 8) / DK;
      const int k0 = (v * 8) % DK;
      const uint4 raw = *(const uint4 *)&K[(size_t)(n0 + n) * DK + k0];
      const int base = (((n >> 4) * KCH + (k0 >> 5)) * 256) + ((k0 & 31) >> 3);
      const int nl = n & 15;
      Ksw[base + (nl) * 4] = raw.x;
      Ksw[base + (nl + 16) * 4] = raw.y;
      Ksw[base + (nl + 32) * 4] = raw.z;
      Ksw[base + (nl + 48) * 4] = raw.w;
    }
    __syncthreads();

    wmma::fragment<wmma::accumulator, 16, 16, 32, float> acc[MT][WNUL];
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
      for (int j = 0; j < WNUL; ++j) wmma::fill_fragment(acc[i][j], 0.f);

    const uint4 *kbase = (const uint4 *)Ksw;
#pragma unroll
    for (int c = 0; c < KCH; ++c) {
      wmma::fragment<wmma::matrix_b, 16, 16, 32, __half, wmma::col_major> fb[WNUL];
#pragma unroll
      for (int j = 0; j < WNUL; ++j) {
        const uint4 f = kbase[((wn * WNUL + j) * KCH + c) * 64 + lane];
        fb[j].x[0] = f.x;
        fb[j].x[1] = f.y;
        fb[j].x[2] = f.z;
        fb[j].x[3] = f.w;
      }
#pragma unroll
      for (int i = 0; i < MT; ++i) {
        wmma::fragment<wmma::matrix_a, 16, 16, 32, __half, wmma::row_major> fa;
#pragma unroll
        for (int q = 0; q < 4; ++q) fa.x[q] = qreg[i][c][q];
#pragma unroll
        for (int j = 0; j < WNUL; ++j) wmma::mma_sync(acc[i][j], fa, fb[j], acc[i][j]);
      }
    }

#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
      for (int j = 0; j < WNUL; ++j) sink += acc[i][j].x[0];
  }
  if (sink == 1234.5678f) S[0] = sink;
}

// Reference path for validation: same math, writes S out.
__global__ __launch_bounds__(NTHREAD) void qk_store(const __half *Q, const __half *K, float *S,
                                                   int kv, int NH) {
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
  constexpr int CH = BN * DK / 8 / NTHREAD;

  for (int n0 = 0; n0 < kv; n0 += BN) {
    __syncthreads();
#pragma unroll
    for (int t = 0; t < CH; ++t) {
      const int v = tid + t * NTHREAD;
      const int n = (v * 8) / DK, k0 = (v * 8) % DK;
      const uint4 raw = *(const uint4 *)&K[(size_t)(n0 + n) * DK + k0];
      const int base = (((n >> 4) * KCH + (k0 >> 5)) * 256) + ((k0 & 31) >> 3);
      const int nl = n & 15;
      Ksw[base + nl * 4] = raw.x;
      Ksw[base + (nl + 16) * 4] = raw.y;
      Ksw[base + (nl + 32) * 4] = raw.z;
      Ksw[base + (nl + 48) * 4] = raw.w;
    }
    __syncthreads();

    wmma::fragment<wmma::accumulator, 16, 16, 32, float> acc[MT][WNUL];
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
      for (int j = 0; j < WNUL; ++j) wmma::fill_fragment(acc[i][j], 0.f);

    const uint4 *kbase = (const uint4 *)Ksw;
#pragma unroll
    for (int c = 0; c < KCH; ++c) {
      wmma::fragment<wmma::matrix_b, 16, 16, 32, __half, wmma::col_major> fb[WNUL];
#pragma unroll
      for (int j = 0; j < WNUL; ++j) {
        const uint4 f = kbase[((wn * WNUL + j) * KCH + c) * 64 + lane];
        fb[j].x[0] = f.x; fb[j].x[1] = f.y; fb[j].x[2] = f.z; fb[j].x[3] = f.w;
      }
#pragma unroll
      for (int i = 0; i < MT; ++i) {
        wmma::fragment<wmma::matrix_a, 16, 16, 32, __half, wmma::row_major> fa;
#pragma unroll
        for (int q = 0; q < 4; ++q) fa.x[q] = qreg[i][c][q];
#pragma unroll
        for (int j = 0; j < WNUL; ++j) wmma::mma_sync(acc[i][j], fa, fb[j], acc[i][j]);
      }
    }

    const int ccol = lane & 15, crow = lane >> 4;
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
      for (int j = 0; j < WNUL; ++j)
#pragma unroll
        for (int q = 0; q < 4; ++q)
          S[(size_t)(qbase + wm * (MT * 16) + i * 16 + crow + 4 * q) * kv + n0 + wn * (WNUL * 16) +
            j * 16 + ccol] = acc[i][j].x[q];
    __syncthreads();
  }
}

int main(int argc, char **argv) {
  const int tok = argc > 1 ? atoi(argv[1]) : 4096;
  const int kv = argc > 2 ? atoi(argv[2]) : 4096;
  const int iters = argc > 3 ? atoi(argv[3]) : 10;
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

  {
    const int k2 = 256, t2 = BM;
    qk_store<<<dim3(tok / BM, 1), NTHREAD>>>(dQ, dK, dS, k2, NH);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) { printf("ERR %s\n", cudaGetErrorString(e)); return 1; }
    float *hS = (float *)malloc((size_t)t2 * k2 * 4);
    cudaMemcpy(hS, dS, (size_t)t2 * k2 * 4, cudaMemcpyDeviceToHost);
    double worst = 0; int bad = 0;
    for (int m = 0; m < t2; ++m)
      for (int n = 0; n < k2; ++n) {
        double ref = 0;
        for (int d = 0; d < DK; ++d)
          ref += (double)__half2float(hQ[(size_t)m * NH * DK + d]) *
                 __half2float(hK[(size_t)n * DK + d]);
        double er = fabs(hS[(size_t)m * k2 + n] - ref) / (fabs(ref) + 1e-6);
        if (er > worst) worst = er;
        if (er > 3e-2) ++bad;
      }
    printf("correctness (head 0, %dx%d): worst rel err %.2e bad %d %s\n", t2, k2, worst, bad,
           bad == 0 ? "CORRECT" : "MISMATCH");
    free(hS);
  }

  dim3 grid(tok / BM, NH);
  auto run = [&]() { qk<<<grid, NTHREAD>>>(dQ, dK, dS, kv, NH); };
  for (int i = 0; i < 3; ++i) run();
  cudaEvent_t s, ev;
  cudaEventCreate(&s); cudaEventCreate(&ev);
  cudaEventRecord(s);
  for (int i = 0; i < iters; ++i) run();
  cudaEventRecord(ev);
  cudaEventSynchronize(ev);
  float ms; cudaEventElapsedTime(&ms, s, ev); ms /= iters;
  double fl = 2.0 * tok * kv * DK * NH;
  double tf = fl / (ms * 1e-3) / 1e12;
  int nb = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, qk, NTHREAD, 0);
  cudaFuncAttributes at;
  cudaFuncGetAttributes(&at, qk);
  printf("tok=%d kv=%d: %.3f ms  %.1f TFLOPS (%.0f%% of 96.6) | %d CTA/SM = %d warps | smem %zuB regs %d\n",
         tok, kv, ms, tf, 100 * tf / 96.6, nb, nb * NWARP, (size_t)at.sharedSizeBytes, at.numRegs);
  return 0;
}
