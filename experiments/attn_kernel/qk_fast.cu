// QK^T at full prefill scale, tuned for occupancy.
//
// Changes over qk_scaled.cu:
//   * K is staged with 16-byte vector copies.  The KV cache is head_dim-major
//     and Ks[n][k] has the same orientation, so no transpose is needed there --
//     the earlier scalar copy was pure waste.
//   * 8 warps split as 4(M) x 2(N) so a 32 KB smem budget still gives 8 warps
//     per CTA, which allows 4 CTAs/SM = 32 warps/SM, the level the issue-rate
//     probe showed is needed to reach ~92 TFLOPS.
//   * the S store is optional, because a fused FA never writes S to HBM.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

using namespace nvcuda;

#define BM 64
#define BN 64
#define DK 128
#define NWARP 8
#define NTHREAD (NWARP * 64)
#define NSPLIT 2
#define QA 8  // row pad (halves) to break smem bank conflicts
#define KB 8  // warps along N: each covers BN/NSPLIT = 32 columns
#define WN (BN / NSPLIT / 16)  // 16-wide tiles per warp = 2

__global__ __launch_bounds__(NTHREAD) void qk(const __half *Q, const __half *K, float *S,
                                              int kv, int NH, int store_s) {
  __shared__ __half Qs[DK][BM + QA];  // k-major; padded stride
  __shared__ __half Ks[BN][DK + KB];  // n-major; padded stride

  const int tid = threadIdx.x;
  const int lane = tid & 63;
  const int warp = tid >> 6;
  const int wm = warp / NSPLIT;   // 0..3  -> M tile
  const int wn = warp % NSPLIT;   // 0..1  -> N half
  const int tile = blockIdx.x;
  const int head = blockIdx.y;
  const int qbase = tile * BM;

  // stage Q, transposed to k-major (scatter: 8192 halves, but once per CTA)
  for (int v = tid; v < BM * DK; v += NTHREAD) {
    const int m = v / DK, d = v % DK;
    Qs[d][m] = Q[(size_t)(qbase + m) * NH * DK + (size_t)head * DK + d];
  }
  __syncthreads();

  float sink = 0.f;

  for (int n0 = 0; n0 < kv; n0 += BN) {
    // vectorised K copy: 16 B at a time, both sides head_dim-contiguous
    for (int v = tid; v < BN * DK / 8; v += NTHREAD) {
      *(ulonglong2 *)&Ks[(v * 8) / DK][(v * 8) % DK] =
          *(const ulonglong2 *)&K[(size_t)(n0 + (v * 8) / DK) * DK + (v * 8) % DK];
    }
    __syncthreads();

    wmma::fragment<wmma::accumulator, 16, 16, 32, float> acc[WN];
#pragma unroll
    for (int j = 0; j < WN; ++j) wmma::fill_fragment(acc[j], 0.f);

    for (int kc = 0; kc < DK; kc += 32) {
      wmma::fragment<wmma::matrix_a, 16, 16, 32, __half, wmma::row_major> fa;
      unsigned int *pa = (unsigned int *)&fa.x[0];
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const int m = (lane / 16) * 2 + 8 * (q % 2);
        const int k = kc + (lane % 16) + 16 * (q / 2);
        pa[q] = *(const unsigned int *)&Qs[k][wm * 16 + m];
      }
#pragma unroll
      for (int j = 0; j < WN; ++j) {
        wmma::fragment<wmma::matrix_b, 16, 16, 32, __half, wmma::col_major> fb;
        unsigned int *pb = (unsigned int *)&fb.x[0];
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          const int k = kc + (lane / 16) * 2 + 8 * q;
          const int n = wn * (BN / NSPLIT) + j * 16 + (lane % 16);
          pb[q] = *(const unsigned int *)&Ks[n][k];
        }
        wmma::mma_sync(acc[j], fa, fb, acc[j]);
      }
    }

    if (store_s) {
      const int ccol = lane & 15, crow = lane >> 4;
#pragma unroll
      for (int j = 0; j < WN; ++j)
#pragma unroll
        for (int q = 0; q < 4; ++q)
          S[(size_t)(qbase + wm * 16 + crow + 4 * q) * kv + n0 + wn * (BN / NSPLIT) + j * 16 +
            ccol] = acc[j].x[q];
    } else {
#pragma unroll
      for (int j = 0; j < WN; ++j) sink += acc[j].x[0];
    }
    __syncthreads();
  }
  if (sink == 1234.5678f) S[0] = sink;  // keep accumulators alive
}

int main(int argc, char **argv) {
  const int tok = argc > 1 ? atoi(argv[1]) : 4096;
  const int kv = argc > 2 ? atoi(argv[2]) : 4096;
  const int iters = argc > 3 ? atoi(argv[3]) : 10;
  const int store_s = argc > 4 ? atoi(argv[4]) : 0;
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

  // ---- correctness on a small single-head case ----
  {
    const int t2 = 64, k2 = 256;
    dim3 g(t2 / BM, 1);
    qk<<<g, NTHREAD>>>(dQ, dK, dS, k2, NH, 1);
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
    printf("correctness (head 0, tok=%d kv=%d): worst rel err %.2e  bad %d  %s\n", t2, k2, worst,
           bad, bad == 0 ? "CORRECT" : "MISMATCH");
    free(hS);
  }

  dim3 grid(tok / BM, NH);
  auto run = [&]() { qk<<<grid, NTHREAD>>>(dQ, dK, dS, kv, NH, store_s); };
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
  cudaFuncAttributes fa_;
  cudaFuncGetAttributes(&fa_, qk);
  printf("tok=%d kv=%d heads=%d store_s=%d: %.3f ms  %.1f TFLOPS  (%.0f%% of 96.6)\n", tok, kv,
         NH, store_s, ms, tf, 100 * tf / 96.6);
  printf("  occupancy: %d CTAs/SM = %d warps/SM | smem/CTA=%zu B | regs/thread=%d\n", nb,
         nb * NWARP, (size_t)fa_.sharedSizeBytes, fa_.numRegs);
  return 0;
}
