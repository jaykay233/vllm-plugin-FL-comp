// QK^T, template on how much M each warp owns, with the B fragment stored in
// shared memory as one 16-byte quantity per lane.
//
// Two corrections drive this version.
//
// 1. Flops accounting.  tile_ok.cu computes and verifies a 16x16x32 tile using
//    exactly two ml_matrix_mad_f32_f16_m16n16k16 instructions, so one such
//    instruction is 16x16x16 = 8192 flops.  mma_rate.cu charged them at 4096, so
//    every peak figure derived from it doubled: the real ceilings are 46 TFLOPS
//    at 8 warps/SM, 92 at 16, 138 at 24, 184 at 32, and full peak is ~196
//    (16 SM x 1 MMA/cycle x 8192 flops x 1.5 GHz = 196.6).  ixBLAS at 96.5 is
//    therefore 49% of peak, not 98%, and the vendor FA at 52.7 is 27%.
//
// 2. Instruction mix is what limits this kernel, not occupancy.  The MT=2 variant
//    reached 64 warps/SM yet only 20 TFLOPS, because per loop iteration it issues
//    32 MMA against roughly as many staging, load and address instructions.
//    Staging cost depends only on BN, so the fix is to buy more MMA per staged
//    byte by giving each warp more M rows.
//
// layout: 8 warps all along M, each owning MT tiles (BM = 8*MT*16), one 16-wide N
// tile, so a B fragment is loaded once and feeds MT MMAs.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

using namespace nvcuda;

#define DK 128
#define BN 64
#define WNUL 1
#define WMUL 8
#define NWARP (WMUL * WNUL)
#define NTHREAD (NWARP * 64)
#define KCH (DK / 32)

// Swizzled shared layout: lane L's four fragment uints are contiguous.
//   half (n,k): nblk=n/16 nl=n%16 c=k/32 kl=k%32 q=kl/8 j=(kl%8)/2
//   uint index = (nblk*KCH + c)*256 + (nl + 16*j)*4 + q
__device__ __forceinline__ int swz_base(int n, int k0) {
  return (((n >> 4) * KCH + (k0 >> 5)) * 256) + ((k0 & 31) >> 3) + ((n & 15) * 4);
}
__device__ __forceinline__ void swz_stage(unsigned int *Ksw, const __half *K, int n0, int tid) {
  constexpr int CH = BN * DK / 8 / NTHREAD;
#pragma unroll
  for (int t = 0; t < CH; ++t) {
    const int v = tid + t * NTHREAD;
    const int n = (v * 8) / DK, k0 = (v * 8) % DK;
    const uint4 raw = *(const uint4 *)&K[(size_t)(n0 + n) * DK + k0];
    const int b = swz_base(n, k0);
    Ksw[b] = raw.x;
    Ksw[b + 64] = raw.y;
    Ksw[b + 128] = raw.z;
    Ksw[b + 192] = raw.w;
  }
}

template <int MT, int BNP, int STORE>
__global__ __launch_bounds__(NTHREAD) void qk(const __half *Q, const __half *K, float *S,
                                              int kv, int NH) {
  constexpr int BM = WMUL * MT * 16;
  __shared__ unsigned int Ksw[BNP * DK / 2];

  const int tid = threadIdx.x;
  const int lane = tid & 63;
  const int wm = tid >> 6;
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
    swz_stage(Ksw, K, n0, tid);
    __syncthreads();

    wmma::fragment<wmma::accumulator, 16, 16, 32, float> acc[MT];
#pragma unroll
    for (int i = 0; i < MT; ++i) wmma::fill_fragment(acc[i], 0.f);

#pragma unroll
    for (int c = 0; c < KCH; ++c) {
      wmma::fragment<wmma::matrix_b, 16, 16, 32, __half, wmma::col_major> fb;
      const uint4 f = kbase[c * 64 + lane];
      fb.x[0] = f.x; fb.x[1] = f.y; fb.x[2] = f.z; fb.x[3] = f.w;
#pragma unroll
      for (int i = 0; i < MT; ++i) {
        wmma::fragment<wmma::matrix_a, 16, 16, 32, __half, wmma::row_major> fa;
#pragma unroll
        for (int q = 0; q < 4; ++q) fa.x[q] = qreg[i][c][q];
        wmma::mma_sync(acc[i], fa, fb, acc[i]);
      }
    }

    const int ccol = lane & 15, crow = lane >> 4;
    if (STORE) {
#pragma unroll
      for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int q = 0; q < 4; ++q)
          S[(size_t)(qbase + wm * (MT * 16) + i * 16 + crow + 4 * q) * kv + n0 + ccol] =
              acc[i].x[q];
    } else {
#pragma unroll
      for (int i = 0; i < MT; ++i) sink += acc[i].x[0];
    }
  }
  if (sink == 1234.5678f) S[0] = sink;
}

template <int MT, int BNP>
static void bench(const __half *dQ, const __half *dK, float *dS, int tok, int kv, int NH,
                  const char *tag) {
  constexpr int BM = WMUL * MT * 16;
  if (tok % BM) { printf("%s: tok not divisible by BM=%d\n", tag, BM); return; }
  dim3 grid(tok / BM, NH);
  auto run = [&]() { qk<MT, BNP, 0><<<grid, NTHREAD>>>(dQ, dK, dS, kv, NH); };
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
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, qk<MT, BNP, 0>, NTHREAD, 0);
  cudaFuncAttributes at;
  cudaFuncGetAttributes(&at, qk<MT, BNP, 0>);
  printf("%-22s BM=%3d BN=%3d: %7.3f ms  %6.1f TFLOPS (%5.1f%% of 196) | %d CTA = %2d warps | regs %d\n",
         tag, BM, BNP, ms, tf, 100 * tf / 196.6, nb, nb * NWARP, at.numRegs);
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

  // correctness on one head, kv=256
  {
    constexpr int MT = 2;
    const int k2 = 256, t2 = WMUL * MT * 16;
    qk<MT, BN, 1><<<dim3(1, 1), NTHREAD>>>(dQ, dK, dS, k2, NH);
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

  bench<1, 64>(dQ, dK, dS, tok, kv, NH, "MT=1  (BM=128)");
  bench<2, 64>(dQ, dK, dS, tok, kv, NH, "MT=2  (BM=256)");
  bench<4, 64>(dQ, dK, dS, tok, kv, NH, "MT=4  (BM=512)");
  bench<2, 128>(dQ, dK, dS, tok, kv, NH, "MT=2  (BM=256, BN=128)");
  bench<4, 128>(dQ, dK, dS, tok, kv, NH, "MT=4  (BM=512, BN=128)");
  return 0;
}
