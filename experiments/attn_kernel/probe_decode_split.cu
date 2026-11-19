// Split-KV decode: does cutting each sequence's KV range into S segments recover
// the bandwidth the 16k shape loses?
//
// Evidence this is aimed at.  The vendor decode kernel, measured on the real 16k
// scoring shape and with page metadata prebuilt, runs at 421 GB/s:
//
//     30 x 16384   (60 streams)   503.3 MB   1.194 ms   421 GB/s
//     60 x 8192   (120 streams)   503.3 MB   0.797 ms   632 GB/s
//
// Same bytes, 1.5x apart, so the 16k shape is short of parallelism rather than
// bandwidth.  The 16k case runs 30 sequences with 2 KV heads each and the vendor
// kernel gives each work unit a run of KV, so it has 60 units for 16 SMs.
//
// Before writing a real split-KV attention kernel, this probe answers the question
// in an afternoon instead of a day: it reproduces the memory access pattern of a
// split-KV decode exactly -- same addresses, same order, same bytes -- and does the
// same amount of arithmetic per byte, but with a trivial reduction instead of a real
// softmax.  If bandwidth does not recover here, splitting will not help and the real
// kernel is not worth writing.
//
// It also settles a design question.  Decode at this shape is 8 flop/byte, so
// sustaining 632 GB/s needs 5.06 TFLOPS, and the SIMT FP32 peak is only
// 16 SM * 128 lanes * 2 * 1.5 GHz = 6.1 TFLOPS -- 82% of it, before counting the
// bf16->f32 conversions.  So if this SIMT probe cannot reach the bandwidth, the
// real kernel has to use MMA; if it can, SIMT is enough and the code stays simple.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>

typedef __ix_bfloat16 bf_t;
typedef __bf16 v2y __attribute__((vector_size(4)));

#define D 128
#define PAGE 16
#define NKVH 2
#define NH_PER_KVH 8  // 16 query heads over 2 KV heads

// HND paged cache: [page][kvh][PAGE position][D]
__device__ __forceinline__ float b2f(__bf16 x) {
  unsigned short h;
  __builtin_memcpy(&h, &x, 2);
  unsigned u = (unsigned)h << 16;
  float f;
  __builtin_memcpy(&f, &u, 4);
  return f;
}

__device__ __forceinline__ long kaddr(const int *bt, int kvh, int pos, int d) {
  return ((long)bt[pos >> 4] * NKVH + kvh) * PAGE * D + (long)(pos & 15) * D + d;
}

template <int NWARP>
__global__ __launch_bounds__(NWARP * 64) void probe(const bf_t *__restrict__ K,
                                                    const bf_t *__restrict__ V,
                                                    const int *__restrict__ bt, int maxblk,
                                                    int S, int kvlen, float *sink) {
  const int s = blockIdx.y / NKVH;
  const int kvh = blockIdx.y % NKVH;
  const int chunk = (kvlen + S - 1) / S;
  const int lo = blockIdx.x * chunk;
  const int hi = lo + chunk < kvlen ? lo + chunk : kvlen;
  const int *b = bt + (long)s * maxblk;

  const int tid = threadIdx.x;
  const int lane = tid & 63;
  const int warp = tid >> 6;
  const int d0 = 2 * lane;  // this lane owns dims [2*lane, 2*lane+2) of every position

  float q[NH_PER_KVH][2];
#pragma unroll
  for (int h = 0; h < NH_PER_KVH; ++h) { q[h][0] = 0.5f + 0.01f * h; q[h][1] = 0.25f + 0.01f * h; }
  float acc[NH_PER_KVH][2];
#pragma unroll
  for (int h = 0; h < NH_PER_KVH; ++h) { acc[h][0] = 0.f; acc[h][1] = 0.f; }

  for (int pos = lo + warp; pos < hi; pos += NWARP) {
    const long a = kaddr(b, kvh, pos, d0);
    const v2y kk = *(const v2y *)(K + a);
    const v2y vv = *(const v2y *)(V + a);
    const float k0 = b2f(kk[0]), k1 = b2f(kk[1]);
    const float v0 = b2f(vv[0]), v1 = b2f(vv[1]);
#pragma unroll
    for (int h = 0; h < NH_PER_KVH; ++h) {
      // partial QK, then partial PV: the same two multiplies per dim the real
      // kernel needs, so the arithmetic per byte matches
      const float dot = q[h][0] * k0 + q[h][1] * k1;
      acc[h][0] += dot * v0;
      acc[h][1] += dot * v1;
    }
  }

  // collapse to one value per CTA so nothing above can be eliminated
  float r = 0.f;
#pragma unroll
  for (int h = 0; h < NH_PER_KVH; ++h) r += acc[h][0] + acc[h][1];
#pragma unroll
  for (int off = 1; off < 64; off <<= 1) r += __shfl_xor_sync(~0ULL, r, off);
  if (lane == 0 && r == 1.2345678e30f) sink[blockIdx.y] = r;
}


// ---------------------------------------------------------------------------
// Probe 2: the structure the real kernel will actually have.
//
// Warp-level independence is what buys the bandwidth, and a warp can only finish a
// KV range on its own if that range is contiguous (softmax has to see all of it).
// So each warp owns one contiguous chunk of one (seq, kvh) and the splittings are
// packed W-per-CTA.  This measures bandwidth for that exact layout.
// ---------------------------------------------------------------------------
template <int W>
__global__ __launch_bounds__(W * 64) void probe2(const bf_t *__restrict__ K,
                                                 const bf_t *__restrict__ V,
                                                 const int *__restrict__ bt, int maxblk,
                                                 int S, int kvlen, int nunits, float *sink) {
  const int tid = threadIdx.x;
  const int lane = tid & 63;
  const int wid = blockIdx.x * W + (tid >> 6);
  if (wid >= nunits * S) return;

  const int unit = wid % nunits;
  const int split = wid / nunits;
  const int s = unit / NKVH;
  const int kvh = unit % NKVH;
  const int chunk = (kvlen + S - 1) / S;
  const int lo = split * chunk;
  const int hi = lo + chunk < kvlen ? lo + chunk : kvlen;
  const int *b = bt + (long)s * maxblk;

  const int d0 = 2 * lane;
  float q[NH_PER_KVH][2];
#pragma unroll
  for (int h = 0; h < NH_PER_KVH; ++h) { q[h][0] = 0.5f + 0.01f * h; q[h][1] = 0.25f + 0.01f * h; }
  float acc[NH_PER_KVH][2];
#pragma unroll
  for (int h = 0; h < NH_PER_KVH; ++h) { acc[h][0] = 0.f; acc[h][1] = 0.f; }

  for (int pos = lo; pos < hi; ++pos) {
    const long a = kaddr(b, kvh, pos, d0);
    const v2y kk = *(const v2y *)(K + a);
    const v2y vv = *(const v2y *)(V + a);
    const float k0 = b2f(kk[0]), k1 = b2f(kk[1]);
    const float v0 = b2f(vv[0]), v1 = b2f(vv[1]);
#pragma unroll
    for (int h = 0; h < NH_PER_KVH; ++h) {
      const float dot = q[h][0] * k0 + q[h][1] * k1;
      acc[h][0] += dot * v0;
      acc[h][1] += dot * v1;
    }
  }
  float r = 0.f;
#pragma unroll
  for (int h = 0; h < NH_PER_KVH; ++h) r += acc[h][0] + acc[h][1];
  if (r == 1.2345678e30f) sink[wid] = r;
}


// ---------------------------------------------------------------------------
// Probe 4: same access pattern as probe2, but with the occupancy the real kernel
// will be stuck at, and with the shared memory staging the real kernel needs.
//
// The real kernel must transpose K into MMA fragment order, and that transpose can
// only be done through shared memory.  Each warp owns a contiguous KV chunk, so the
// staging buffer is private to the warp: a K tile (16 pos x 128 dim = 4 KB) plus a V
// tile (4 KB).  At 8 KB per warp and 128 KB of shared memory per SM, that caps
// residency at 16 warps/SM -- whereas probe2, which had no shared memory, ran at
// full occupancy.
//
// Both the stage-out and the read-back below walk shared memory linearly, which is
// conflict free.  The real kernel's swizzle is also conflict free by construction,
// so the instruction and bank counts are representative; only the address mapping
// differs.  Bytes transported and requests issued are exactly the real ones.
//
// probe2 (no smem, full occupancy) numbers for reference:
//     480 warps  410 GB/s      960 warps  611 GB/s      1920 warps  598 GB/s
template <int W>
__global__ __launch_bounds__(W * 64) void probe4(const bf_t *__restrict__ K,
                                                 const bf_t *__restrict__ V,
                                                 const int *__restrict__ bt, int maxblk,
                                                 int S, int kvlen, int nunits, float *sink) {
  __shared__ bf_t buf[W * 4096];  // 2048 bf16 for K + 2048 for V, per warp
  const int tid = threadIdx.x;
  const int lane = tid & 63;
  const int wid = blockIdx.x * W + (tid >> 6);
  if (wid >= nunits * S) return;

  const int unit = wid % nunits;
  const int split = wid / nunits;
  const int s = unit / NKVH;
  const int kvh = unit % NKVH;
  const int chunk = (kvlen + S - 1) / S;
  const int lo = split * chunk;
  const int hi = lo + chunk < kvlen ? lo + chunk : kvlen;
  const int *b = bt + (long)s * maxblk;

  bf_t *sK = buf + (tid >> 6) * 4096;
  bf_t *sV = sK + 2048;

  float q[NH_PER_KVH][2];
#pragma unroll
  for (int h = 0; h < NH_PER_KVH; ++h) { q[h][0] = 0.5f + 0.01f * h; q[h][1] = 0.25f + 0.01f * h; }
  float acc[NH_PER_KVH][2];
#pragma unroll
  for (int h = 0; h < NH_PER_KVH; ++h) { acc[h][0] = 0.f; acc[h][1] = 0.f; }

  for (int pos = lo; pos < hi; pos += 16) {
    // One page's kv-head slice is 16 x 128 contiguous bf16, so it streams with
    // eight 512-byte vector loads per warp.
    const long base = ((long)b[pos >> 4] * NKVH + kvh) * PAGE * D;
#pragma unroll
    for (int it = 0; it < 8; ++it) {
      const int e = (it * 64 + lane) * 4;
      *(v2y *)(sK + e) = *(const v2y *)(K + base + e);
      *(v2y *)(sV + e) = *(const v2y *)(V + base + e);
    }
    __syncwarp();

    // read the tiles back the way the MMAs will: one 16x16 fragment per channel
    v2y kf[8], vf[8];
#pragma unroll
    for (int ch = 0; ch < 8; ++ch) {
      kf[ch] = *(const v2y *)(sK + (ch * 64 + lane) * 4);
      vf[ch] = *(const v2y *)(sV + (ch * 64 + lane) * 4);
    }
    // stand in for QK (8 MMAs) and PV (8 MMAs) with the same operand traffic
    float t0 = 0.f, t1 = 0.f;
#pragma unroll
    for (int ch = 0; ch < 8; ++ch) {
      t0 += b2f(kf[ch][0]) * b2f(kf[ch][1]);
      t1 += b2f(vf[ch][0]) * b2f(vf[ch][1]);
    }
#pragma unroll
    for (int h = 0; h < NH_PER_KVH; ++h) {
      acc[h][0] += q[h][0] * t0;
      acc[h][1] += q[h][1] * t1;
    }
    __syncwarp();
  }
  float r = 0.f;
#pragma unroll
  for (int h = 0; h < NH_PER_KVH; ++h) r += acc[h][0] + acc[h][1];
  if (r == 1.2345678e30f) sink[wid] = r;
}


// Probe 5: probe4 plus a register prefetch of the next page.
//
// probe4 stalls once per page: it issues all sixteen vector loads, then blocks on the
// last one before it can start reading fragments.  Nothing is in flight during that
// wait, which is why it tops out at 569 GB/s where probe2 (one request at a time but
// far more warps) reached 611.
//
// Here the next page's K and V are fetched into registers before the current page's
// fragments are read out of shared memory, so the global loads overlap with the
// shared memory reads and the arithmetic instead of stalling after them.  The
// register cost is 2 x 8 x v2y = 32 bf16 = 16 registers, which is affordable.
template <int W>
__global__ __launch_bounds__(W * 64) void probe5(const bf_t *__restrict__ K,
                                                 const bf_t *__restrict__ V,
                                                 const int *__restrict__ bt, int maxblk,
                                                 int S, int kvlen, int nunits, float *sink) {
  __shared__ bf_t buf[W * 4096];
  const int tid = threadIdx.x;
  const int lane = tid & 63;
  const int wid = blockIdx.x * W + (tid >> 6);
  if (wid >= nunits * S) return;

  const int unit = wid % nunits;
  const int split = wid / nunits;
  const int s = unit / NKVH;
  const int kvh = unit % NKVH;
  const int chunk = (kvlen + S - 1) / S;
  const int lo = split * chunk;
  const int hi = lo + chunk < kvlen ? lo + chunk : kvlen;
  const int *b = bt + (long)s * maxblk;

  bf_t *sK = buf + (tid >> 6) * 4096;
  bf_t *sV = sK + 2048;

  float q[NH_PER_KVH][2];
#pragma unroll
  for (int h = 0; h < NH_PER_KVH; ++h) { q[h][0] = 0.5f + 0.01f * h; q[h][1] = 0.25f + 0.01f * h; }
  float acc[NH_PER_KVH][2];
#pragma unroll
  for (int h = 0; h < NH_PER_KVH; ++h) { acc[h][0] = 0.f; acc[h][1] = 0.f; }

  v2y pk[8], pv[8];
  // prime the first page
  {
    const long base = ((long)b[lo >> 4] * NKVH + kvh) * PAGE * D;
#pragma unroll
    for (int it = 0; it < 8; ++it) {
      const int e = (it * 64 + lane) * 4;
      pk[it] = *(const v2y *)(K + base + e);
      pv[it] = *(const v2y *)(V + base + e);
    }
  }

  for (int pos = lo; pos < hi; pos += 16) {
    // hand the registers just fetched over to shared memory
#pragma unroll
    for (int it = 0; it < 8; ++it) {
      const int e = (it * 64 + lane) * 4;
      *(v2y *)(sK + e) = pk[it];
      *(v2y *)(sV + e) = pv[it];
    }
    __syncwarp();

    // issue the next page's loads before touching this one's fragments
    const int npos = pos + 16;
    if (npos < hi) {
      const long nb2 = ((long)b[npos >> 4] * NKVH + kvh) * PAGE * D;
#pragma unroll
      for (int it = 0; it < 8; ++it) {
        const int e = (it * 64 + lane) * 4;
        pk[it] = *(const v2y *)(K + nb2 + e);
        pv[it] = *(const v2y *)(V + nb2 + e);
      }
    }

    v2y kf[8], vf[8];
#pragma unroll
    for (int ch = 0; ch < 8; ++ch) {
      kf[ch] = *(const v2y *)(sK + (ch * 64 + lane) * 4);
      vf[ch] = *(const v2y *)(sV + (ch * 64 + lane) * 4);
    }
    float t0 = 0.f, t1 = 0.f;
#pragma unroll
    for (int ch = 0; ch < 8; ++ch) {
      t0 += b2f(kf[ch][0]) * b2f(kf[ch][1]);
      t1 += b2f(vf[ch][0]) * b2f(vf[ch][1]);
    }
#pragma unroll
    for (int h = 0; h < NH_PER_KVH; ++h) {
      acc[h][0] += q[h][0] * t0;
      acc[h][1] += q[h][1] * t1;
    }
    __syncwarp();  // fragments are in registers now; shared memory is free to reuse
  }
  float r = 0.f;
#pragma unroll
  for (int h = 0; h < NH_PER_KVH; ++h) r += acc[h][0] + acc[h][1];
  if (r == 1.2345678e30f) sink[wid] = r;
}



// Probe 6: probe4 with the shared memory halved, to isolate occupancy.
//
// probe4 and probe5 both plateau near 570 GB/s, and prefetching the next page into
// registers changed nothing, so the stall is not the limiter.  The remaining
// difference from probe2 (611 GB/s) is residency: 8 KB per warp caps it at 16
// warps/SM, while probe2 ran at full occupancy.
//
// This keeps the same 16 batched loads per page and the same bytes, but stages K and
// V through ONE 4 KB buffer in two phases, so residency doubles to 32 warps/SM.  If
// the number moves, occupancy is the limiter; if it does not, 570 is this access
// pattern's ceiling and there is nothing left to chase here.
template <int W>
__global__ __launch_bounds__(W * 64) void probe6(const bf_t *__restrict__ K,
                                                 const bf_t *__restrict__ V,
                                                 const int *__restrict__ bt, int maxblk,
                                                 int S, int kvlen, int nunits, float *sink) {
  __shared__ bf_t buf[W * 2048];  // one tile per warp, reused for K then for V
  const int tid = threadIdx.x;
  const int lane = tid & 63;
  const int wid = blockIdx.x * W + (tid >> 6);
  if (wid >= nunits * S) return;

  const int unit = wid % nunits;
  const int split = wid / nunits;
  const int s = unit / NKVH;
  const int kvh = unit % NKVH;
  const int chunk = (kvlen + S - 1) / S;
  const int lo = split * chunk;
  const int hi = lo + chunk < kvlen ? lo + chunk : kvlen;
  const int *b = bt + (long)s * maxblk;

  bf_t *sh = buf + (tid >> 6) * 2048;

  float q[NH_PER_KVH][2];
#pragma unroll
  for (int h = 0; h < NH_PER_KVH; ++h) { q[h][0] = 0.5f + 0.01f * h; q[h][1] = 0.25f + 0.01f * h; }
  float acc[NH_PER_KVH][2];
#pragma unroll
  for (int h = 0; h < NH_PER_KVH; ++h) { acc[h][0] = 0.f; acc[h][1] = 0.f; }

  for (int pos = lo; pos < hi; pos += 16) {
    const long base = ((long)b[pos >> 4] * NKVH + kvh) * PAGE * D;
    float t0 = 0.f, t1 = 0.f;
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const bf_t *src = half == 0 ? K : V;
      v2y pf[8];
#pragma unroll
      for (int it = 0; it < 8; ++it) {
        const int e = (it * 64 + lane) * 4;
        pf[it] = *(const v2y *)(src + base + e);
      }
      __syncwarp();
#pragma unroll
      for (int it = 0; it < 8; ++it) *(v2y *)(sh + (it * 64 + lane) * 4) = pf[it];
      __syncwarp();
      v2y f[8];
#pragma unroll
      for (int ch = 0; ch < 8; ++ch) f[ch] = *(const v2y *)(sh + (ch * 64 + lane) * 4);
      float tt = 0.f;
#pragma unroll
      for (int ch = 0; ch < 8; ++ch) tt += b2f(f[ch][0]) * b2f(f[ch][1]);
      if (half == 0) t0 = tt; else t1 = tt;
    }
#pragma unroll
    for (int h = 0; h < NH_PER_KVH; ++h) {
      acc[h][0] += q[h][0] * t0;
      acc[h][1] += q[h][1] * t1;
    }
  }
  float r = 0.f;
#pragma unroll
  for (int h = 0; h < NH_PER_KVH; ++h) r += acc[h][0] + acc[h][1];
  if (r == 1.2345678e30f) sink[wid] = r;
}


int main() {
  const int batch = 30, kvlen = 16384;
  const int nb = kvlen / PAGE, pool = nb * batch;
  const size_t nkv = (size_t)pool * NKVH * PAGE * D;

  bf_t *dK, *dV;
  int *dbt;
  float *sink;
  cudaMalloc(&dK, nkv * 2);
  cudaMalloc(&dV, nkv * 2);
  cudaMalloc(&dbt, (size_t)batch * nb * 4);
  cudaMalloc(&sink, 4096 * 4);
  cudaMemset(dK, 1, nkv * 2);
  cudaMemset(dV, 1, nkv * 2);
  int *hbt = (int *)malloc((size_t)batch * nb * 4);
  for (int i = 0; i < batch * nb; ++i) hbt[i] = i;
  cudaMemcpy(dbt, hbt, (size_t)batch * nb * 4, cudaMemcpyHostToDevice);

  const double bytes = 2.0 * batch * kvlen * NKVH * D * 2;  // K and V, bf16
  printf("16k shape: %d seq x %d kv x %d kvh x %d dim, %.1f MB\n\n", batch, kvlen, NKVH, D,
         bytes / 1e6);
  printf("%-22s %-9s %8s %8s %9s %8s\n", "config", "CTAs", "ms", "GB/s", "% of 819", "% of 632");

  for (int nwarp : {4, 8, 16}) {
    for (int S : {1, 2, 4, 8}) {
      dim3 grid(S, batch * NKVH);
      const int thr = nwarp * 64;
      auto launch = [&]() {
        if (nwarp == 4) probe<4><<<grid, thr>>>(dK, dV, dbt, nb, S, kvlen, sink);
        else if (nwarp == 8) probe<8><<<grid, thr>>>(dK, dV, dbt, nb, S, kvlen, sink);
        else probe<16><<<grid, thr>>>(dK, dV, dbt, nb, S, kvlen, sink);
      };
      for (int i = 0; i < 3; ++i) launch();
      cudaError_t e = cudaDeviceSynchronize();
      if (e != cudaSuccess) { printf("  NW=%2d S=%d ERR %s\n", nwarp, S, cudaGetErrorString(e)); continue; }
      cudaEvent_t a, b2;
      cudaEventCreate(&a); cudaEventCreate(&b2);
      cudaEventRecord(a);
      for (int i = 0; i < 10; ++i) launch();
      cudaEventRecord(b2); cudaEventSynchronize(b2);
      float ms; cudaEventElapsedTime(&ms, a, b2); ms /= 10;
      const int ctas = S * batch * NKVH;
      printf("NW=%2d S=%d             %-9d %8.3f %8.1f %8.0f%% %8.0f%%\n", nwarp, S, ctas, ms,
             bytes / (ms * 1e-3) / 1e9, 100 * bytes / (ms * 1e-3) / 1e9 / 819.2,
             100 * bytes / (ms * 1e-3) / 1e9 / 632.0);
    }
  }
  printf("\n-- probe2: one contiguous chunk per warp, W warp/CTA --\n");
  const int nunits = batch * NKVH;
  for (int W : {2, 4, 8, 16}) {
    for (int S : {8, 16, 32, 64}) {
      const int warps = nunits * S;
      const int ctas = (warps + W - 1) / W;
      auto l2 = [&]() {
        if (W == 2) probe2<2><<<ctas, W * 64>>>(dK, dV, dbt, nb, S, kvlen, nunits, sink);
        else if (W == 4) probe2<4><<<ctas, W * 64>>>(dK, dV, dbt, nb, S, kvlen, nunits, sink);
        else if (W == 8) probe2<8><<<ctas, W * 64>>>(dK, dV, dbt, nb, S, kvlen, nunits, sink);
        else probe2<16><<<ctas, W * 64>>>(dK, dV, dbt, nb, S, kvlen, nunits, sink);
      };
      for (int i = 0; i < 3; ++i) l2();
      cudaError_t e = cudaDeviceSynchronize();
      if (e != cudaSuccess) { printf("  W=%d S=%d ERR %s\n", W, S, cudaGetErrorString(e)); continue; }
      cudaEvent_t a, b3;
      cudaEventCreate(&a); cudaEventCreate(&b3);
      cudaEventRecord(a);
      for (int i = 0; i < 10; ++i) l2();
      cudaEventRecord(b3); cudaEventSynchronize(b3);
      float ms; cudaEventElapsedTime(&ms, a, b3); ms /= 10;
      printf("W=%2d S=%2d  warps=%4d ctas=%4d  %7.3f ms %7.1f GB/s  %5.0f%% of 819\n", W, S,
             warps, ctas, ms, bytes / (ms * 1e-3) / 1e9, 100 * bytes / (ms * 1e-3) / 1e9 / 819.2);
    }
  }

  printf("\n-- probe4: 8 KB smem/warp, occupancy capped at 16 warps/SM --\n");
  for (int W : {4, 8}) {
    for (int S : {8, 16, 32}) {
      const int warps = nunits * S;
      const int ctas = (warps + W - 1) / W;
      auto l4 = [&]() {
        if (W == 4) probe4<4><<<ctas, W * 64>>>(dK, dV, dbt, nb, S, kvlen, nunits, sink);
        else probe4<8><<<ctas, W * 64>>>(dK, dV, dbt, nb, S, kvlen, nunits, sink);
      };
      for (int i = 0; i < 3; ++i) l4();
      cudaError_t e = cudaDeviceSynchronize();
      if (e != cudaSuccess) { printf("  W=%d S=%d ERR %s\n", W, S, cudaGetErrorString(e)); continue; }
      cudaEvent_t a, b4;
      cudaEventCreate(&a); cudaEventCreate(&b4);
      cudaEventRecord(a);
      for (int i = 0; i < 10; ++i) l4();
      cudaEventRecord(b4); cudaEventSynchronize(b4);
      float ms; cudaEventElapsedTime(&ms, a, b4); ms /= 10;
      printf("W=%2d S=%2d  warps=%4d ctas=%4d  %7.3f ms %7.1f GB/s  %5.0f%% of 819\n", W, S,
             warps, ctas, ms, bytes / (ms * 1e-3) / 1e9, 100 * bytes / (ms * 1e-3) / 1e9 / 819.2);
    }
  }

  printf("\n-- probe5: probe4 + register prefetch of the next page --\n");
  for (int W : {4, 8}) {
    for (int S : {8, 16, 32}) {
      const int warps = nunits * S;
      const int ctas = (warps + W - 1) / W;
      auto l5 = [&]() {
        if (W == 4) probe5<4><<<ctas, W * 64>>>(dK, dV, dbt, nb, S, kvlen, nunits, sink);
        else probe5<8><<<ctas, W * 64>>>(dK, dV, dbt, nb, S, kvlen, nunits, sink);
      };
      for (int i = 0; i < 3; ++i) l5();
      cudaError_t e = cudaDeviceSynchronize();
      if (e != cudaSuccess) { printf("  W=%d S=%d ERR %s\n", W, S, cudaGetErrorString(e)); continue; }
      cudaEvent_t a, b5;
      cudaEventCreate(&a); cudaEventCreate(&b5);
      cudaEventRecord(a);
      for (int i = 0; i < 10; ++i) l5();
      cudaEventRecord(b5); cudaEventSynchronize(b5);
      float ms; cudaEventElapsedTime(&ms, a, b5); ms /= 10;
      printf("W=%2d S=%2d  warps=%4d ctas=%4d  %7.3f ms %7.1f GB/s  %5.0f%% of 819  %.2fx vendor\n",
             W, S, warps, ctas, ms, bytes / (ms * 1e-3) / 1e9,
             100 * bytes / (ms * 1e-3) / 1e9 / 819.2, 1.194 / ms);
    }
  }

  printf("\n-- probe6: 4 KB smem/warp (32 warps/SM), two-phase staging --\n");
  for (int W : {4, 8}) {
    for (int S : {8, 16, 32}) {
      const int warps = nunits * S;
      const int ctas = (warps + W - 1) / W;
      auto l6 = [&]() {
        if (W == 4) probe6<4><<<ctas, W * 64>>>(dK, dV, dbt, nb, S, kvlen, nunits, sink);
        else probe6<8><<<ctas, W * 64>>>(dK, dV, dbt, nb, S, kvlen, nunits, sink);
      };
      for (int i = 0; i < 3; ++i) l6();
      cudaError_t e = cudaDeviceSynchronize();
      if (e != cudaSuccess) { printf("  W=%d S=%d ERR %s\n", W, S, cudaGetErrorString(e)); continue; }
      cudaEvent_t a, b6;
      cudaEventCreate(&a); cudaEventCreate(&b6);
      cudaEventRecord(a);
      for (int i = 0; i < 10; ++i) l6();
      cudaEventRecord(b6); cudaEventSynchronize(b6);
      float ms; cudaEventElapsedTime(&ms, a, b6); ms /= 10;
      printf("W=%2d S=%2d  warps=%4d ctas=%4d  %7.3f ms %7.1f GB/s  %5.0f%% of 819  %.2fx vendor\n",
             W, S, warps, ctas, ms, bytes / (ms * 1e-3) / 1e9,
             100 * bytes / (ms * 1e-3) / 1e9 / 819.2, 1.194 / ms);
    }
  }

  printf("\nvendor kernel, same shape and bytes: 1.194 ms / 421 GB/s (60 work units)\n");
  return 0;
}
