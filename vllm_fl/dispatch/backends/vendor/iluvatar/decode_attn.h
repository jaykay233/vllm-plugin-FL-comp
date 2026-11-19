// The decode attention kernel itself, shared by the tuning harness (fused_decode.cu)
// and the vLLM drop-in (vllm_fl/.../native_decode_attn.cu).  One copy on purpose: the
// harness is where it gets tuned and the plugin is where it ships, and two copies of a
// kernel this size drift apart within a week.
//
// Everything here is device code plus the constants the launch geometry depends on.
// The host side that allocates the split partials and picks S lives with each caller.
#ifndef DECODE_ATTN_IMPL_H
#define DECODE_ATTN_IMPL_H

#include <cuda_runtime.h>
#include <cuda_bf16.h>

typedef __ix_bfloat16 bf_t;
typedef __bf16 v4y __attribute__((vector_size(8)));
// 8 bytes = four bf16, the unit every stage/load here moves.
typedef __bf16 v8h __attribute__((vector_size(8)));
typedef float v4f __attribute__((vector_size(16)));

#define D 128
#define NCH (D / 16)
#define PAGE 16
#define NWARP 8
#define NTHREAD (NWARP * 64)
#define TILE (PAGE * D)  // 2048 bf16 = 4 KB: one page of one kv head

// How many of the page's eight V vectors are in flight at once during the PV staging.
// Each one costs two registers per lane.
#ifndef DEC_VWIN
#define DEC_VWIN 4
#endif

__device__ __forceinline__ v4y pack4(const bf_t *p) {
  v4y r;
  __builtin_memcpy(&r, p, 8);
  return r;
}

// Shared memory element index for the K tile: a K fragment for (channel ch, read lane
// L) holds dims 16ch + 2*(L/16) + {0,1,8,9} at position L%16, and those four land in
// four consecutive slots.
//
// DEC_SWZ XORs the low four bits of the slot with a function of (ch, dim/pos group).
// Without it the store pattern is degenerate: the slot's 8-byte stride means a whole
// channel block spans exactly 512 bytes, so all eight channels land on the same bank,
// and one store instruction -- which varies only channel and dim group -- hits two
// banks with 64 lanes.  The swizzle spreads it over sixteen slots x two halves, i.e.
// the two phases a 64-lane store needs anyway.  Both sides use the same helper, so the
// layout stays a permutation and the read side of the pair is unchanged in spirit.
#ifndef DEC_SWZ
#define DEC_SWZ 1
#endif
#if DEC_SWZ
__device__ __forceinline__ int swzm(int ch, int g) { return ((g >> 1) << 3) | (ch & 7); }
#else
__device__ __forceinline__ int swzm(int ch, int g) { (void)ch; (void)g; return 0; }
#endif

__device__ __forceinline__ int kidx(int ch, int pos, int dlin) {
  const int g = (dlin >> 1) & 3;
  return (ch * 64 + 16 * g + ((pos ^ swzm(ch, g)) & 15)) * 4 + 2 * (dlin >> 3) + (dlin & 1);
}

// Same idea for V, but a V fragment holds four positions at one dim, so here the
// position bits are the ones folded into the group index.
__device__ __forceinline__ int vidx(int ch, int pos, int dlin) {
  const int g = (pos >> 1) & 3;
  return (ch * 64 + 16 * g + ((dlin ^ swzm(ch, g)) & 15)) * 4 + 2 * (pos >> 3) + (pos & 1);
}

// Where lane L reads its four-element fragment for channel ch: the logical slot is
// ch*64 + L, and the low four bits take the same swizzle the stores applied.
__device__ __forceinline__ int ridx(int ch, int lane) {
  const int g = lane >> 4;
  return (ch * 64 + 16 * g + ((lane & 15) ^ swzm(ch, g))) * 4;
}

// Warps per CTA.  DEC_MINB is the minimum block count per SM the register allocator
// has to leave room for; at 2 the kernel stays inside 128 vector registers, and going
// over that drops occupancy from 16 warps/SM to 8 and the kernel from 336 to 195 GB/s.
template <int W>
#ifndef DEC_MINB
#define DEC_MINB 0
#endif
#if DEC_MINB > 0
__global__ __launch_bounds__(W * 64, DEC_MINB) void decode_split(
#else
__global__ __launch_bounds__(W * 64) void decode_split(
#endif
    const bf_t *__restrict__ Q, const bf_t *__restrict__ K, const bf_t *__restrict__ V,
    float *__restrict__ Op, float *__restrict__ Mp, float *__restrict__ Lp,
    const int *__restrict__ block_table, const int *__restrict__ klen,
    const int *__restrict__ qrow, int nunit, int S, int maxblk, int NH, int NKVH,
    float log2scale) {
  __shared__ __align__(16) bf_t buf[W * TILE];

  const int tid = threadIdx.x;
  const int lane = tid & 63;
  const int wid = blockIdx.x * W + (tid >> 6);
  if (wid >= nunit * S) return;

  const int unit = wid % nunit;
  const int split = wid / nunit;
  const int si = unit / NKVH;
  const int kvh = unit % NKVH;
  const int NHK = NH / NKVH;
  const int kl = klen[si];
  const int npages = (kl + PAGE - 1) / PAGE;
  const int p0 = (int)((long)split * npages / S);
  const int p1 = (int)((long)(split + 1) * npages / S);
  const int *bt = block_table + (long)si * maxblk;

  bf_t *sh = buf + (tid >> 6) * TILE;

  // ---- Q fragments, resident for the whole loop ----
  v4y qf[NCH];
  {
    const bf_t *qr = Q + (size_t)qrow[si] * NH * D;
#pragma unroll
    for (int ch = 0; ch < NCH; ++ch) {
      bf_t h[4];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int hd = 2 * (lane / 16) + 8 * (j / 2) + (j % 2);
        h[j] = hd < NHK ? qr[(size_t)(kvh * NHK + hd) * D + 16 * ch + (lane % 16)]
                        : __float2bfloat16(0.f);
      }
      qf[ch] = pack4(h);
    }
  }

  v4f acc[NCH];
#pragma unroll
  for (int ch = 0; ch < NCH; ++ch) acc[ch] = (v4f){0.f, 0.f, 0.f, 0.f};
  float mrun[4], lrun[4];
#pragma unroll
  for (int q = 0; q < 4; ++q) { mrun[q] = -1e30f; lrun[q] = 0.f; }

  // ---- prologue: fetch the first page of this split before entering the loop ----
  v8h pk[8];
  if (p0 < p1) {
    const long b0 = ((long)bt[p0] * NKVH + kvh) * PAGE * D;
#pragma unroll
    for (int it = 0; it < 8; ++it) pk[it] = *(const v8h *)(K + b0 + (it * 64 + lane) * 4);
  } else {
#pragma unroll
    for (int it = 0; it < 8; ++it) pk[it] = (v8h){};
  }

  for (int p = p0; p < p1; ++p) {
    const long base = ((long)bt[p] * NKVH + kvh) * PAGE * D;
    const int pos0 = p * PAGE;

    // ---- K into shared memory, in fragment order ----
    // pk already holds this page: the fetch for page p was issued at the bottom of
    // iteration p-1 and has had the whole softmax and PV phase to land.  Four
    // consecutive dims at one position are two contiguous pairs in the K layout, so
    // each iteration is two 4-byte stores.
#pragma unroll
    for (int it = 0; it < 8; ++it) {
      const int flat = (it * 64 + lane) * 4;
      const int pos = flat / D;
      const int dim = flat % D;
      const int ch = dim / 16, dlin = dim % 16;
      unsigned lo, hi;
      __builtin_memcpy(&lo, &pk[it], 4);
      __builtin_memcpy(&hi, (const char *)&pk[it] + 4, 4);
      *(unsigned *)(sh + kidx(ch, pos, dlin)) = lo;
      *(unsigned *)(sh + kidx(ch, pos, dlin + 2)) = hi;
    }
    __syncwarp();

    // pk is dead once those stores are in, so the next page's fetch goes here: it then
    // overlaps the QK, softmax and PV work below instead of stalling in front of them.
    // Fetching it at the top of the loop instead left the warp with no request in
    // flight for the whole compute phase, and that is what held the kernel at 358 GB/s
    // while the bare memory probe on the same pattern reached 598.
    if (p + 1 < p1) {
      const long nb = ((long)bt[p + 1] * NKVH + kvh) * PAGE * D;
#pragma unroll
      for (int it = 0; it < 8; ++it) pk[it] = *(const v8h *)(K + nb + (it * 64 + lane) * 4);
    }

    // Same for this page's V: sixteen registers of fragments, fetched now so the loads
    // land during the QK and softmax phases.  Nothing here depends on the softmax.
    v4y vf[NCH];
#pragma unroll
    for (int ch = 0; ch < NCH; ++ch) {
      const int dim = 16 * ch + (lane & 15);
      const long row = base + dim;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int vp = 2 * (lane >> 4) + 8 * (j >> 1) + (j & 1);
        ((bf_t *)&vf[ch])[j] = V[row + (long)vp * D];
      }
    }

    // ---- S = Q @ K^T, eight MMAs over the 128 dims ----
    v4f sc = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int ch = 0; ch < NCH; ++ch) {
      const v4y kf = pack4(sh + ridx(ch, lane));
      sc = __ivcorex_matrix_mad_f32x4_bf16x4(qf[ch], kf, sc);
    }

    // ---- online softmax over this page's 16 positions ----
    // Lane L owns head L/16+4q at position L%16, so the max/sum reduction runs over
    // the sixteen lanes that share L/16, i.e. over positions.
    const int pos = pos0 + (lane % 16);
    float x[4], alpha[4];
#pragma unroll
    for (int q = 0; q < 4; ++q) x[q] = pos < kl ? sc[q] * log2scale : -1e30f;
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      float mx = x[q];
#pragma unroll
      for (int off = 1; off < 16; off <<= 1) mx = fmaxf(mx, __shfl_xor_sync(~0ULL, mx, off));
      const float mnew = fmaxf(mrun[q], mx);
      alpha[q] = exp2f(mrun[q] - mnew);
      mrun[q] = mnew;
    }
#pragma unroll
    for (int ch = 0; ch < NCH; ++ch)
#pragma unroll
      for (int q = 0; q < 4; ++q) acc[ch][q] *= alpha[q];

    // P values: exponentiate once, then reuse for both the l accumulation and the
    // fragment build.  Positions past the end of the sequence are already -1e30 in x,
    // so they exponentiate to zero and contribute nothing.
    float pe[4];
#pragma unroll
    for (int q = 0; q < 4; ++q) pe[q] = x[q] > -1e29f ? exp2f(x[q] - mrun[q]) : 0.f;

#pragma unroll
    for (int q = 0; q < 4; ++q) {
      float sum = pe[q];
#pragma unroll
      for (int off = 1; off < 16; off <<= 1) sum += __shfl_xor_sync(~0ULL, sum, off);
      lrun[q] = lrun[q] * alpha[q] + sum;
    }

    // ---- P into A-fragment order, by shuffle ----
    // Target lane L element j needs P[head 2*(L/16)+8*(j/2)+(j%2)][pos L%16], which
    // lives at lane 16*(head%4)+pos in register head/4.
    v4y pa;
    {
      const int gp = (lane >> 4) & 3;
      const int qsel = (lane >> 5) & 1;
      bf_t h[4];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int src = 16 * (2 * (gp & 1) + (j & 1)) + (lane & 15);
        const int q0s = 2 * (j >> 1);
        const float v0 = __shfl_sync(~0ULL, pe[q0s], src);
        const float v1 = __shfl_sync(~0ULL, pe[q0s + 1], src);
        h[j] = __float2bfloat16(qsel ? v1 : v0);
      }
      pa = pack4(h);
    }
    __syncwarp();  // K is dead; nothing else touches the buffer

    // ---- O += P @ V ----
    // A V fragment is four positions at one dim, so its elements live on four
    // different 256-byte rows of the page and cannot be one vector load.  Read
    // directly, though, one instruction covers sixteen contiguous dims on four rows:
    // 128 bytes, every byte of the four sectors used.  Staging the page through shared
    // memory instead needed thirty-two scattered two-byte stores per lane, and the
    // instrumented kernel showed that path running at 298 GB/s.
    //
    // The fetch itself was hoisted to just after the K stores: nothing in it depends on
    // the softmax, and leaving it here made every PV MMA wait on four fresh loads.  With
    // it hoisted, the measured cost of V's 251 MB fell from 0.55 ms to under the noise.
#pragma unroll
    for (int ch = 0; ch < NCH; ++ch) {
      acc[ch] = __ivcorex_matrix_mad_f32x4_bf16x4(pa, vf[ch], acc[ch]);
    }
  }

  // ---- partials: one (m, l, O) triple per head per split ----
  const int pbase = (unit * S + split) * NHK;
#pragma unroll
  for (int q = 0; q < 4; ++q) {
    const int hd = lane / 16 + 4 * q;
    if (hd >= NHK) continue;
    Mp[pbase + hd] = mrun[q];
    Lp[pbase + hd] = lrun[q];
#pragma unroll
    for (int ch = 0; ch < NCH; ++ch)
      Op[(size_t)(pbase + hd) * D + 16 * ch + (lane % 16)] = acc[ch][q];
  }
}


// The split partials are float; the caller's output may not be.
__device__ __forceinline__ void store_out(float *p, float v) { *p = v; }
__device__ __forceinline__ void store_out(bf_t *p, float v) { *p = __float2bfloat16(v); }

// Combine the S splits of each (unit, head).  Cheap enough to ignore: the partials are
// S x smaller than the KV cache that produced them (8 x 8 x 30 x 2 x 128 floats = 2 MB).
template <typename OT>
__global__ void decode_merge(const float *__restrict__ Op, const float *__restrict__ Mp,
                             const float *__restrict__ Lp, OT *__restrict__ O,
                             const int *__restrict__ qrow, int S, int NH, int NKVH) {
  const int unit = blockIdx.x;
  const int si = unit / NKVH, kvh = unit % NKVH;
  const int NHK = NH / NKVH;
  const int tid = threadIdx.x;

  for (int h = 0; h < NHK; ++h) {
    const int b = (unit * S) * NHK + h;
    float M = -1e30f;
    for (int s = 0; s < S; ++s) M = fmaxf(M, Mp[b + s * NHK]);
    float w[64];
    float den = 0.f;
    for (int s = 0; s < S; ++s) {
      w[s] = exp2f(Mp[b + s * NHK] - M);
      den += w[s] * Lp[b + s * NHK];
    }
    const float inv = den > 0.f ? 1.f / den : 0.f;
    for (int d = tid; d < D; d += blockDim.x) {
      float a = 0.f;
      for (int s = 0; s < S; ++s) a += w[s] * Op[(size_t)(b + s * NHK) * D + d];
      store_out(O + ((size_t)qrow[si] * NH + kvh * NHK + h) * D + d, a * inv);
    }
  }
}

// ---------------------------------------------------------------------------

#endif  // DECODE_ATTN_IMPL_H
