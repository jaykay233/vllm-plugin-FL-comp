// Paged, varlen attention: the path serving actually takes.
//
// The dense kernel in fused_wide.cu takes K/V as a contiguous [token][kv_head][dim]
// array.  Serving does not: the KV cache is paged, `block_table` maps each
// sequence's logical page to a physical one, and a step's batch is varlen -- the
// sequences in it have different query lengths and different KV lengths, because
// chunked prefill puts a 1-token decode next to a 4096-token prefill chunk.
//
// The vendor kernel this replaces has the signature
//   flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_q, max_k,
//                          causal=..., softmax_scale=..., block_table=...)
//
// Its paged cache layout had to be determined by experiment, because it is not
// what a reader of the config would guess.  Given identical data, vendor paged vs
// vendor dense gives
//
//     K/V as [blocks, PAGE, n_kv, D]   maxdiff vs dense 1.42e+00   wrong
//     K as [blocks, n_kv, D, PAGE]     1.53e+00                   wrong
//     K and V as [blocks, n_kv, PAGE, D]   0.00e+00               exact
//
// so this hardware's paged FA wants HND, not the BNDH the plugin config and the
// task description mention.  Both layouts are supported here (HND as a template
// parameter) so the kernel works whichever the caller holds.
//
// Also note the vendor's paged path rejects n_kv_heads < n_heads outright
// (CUINFER_STATUS_BAD_PARAM at ixinfer_flash_attn_unpad.cpp:1080); only a cache
// already expanded to 16 heads is accepted.  This kernel takes the real GQA cache
// with 2 kv heads, which the vendor path cannot do at all.
//
// Two properties of this shape make the paging almost free:
//
//   * PAGE (16) equals the kernel's tile width BNT, so a tile never straddles a
//     page.  The physical address of a tile is resolved once from block_table
//     instead of advancing a pointer across the KV loop.
//   * block_table is read once per tile, from a small int array that stays in L1.
//
// Query tiling is per sequence: each CTA owns BM consecutive query rows of one
// sequence, so the causal diagonal and the KV bound are per sequence rather than
// per batch.  A sequence shorter than BM wastes the tail of its last block; for
// prefill that is one block out of many, and for a decode row (q_len == 1) it is
// the whole block -- correct, but the reason a mixed step is not yet competitive.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

typedef __ix_bfloat16 bf_t;
typedef __bf16 v4y __attribute__((vector_size(8)));
typedef float v4f __attribute__((vector_size(16)));

// With GQA 16/2 two adjacent query heads read the same K/V page, so one CTA serving
// both leaves the MMA count untouched and halves the staged bytes per unit of work.
// The row warps are replicated per head, so per-thread register pressure and the inner
// loop are unchanged.  Measured +14% at every shape.
#ifndef HSPAN
#define HSPAN 2
#endif
#define NWARP_ROW 8
#define NWARP (NWARP_ROW * HSPAN)
#define NTHREAD (NWARP * 64)
#define D 128
#define NCH (D / 16)
#define BNT 16
#define TF (NCH * 64 * 4)
#define PAGE 16

__device__ __forceinline__ v4y pack4(const bf_t *h) {
  v4y r;
  __builtin_memcpy(&r, h, 8);
  return r;
}

// qblk_seq[i] / qblk_row[i] give the sequence and the first local query row of
// global query block i, precomputed on the host because the block count per
// sequence is data dependent and searching it on device would serialise.
#ifndef PG_MINB
#define PG_MINB 0
#endif

template <int CAUSAL, int NT, int RB, int HND>
#if PG_MINB > 0
__global__ __launch_bounds__(NTHREAD, PG_MINB) void fused_paged(
#else
__global__ __launch_bounds__(NTHREAD) void fused_paged(
#endif

    const bf_t *Q, const bf_t *K, const bf_t *V, float *O, const int *cu_q, const int *cu_kv,
    const int *block_table, const int *qblk_seq, const int *qblk_row, int maxblk, int NH, int NKVH,
    float log2scale) {
  const int BM = NWARP_ROW * 16 * RB;
  __shared__ bf_t Ksw[NT * TF];
  __shared__ bf_t Vsw[NT * TF];

  const int tid = threadIdx.x;
  const int lane = tid & 63;
  const int warp = tid >> 6;
  const int head = blockIdx.y * HSPAN + (warp / NWARP_ROW);
  const int wrow = warp % NWARP_ROW;  // row warp inside its head
  // every head in this CTA must share one kv head, or the staging is not shared
  const int kvhead = NKVH == NH ? head : (blockIdx.y * HSPAN) * NKVH / NH;
  // HND: [blocks, n_kv, PAGE, D] -> page stride NKVH*PAGE*D, head stride PAGE*D
  // BNDH:[blocks, PAGE, n_kv, D]  -> page stride PAGE*NKVH*D, head stride D
  const int pstride = HND ? D : NKVH * D;
  const int hstride = HND ? PAGE * D : D;
  const long pagestride = HND ? (long)NKVH * PAGE * D : (long)PAGE * NKVH * D;

  // Heaviest block first.  A query block's KV bound grows with its row offset, so the
  // last block of a sequence does up to BM/16 times the tiles of the first.  Launching
  // them in that order (grid order is otherwise monotone in the row offset) is the
  // classic longest-first schedule and it is free: the mapping is still a bijection.
#ifndef PR_ORD
#define PR_ORD 1
#endif
  const int qb = PR_ORD ? (gridDim.x - 1 - blockIdx.x) : blockIdx.x;
  const int s = qblk_seq[qb];
  const int q0 = cu_q[s], q1 = cu_q[s + 1], qlen = q1 - q0;
  const int klen = cu_kv[s + 1] - cu_kv[s];
  const int rowbase = qblk_row[qb];  // local row of this CTA's first query
  const int nrow = min(16 * NWARP * RB, qlen - rowbase);  // rows left in this sequence
  const int *bt = block_table + (long)s * maxblk;

  // ---- Q fragments, resident for the whole KV loop ----
  v4y qf[RB][NCH];
#pragma unroll
  for (int rb = 0; rb < RB; ++rb)
#pragma unroll
    for (int ch = 0; ch < NCH; ++ch) {
      bf_t h[4];
      const int lr = rowbase + wrow * 16 * RB + rb * 16 + 2 * (lane / 16);
      const int row = q0 + (lr < qlen ? lr : qlen - 1);  // clamp: tail rows are masked at output
      const int col = 16 * ch + (lane % 16);
#pragma unroll
      for (int j = 0; j < 4; ++j)
        h[j] = Q[(size_t)(row + 8 * (j / 2) + (j % 2)) * NH * D + (size_t)head * D + col];
      qf[rb][ch] = pack4(h);
    }

  v4f acc[RB][NCH];
#pragma unroll
  for (int rb = 0; rb < RB; ++rb)
#pragma unroll
    for (int ch = 0; ch < NCH; ++ch) acc[rb][ch] = (v4f){0.f, 0.f, 0.f, 0.f};
  float mrun[RB][4], lrun[RB][4];
#pragma unroll
  for (int rb = 0; rb < RB; ++rb)
#pragma unroll
    for (int q = 0; q < 4; ++q) { mrun[rb][q] = -1e30f; lrun[rb][q] = 0.f; }

  const int st = tid & 511;  // the copy pattern is 512 wide; head halves split the tiles
  const int sa = st & 3, sc16 = (st >> 2) & 15, sch = st >> 6;
  const int t0 = (warp / NWARP_ROW) * (NT / HSPAN);
  const int doff = kvhead * hstride + sc16 * pstride + 16 * sch + 2 * sa;  // K: pos fixed, dims 2a..
  const int voff = kvhead * hstride + 16 * sch + sc16;                     // V: dim fixed, pos varies

  // Bottom-right causal: within the sequence, row r sees keys 0 .. (klen-qlen)+r.
  const int causal_off = klen - qlen;
  const int lastrow = rowbase + BM - 1;
  const int klast = CAUSAL ? (causal_off + lastrow < klen - 1 ? causal_off + lastrow : klen - 1)
                           : klen - 1;
  const int nblk = (klast + 1 + BNT * NT - 1) / (BNT * NT);

  for (int blk = 0; blk < nblk; ++blk) {
    // ---- stage NT tiles; each tile sits inside one physical page ----
    bf_t kv2[NT][4], vv2[NT][4];
#pragma unroll
    for (int i = 0; i < NT / HSPAN; ++i) {
      const int t = t0 + i;
      const long p0 = (long)blk * NT + t;  // page index in this sequence
      const long ph = (long)bt[p0] * pagestride;  // page base, since BNT == PAGE
      const bf_t *pK = K + ph + doff;
      const bf_t *pV = V + ph + voff;
      unsigned int a0, a1;
      __builtin_memcpy(&a0, &pK[0], 4);
      __builtin_memcpy(&a1, &pK[8], 4);
      __builtin_memcpy(&kv2[t][0], &a0, 2);
      __builtin_memcpy(&kv2[t][1], (const char *)&a0 + 2, 2);
      __builtin_memcpy(&kv2[t][2], &a1, 2);
      __builtin_memcpy(&kv2[t][3], (const char *)&a1 + 2, 2);
      vv2[t][0] = pV[2 * sa * pstride];
      vv2[t][1] = pV[(2 * sa + 1) * pstride];
      vv2[t][2] = pV[(8 + 2 * sa) * pstride];
      vv2[t][3] = pV[(9 + 2 * sa) * pstride];
    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < NT / HSPAN; ++i) {
      const int t = t0 + i;
      const int o = ((sch * 64) + 16 * sa + sc16) * 4;
      __builtin_memcpy(&Ksw[t * TF + o], kv2[t], 8);
      __builtin_memcpy(&Vsw[t * TF + o], vv2[t], 8);
    }
    __syncthreads();

    // ---- S = Q @ K^T: one K fragment feeds RB MMAs ----
    v4f s[NT][RB];
#pragma unroll
    for (int t = 0; t < NT; ++t) {
      v4y kf[NCH];
#pragma unroll
      for (int ch = 0; ch < NCH; ++ch) kf[ch] = pack4(&Ksw[t * TF + ((ch * 64) + lane) * 4]);
      v4f a[RB];
#pragma unroll
      for (int rb = 0; rb < RB; ++rb) a[rb] = (v4f){0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int ch = 0; ch < NCH; ++ch)
#pragma unroll
        for (int rb = 0; rb < RB; ++rb)
          a[rb] = __ivcorex_matrix_mad_f32x4_bf16x4(qf[rb][ch], kf[ch], a[rb]);
#pragma unroll
      for (int rb = 0; rb < RB; ++rb) s[t][rb] = a[rb];
    }

    const int n0 = blk * (BNT * NT);
    float x[NT][RB][4], alpha[RB][4], pv[NT][RB][4];

    // Causal bounds, hoisted out of the tile loop.  The old inner test was three
    // comparisons per element -- col > causal_off+row, col >= klen, row >= qlen.  The
    // first two fold into one bound known before the loop, and the third becomes a
    // bound of -1.  bndmin is that bound for the top row of a lane group, which makes
    // it uniform across the warp, so a round where every row is inside it skips
    // masking altogether: only the diagonal round pays.
    int bnd[RB][4], bndmin[RB][4];
#pragma unroll
    for (int rb = 0; rb < RB; ++rb)
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const int rtop = wrow * 16 * RB + rb * 16 + 4 * q;
        const int rhi = CAUSAL ? causal_off + rowbase + rtop : klen - 1;
        bndmin[rb][q] = rtop + rowbase >= qlen ? -1 : (rhi < klen - 1 ? rhi : klen - 1);
        const int rlane = rtop + (lane / 16);
        const int rlo = CAUSAL ? causal_off + rowbase + rlane : klen - 1;
        bnd[rb][q] = rlane + rowbase >= qlen ? -1 : (rlo < klen - 1 ? rlo : klen - 1);
      }
    bool needmask = false;
#pragma unroll
    for (int rb = 0; rb < RB; ++rb)
#pragma unroll
      for (int q = 0; q < 4; ++q) needmask |= (n0 + BNT * NT - 1 > bndmin[rb][q]);

#pragma unroll
    for (int rb = 0; rb < RB; ++rb) {
      if (needmask) {
#pragma unroll
        for (int t = 0; t < NT; ++t)
#pragma unroll
          for (int q = 0; q < 4; ++q) {
            const int col = n0 + t * BNT + (lane % 16);
            float v = s[t][rb][q] * log2scale;
            if (col > bnd[rb][q]) v = -1e30f;
            x[t][rb][q] = v;
          }
      } else {
#pragma unroll
        for (int t = 0; t < NT; ++t)
#pragma unroll
          for (int q = 0; q < 4; ++q) x[t][rb][q] = s[t][rb][q] * log2scale;
      }
      bool moved = false;
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        float mx = x[0][rb][q];
#pragma unroll
        for (int t = 1; t < NT; ++t) mx = fmaxf(mx, x[t][rb][q]);
#pragma unroll
        for (int off = 1; off < 16; off <<= 1) mx = fmaxf(mx, __shfl_xor_sync(~0ULL, mx, off));
        const float mnew = fmaxf(mrun[rb][q], mx);
        alpha[rb][q] = exp2f(mrun[rb][q] - mnew);
        mrun[rb][q] = mnew;
        moved |= (alpha[rb][q] != 1.f);
      }
      // Rescaling the accumulator costs 32 multiplies a round, and the running max
      // stops moving after the first few rounds, where alpha is exactly 1.  alpha is
      // reduced across the lane group, so this branch is uniform within the warp.
      if (moved) {
#pragma unroll
        for (int ch = 0; ch < NCH; ++ch)
#pragma unroll
          for (int q = 0; q < 4; ++q) acc[rb][ch][q] *= alpha[rb][q];
      }
    }

    // ---- P to A-fragment order by shuffle, no smem round trip ----
    v4y pa[NT][RB];
#pragma unroll
    for (int t = 0; t < NT; ++t)
#pragma unroll
      for (int rb = 0; rb < RB; ++rb) {
#pragma unroll
        for (int q = 0; q < 4; ++q) pv[t][rb][q] = exp2f(x[t][rb][q] - mrun[rb][q]);
        const int gp = (lane >> 4) & 3;
        const int qsel = (lane >> 5) & 1;
        bf_t h[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const int src = 16 * (2 * (gp & 1) + (j & 1)) + (lane & 15);
          const int q0s = 2 * (j >> 1);
          const float v0 = __shfl_sync(~0ULL, pv[t][rb][q0s], src);
          const float v1 = __shfl_sync(~0ULL, pv[t][rb][q0s + 1], src);
          h[j] = __float2bfloat16(qsel ? v1 : v0);
        }
        pa[t][rb] = pack4(h);
      }

#pragma unroll
    for (int rb = 0; rb < RB; ++rb)
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        float sum = pv[0][rb][q];
#pragma unroll
        for (int t = 1; t < NT; ++t) sum += pv[t][rb][q];
#pragma unroll
        for (int off = 1; off < 16; off <<= 1) sum += __shfl_xor_sync(~0ULL, sum, off);
        lrun[rb][q] = lrun[rb][q] * alpha[rb][q] + sum;
      }

    // ---- O += P @ V: one V fragment feeds RB MMAs ----
#pragma unroll
    for (int t = 0; t < NT; ++t) {
      v4y vf[NCH];
#pragma unroll
      for (int ch = 0; ch < NCH; ++ch) vf[ch] = pack4(&Vsw[t * TF + ((ch * 64) + lane) * 4]);
#pragma unroll
      for (int ch = 0; ch < NCH; ++ch)
#pragma unroll
        for (int rb = 0; rb < RB; ++rb)
          acc[rb][ch] = __ivcorex_matrix_mad_f32x4_bf16x4(pa[t][rb], vf[ch], acc[rb][ch]);
    }
  }

  // ---- epilogue; rows past qlen are dropped ----
#pragma unroll
  for (int rb = 0; rb < RB; ++rb)
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const float iv = lrun[rb][q] > 0.f ? 1.f / lrun[rb][q] : 0.f;
      const int lr = rowbase + wrow * 16 * RB + rb * 16 + (lane / 16) + 4 * q;
      if (lr >= qlen) continue;
#pragma unroll
      for (int ch = 0; ch < NCH; ++ch)
        O[(size_t)(q0 + lr) * NH * D + (size_t)head * D + 16 * ch + (lane % 16)] =
            acc[rb][ch][q] * iv;
    }
}

static const char *arg(int argc, char **argv, int i, const char *dflt) {
  return i < argc ? argv[i] : dflt;
}

using kern_p = void (*)(const bf_t *, const bf_t *, const bf_t *, float *, const int *, const int *,
                        const int *, const int *, const int *, int, int, int, float);

template <int CAUSAL, int RB, int HND>
static kern_p pick_nt(int nt) {
  switch (nt) {
    case 2: return (kern_p)fused_paged<CAUSAL, 2, RB, HND>;
    case 4: return (kern_p)fused_paged<CAUSAL, 4, RB, HND>;
    default: return (kern_p)fused_paged<CAUSAL, 8, RB, HND>;
  }
}

template <int CAUSAL, int RB>
static kern_p pick_lay(int nt, int hnd) {
  return hnd ? pick_nt<CAUSAL, RB, 1>(nt) : pick_nt<CAUSAL, RB, 0>(nt);
}

template <int CAUSAL>
static kern_p pick_rb(int nt, int rb, int hnd) {
  switch (rb) {
    case 2: return pick_lay<CAUSAL, 2>(nt, hnd);
    case 3: return pick_lay<CAUSAL, 3>(nt, hnd);
    default: return pick_lay<CAUSAL, 1>(nt, hnd);
  }
}

int main(int argc, char **argv) {
  // Dense-vs-paged comparison mode for the numeric gate: argv[1] = case path.
  const char *casep = argc > 1 ? argv[1] : nullptr;
  if (!casep) {
    printf("usage: fused_paged <case.txt> [nt] [rb]\n"
           "case.txt: line1 = nseq NH NKVH maxblk; then per seq: qlen klen; then Q,K,V,bt,cu_q,cu_kv\n");
    return 1;
  }
  // Every head in a CTA stages NT/HSPAN tiles, so NT must be a multiple of HSPAN.
  // With NT=1 and HSPAN=2 the intra-round loop is empty and nothing is staged at all.
  const int nt_arg = atoi(arg(argc, argv, 2, "4"));
  const int nt_sel = (nt_arg % HSPAN == 0 && nt_arg > 0) ? nt_arg : HSPAN * 2;
  // pick_rb folds anything outside {2,3} to RB=1, so the host-side row block must come
  // from the same mapping.  Passing rb=4 used to time a BM=512 grid against a BM=128
  // kernel: it computed a quarter of the rows and reported four times the throughput.
  const int rb_arg = atoi(arg(argc, argv, 3, "3"));
  const int rb_sel = (rb_arg == 2 || rb_arg == 3) ? rb_arg : 1;
  const int hnd = atoi(arg(argc, argv, 4, "1"));

  const int BM = NWARP_ROW * 16 * rb_sel;

  FILE *cf = fopen(casep, "r");
  if (!cf) { printf("cannot open case\n"); return 1; }
  int nseq, NH, NKVH, maxblk;
  if (fscanf(cf, "%d %d %d %d", &nseq, &NH, &NKVH, &maxblk) != 4) { printf("bad header\n"); return 1; }
  int *hq = (int *)malloc((nseq + 1) * 4), *hk = (int *)malloc((nseq + 1) * 4);
  hq[0] = hk[0] = 0;
  for (int s = 0; s < nseq; ++s) {
    int ql, kl;
    if (fscanf(cf, "%d %d", &ql, &kl) != 2) { printf("bad seq\n"); return 1; }
    hq[s + 1] = hq[s] + ql;
    hk[s + 1] = hk[s] + kl;
  }
  char qp[512], kp[512], vp[512], bp[512], op[512];
  if (fscanf(cf, "%511s %511s %511s %511s %511s", qp, kp, vp, bp, op) != 5) {
    printf("bad paths\n");
    return 1;
  }
  fclose(cf);

  const int totq = hq[nseq], totk = hk[nseq];
  const size_t qsz = (size_t)totq * NH * D;
  bf_t *hQ = (bf_t *)malloc(qsz * 2);
  float *hO = (float *)malloc(qsz * 4);
  int *bt = (int *)malloc((size_t)nseq * maxblk * 4);
  auto rd = [&](const char *p, void *dst, size_t n) {
    FILE *f = fopen(p, "rb");
    if (!f) { printf("cannot open %s\n", p); exit(1); }
    if (fread(dst, 1, n, f) != n) { printf("short read %s\n", p); exit(1); }
    fclose(f);
  };
  rd(qp, hQ, qsz * 2);
  rd(bp, bt, (size_t)nseq * maxblk * 4);

  int npages = 0;
  for (int i = 0; i < nseq * maxblk; ++i) if (bt[i] + 1 > npages) npages = bt[i] + 1;
  const size_t ksz = (size_t)npages * PAGE * NKVH * D;  // either layout, same count
  bf_t *hK = (bf_t *)malloc(ksz * 2), *hV = (bf_t *)malloc(ksz * 2);
  rd(kp, hK, ksz * 2);
  rd(vp, hV, ksz * 2);

  // host-side query block -> (sequence, local row) map
  int nqblk = 0;
  for (int s = 0; s < nseq; ++s) nqblk += (hq[s + 1] - hq[s] + BM - 1) / BM;
  int *qb_s = (int *)malloc(nqblk * 4), *qb_r = (int *)malloc(nqblk * 4);
  int bi = 0;
  for (int s = 0; s < nseq; ++s) {
    const int ql = hq[s + 1] - hq[s];
    for (int r = 0; r < ql; r += BM) { qb_s[bi] = s; qb_r[bi] = r; ++bi; }
  }

  bf_t *dQ, *dK, *dV;
  float *dO;
  int *dbt, *dcq, *dck, *dqs, *dqr;
  cudaMalloc(&dQ, qsz * 2); cudaMalloc(&dK, ksz * 2); cudaMalloc(&dV, ksz * 2);
  cudaMalloc(&dO, qsz * 4);
  cudaMalloc(&dbt, (size_t)nseq * maxblk * 4); cudaMalloc(&dcq, (nseq + 1) * 4);
  cudaMalloc(&dck, (nseq + 1) * 4); cudaMalloc(&dqs, nqblk * 4); cudaMalloc(&dqr, nqblk * 4);
  cudaMemcpy(dQ, hQ, qsz * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(dK, hK, ksz * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(dV, hV, ksz * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(dO, hO, qsz * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(dbt, bt, (size_t)nseq * maxblk * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(dcq, hq, (nseq + 1) * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(dck, hk, (nseq + 1) * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(dqs, qb_s, nqblk * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(dqr, qb_r, nqblk * 4, cudaMemcpyHostToDevice);

  const float log2scale = (float)(1.4426950408889634 / std::sqrt((double)D));
  const dim3 grid(nqblk, (NH + HSPAN - 1) / HSPAN);
  for (int causal = 0; causal < 2; ++causal) {
    kern_p k = causal ? pick_rb<1>(nt_sel, rb_sel, hnd) : pick_rb<0>(nt_sel, rb_sel, hnd);
    k<<<grid, NTHREAD>>>(dQ, dK, dV, dO, dcq, dck, dbt, dqs, dqr, maxblk, NH, NKVH, log2scale);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) { printf("ERR %s\n", cudaGetErrorString(e)); return 1; }
    // append rather than overwrite: both causal settings are compared against one file
    char path[600];
    snprintf(path, sizeof(path), "%s.%s", op, causal ? "causal" : "dense");
    cudaMemcpy(hO, dO, qsz * 4, cudaMemcpyDeviceToHost);
    FILE *f = fopen(path, "wb");
    if (fwrite(hO, 4, qsz, f) != qsz) { printf("short write\n"); return 1; }
    fclose(f);
    printf("wrote %s\n", path);
  }

  // informal timing, no reference
  if (nseq > 0) {
    double flops = 0;
    for (int b = 0; b < nqblk; ++b) {
      const int s = qb_s[b];
      const int ql = hq[s + 1] - hq[s], kl = hk[s + 1] - hk[s];
      const int last = qb_r[b] + BM - 1 < ql - 1 ? qb_r[b] + BM - 1 : ql - 1;
      const int kk = (kl - ql + last + 1) < kl ? (kl - ql + last + 1) : kl;
      flops += 4.0 * BM * (double)kk * D * NH;
    }
    kern_p k = pick_rb<1>(nt_sel, rb_sel, hnd);
    for (int i = 0; i < 3; ++i)
      k<<<grid, NTHREAD>>>(dQ, dK, dV, dO, dcq, dck, dbt, dqs, dqr, maxblk, NH, NKVH, log2scale);
    cudaEvent_t st, en;
    cudaEventCreate(&st); cudaEventCreate(&en);
    cudaEventRecord(st);
    for (int i = 0; i < 10; ++i)
      k<<<grid, NTHREAD>>>(dQ, dK, dV, dO, dcq, dck, dbt, dqs, dqr, maxblk, NH, NKVH, log2scale);
    cudaEventRecord(en); cudaEventSynchronize(en);
    float ms; cudaEventElapsedTime(&ms, st, en); ms /= 10;
    printf("paged causal NT=%d RB=%d: %.3f ms  %.1f TFLOPS\n", nt_sel, rb_sel, ms,
           flops / (ms * 1e-3) / 1e12);
  }
  return 0;
}
