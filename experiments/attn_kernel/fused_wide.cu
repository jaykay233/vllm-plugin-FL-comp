// Attention kernel with RB row-blocks per warp.
//
// Why this file exists.  The committed kernel runs at 16.6 TFLOPS and the
// measured ladder says the cost is not the arithmetic:
//
//     pure MMA                     0.94 ms   76.3 TF
//     + fragment reads             2.08 ms   34.0 TF
//     + staging                    3.87 ms   18.3 TF
//     + softmax (full kernel)      4.27 ms   16.6 TF
//
// So adding the shared-memory fragment reads alone costs 1.14 ms and lands at
// 34 TF, which matches a bandwidth argument: one 16x16x16 warp MMA (8192 flop)
// consumes a 512 B B-fragment, so at smem's ~90-128 B/cycle an SM can only be
// fed 0.18-0.25 MMA/cycle = 22-31 TF.  Measured 34 TF is that ceiling.
//
// The kernel at that point had one 16-row block per warp, so each B-fragment
// fed exactly one MMA -- the worst possible ratio.  RB row-blocks per warp reuse
// the same K and V fragment across RB MMAs, cutting bytes/MMA by RB:
//
//     RB=1  512 B/MMA  ceiling ~34 TF   (current kernel)
//     RB=2  256 B/MMA  ceiling ~60 TF
//     RB=4  128 B/MMA  ceiling ~90 TF
//
// The binding constraint moves to registers: each row block adds 8 Q fragments
// (16 regs) and 8 accumulators (32 regs), so RB=2 costs ~48 more registers than
// RB=1.  Whether occupancy survives that is an empirical question, which is the
// point of making RB a template parameter: RB=1 compiles to the committed
// kernel, so any difference at RB=1 is noise and the comparison is clean.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

typedef __ix_bfloat16 bf_t;
typedef __bf16 v4y __attribute__((vector_size(8)));
typedef float v4f __attribute__((vector_size(16)));

#define NWARP 8
#define NTHREAD (NWARP * 64)
#define D 128
#define NCH (D / 16)
#define BNT 16
#define TF (NCH * 64 * 4)

__device__ __forceinline__ v4y pack4(const bf_t *h) {
  v4y r;
  __builtin_memcpy(&r, h, 8);
  return r;
}

template <int CAUSAL, int NT, int RB, int NW = NWARP>
__global__ __launch_bounds__(NW * 64) void fused(const bf_t *Q, const bf_t *K, const bf_t *V,
                                                 float *O, int kv, int tok, int NH, int NKVH,
                                                 float log2scale) {
  const int BM = NW * 16 * RB;
  __shared__ bf_t Ksw[NT * TF];
  __shared__ bf_t Vsw[NT * TF];

  const int tid = threadIdx.x;
  const int lane = tid & 63;
  const int warp = tid >> 6;
  const int qbase = blockIdx.x * BM;
  const int head = blockIdx.y;
  const int kvhead = NKVH == NH ? head : head * NKVH / NH;
  const int ldKV = NKVH * D;

  // BM does not have to divide tok -- RB=3 gives BM=384, and 4096 is not a multiple
  // of it -- so the tail block reads clamped Q rows and drops them at the epilogue.
  // Without this the last block reads past Q and writes past O.
  const int nrow = tok - qbase;  // rows actually owned by this block

  // ---- Q fragments, resident for the whole KV loop ----
  v4y qf[RB][NCH];
#pragma unroll
  for (int rb = 0; rb < RB; ++rb)
#pragma unroll
    for (int ch = 0; ch < NCH; ++ch) {
      bf_t h[4];
      const int lr = warp * 16 * RB + rb * 16 + 2 * (lane / 16);
      const int row = qbase + (lr < nrow ? lr : nrow - 1);
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

  const int sa = tid & 3, sc16 = (tid >> 2) & 15, sch = tid >> 6;
  const int sld = ldKV;
  const bf_t *pK = K + (long)kvhead * D + (long)sc16 * sld + 16 * sch + 2 * sa;
  const bf_t *pV = V + (long)kvhead * D + 16 * sch + sc16;
  const int sstep = BNT * sld;

  // Bottom-right causal alignment: with a KV cache longer than the new tokens,
  // row r sees keys 0 .. (kv-tok)+r, so the last key this block touches comes from
  // the block's last row, not its first.
  const int causal_off = kv - tok;
  const int qlast = qbase + BM - 1;
  const int klast = CAUSAL ? (causal_off + qlast < kv - 1 ? causal_off + qlast : kv - 1)
                           : kv - 1;
  const int nblk = (klast + 1 + BNT * NT - 1) / (BNT * NT);

  for (int blk = 0; blk < nblk; ++blk) {
    // ---- stage NT tiles of K/V ----
#ifdef ABL_STAGE_OFF
    if (blk) { __syncthreads(); }
#endif
    bf_t kv2[NT][4], vv2[NT][4];
    const bf_t *qK = pK, *qV = pV;
#ifndef ABL_STAGE_OFF
#pragma unroll
    for (int t = 0; t < NT; ++t) {
      unsigned int p0, p1;
      __builtin_memcpy(&p0, &qK[0], 4);
      __builtin_memcpy(&p1, &qK[8], 4);
      __builtin_memcpy(&kv2[t][0], &p0, 2);
      __builtin_memcpy(&kv2[t][1], (const char *)&p0 + 2, 2);
      __builtin_memcpy(&kv2[t][2], &p1, 2);
      __builtin_memcpy(&kv2[t][3], (const char *)&p1 + 2, 2);
      vv2[t][0] = qV[2 * sa * sld];
      vv2[t][1] = qV[(2 * sa + 1) * sld];
      vv2[t][2] = qV[(8 + 2 * sa) * sld];
      vv2[t][3] = qV[(9 + 2 * sa) * sld];
      qK += sstep;
      qV += sstep;
    }
#endif
    __syncthreads();
#pragma unroll
    for (int t = 0; t < NT; ++t) {
      const int o = ((sch * 64) + 16 * sa + sc16) * 4;
      __builtin_memcpy(&Ksw[t * TF + o], kv2[t], 8);
      __builtin_memcpy(&Vsw[t * TF + o], vv2[t], 8);
    }
    __syncthreads();
    pK = qK;
    pV = qV;

    // ---- S = Q @ K^T: one K fragment load feeds RB MMAs ----
    v4f s[NT][RB];
#pragma unroll
    for (int t = 0; t < NT; ++t) {
      v4y kf[NCH];
#pragma unroll
#ifdef ABL_LDS_OFF
      for (int ch = 0; ch < NCH; ++ch) kf[ch] = qf[0][ch];
#else
#ifdef ABL_LDS_OFF
      for (int ch = 0; ch < NCH; ++ch) kf[ch] = qf[0][ch];
#else
      for (int ch = 0; ch < NCH; ++ch) kf[ch] = pack4(&Ksw[t * TF + ((ch * 64) + lane) * 4]);
#endif
#endif
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
#pragma unroll
    for (int rb = 0; rb < RB; ++rb) {
#pragma unroll
      for (int t = 0; t < NT; ++t)
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          const int col = n0 + t * BNT + (lane % 16);
          float v = s[t][rb][q] * log2scale;
          const int row = warp * 16 * RB + rb * 16 + (lane / 16) + 4 * q;
          if (CAUSAL && col > causal_off + qbase + row) v = -1e30f;
          x[t][rb][q] = v;
        }
      // row max: merge across tiles, then one 16-lane reduction.  A row lives in 16
      // consecutive lanes that share lane/16, so xor-shuffles over 1,2,4,8 cover it.
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        float mx = x[0][rb][q];
#pragma unroll
        for (int t = 1; t < NT; ++t) mx = fmaxf(mx, x[t][rb][q]);
#pragma unroll
        for (int off = 1; off < 16; off <<= 1) mx = fmaxf(mx, __shfl_xor_sync(~0ULL, mx, off));
        const float mnew = fmaxf(mrun[rb][q], mx);
        alpha[rb][q] = exp2f(mrun[rb][q] - mnew);  // x is log2-domain: 2^d, not e^d
        mrun[rb][q] = mnew;
      }
#pragma unroll
      for (int ch = 0; ch < NCH; ++ch)
#pragma unroll
        for (int q = 0; q < 4; ++q) acc[rb][ch][q] *= alpha[rb][q];
    }

    // ---- P to A-fragment order by shuffle ----
    //   accumulator (group g, reg q) -> row g + 4q
    //   A fragment  (group g', reg j) -> row 2g' + 8*(j>>1) + (j&1)
    // inverting gives g = 2*(g'&1) + (j&1) and q = 2*(j>>1) + (g'>>1), column
    // untouched, so two shuffles plus a select per register and no smem round trip.
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
          const int q0 = 2 * (j >> 1);
          const float v0 = __shfl_sync(~0ULL, pv[t][rb][q0], src);
          const float v1 = __shfl_sync(~0ULL, pv[t][rb][q0 + 1], src);
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

    // ---- O += P @ V: one V fragment load feeds RB MMAs ----
#pragma unroll
    for (int t = 0; t < NT; ++t) {
      v4y vf[NCH];
#pragma unroll
#ifdef ABL_LDS_OFF
      for (int ch = 0; ch < NCH; ++ch) vf[ch] = qf[0][ch];
#else
#ifdef ABL_LDS_OFF
      for (int ch = 0; ch < NCH; ++ch) vf[ch] = qf[0][ch];
#else
      for (int ch = 0; ch < NCH; ++ch) vf[ch] = pack4(&Vsw[t * TF + ((ch * 64) + lane) * 4]);
#endif
#endif
#pragma unroll
      for (int ch = 0; ch < NCH; ++ch)
#pragma unroll
        for (int rb = 0; rb < RB; ++rb)
          acc[rb][ch] = __ivcorex_matrix_mad_f32x4_bf16x4(pa[t][rb], vf[ch], acc[rb][ch]);
    }
  }

  // ---- epilogue ----
  float inv[RB][4];
#pragma unroll
  for (int rb = 0; rb < RB; ++rb)
#pragma unroll
    for (int q = 0; q < 4; ++q) inv[rb][q] = lrun[rb][q] > 0.f ? 1.f / lrun[rb][q] : 0.f;
#pragma unroll
  for (int rb = 0; rb < RB; ++rb)
#pragma unroll
    for (int ch = 0; ch < NCH; ++ch)
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const int row = warp * 16 * RB + rb * 16 + (lane / 16) + 4 * q;
        if (row >= nrow) continue;  // do not write past the end of O
        O[(size_t)(qbase + row) * NH * D + (size_t)head * D + 16 * ch + (lane % 16)] =
            acc[rb][ch][q] * inv[rb][q];
      }
}

// ---- host reference, double precision ----
static void ref_attn(const bf_t *hQ, const bf_t *hK, const bf_t *hV, float *hO, int tok, int kv,
                     int NH, int NKVH, int causal) {
  const double scale = 1.0 / std::sqrt((double)D);
  for (int t = 0; t < tok; ++t)
    for (int h = 0; h < NH; ++h) {
      const int kh = NKVH == NH ? h : h * NKVH / NH;
      const int lim = (kv - tok) + t + 1;
      const int nk = causal ? (lim < kv ? lim : kv) : kv;
      double mx = -1e30, s[16384];
      for (int j = 0; j < nk; ++j) {
        double d = 0;
        for (int x = 0; x < D; ++x) {
          unsigned short qa, kb;
          std::memcpy(&qa, &hQ[(size_t)t * NH * D + (size_t)h * D + x], 2);
          std::memcpy(&kb, &hK[(size_t)j * NKVH * D + (size_t)kh * D + x], 2);
          unsigned ua = (unsigned)qa << 16, ub = (unsigned)kb << 16;
          float fa, fb;
          std::memcpy(&fa, &ua, 4);
          std::memcpy(&fb, &ub, 4);
          d += (double)fa * fb;
        }
        s[j] = d * scale;
        if (s[j] > mx) mx = s[j];
      }
      double sum = 0;
      for (int j = 0; j < nk; ++j) { s[j] = std::exp(s[j] - mx); sum += s[j]; }
      for (int x = 0; x < D; ++x) {
        double o = 0;
        for (int j = 0; j < nk; ++j) {
          unsigned short vb;
          std::memcpy(&vb, &hV[(size_t)j * NKVH * D + (size_t)kh * D + x], 2);
          unsigned ub = (unsigned)vb << 16;
          float fb;
          std::memcpy(&fb, &ub, 4);
          o += s[j] / sum * (double)fb;
        }
        hO[(size_t)t * NH * D + (size_t)h * D + x] = (float)o;
      }
    }
}

static void set_bf(bf_t *p, float f) {
  unsigned u;
  std::memcpy(&u, &f, 4);
  u += 0x7fffu + ((u >> 16) & 1u);
  const unsigned short h = (unsigned short)(u >> 16);
  std::memcpy(p, &h, 2);
}

static const char *arg(int argc, char **argv, int i, const char *dflt) {
  return i < argc ? argv[i] : dflt;
}

using kern = void (*)(const bf_t *, const bf_t *, const bf_t *, float *, int, int, int, int, float);

template <int CAUSAL, int RB, int NW>
static kern pick_nt(int nt) {
  switch (nt) {
    case 1: return (kern)fused<CAUSAL, 1, RB, NW>;
    case 2: return (kern)fused<CAUSAL, 2, RB, NW>;
    case 4: return (kern)fused<CAUSAL, 4, RB, NW>;
    default: return (kern)fused<CAUSAL, 8, RB, NW>;
  }
}

int main(int argc, char **argv) {
  const int tok = atoi(arg(argc, argv, 1, "1024"));
  const int kv = atoi(arg(argc, argv, 2, "1024"));
  const bool want_ref = atoi(arg(argc, argv, 3, "1")) != 0;
  const int NH = 16;
  const int NKVH = atoi(arg(argc, argv, 4, "16")) > 0 ? atoi(arg(argc, argv, 4, "16")) : NH;
  const int nt_sel = atoi(arg(argc, argv, 5, "2"));
  const int rb_sel = atoi(arg(argc, argv, 6, "2"));

  const int nw_sel = atoi(arg(argc, argv, 7, "8"));
  auto pick = [&](int causal, int nt, int rb, int nw) -> kern {
    if (causal) {
      switch (rb * 100 + nw) {
        case 104: return pick_nt<1, 1, 4>(nt);
        case 108: return pick_nt<1, 1, 8>(nt);
        case 116: return pick_nt<1, 1, 16>(nt);
        case 204: return pick_nt<1, 2, 4>(nt);
        case 208: return pick_nt<1, 2, 8>(nt);
        case 216: return pick_nt<1, 2, 16>(nt);
        case 304: return pick_nt<1, 3, 4>(nt);
        case 308: return pick_nt<1, 3, 8>(nt);
        default: return pick_nt<1, 4, 8>(nt);
      }
    }
    switch (rb * 100 + nw) {
      case 104: return pick_nt<0, 1, 4>(nt);
      case 108: return pick_nt<0, 1, 8>(nt);
      case 116: return pick_nt<0, 1, 16>(nt);
      case 204: return pick_nt<0, 2, 4>(nt);
      case 208: return pick_nt<0, 2, 8>(nt);
      case 216: return pick_nt<0, 2, 16>(nt);
      case 304: return pick_nt<0, 3, 4>(nt);
      case 308: return pick_nt<0, 3, 8>(nt);
      default: return pick_nt<0, 4, 8>(nt);
    }
  };
  const int BM = nw_sel * 16 * rb_sel;
  const int NTH = nw_sel * 64;

  const size_t qsz = (size_t)tok * NH * D, ksz = (size_t)kv * NKVH * D;
  bf_t *hQ = (bf_t *)malloc(qsz * 2), *hK = (bf_t *)malloc(ksz * 2), *hV = (bf_t *)malloc(ksz * 2);
  float *hO = (float *)malloc(qsz * 4), *hRef = (float *)malloc(qsz * 4);
  srand(3);
  for (size_t i = 0; i < qsz; ++i) set_bf(&hQ[i], (rand() % 200 - 100) / 64.0f);
  for (size_t i = 0; i < ksz; ++i) {
    set_bf(&hK[i], (rand() % 200 - 100) / 64.0f);
    set_bf(&hV[i], (rand() % 200 - 100) / 64.0f);
  }

  // file mode: argv[8] is a path prefix; read <p>.Q/K/V.bin, write <p>.O.*.bin so
  // the identical tensors can be handed to the vendor kernel.
  const char *fpref = argc > 8 ? argv[8] : nullptr;
  if (fpref) {
    auto load = [&](const char *sfx, bf_t *dst, size_t n) {
      char path[512];
      snprintf(path, sizeof(path), "%s.%s", fpref, sfx);
      FILE *f = fopen(path, "rb");
      if (!f) { printf("cannot open %s\n", path); exit(1); }
      if (fread(dst, 2, n, f) != n) { printf("short read %s\n", path); exit(1); }
      fclose(f);
    };
    load("Q.bin", hQ, qsz);
    load("K.bin", hK, ksz);
    load("V.bin", hV, ksz);
  }

  bf_t *dQ, *dK, *dV;
  float *dO;
  cudaMalloc(&dQ, qsz * 2); cudaMalloc(&dK, ksz * 2);
  cudaMalloc(&dV, ksz * 2); cudaMalloc(&dO, qsz * 4);
  cudaMemcpy(dQ, hQ, qsz * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(dK, hK, ksz * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(dV, hV, ksz * 2, cudaMemcpyHostToDevice);

  const float log2scale = (float)(1.4426950408889634 / std::sqrt((double)D));
  const dim3 grid((tok + BM - 1) / BM, NH);

  auto row_err = [&](int *nbad, double *worst) {
    *worst = 0; *nbad = 0;
    for (int t = 0; t < tok; ++t)
      for (int h = 0; h < NH; ++h) {
        double dn = 0, rn = 0;
        for (int d = 0; d < D; ++d) {
          const size_t i = (size_t)t * NH * D + (size_t)h * D + d;
          const double diff = (double)hO[i] - hRef[i];
          dn += diff * diff;
          rn += (double)hRef[i] * hRef[i];
        }
        const double e = std::sqrt(dn) / (std::sqrt(rn) + 1e-12);
        if (e > *worst) *worst = e;
        if (e > 1e-2) ++*nbad;
      }
  };

  if (fpref) {
    for (int causal = 0; causal < 2; ++causal) {
      pick(causal, nt_sel, rb_sel, nw_sel)<<<grid, NTH>>>(dQ, dK, dV, dO, kv, tok, NH, NKVH, log2scale);
      if (cudaDeviceSynchronize() != cudaSuccess) { printf("ERR\n"); return 1; }
      cudaMemcpy(hO, dO, qsz * 4, cudaMemcpyDeviceToHost);
      char path[512];
      snprintf(path, sizeof(path), "%s.O.%s.bin", fpref, causal ? "causal" : "dense");
      FILE *f = fopen(path, "wb");
      if (fwrite(hO, 4, qsz, f) != qsz) { printf("short write\n"); return 1; }
      fclose(f);
      printf("wrote %s\n", path);
    }
    return 0;
  }

  for (int causal = 0; causal < 2; ++causal) {
    pick(causal, nt_sel, rb_sel, nw_sel)<<<grid, NTH>>>(dQ, dK, dV, dO, kv, tok, NH, NKVH, log2scale);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) { printf("ERR %s\n", cudaGetErrorString(e)); return 1; }
    if (!want_ref) continue;
    cudaMemcpy(hO, dO, qsz * 4, cudaMemcpyDeviceToHost);
    ref_attn(hQ, hK, hV, hRef, tok, kv, NH, NKVH, causal);
    int nbad = 0; double worst = 0;
    row_err(&nbad, &worst);
    printf("fused %-6s tok=%d kv=%d heads=%d kvh=%d NT=%d RB=%d: worst row err %.2e  bad %d/%d  %s\n",
           causal ? "causal" : "dense", tok, kv, NH, NKVH, nt_sel, rb_sel, worst, nbad, tok * NH,
           nbad == 0 ? "OK" : "FAIL");
  }

  auto timeit = [&](const char *tag, kern k, double flops) {
    for (int i = 0; i < 3; ++i) k<<<grid, NTH>>>(dQ, dK, dV, dO, kv, tok, NH, NKVH, log2scale);
    cudaEvent_t st, en;
    cudaEventCreate(&st); cudaEventCreate(&en);
    cudaEventRecord(st);
    for (int i = 0; i < 10; ++i) k<<<grid, NTH>>>(dQ, dK, dV, dO, kv, tok, NH, NKVH, log2scale);
    cudaEventRecord(en); cudaEventSynchronize(en);
    float ms; cudaEventElapsedTime(&ms, st, en); ms /= 10;
    printf("  %-24s %7.3f ms   %6.1f TFLOPS (%5.1f%% of 98.3)\n", tag, ms,
           flops / (ms * 1e-3) / 1e12, 100 * flops / (ms * 1e-3) / 1e12 / 98.3);
  };

  printf("\n=== tok=%d kv=%d heads=%d kvh=%d (%d warps) ===\n", tok, kv, NH, NKVH, NWARP);
  char tag[64];
  for (int rb = 2; rb <= 3; ++rb) for (int nw = 8; nw <= 8; nw *= 2) {
    const int bm = nw * 16 * rb;
    const int nth = nw * 64;
    const dim3 g((tok + bm - 1) / bm, NH);
    double cfl = 0;
    for (int qb = 0; qb < tok; qb += bm)
      cfl += 4.0 * bm * (double)(qb + bm < kv ? qb + bm : kv) * D * NH;
    for (int t = 1; t <= 8; t *= 2) {
      kern k = pick(1, t, rb, nw);
      for (int i = 0; i < 3; ++i) k<<<g, nth>>>(dQ, dK, dV, dO, kv, tok, NH, NKVH, log2scale);
      cudaEvent_t st, en;
      cudaEventCreate(&st); cudaEventCreate(&en);
      cudaEventRecord(st);
      for (int i = 0; i < 10; ++i) k<<<g, nth>>>(dQ, dK, dV, dO, kv, tok, NH, NKVH, log2scale);
      cudaEventRecord(en); cudaEventSynchronize(en);
      float ms; cudaEventElapsedTime(&ms, st, en); ms /= 10;
      int nb = 0;
      cudaFuncAttributes a;
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, k, nth, 0);
      cudaFuncGetAttributes(&a, k);
      printf("  RB=%d NW=%2d NT=%d BM=%3d  %7.3f ms  %6.1f TFLOPS | %d CTA/SM, %d warps, %zu B, %d regs\n",
             rb, nw, t, bm, ms, cfl / (ms * 1e-3) / 1e12, nb, nb * nw,
             (size_t)a.sharedSizeBytes, a.numRegs);
    }
  }
  printf("  vendor FA (4k prefill causal, per layer): 1.304 ms / 52.7 TFLOPS\n");
  return 0;
}
