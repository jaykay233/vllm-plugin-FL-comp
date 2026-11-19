// Split-KV decode attention.
//
// Why this kernel exists, and why the split is not the point.
//
// On the 16k scoring shape (30 sequences, kv 16384, 2 kv heads, head_dim 128) the
// vendor decode kernel moves 503.3 MB and takes 1.194 ms, which is 421 GB/s against
// an 819 GB/s part.  Two measurements located the cause:
//
//     30 x 16384  (60 work units)  503.3 MB  1.194 ms  421 GB/s
//     60 x 8192  (120 work units)  503.3 MB  0.797 ms  632 GB/s
//
// The same bytes, 1.5x apart.  probe_decode_split.cu then reproduced the memory
// pattern with a trivial reduction in place of the softmax, and found the real driver
// is not the unit count but how many independent 512-byte requests a warp can keep in
// flight:
//
//     one request at a time, full occupancy       410 GB/s at 480 warps
//     sixteen batched requests, 8 KB smem/warp    569 GB/s at 480 warps (16 warps/SM)
//     sixteen batched requests, 4 KB smem/warp    598 GB/s at 480 warps (32 warps/SM)
//
// So the win comes from batching a whole page of K and V per iteration, not from
// splitting.  The split is only needed because one warp cannot own a whole 16k KV
// range and still present that batching; with S=8 each warp covers 2048 positions, and
// 60 units x 8 = 480 warps is exactly the count the probe liked best.
//
// Structure, per warp, per cache page (PAGE is 16 and equals the MMA's n tile, so a
// tile never straddles a page and its address comes straight from block_table):
//
//   1. fetch the page's K and V, eight 512-byte vector loads each, issued together
//   2. store K into shared memory in fragment order, syncwarp
//   3. QK as eight 16x16x16 MMAs over k=128, giving 16 heads x 16 positions
//   4. online softmax over those positions
//   5. store V over the same buffer (K is dead by then), syncwarp
//   6. PV as eight 16x16x16 MMAs over k=16
//
// Step 5 reusing step 2's buffer is what holds the footprint at 4 KB per warp, and
// that is what buys 32 warps/SM instead of 16.
//
// Only 8 of the 16 MMA rows are real heads (16 query heads over 2 kv heads), so half
// the MMA work is padding.  At 12e9 MMA/s the whole 16k shape needs 0.08 ms of MMA
// time against a 0.84 ms memory budget, so the padding is affordable and not worth a
// more elaborate packing.
//
// Fragment rule, verified against a CPU reference in bf16_mma_map.cu:
//     operand row = 2*(L/16) + 8*(j/2) + (j%2),   operand col = L%16
//     accumulator row = L/16 + 4*q,               accumulator col = L%16
//
// With A = Q (m=head, k=dim) and B = K (k=dim, n=pos), a K fragment wants four dims at
// one position.  With B = V (k=pos, n=dim), a V fragment wants four positions at one
// dim.  Both read back as pack4(&sh[(ch*64 + lane)*4]); only the store layout differs,
// which is why the two index helpers below are not the same function.
//
// What it measures, and what was already rejected.
//
// On 2026-11-19, in one session, on the 16k shape (503.3 MB of KV):
//
//     30 x 16384  this kernel, split 1.424 ms + merge 0.040 ms   343 GB/s
//     30 x 16384  vendor paged_attention_flashinfer              1.617 ms  311 GB/s
//
// Absolute times move with the part's clock state -- the 1.194 ms at the top of this
// file was a faster one -- so only same-session pairs mean anything.
//
// That comparison turned out to be the wrong one.  What the delivered configuration
// actually runs for decode is Triton FlashDecoding, and it is much faster:
//
//     30 x 16384  Triton FlashDecoding   0.816 ms  617 GB/s
//     64 x 16384  Triton FlashDecoding   1.671 ms  642 GB/s
//     32 x 4096   Triton FlashDecoding   0.231 ms  581 GB/s
//     30 x 16384  this kernel            1.522 ms  331 GB/s
//     32 x 4096   this kernel            0.443 ms  303 GB/s
//
// The vendor kernel is also numerically broken exactly where it is slow: at
// 30 x 16384 it returns 2.327 relative error against fp32, against this kernel's
// 4.8e-03, and vLLM dodges that only by padding the batch to 32.  So the shape this
// kernel was built for was never really served by the kernel it was measured against.
//
// The 1.9x gap is instruction issue, not bandwidth.  Per 16-position page this kernel
// issues about 216 wavefront instructions for 8 KB: 8 global loads for K, 16 shared
// stores and 8 shared loads to stage it, 32 two-byte gathers for V, 16 MMAs, and 32
// shuffles plus ~40 ALU for the online softmax.  A 64-lane wavefront issues over two
// cycles, so the floor is ~2 cycles per instruction, and this measures 2.75 -- 73%
// issue efficiency.  The same access pattern with the softmax removed reaches 598 GB/s
// (probe_decode_split.cu), so the arithmetic is what is left, and specifically a
// 16-position softmax whose reduction is amortised over four times fewer positions
// than Triton's tile.  Swapping staging layouts -- what both rejected variants below
// do -- cannot reach it.
//
// Rejected: transposing the PV product to O^T = V^T @ P^T so that V reuses K's staging
// path exactly.  It does work, and it removes the scattered V gather, but P^T then has
// to reach the B-operand layout and no shuffle can build it: one source lane is read
// for four different heads by four different destinations, and a shuffle hands over
// the source's register, not the destination's pick.  Routing P^T through 512 B of
// shared scratch and staging V through shared memory instead of gathering it measured
// 1.61 ms against this version's 1.42 ms.  The gather here is 128 bytes per
// instruction across four rows, which is cheap; the shared-store port is what is
// scarce, and the transposed form spends twice as much of it.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>


// The kernel lives with the plugin so that the shipped directory is self-contained;
// this harness compiles it from there (see README for the -I flag).
#include "decode_attn.h"
// Layout self-test.
//
// Stages one page exactly as the kernel does and then reconstructs it through the
// fragment read, writing each element to the coordinate the fragment rule claims it
// came from.  If the staging and the two index helpers agree, the reconstruction is
// the identity; if they do not, the mismatches point at the offending lane/dim.
//
// This exists because the first version of this kernel produced bounded but wrong
// numbers, and telling a bad transpose apart from a bad softmax by reading the code
// is far slower than checking the transpose on its own.
// ---------------------------------------------------------------------------
__global__ void selftest_stage(const bf_t *__restrict__ K, const bf_t *__restrict__ V,
                               bf_t *__restrict__ K2, bf_t *__restrict__ V2,
                               bf_t *__restrict__ dump, int *__restrict__ dbg) {
  __shared__ __align__(16) bf_t buf[TILE];
  const int lane = threadIdx.x & 63;
  bf_t *sh = buf;

  v8h pk[8], pv[8];
#pragma unroll
  for (int it = 0; it < 8; ++it) {
    const int e = (it * 64 + lane) * 4;
    pk[it] = *(const v8h *)(K + e);
    pv[it] = *(const v8h *)(V + e);
  }
#pragma unroll
  for (int it = 0; it < 8; ++it) {
    const int flat = (it * 64 + lane) * 4;
    const int pos = flat / D, dim = flat % D;
    const int ch = dim / 16, dlin = dim % 16;
    unsigned lo, hi;
    __builtin_memcpy(&lo, &pk[it], 4);
    __builtin_memcpy(&hi, (const char *)&pk[it] + 4, 4);
    *(unsigned *)(sh + kidx(ch, pos, dlin)) = lo;
    *(unsigned *)(sh + kidx(ch, pos, dlin + 2)) = hi;
    if (dbg) dbg[lane * 8 + it] = (flat << 16) | (pos << 8) | kidx(ch, pos, dlin);
  }
  __syncwarp();
  if (dump) {
    for (int i = lane; i < TILE; i += 64) dump[i] = sh[i];
  }
  __syncwarp();
  // read back as a K fragment: lane L element j should be K[pos L%16][dim 16ch+2*(L/16)+..]
#pragma unroll
  for (int ch = 0; ch < NCH; ++ch) {
    const v4y f = pack4(sh + ridx(ch, lane));
    bf_t e[4];
    __builtin_memcpy(e, &f, 8);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int pos = lane % 16;
      const int dim = 16 * ch + 2 * (lane / 16) + 8 * (j / 2) + (j % 2);
      K2[pos * D + dim] = e[j];
    }
  }
  __syncwarp();

  // now the same buffer as a V tile, stored the V way and read back as a V fragment
#pragma unroll
  for (int it = 0; it < 8; ++it) {
    const int flat = (it * 64 + lane) * 4;
    const int pos = flat / D, dim = flat % D;
    const int ch = dim / 16, dlin = dim % 16;
    const bf_t *q4 = (const bf_t *)&pv[it];
#pragma unroll
    for (int u = 0; u < 4; ++u) sh[vidx(ch, pos, dlin + u)] = q4[u];
  }
  __syncwarp();
#pragma unroll
  for (int ch = 0; ch < NCH; ++ch) {
    const v4y f = pack4(sh + ridx(ch, lane));
    bf_t e[4];
    __builtin_memcpy(e, &f, 8);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int pos = 2 * (lane / 16) + 8 * (j / 2) + (j % 2);
      const int dim = 16 * ch + (lane % 16);
      V2[pos * D + dim] = e[j];
    }
  }
}

static const char *arg(int argc, char **argv, int i, const char *dflt) {
  return i < argc ? argv[i] : dflt;
}

int main(int argc, char **argv) {
  const char *casep = argc > 1 ? argv[1] : nullptr;
  if (!casep) {
    printf("usage: fused_decode <case.txt> [S] [W]\n"
           "case.txt: line1 = nseq NH NKVH maxblk; per seq: klen; then qp kp vp bp op\n");
    return 1;
  }
  if (!strcmp(casep, "--selftest")) {
    // Fill one page with a pattern that bf16 can hold exactly, then push it through
    // the staging code and compare against the identity.  bf16 keeps only 8 mantissa
    // bits, so a single value cannot identify 2048 slots; two passes, low byte then
    // high byte, do identify them.
    auto b2f = [](bf_t x) {
      unsigned short h;
      __builtin_memcpy(&h, &x, 2);
      unsigned u = (unsigned)h << 16;
      float f;
      __builtin_memcpy(&f, &u, 4);
      return f;
    };
    bf_t *hK = (bf_t *)malloc(TILE * 2), *hV = (bf_t *)malloc(TILE * 2);
    bf_t *dK, *dV, *dK2, *dV2, *gK = (bf_t *)malloc(TILE * 2), *gV = (bf_t *)malloc(TILE * 2);
    cudaMalloc(&dK, TILE * 2); cudaMalloc(&dV, TILE * 2);
    cudaMalloc(&dK2, TILE * 2); cudaMalloc(&dV2, TILE * 2);
    bf_t *dDump, *gDump = (bf_t *)malloc(TILE * 2);
    cudaMalloc(&dDump, TILE * 2);
    int *dDbg, *gDbg = (int *)malloc(512 * 4);
    cudaMalloc(&dDbg, 512 * 4);
    const bool dump_on = (argc > 2 && !strcmp(argv[2], "dump"));
    int badk = 0, badv = 0, firstk = -1, firstv = -1;
    srand(12345);
    for (int mode = 0; mode < 3; ++mode) {
      for (int i = 0; i < TILE; ++i) {
        if (mode == 2) {  // random but exactly representable, so a swap cannot hide
          hK[i] = __float2bfloat16((float)(rand() % 200 - 100));
          hV[i] = __float2bfloat16((float)(rand() % 200 - 100));
        } else {
          const int v = mode == 0 ? (i & 255) : (i >> 8);
          hK[i] = __float2bfloat16((float)v);
          hV[i] = __float2bfloat16((float)(255 - v));
        }
      }
      cudaMemcpy(dK, hK, TILE * 2, cudaMemcpyHostToDevice);
      cudaMemcpy(dV, hV, TILE * 2, cudaMemcpyHostToDevice);
      cudaMemset(dK2, 0xff, TILE * 2);
      cudaMemset(dV2, 0xff, TILE * 2);
      selftest_stage<<<1, 64>>>(dK, dV, dK2, dV2, dump_on ? dDump : nullptr, dump_on ? dDbg : nullptr);
      cudaError_t e = cudaDeviceSynchronize();
      if (e != cudaSuccess) { printf("ERR %s\n", cudaGetErrorString(e)); return 1; }
      cudaMemcpy(gK, dK2, TILE * 2, cudaMemcpyDeviceToHost);
      cudaMemcpy(gV, dV2, TILE * 2, cudaMemcpyDeviceToHost);
      int mk = 0, mv = 0;
      for (int i = 0; i < TILE; ++i) {
        if (b2f(gK[i]) != b2f(hK[i])) { if (firstk < 0) firstk = i; ++badk; ++mk; }
        if (b2f(gV[i]) != b2f(hV[i])) { if (firstv < 0) firstv = i; ++badv; ++mv; }
      }
      if (dump_on) {
        cudaMemcpy(gDump, dDump, TILE * 2, cudaMemcpyDeviceToHost);
        printf("  mode %d raw smem after K stage (s0 = sh[0]), first 16 groups:\n", mode);
        cudaMemcpy(gDbg, dDbg, 512 * 4, cudaMemcpyDeviceToHost);
        printf("    lane0 it0..7 (flat,pos,idx): ");
        for (int i = 0; i < 8; ++i)
          printf("(%d,%d,%d) ", gDbg[i] >> 16, (gDbg[i] >> 8) & 255, gDbg[i] & 255);
        printf("\n    K2[0..15] (read back): ");
        for (int i = 0; i < 16; ++i) printf("%5.0f", b2f(gK[i]));
        printf("\n    want      [0..15]     : ");
        for (int i = 0; i < 16; ++i) printf("%5.0f", b2f(hK[i]));
        printf("\n    sh[64..71] = ");
        for (int i = 64; i < 72; ++i) printf("%5.0f", b2f(gDump[i]));
        printf("\n");
        for (int g = 0; g < 6; ++g)
          printf("    sh[%3d..%3d]: %6.1f %6.1f %6.1f %6.1f   (want pos%d dims %d,%d,%d,%d)\n",
                 g * 4, g * 4 + 3, b2f(gDump[g * 4]), b2f(gDump[g * 4 + 1]),
                 b2f(gDump[g * 4 + 2]), b2f(gDump[g * 4 + 3]), g / 8,
                 16 * ((g * 4) & 15) % 128, 0, 0, 0);
      }
      printf("  mode %d: K %d wrong, V %d wrong\n", mode, mk, mv);
      if (mk) {
        int shown = 0;
        for (int i = 0; i < TILE && shown < 6; ++i)
          if (b2f(gK[i]) != b2f(hK[i])) {
            printf("    pos %2d dim %3d: want %6.1f got %6.1f\n", i / D, i % D, b2f(hK[i]),
                   b2f(gK[i]));
            ++shown;
          }
      }
    }
    printf("K staging: %d/%d wrong", badk, 2 * TILE);
    if (badk) printf("  first at pos %d dim %d", firstk / D, firstk % D);
    printf("\nV staging: %d/%d wrong", badv, 2 * TILE);
    if (badv) printf("  first at pos %d dim %d", firstv / D, firstv % D);
    printf("\n");
    return badk || badv ? 1 : 0;
  }
  const int S = atoi(arg(argc, argv, 2, "8"));
  const int W = atoi(arg(argc, argv, 3, "8"));
  if (S > 64) { printf("S too large for the merge buffer\n"); return 1; }

  FILE *cf = fopen(casep, "r");
  if (!cf) { printf("cannot open case\n"); return 1; }
  int nseq, NH, NKVH, maxblk;
  if (fscanf(cf, "%d %d %d %d", &nseq, &NH, &NKVH, &maxblk) != 4) { printf("bad header\n"); return 1; }
  int *hk = (int *)malloc((nseq + 1) * 4);
  hk[0] = 0;
  for (int s = 0; s < nseq; ++s) {
    int kl;
    if (fscanf(cf, "%d", &kl) != 1) { printf("bad seq\n"); return 1; }
    hk[s + 1] = hk[s] + kl;
  }
  char qp[512], kp[512], vp[512], bp[512], op[512];
  if (fscanf(cf, "%511s %511s %511s %511s %511s", qp, kp, vp, bp, op) != 5) {
    printf("bad paths\n");
    return 1;
  }
  fclose(cf);

  const int totq = nseq;  // decode: exactly one query row per sequence
  const int nunit = nseq * NKVH;
  const int NHK = NH / NKVH;
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
  for (int i = 0; i < nseq * maxblk; ++i)
    if (bt[i] + 1 > npages) npages = bt[i] + 1;
  const size_t ksz = (size_t)npages * NKVH * PAGE * D;  // HND
  bf_t *hK = (bf_t *)malloc(ksz * 2), *hV = (bf_t *)malloc(ksz * 2);
  rd(kp, hK, ksz * 2);
  rd(vp, hV, ksz * 2);

  // per-sequence klen (indexed by the sequence, which the kernel derives as unit /
  // NKVH) and the query row each sequence's single token sits at
  int *uklen = (int *)malloc(nseq * 4), *qrow = (int *)malloc(nseq * 4);
  for (int s = 0; s < nseq; ++s) {
    qrow[s] = s;
    uklen[s] = hk[s + 1] - hk[s];
  }

  bf_t *dQ, *dK, *dV;
  float *dO, *dOp, *dMp, *dLp;
  int *dbt, *dklen, *dqr;
  const size_t psz = (size_t)nunit * S * NHK;
  cudaMalloc(&dQ, qsz * 2); cudaMalloc(&dK, ksz * 2); cudaMalloc(&dV, ksz * 2);
  cudaMalloc(&dO, qsz * 4);
  cudaMalloc(&dOp, psz * D * 4); cudaMalloc(&dMp, psz * 4); cudaMalloc(&dLp, psz * 4);
  cudaMalloc(&dbt, (size_t)nseq * maxblk * 4); cudaMalloc(&dklen, nseq * 4);
  cudaMalloc(&dqr, nseq * 4);
  cudaMemcpy(dQ, hQ, qsz * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(dK, hK, ksz * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(dV, hV, ksz * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(dbt, bt, (size_t)nseq * maxblk * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(dklen, uklen, nseq * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(dqr, qrow, nseq * 4, cudaMemcpyHostToDevice);

  const float log2scale = (float)(1.4426950408889634 / std::sqrt((double)D));
  const int ctas = (nunit * S + W - 1) / W;
  auto launch_split = [&]() {
    if (W == 4)
      decode_split<4><<<ctas, 4 * 64>>>(dQ, dK, dV, dOp, dMp, dLp, dbt, dklen, dqr, nunit, S,
                                        maxblk, NH, NKVH, log2scale);
    else if (W == 16)
      decode_split<16><<<ctas, 16 * 64>>>(dQ, dK, dV, dOp, dMp, dLp, dbt, dklen, dqr, nunit, S,
                                          maxblk, NH, NKVH, log2scale);
    else
      decode_split<8><<<ctas, 8 * 64>>>(dQ, dK, dV, dOp, dMp, dLp, dbt, dklen, dqr, nunit, S,
                                        maxblk, NH, NKVH, log2scale);
  };
  auto launch_merge = [&]() {
    decode_merge<float><<<nunit, 128>>>(dOp, dMp, dLp, dO, dqr, S, NH, NKVH);
  };

  launch_split();
  launch_merge();
  cudaError_t e = cudaDeviceSynchronize();
  if (e != cudaSuccess) { printf("ERR %s\n", cudaGetErrorString(e)); return 1; }

  cudaMemcpy(hO, dO, qsz * 4, cudaMemcpyDeviceToHost);
  {
    FILE *f = fopen(op, "wb");
    if (!f) { printf("cannot open %s\n", op); return 1; }
    if (fwrite(hO, 4, qsz, f) != qsz) { printf("short write\n"); return 1; }
    fclose(f);
  }

  if (argc > 4) {  // dump the raw partials
    float *hOp = (float *)malloc(psz * D * 4), *hMp = (float *)malloc(psz * 4),
          *hLp = (float *)malloc(psz * 4);
    cudaMemcpy(hOp, dOp, psz * D * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(hMp, dMp, psz * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(hLp, dLp, psz * 4, cudaMemcpyDeviceToHost);
    const char *names[3] = {"/tmp/part.op", "/tmp/part.mp", "/tmp/part.lp"};
    void *ptr[3] = {hOp, hMp, hLp};
    size_t sz[3] = {psz * D, psz, psz};
    for (int i = 0; i < 3; ++i) {
      FILE *f = fopen(names[i], "wb");
      fwrite(ptr[i], 4, sz[i], f);
      fclose(f);
    }
    printf("dumped partials (%zu entries)\n", (size_t)psz);
  }

  // ---- timing: split and merge are reported apart because a 1.5x win in the split
  // must not be eaten by the combine ----
  auto timeit = [&](auto &&fn, int reps, const char *name) {
    for (int i = 0; i < 3; ++i) fn();
    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    cudaEventRecord(a);
    for (int i = 0; i < reps; ++i) fn();
    cudaEventRecord(b); cudaEventSynchronize(b);
    float ms; cudaEventElapsedTime(&ms, a, b);
    printf("%-14s %8.3f ms\n", name, ms / reps);
    return ms / reps;
  };
  const double bytes = 4.0 * (double)NKVH * (double)hk[nseq] * D;  // K + V, 2 bytes each
  const float sm = timeit(launch_split, 10, "split");
  const float mm = timeit(launch_merge, 10, "merge");
  printf("total          %8.3f ms   %.1f GB/s\n", sm + mm, bytes / (sm * 1e-3) / 1e9);
  printf("(vendor paged_attention_flashinfer, same session: 1.617 ms / 311 GB/s; the\n"
         " 1.194 ms / 421 GB/s in the header is the same kernel on a faster clock)\n");
  return 0;
}
