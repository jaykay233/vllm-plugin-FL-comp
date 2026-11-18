// ivcore u8 decode v49 — INT8-K/V + native i8 MMA QK (I8M16N16K64).
// INT8 (v_mode=2): Q quantized once/CTA; K raw i8 MMA; scale→P; V=i8→bf16.
// BI-V150 dual-constraint: ~423µs / cos≈0.99996 @ split=6 (FINDINGS 17.64).
// FP8 (v_mode=1): branchless e4m3fn→bf16 (denorm flush); ~605µs.

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <cstdint>

using namespace ixix::wmma;

static constexpr int kPage = 16;
static constexpr int kHd = 128;
static constexpr int kWarp = 64;
static constexpr int kNTiles = 8;
static constexpr int kGroup = 8;
static constexpr int kPageBytesI8 = kHd * kPage;  // 2048
static constexpr int kSmeChunk = 1024;            // 16×64B

enum VMode : int { kVHiByte = 0, kVFp8 = 1, kVI8 = 2 };

__device__ __forceinline__ void wait_g2s_zero() {
  WaitCount w = {};
  w.Cnt.G2S = 1;
  w.Cnt.G2S_CNT = 0;
  __ivcorex_sl_waitcnt(w.Count);
}

// Contiguous i8 page (2KB) → smem via SME (2 × 1024 B, stride 64 B).
__device__ __forceinline__ void sme_load_page_i8(
    uint8_t* __restrict__ sm, const uint8_t* __restrict__ g) {
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    ld_sme_16x1b64(
        sm, g,
        /*Stride*/ 64,
        /*GlobalMemoryOffset*/ 0,
        /*RowOffsetImm*/ r * 16,
        /*ColOffsetImm*/ 0,
        /*ShareMemoryOffsetImm*/ r * kSmeChunk);
  }
}

__device__ __forceinline__ uint16_t f32_to_bf16bits(float f) {
  uint32_t u;
  __builtin_memcpy(&u, &f, sizeof(u));
  return static_cast<uint16_t>(u >> 16);
}

__device__ __forceinline__ float bf16bits_to_f32(uint16_t h) {
  uint32_t u = static_cast<uint32_t>(h) << 16;
  float f;
  __builtin_memcpy(&f, &u, sizeof(f));
  return f;
}

// Branchless e4m3fn→bf16; denorms flushed to signed 0 (cos>0.999 on N(0,.12) KV).
__device__ __forceinline__ uint16_t e4m3fn_to_bf16bits_fast(uint8_t u) {
  const unsigned sign = (unsigned)(u & 0x80) << 8;
  const unsigned exp = (u >> 3) & 0xF;
  const unsigned mant = u & 0x7;
  const unsigned normal = sign | ((exp + 120u) << 7) | (mant << 4);
  const unsigned is_den = (exp == 0u) ? ~0u : 0u;
  const unsigned is_nan = (exp == 0xFu && mant == 0x7u) ? ~0u : 0u;
  unsigned r = normal;
  r = (r & ~is_den) | (sign & is_den);
  r = (r & ~is_nan) | (0x7fc0u & is_nan);
  return static_cast<uint16_t>(r);
}

__device__ __forceinline__ void mma_bf16(
    fragment<accumulator, 16, 16, 32, float>& d,
    const fragment<matrix_a, 16, 16, 32, half, row_major>& a,
    const fragment<matrix_b, 16, 16, 32, half, col_major>& b,
    const fragment<accumulator, 16, 16, 32, float>& c) {
  *(reinterpret_cast<v4f32*>(&(d.x))) = __ivcorex_matrix_mad_f32x4_bf16x4(
      *(reinterpret_cast<const v4bf16*>(&(a.x))),
      *(reinterpret_cast<const v4bf16*>(&(b.x))),
      reinterpret_cast<const v4f32&>(c.x));
  *(reinterpret_cast<v4f32*>(&(d.x))) = __ivcorex_matrix_mad_f32x4_bf16x4(
      *(reinterpret_cast<const v4bf16*>(&(a.x)) + 1),
      *(reinterpret_cast<const v4bf16*>(&(b.x)) + 1),
      reinterpret_cast<const v4f32&>(d.x));
}

// Native i8 MMA (BI-V150 / ivcore11: I8M16N16K32). Avoids i8→bf16 on K.
__device__ __forceinline__ void mma_i8(
    fragment<accumulator, 16, 16, 64, int>& d,
    const fragment<matrix_a, 16, 16, 64, signed char, row_major>& a,
    const fragment<matrix_b, 16, 16, 64, signed char, col_major>& b,
    const fragment<accumulator, 16, 16, 64, int>& c) {
  mma_sync(d, a, b, c);
}

// Pack Q tile into i8 A fragment (matches __imma_ld_row_b8 lane map).
__device__ __forceinline__ void load_a_q_i8(
    fragment<matrix_a, 16, 16, 64, signed char, row_major>& frag,
    const int8_t* Q, int k0) {
  int laneId = __ivcorex_lane_id();
#pragma unroll
  for (int qt = 0; qt < 4; qt++) {
    int row = (laneId / 16) * 4;
    int column = laneId % 16 + 16 * qt;
    int8_t b0 = 0, b1 = 0, b2 = 0, b3 = 0;
    if (row < kGroup) {
      b0 = Q[row * kHd + k0 + column];
      if (row + 1 < kGroup) b1 = Q[(row + 1) * kHd + k0 + column];
      if (row + 2 < kGroup) b2 = Q[(row + 2) * kHd + k0 + column];
      if (row + 3 < kGroup) b3 = Q[(row + 3) * kHd + k0 + column];
    }
    frag.x[qt] = (unsigned)(uint8_t)b0 | ((unsigned)(uint8_t)b1 << 8) |
                 ((unsigned)(uint8_t)b2 << 16) | ((unsigned)(uint8_t)b3 << 24);
  }
}

// Pack K HD-major i8 into B fragment (matches __imma_ld_col_b8 lane map).
__device__ __forceinline__ void load_b_k_i8_raw(
    fragment<matrix_b, 16, 16, 64, signed char, col_major>& frag,
    const uint8_t* K, int k0) {
  int laneId = __ivcorex_lane_id();
#pragma unroll
  for (int qt = 0; qt < 4; qt++) {
    int row = (laneId / 16) * 4 + 16 * qt;
    int column = laneId % 16;
    int8_t b0 = static_cast<int8_t>(__ldg(K + (k0 + row) * kPage + column));
    int8_t b1 = static_cast<int8_t>(__ldg(K + (k0 + row + 1) * kPage + column));
    int8_t b2 = static_cast<int8_t>(__ldg(K + (k0 + row + 2) * kPage + column));
    int8_t b3 = static_cast<int8_t>(__ldg(K + (k0 + row + 3) * kPage + column));
    frag.x[qt] = (unsigned)(uint8_t)b0 | ((unsigned)(uint8_t)b1 << 8) |
                 ((unsigned)(uint8_t)b2 << 16) | ((unsigned)(uint8_t)b3 << 24);
  }
}

__device__ __forceinline__ void load_a_q(
    fragment<matrix_a, 16, 16, 32, half, row_major>& frag,
    const uint16_t* Q, int k0) {
  int laneId = __ivcorex_lane_id();
#pragma unroll
  for (int qt = 0; qt < 4; qt++) {
    int row = (laneId / 16) * 2 + 8 * (qt % 2);
    int column = laneId % 16 + 16 * (qt / 2);
    uint16_t h0 = 0, h1 = 0;
    if (row < kGroup) h0 = Q[row * kHd + k0 + column];
    if (row + 1 < kGroup) h1 = Q[(row + 1) * kHd + k0 + column];
    frag.x[qt] = unsigned(h0) | (unsigned(h1) << 16);
  }
}

// Legacy hibyte K (broken on real MiniCPM KV — cos~0.5). Kept for reference.
__device__ __forceinline__ void load_b_k_hd(
    fragment<matrix_b, 16, 16, 32, half, col_major>& frag,
    const uint8_t* K, int k0) {
  int laneId = __ivcorex_lane_id();
#pragma unroll
  for (int qt = 0; qt < 4; qt++) {
    int row = (laneId / 16) * 2 + 8 * qt;
    int column = laneId % 16;
    uint8_t u0 = K[(k0 + row) * kPage + column];
    uint8_t u1 = K[(k0 + row + 1) * kPage + column];
    frag.x[qt] = (unsigned(u0) << 8) | (unsigned(u1) << 24);
  }
}

// INT8 → bf16 bits via float (exact for |q|<=127). Prefer cast over
// __int2float_rn — ivcore11 has no cvt_i8; this is the FINDINGS §17.64 tax.
__device__ __forceinline__ uint16_t i8_to_bf16bits(int8_t q) {
  return f32_to_bf16bits(static_cast<float>(q));
}

// INT8 K (HD-major); per-token scale applied to scores after QK.
__device__ __forceinline__ void load_b_k_i8(
    fragment<matrix_b, 16, 16, 32, half, col_major>& frag,
    const uint8_t* K, int k0) {
  int laneId = __ivcorex_lane_id();
#pragma unroll
  for (int qt = 0; qt < 4; qt++) {
    int row = (laneId / 16) * 2 + 8 * qt;
    int column = laneId % 16;
    int8_t q0 = static_cast<int8_t>(K[(k0 + row) * kPage + column]);
    int8_t q1 = static_cast<int8_t>(K[(k0 + row + 1) * kPage + column]);
    uint16_t h0 = i8_to_bf16bits(q0);
    uint16_t h1 = i8_to_bf16bits(q1);
    frag.x[qt] = unsigned(h0) | (unsigned(h1) << 16);
  }
}

__device__ __forceinline__ void load_b_v_hibyte(
    fragment<matrix_b, 16, 16, 32, half, col_major>& frag,
    const uint8_t* V, int n0) {
  int laneId = __ivcorex_lane_id();
#pragma unroll
  for (int qt = 0; qt < 4; qt++) {
    int row = (laneId / 16) * 2 + 8 * qt;
    int column = laneId % 16;
    uint8_t u0 = 0, u1 = 0;
    if (row < kPage) u0 = V[row * kHd + n0 + column];
    if (row + 1 < kPage) u1 = V[(row + 1) * kHd + n0 + column];
    frag.x[qt] = (unsigned(u0) << 8) | (unsigned(u1) << 24);
  }
}

__device__ __forceinline__ void load_b_v_fp8(
    fragment<matrix_b, 16, 16, 32, half, col_major>& frag,
    const uint8_t* V, int n0) {
  int laneId = __ivcorex_lane_id();
#pragma unroll
  for (int qt = 0; qt < 4; qt++) {
    int row = (laneId / 16) * 2 + 8 * qt;
    int column = laneId % 16;
    uint16_t h0 = 0, h1 = 0;
    if (row < kPage) h0 = e4m3fn_to_bf16bits_fast(V[row * kHd + n0 + column]);
    if (row + 1 < kPage)
      h1 = e4m3fn_to_bf16bits_fast(V[(row + 1) * kHd + n0 + column]);
    frag.x[qt] = unsigned(h0) | (unsigned(h1) << 16);
  }
}

// INT8 as integer bf16 (scale already absorbed into P).
__device__ __forceinline__ void load_b_v_i8(
    fragment<matrix_b, 16, 16, 32, half, col_major>& frag,
    const uint8_t* V, int n0) {
  int laneId = __ivcorex_lane_id();
#pragma unroll
  for (int qt = 0; qt < 4; qt++) {
    int row = (laneId / 16) * 2 + 8 * qt;
    int column = laneId % 16;
    uint16_t h0 = 0, h1 = 0;
    if (row < kPage) {
      h0 = i8_to_bf16bits(
          static_cast<int8_t>(__ldg(V + row * kHd + n0 + column)));
    }
    if (row + 1 < kPage) {
      h1 = i8_to_bf16bits(
          static_cast<int8_t>(__ldg(V + (row + 1) * kHd + n0 + column)));
    }
    frag.x[qt] = unsigned(h0) | (unsigned(h1) << 16);
  }
}

// Smem variants — no __ldg (illegal / pointless on shared).
__device__ __forceinline__ void load_b_k_i8_raw_sm(
    fragment<matrix_b, 16, 16, 64, signed char, col_major>& frag,
    const uint8_t* K, int k0) {
  int laneId = __ivcorex_lane_id();
#pragma unroll
  for (int qt = 0; qt < 4; qt++) {
    int row = (laneId / 16) * 4 + 16 * qt;
    int column = laneId % 16;
    int8_t b0 = static_cast<int8_t>(K[(k0 + row) * kPage + column]);
    int8_t b1 = static_cast<int8_t>(K[(k0 + row + 1) * kPage + column]);
    int8_t b2 = static_cast<int8_t>(K[(k0 + row + 2) * kPage + column]);
    int8_t b3 = static_cast<int8_t>(K[(k0 + row + 3) * kPage + column]);
    frag.x[qt] = (unsigned)(uint8_t)b0 | ((unsigned)(uint8_t)b1 << 8) |
                 ((unsigned)(uint8_t)b2 << 16) | ((unsigned)(uint8_t)b3 << 24);
  }
}

__device__ __forceinline__ void load_b_v_i8_sm(
    fragment<matrix_b, 16, 16, 32, half, col_major>& frag,
    const uint8_t* V, int n0) {
  int laneId = __ivcorex_lane_id();
#pragma unroll
  for (int qt = 0; qt < 4; qt++) {
    int row = (laneId / 16) * 2 + 8 * qt;
    int column = laneId % 16;
    uint16_t h0 = 0, h1 = 0;
    if (row < kPage) {
      h0 = i8_to_bf16bits(static_cast<int8_t>(V[row * kHd + n0 + column]));
    }
    if (row + 1 < kPage) {
      h1 = i8_to_bf16bits(
          static_cast<int8_t>(V[(row + 1) * kHd + n0 + column]));
    }
    frag.x[qt] = unsigned(h0) | (unsigned(h1) << 16);
  }
}

__device__ __forceinline__ void pack_ap_g8(
    fragment<matrix_a, 16, 16, 32, half, row_major>& ap,
    float e0, float e1) {
  int lane = __ivcorex_lane_id();
  uint32_t p01 = (uint32_t)f32_to_bf16bits(e0) |
                 ((uint32_t)f32_to_bf16bits(e1) << 16);
  int g = lane / 16, col = lane % 16, a_row = g * 2;
  uint32_t s0 = __shfl_sync(0xffffffffu, p01, (a_row & 3) * 16 + col);
  uint32_t s1 = __shfl_sync(0xffffffffu, p01, ((a_row + 1) & 3) * 16 + col);
  uint16_t h0, h1;
  if (a_row < 4) {
    h0 = (uint16_t)s0;
    h1 = (uint16_t)s1;
  } else {
    h0 = (uint16_t)(s0 >> 16);
    h1 = (uint16_t)(s1 >> 16);
  }
  ap.x[0] = unsigned(h0) | (unsigned(h1) << 16);
  ap.x[1] = 0;
  ap.x[2] = 0;
  ap.x[3] = 0;
}

template <int VMODE, int SME = 0>
__global__ void ivcore_u8_decode_kernel(
    const uint16_t* __restrict__ q,
    const uint8_t* __restrict__ k_u8,
    const uint8_t* __restrict__ v_u8,
    const float* __restrict__ k_scales,  // [blocks,n_kv,PAGE] per-token K scales
    const float* __restrict__ v_scales,  // [blocks,n_kv,PAGE] if VMODE==kVI8 else unused
    const int32_t* __restrict__ block_table,
    const int32_t* __restrict__ seqused,
    uint16_t* __restrict__ out_partial,  // bf16 bits — halves combine BW
    float* __restrict__ lse_partial,
    int batch, int n_qo, int n_kv, int /*max_pages*/, int stride_bt,
    float scale_log2, int split) {
  const int lane = threadIdx.x;
  const int pid = static_cast<int>(blockIdx.x);
  const int sp = pid % split;
  const int rest = pid / split;
  const int kvh = rest % n_kv;
  const int b = rest / n_kv;

  const int seqlen = seqused[b];
  const int npages = (seqlen + kPage - 1) / kPage;
  const int chunk = (npages + split - 1) / split;
  int p0 = sp * chunk;
  int p1 = p0 + chunk;
  if (p1 > npages) p1 = npages;
  // Idle splits must zero their partials so host can skip workspace zero_().
  if (p0 >= p1) {
    uint16_t* o_base =
        out_partial + ((sp * batch + b) * n_qo + kvh * kGroup) * kHd;
    for (int i = lane; i < kGroup * kHd; i += kWarp) o_base[i] = 0;
    float* l_base = lse_partial + (b * n_qo + kvh * kGroup) * split + sp;
    if ((lane % 16) == 0) {
      int g = lane / 16;
      if (g < 4) {
        l_base[g * split] = 0.f;
        l_base[(g + 4) * split] = 0.f;
      }
    }
    return;
  }

  const uint16_t* q_base = q + b * n_qo * kHd + kvh * kGroup * kHd;

  // INT8-V path: quantize Q once → native i8 MMA for QK (skip i8→bf16 on K).
  __shared__ int8_t sm_q[kGroup * kHd];
  __shared__ float sm_q_scale;
  float q_scale = 1.f;
  if constexpr (VMODE == kVI8) {
    float amax = 0.f;
    for (int i = lane; i < kGroup * kHd; i += kWarp) {
      amax = fmaxf(amax, fabsf(bf16bits_to_f32(q_base[i])));
    }
    __shared__ float red64[64];
    red64[lane] = amax;
    __syncwarp();
#pragma unroll
    for (int s = 32; s > 0; s >>= 1) {
      if (lane < s) red64[lane] = fmaxf(red64[lane], red64[lane + s]);
      __syncwarp();
    }
    float qs = red64[0] * (1.f / 127.f);
    if (qs < 1e-8f) qs = 1e-8f;
    if (lane == 0) sm_q_scale = qs;
    __syncwarp();
    q_scale = sm_q_scale;
    const float inv_q = __ivcorex_rcpf(q_scale);
    for (int i = lane; i < kGroup * kHd; i += kWarp) {
      int qv = __float2int_rn(bf16bits_to_f32(q_base[i]) * inv_q);
      qv = max(-127, min(127, qv));
      sm_q[i] = static_cast<int8_t>(qv);
    }
    __syncwarp();
  }

  fragment<matrix_a, 16, 16, 32, half, row_major> aq[4];
  if constexpr (VMODE != kVI8) {
#pragma unroll
    for (int t = 0; t < 4; ++t) load_a_q(aq[t], q_base, t * 32);
  }

  fragment<accumulator, 16, 16, 32, float> o_acc[kNTiles];
#pragma unroll
  for (int t = 0; t < kNTiles; ++t) fill_fragment(o_acc[t], 0.f);

  float lse_r[kGroup];
#pragma unroll
  for (int i = 0; i < kGroup; ++i) lse_r[i] = 0.f;

  fragment<matrix_b, 16, 16, 32, half, col_major> bk, bv;
  fragment<matrix_a, 16, 16, 32, half, row_major> fa;
  fragment<accumulator, 16, 16, 32, float> cq, cp;
  fragment<matrix_a, 16, 16, 64, signed char, row_major> aq8[2];
  fragment<matrix_b, 16, 16, 64, signed char, col_major> bk8;
  fragment<accumulator, 16, 16, 64, int> cq8;
  if constexpr (VMODE == kVI8) {
#pragma unroll
    for (int t = 0; t < 2; ++t) load_a_q_i8(aq8[t], sm_q, t * 64);
  }

  const size_t k_stride = static_cast<size_t>(n_kv) * kHd * kPage;
  const size_t v_stride = static_cast<size_t>(n_kv) * kPage * kHd;
  const size_t s_stride = static_cast<size_t>(n_kv) * kPage;

  if constexpr (SME != 0 && VMODE == kVI8) {
    // Double-buffer K/V pages in smem; overlap G2S of p+1 with QK+PV of p.
    alignas(64) __shared__ uint8_t sm_k[2][kPageBytesI8];
    alignas(64) __shared__ uint8_t sm_v[2][kPageBytesI8];
    int sme_buf = 0;
    {
      const int page_id0 = block_table[b * stride_bt + p0];
      sme_load_page_i8(
          sm_k[0],
          k_u8 + static_cast<size_t>(page_id0) * k_stride + kvh * kPageBytesI8);
      sme_load_page_i8(
          sm_v[0],
          v_u8 + static_cast<size_t>(page_id0) * v_stride + kvh * kPageBytesI8);
    }
    for (int p = p0; p < p1; ++p) {
      const int page_id = block_table[b * stride_bt + p];
      const float* ksptr =
          k_scales + static_cast<size_t>(page_id) * s_stride + kvh * kPage;
      const float* sptr =
          v_scales + static_cast<size_t>(page_id) * s_stride + kvh * kPage;
      const int valid_n =
          (p == npages - 1) ? (seqlen - p * kPage) : kPage;

      wait_g2s_zero();
      if (p + 1 < p1) {
        const int page_id_n = block_table[b * stride_bt + (p + 1)];
        const int nb = 1 - sme_buf;
        sme_load_page_i8(
            sm_k[nb],
            k_u8 + static_cast<size_t>(page_id_n) * k_stride +
                kvh * kPageBytesI8);
        sme_load_page_i8(
            sm_v[nb],
            v_u8 + static_cast<size_t>(page_id_n) * v_stride +
                kvh * kPageBytesI8);
      }
      const uint8_t* kptr = sm_k[sme_buf];
      const uint8_t* vptr = sm_v[sme_buf];

      fill_fragment(cq8, 0);
#pragma unroll
      for (int t = 0; t < 2; ++t) {
        load_b_k_i8_raw_sm(bk8, kptr, t * 64);
        mma_i8(cq8, aq8[t], bk8, cq8);
      }
      int col = lane % 16;
      int r0 = lane / 16, r1 = r0 + 4;
      const float ks = (col < kPage) ? ksptr[col] : 0.f;
      const float s = q_scale * ks * scale_log2;
      float score0 = static_cast<float>(cq8.x[0]) * s;
      float score1 = static_cast<float>(cq8.x[1]) * s;
      float e0 = 0.f, e1 = 0.f;
      if (col < valid_n) {
        if (r0 < kGroup) e0 = exp2f(fminf(score0, 80.f));
        if (r1 < kGroup) e1 = exp2f(fminf(score1, 80.f));
      }
      float x0 = e0, x1 = e1;
#pragma unroll
      for (int m = 8; m >= 1; m >>= 1) {
        x0 += __shfl_xor_sync(0xffffffffu, x0, m);
        x1 += __shfl_xor_sync(0xffffffffu, x1, m);
      }
      if ((lane % 16) == 0) {
        lse_r[r0] += x0;
        lse_r[r1] += x1;
      }
      {
        float vs = (col < kPage) ? sptr[col] : 0.f;
        pack_ap_g8(fa, e0 * vs, e1 * vs);
      }
#pragma unroll
      for (int ti = 0; ti < kNTiles; ++ti) {
        fill_fragment(cp, 0.f);
        load_b_v_i8_sm(bv, vptr, ti * 16);
        mma_bf16(cp, fa, bv, cp);
#pragma unroll
        for (int i = 0; i < 4; ++i) o_acc[ti].x[i] += cp.x[i];
      }
      sme_buf = 1 - sme_buf;
    }
  } else {
    for (int p = p0; p < p1; ++p) {
      const int page_id = block_table[b * stride_bt + p];
      const uint8_t* kptr =
          k_u8 + static_cast<size_t>(page_id) * k_stride + kvh * (kHd * kPage);
      const uint8_t* vptr =
          v_u8 + static_cast<size_t>(page_id) * v_stride + kvh * (kPage * kHd);
      const float* ksptr =
          k_scales + static_cast<size_t>(page_id) * s_stride + kvh * kPage;
      const float* sptr = nullptr;
      if constexpr (VMODE == kVI8) {
        sptr = v_scales + static_cast<size_t>(page_id) * s_stride + kvh * kPage;
      }
      const int valid_n =
          (p == npages - 1) ? (seqlen - p * kPage) : kPage;

      float score0 = 0.f, score1 = 0.f;
      int col = lane % 16;
      int r0 = lane / 16, r1 = r0 + 4;
      if constexpr (VMODE == kVI8) {
        fill_fragment(cq8, 0);
#pragma unroll
        for (int t = 0; t < 2; ++t) {
          load_b_k_i8_raw(bk8, kptr, t * 64);
          mma_i8(cq8, aq8[t], bk8, cq8);
        }
        const float ks = (col < kPage) ? ksptr[col] : 0.f;
        const float s = q_scale * ks * scale_log2;
        score0 = static_cast<float>(cq8.x[0]) * s;
        score1 = static_cast<float>(cq8.x[1]) * s;
      } else {
        fill_fragment(cq, 0.f);
#pragma unroll
        for (int t = 0; t < 4; ++t) {
          load_b_k_i8(bk, kptr, t * 32);
          mma_bf16(cq, aq[t], bk, cq);
        }
        const float ks = (col < kPage) ? ksptr[col] : 0.f;
        score0 = cq.x[0] * ks * scale_log2;
        score1 = cq.x[1] * ks * scale_log2;
      }

      float e0 = 0.f, e1 = 0.f;
      if (col < valid_n) {
        if (r0 < kGroup) e0 = exp2f(fminf(score0, 80.f));
        if (r1 < kGroup) e1 = exp2f(fminf(score1, 80.f));
      }
      float x0 = e0, x1 = e1;
#pragma unroll
      for (int m = 8; m >= 1; m >>= 1) {
        x0 += __shfl_xor_sync(0xffffffffu, x0, m);
        x1 += __shfl_xor_sync(0xffffffffu, x1, m);
      }
      if ((lane % 16) == 0) {
        lse_r[r0] += x0;
        lse_r[r1] += x1;
      }
      if constexpr (VMODE == kVI8) {
        float s = (col < kPage) ? sptr[col] : 0.f;
        pack_ap_g8(fa, e0 * s, e1 * s);
      } else {
        pack_ap_g8(fa, e0, e1);
      }

#pragma unroll
      for (int ti = 0; ti < kNTiles; ++ti) {
        fill_fragment(cp, 0.f);
        if constexpr (VMODE == kVHiByte) {
          load_b_v_hibyte(bv, vptr, ti * 16);
        } else if constexpr (VMODE == kVFp8) {
          load_b_v_fp8(bv, vptr, ti * 16);
        } else {
          load_b_v_i8(bv, vptr, ti * 16);
        }
        mma_bf16(cp, fa, bv, cp);
#pragma unroll
        for (int i = 0; i < 4; ++i) o_acc[ti].x[i] += cp.x[i];
      }
    }
  }

  uint16_t* o_base =
      out_partial + ((sp * batch + b) * n_qo + kvh * kGroup) * kHd;
#pragma unroll
  for (int ti = 0; ti < kNTiles; ++ti) {
    const int n0 = ti * 16;
#pragma unroll
    for (int qt = 0; qt < 4; ++qt) {
      int row = lane / 16 + 4 * qt;
      int column = lane % 16;
      if (row < kGroup)
        o_base[row * kHd + n0 + column] = f32_to_bf16bits(o_acc[ti].x[qt]);
    }
  }

  float* l_base = lse_partial + (b * n_qo + kvh * kGroup) * split + sp;
  if ((lane % 16) == 0) {
    int g = lane / 16;
    if (g < 4) {
      l_base[g * split] = lse_r[g];
      l_base[(g + 4) * split] = lse_r[g + 4];
    }
  }
}


// Fused FD: one block per (b,kvh) with `split` warps; combine in smem → bf16 out.
template <int VMODE, int SPLIT>
__global__ void ivcore_u8_decode_fused_kernel(
    const uint16_t* __restrict__ q,
    const uint8_t* __restrict__ k_u8,
    const uint8_t* __restrict__ v_u8,
    const float* __restrict__ k_scales,
    const float* __restrict__ v_scales,
    const int32_t* __restrict__ block_table,
    const int32_t* __restrict__ seqused,
    uint16_t* __restrict__ out,
    int batch, int n_qo, int n_kv, int /*max_pages*/, int stride_bt,
    float scale_log2) {
  const int lane = threadIdx.x & 63;
  const int sp = threadIdx.x >> 6;  // warp id = split index
  const int pid = static_cast<int>(blockIdx.x);
  const int kvh = pid % n_kv;
  const int b = pid / n_kv;

  const int seqlen = seqused[b];
  const int npages = (seqlen + kPage - 1) / kPage;
  const int chunk = (npages + SPLIT - 1) / SPLIT;
  int p0 = sp * chunk;
  int p1 = p0 + chunk;
  if (p1 > npages) p1 = npages;

  const uint16_t* q_base = q + b * n_qo * kHd + kvh * kGroup * kHd;
  fragment<matrix_a, 16, 16, 32, half, row_major> aq[4];
#pragma unroll
  for (int t = 0; t < 4; ++t) load_a_q(aq[t], q_base, t * 32);

  fragment<accumulator, 16, 16, 32, float> o_acc[kNTiles];
#pragma unroll
  for (int t = 0; t < kNTiles; ++t) fill_fragment(o_acc[t], 0.f);

  float lse_r[kGroup];
#pragma unroll
  for (int i = 0; i < kGroup; ++i) lse_r[i] = 0.f;

  fragment<matrix_b, 16, 16, 32, half, col_major> bk, bv;
  fragment<matrix_a, 16, 16, 32, half, row_major> fa;
  fragment<accumulator, 16, 16, 32, float> cq, cp;

  const size_t k_stride = static_cast<size_t>(n_kv) * kHd * kPage;
  const size_t v_stride = static_cast<size_t>(n_kv) * kPage * kHd;
  const size_t s_stride = static_cast<size_t>(n_kv) * kPage;

  if (p0 < p1) {
    for (int p = p0; p < p1; ++p) {
      const int page_id = block_table[b * stride_bt + p];
      const uint8_t* kptr =
          k_u8 + static_cast<size_t>(page_id) * k_stride + kvh * (kHd * kPage);
      const uint8_t* vptr =
          v_u8 + static_cast<size_t>(page_id) * v_stride + kvh * (kPage * kHd);
      const float* ksptr =
          k_scales + static_cast<size_t>(page_id) * s_stride + kvh * kPage;
      const float* sptr = nullptr;
      if constexpr (VMODE == kVI8) {
        sptr = v_scales + static_cast<size_t>(page_id) * s_stride + kvh * kPage;
      }
      const int valid_n =
          (p == npages - 1) ? (seqlen - p * kPage) : kPage;

      fill_fragment(cq, 0.f);
#pragma unroll
      for (int t = 0; t < 4; ++t) {
        load_b_k_i8(bk, kptr, t * 32);
        mma_bf16(cq, aq[t], bk, cq);
      }

      int col = lane % 16;
      int r0 = lane / 16, r1 = r0 + 4;
      float e0 = 0.f, e1 = 0.f;
      const float ks = (col < kPage) ? ksptr[col] : 0.f;
      if (col < valid_n) {
        if (r0 < kGroup) e0 = exp2f(fminf(cq.x[0] * ks * scale_log2, 80.f));
        if (r1 < kGroup) e1 = exp2f(fminf(cq.x[1] * ks * scale_log2, 80.f));
      }
      float x0 = e0, x1 = e1;
#pragma unroll
      for (int m = 8; m >= 1; m >>= 1) {
        x0 += __shfl_xor_sync(0xffffffffu, x0, m);
        x1 += __shfl_xor_sync(0xffffffffu, x1, m);
      }
      if ((lane % 16) == 0) {
        lse_r[r0] += x0;
        lse_r[r1] += x1;
      }
      if constexpr (VMODE == kVI8) {
        float s = (col < kPage) ? sptr[col] : 0.f;
        pack_ap_g8(fa, e0 * s, e1 * s);
      } else {
        pack_ap_g8(fa, e0, e1);
      }

#pragma unroll
      for (int ti = 0; ti < kNTiles; ++ti) {
        fill_fragment(cp, 0.f);
        if constexpr (VMODE == kVHiByte) {
          load_b_v_hibyte(bv, vptr, ti * 16);
        } else if constexpr (VMODE == kVFp8) {
          load_b_v_fp8(bv, vptr, ti * 16);
        } else {
          load_b_v_i8(bv, vptr, ti * 16);
        }
        mma_bf16(cp, fa, bv, cp);
#pragma unroll
        for (int i = 0; i < 4; ++i) o_acc[ti].x[i] += cp.x[i];
      }
    }
  }

  // smem: [SPLIT][kGroup][kHd] outs + [SPLIT][kGroup] lse
  extern __shared__ float sm[];
  float* s_out = sm;                         // SPLIT * kGroup * kHd
  float* s_lse = sm + SPLIT * kGroup * kHd;  // SPLIT * kGroup

#pragma unroll
  for (int ti = 0; ti < kNTiles; ++ti) {
    const int n0 = ti * 16;
#pragma unroll
    for (int qt = 0; qt < 4; ++qt) {
      int row = lane / 16 + 4 * qt;
      int column = lane % 16;
      if (row < kGroup) {
        s_out[((sp * kGroup + row) * kHd) + n0 + column] = o_acc[ti].x[qt];
      }
    }
  }
  if ((lane % 16) == 0) {
    int g = lane / 16;
    if (g < 4) {
      s_lse[sp * kGroup + g] = lse_r[g];
      s_lse[sp * kGroup + g + 4] = lse_r[g + 4];
    }
  }
  __syncthreads();

  // All warps help combine → bf16 out for this kv group.
  uint16_t* o_base = out + (b * n_qo + kvh * kGroup) * kHd;
  for (int idx = threadIdx.x; idx < kGroup * kHd; idx += blockDim.x) {
    const int row = idx / kHd;
    const int col = idx - row * kHd;
    float denom = 0.f;
#pragma unroll
    for (int s = 0; s < SPLIT; ++s) denom += s_lse[s * kGroup + row];
    const float inv = (denom > 0.f) ? __ivcorex_rcpf(denom) : 0.f;
    float a = 0.f;
#pragma unroll
    for (int s = 0; s < SPLIT; ++s)
      a += s_out[(s * kGroup + row) * kHd + col];
    o_base[row * kHd + col] = f32_to_bf16bits(a * inv);
  }
}

__global__ void ivcore_combine_kernel(
    const uint16_t* __restrict__ out_partial,
    const float* __restrict__ lse_partial,
    uint16_t* __restrict__ out,
    int batch, int n_qo, int split) {
  const int tid = threadIdx.x;
  const int idx = static_cast<int>(blockIdx.x);
  const int b = idx / n_qo;
  const int qh = idx - b * n_qo;
  float denom = 0.f;
#pragma unroll 1
  for (int s = 0; s < split; ++s)
    denom += lse_partial[(b * n_qo + qh) * split + s];
  const float inv = (denom > 0.f) ? __ivcorex_rcpf(denom) : 0.f;
  uint16_t* o = out + (b * n_qo + qh) * kHd;
  // 4-wide accumulate; bf16 partials → f32 via shift.
  for (int c = tid * 4; c < kHd; c += blockDim.x * 4) {
    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
#pragma unroll 1
    for (int s = 0; s < split; ++s) {
      const uint16_t* p =
          out_partial + ((s * batch + b) * n_qo + qh) * kHd + c;
      a0 += bf16bits_to_f32(p[0]);
      a1 += bf16bits_to_f32(p[1]);
      a2 += bf16bits_to_f32(p[2]);
      a3 += bf16bits_to_f32(p[3]);
    }
    o[c + 0] = f32_to_bf16bits(a0 * inv);
    o[c + 1] = f32_to_bf16bits(a1 * inv);
    o[c + 2] = f32_to_bf16bits(a2 * inv);
    o[c + 3] = f32_to_bf16bits(a3 * inv);
  }
}

extern "C" void launch_ivcore_u8_decode(
    const uint16_t* q, const uint8_t* k_u8, const uint8_t* v_u8,
    const float* k_scales, const float* v_scales, int v_mode,
    const int32_t* block_table, const int32_t* seqused,
    uint16_t* out_partial, float* lse_partial,
    int batch, int n_qo, int n_kv, int max_pages, int stride_bt,
    float scale_log2, int split, int group, int use_sme,
    cudaStream_t stream) {
  (void)group;
  (void)max_pages;
  dim3 grid(batch * n_kv * split);
  if (v_mode == kVFp8) {
    ivcore_u8_decode_kernel<kVFp8, 0><<<grid, kWarp, 0, stream>>>(
        q, k_u8, v_u8, k_scales, v_scales, block_table, seqused, out_partial,
        lse_partial, batch, n_qo, n_kv, max_pages, stride_bt, scale_log2, split);
  } else if (v_mode == kVI8) {
    if (use_sme) {
      ivcore_u8_decode_kernel<kVI8, 1><<<grid, kWarp, 0, stream>>>(
          q, k_u8, v_u8, k_scales, v_scales, block_table, seqused, out_partial,
          lse_partial, batch, n_qo, n_kv, max_pages, stride_bt, scale_log2,
          split);
    } else {
      ivcore_u8_decode_kernel<kVI8, 0><<<grid, kWarp, 0, stream>>>(
          q, k_u8, v_u8, k_scales, v_scales, block_table, seqused, out_partial,
          lse_partial, batch, n_qo, n_kv, max_pages, stride_bt, scale_log2,
          split);
    }
  } else {
    ivcore_u8_decode_kernel<kVHiByte, 0><<<grid, kWarp, 0, stream>>>(
        q, k_u8, v_u8, k_scales, v_scales, block_table, seqused, out_partial,
        lse_partial, batch, n_qo, n_kv, max_pages, stride_bt, scale_log2, split);
  }
}

extern "C" void launch_ivcore_combine(
    const uint16_t* out_partial, const float* lse_partial, uint16_t* out,
    int batch, int n_qo, int split, cudaStream_t stream) {
  ivcore_combine_kernel<<<batch * n_qo, 64, 0, stream>>>(
      out_partial, lse_partial, out, batch, n_qo, split);
}

// ---------------------------------------------------------------------------
// Graph-safe pack v46: grid=(batch, n_kv) loops real pages (no max_model_len
// empty-CTA storm). Vectorized bf16 loads + warp shfl amax + __ivcorex_rcpf.
// ---------------------------------------------------------------------------
__device__ __forceinline__ float block_amax_hd(float v, int tid, float* red) {
  red[tid] = v;
  __syncthreads();
#pragma unroll
  for (int s = 64; s > 0; s >>= 1) {
    if (tid < s) {
      float o = red[tid + s];
      if (o > red[tid]) red[tid] = o;
    }
    __syncthreads();
  }
  return red[0];
}

__device__ __forceinline__ int quant_i8(float x) {
  int q = __float2int_rn(x);
  return max(-127, min(127, q));
}

// Phys K/V [PAGE, HD]: thread tid owns dim tid, walks PAGE tokens.
__device__ __forceinline__ void pack_page_kv_pagescale(
    const uint16_t* __restrict__ K, const uint16_t* __restrict__ V,
    uint8_t* __restrict__ KU, uint8_t* __restrict__ VU,
    float* __restrict__ KSC, float* __restrict__ VSC,
    int tid, float* red) {
  float amax_k = 0.f, amax_v = 0.f;
  float k_f[kPage], v_f[kPage];
  if (tid < kHd) {
#pragma unroll
    for (int t = 0; t < kPage; ++t) {
      k_f[t] = bf16bits_to_f32(K[t * kHd + tid]);
      v_f[t] = bf16bits_to_f32(V[t * kHd + tid]);
      amax_k = fmaxf(amax_k, fabsf(k_f[t]));
      amax_v = fmaxf(amax_v, fabsf(v_f[t]));
    }
  }
  float sk = block_amax_hd(amax_k, tid, red) * (1.f / 127.f);
  if (sk < 1e-8f) sk = 1e-8f;
  float sv = block_amax_hd(amax_v, tid, red) * (1.f / 127.f);
  if (sv < 1e-8f) sv = 1e-8f;

  if (tid < kPage) {
    KSC[tid] = sk;
    VSC[tid] = sv;
  }
  const float inv_k = __ivcorex_rcpf(sk);
  const float inv_v = __ivcorex_rcpf(sv);
  if (tid < kHd) {
#pragma unroll
    for (int t = 0; t < kPage; ++t) {
      KU[tid * kPage + t] =
          static_cast<uint8_t>(static_cast<int8_t>(quant_i8(k_f[t] * inv_k)));
      VU[t * kHd + tid] =
          static_cast<uint8_t>(static_cast<int8_t>(quant_i8(v_f[t] * inv_v)));
    }
  }
}

// Asym: K HD-major [HD,PAGE] contiguous — load row as 8×u32 (16 bf16).
__device__ __forceinline__ void pack_page_kv_pagescale_asym(
    const uint16_t* __restrict__ K, const uint16_t* __restrict__ V,
    uint8_t* __restrict__ KU, uint8_t* __restrict__ VU,
    float* __restrict__ KSC, float* __restrict__ VSC,
    int v_slot_stride, int tid, float* red) {
  float amax_k = 0.f, amax_v = 0.f;
  float k_f[kPage], v_f[kPage];
  if (tid < kHd) {
    const uint32_t* kr =
        reinterpret_cast<const uint32_t*>(K + tid * kPage);
    uint32_t kw[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) kw[i] = kr[i];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      k_f[2 * i] = bf16bits_to_f32(static_cast<uint16_t>(kw[i] & 0xffffu));
      k_f[2 * i + 1] =
          bf16bits_to_f32(static_cast<uint16_t>((kw[i] >> 16) & 0xffffu));
      amax_k = fmaxf(amax_k, fabsf(k_f[2 * i]));
      amax_k = fmaxf(amax_k, fabsf(k_f[2 * i + 1]));
    }
#pragma unroll
    for (int t = 0; t < kPage; ++t) {
      v_f[t] = bf16bits_to_f32(V[t * v_slot_stride + tid]);
      amax_v = fmaxf(amax_v, fabsf(v_f[t]));
    }
  }
  float sk = block_amax_hd(amax_k, tid, red) * (1.f / 127.f);
  if (sk < 1e-8f) sk = 1e-8f;
  float sv = block_amax_hd(amax_v, tid, red) * (1.f / 127.f);
  if (sv < 1e-8f) sv = 1e-8f;

  if (tid < kPage) {
    KSC[tid] = sk;
    VSC[tid] = sv;
  }
  const float inv_k = __ivcorex_rcpf(sk);
  const float inv_v = __ivcorex_rcpf(sv);
  if (tid < kHd) {
    uint32_t out_w[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      int q0 = quant_i8(k_f[4 * i] * inv_k);
      int q1 = quant_i8(k_f[4 * i + 1] * inv_k);
      int q2 = quant_i8(k_f[4 * i + 2] * inv_k);
      int q3 = quant_i8(k_f[4 * i + 3] * inv_k);
      out_w[i] = (static_cast<uint32_t>(static_cast<uint8_t>(q0))) |
                 (static_cast<uint32_t>(static_cast<uint8_t>(q1)) << 8) |
                 (static_cast<uint32_t>(static_cast<uint8_t>(q2)) << 16) |
                 (static_cast<uint32_t>(static_cast<uint8_t>(q3)) << 24);
    }
    uint32_t* kur = reinterpret_cast<uint32_t*>(KU + tid * kPage);
#pragma unroll
    for (int i = 0; i < 4; ++i) kur[i] = out_w[i];
#pragma unroll
    for (int t = 0; t < kPage; ++t) {
      VU[t * kHd + tid] = static_cast<uint8_t>(
          static_cast<int8_t>(quant_i8(v_f[t] * inv_v)));
    }
  }
}

__global__ void ivcore_pack_bt_kernel(
    const uint16_t* __restrict__ k_phys,  // [blocks, n_kv, PAGE, HD]
    const uint16_t* __restrict__ v_phys,
    uint8_t* __restrict__ ku8,
    uint8_t* __restrict__ vu8,
    float* __restrict__ ksc,
    float* __restrict__ vsc,
    const int32_t* __restrict__ block_table,
    const int32_t* __restrict__ seqused,
    int batch, int max_pages, int stride_bt, int n_kv, int n_blocks,
    int last_page_only) {
  // Grid=(batch, n_kv); each CTA walks that seq's pages (or only the last).
  const int b = static_cast<int>(blockIdx.x);
  const int h = static_cast<int>(blockIdx.y);
  if (b >= batch || h >= n_kv) return;

  int pages = (seqused[b] + kPage - 1) / kPage;
  if (pages > max_pages) pages = max_pages;
  if (pages <= 0) return;

  const int tid = threadIdx.x;
  __shared__ float red[128];
  const int p0 = last_page_only ? (pages - 1) : 0;
  for (int p = p0; p < pages; ++p) {
    const int page_id = block_table[b * stride_bt + p];
    if (page_id < 0 || page_id >= n_blocks) continue;
    const int64_t base =
        (static_cast<int64_t>(page_id) * n_kv + h) * kPage * kHd;
    pack_page_kv_pagescale(
        k_phys + base, v_phys + base,
        ku8 + (static_cast<int64_t>(page_id) * n_kv + h) * kHd * kPage,
        vu8 + base,
        ksc + (static_cast<int64_t>(page_id) * n_kv + h) * kPage,
        vsc + (static_cast<int64_t>(page_id) * n_kv + h) * kPage,
        tid, red);
  }
}

extern "C" void launch_ivcore_pack_bt(
    const uint16_t* k_phys, const uint16_t* v_phys,
    uint8_t* ku8, uint8_t* vu8, float* ksc, float* vsc,
    const int32_t* block_table, const int32_t* seqused,
    int batch, int max_pages, int stride_bt, int n_kv, int n_blocks,
    cudaStream_t stream) {
  dim3 grid(batch, n_kv);
  ivcore_pack_bt_kernel<<<grid, 128, 0, stream>>>(
      k_phys, v_phys, ku8, vu8, ksc, vsc, block_table, seqused,
      batch, max_pages, stride_bt, n_kv, n_blocks, /*last_page_only=*/0);
}

extern "C" void launch_ivcore_pack_bt_last(
    const uint16_t* k_phys, const uint16_t* v_phys,
    uint8_t* ku8, uint8_t* vu8, float* ksc, float* vsc,
    const int32_t* block_table, const int32_t* seqused,
    int batch, int max_pages, int stride_bt, int n_kv, int n_blocks,
    cudaStream_t stream) {
  dim3 grid(batch, n_kv);
  ivcore_pack_bt_kernel<<<grid, 128, 0, stream>>>(
      k_phys, v_phys, ku8, vu8, ksc, vsc, block_table, seqused,
      batch, max_pages, stride_bt, n_kv, n_blocks, /*last_page_only=*/1);
}

__global__ void ivcore_pack_bt_asym_kernel(
    const uint16_t* __restrict__ k_hd,
    const uint16_t* __restrict__ v_bndh,
    uint8_t* __restrict__ ku8,
    uint8_t* __restrict__ vu8,
    float* __restrict__ ksc,
    float* __restrict__ vsc,
    const int32_t* __restrict__ block_table,
    const int32_t* __restrict__ seqused,
    int batch, int max_pages, int stride_bt, int n_kv, int n_blocks,
    int last_page_only) {
  const int b = static_cast<int>(blockIdx.x);
  const int h = static_cast<int>(blockIdx.y);
  if (b >= batch || h >= n_kv) return;

  int pages = (seqused[b] + kPage - 1) / kPage;
  if (pages > max_pages) pages = max_pages;
  if (pages <= 0) return;

  const int tid = threadIdx.x;
  __shared__ float red[128];
  const int v_slot_stride = n_kv * kHd;
  const int p0 = last_page_only ? (pages - 1) : 0;
  for (int p = p0; p < pages; ++p) {
    const int page_id = block_table[b * stride_bt + p];
    if (page_id < 0 || page_id >= n_blocks) continue;
    const int64_t k_base =
        (static_cast<int64_t>(page_id) * n_kv + h) * kHd * kPage;
    const int64_t v_base =
        (static_cast<int64_t>(page_id) * kPage * n_kv + h) * kHd;
    const int64_t vu_base =
        (static_cast<int64_t>(page_id) * n_kv + h) * kPage * kHd;
    pack_page_kv_pagescale_asym(
        k_hd + k_base, v_bndh + v_base,
        ku8 + k_base, vu8 + vu_base,
        ksc + (static_cast<int64_t>(page_id) * n_kv + h) * kPage,
        vsc + (static_cast<int64_t>(page_id) * n_kv + h) * kPage,
        v_slot_stride, tid, red);
  }
}

extern "C" void launch_ivcore_pack_bt_asym(
    const uint16_t* k_hd, const uint16_t* v_bndh,
    uint8_t* ku8, uint8_t* vu8, float* ksc, float* vsc,
    const int32_t* block_table, const int32_t* seqused,
    int batch, int max_pages, int stride_bt, int n_kv, int n_blocks,
    cudaStream_t stream) {
  dim3 grid(batch, n_kv);
  ivcore_pack_bt_asym_kernel<<<grid, 128, 0, stream>>>(
      k_hd, v_bndh, ku8, vu8, ksc, vsc, block_table, seqused,
      batch, max_pages, stride_bt, n_kv, n_blocks, /*last_page_only=*/0);
}

extern "C" void launch_ivcore_pack_bt_asym_last(
    const uint16_t* k_hd, const uint16_t* v_bndh,
    uint8_t* ku8, uint8_t* vu8, float* ksc, float* vsc,
    const int32_t* block_table, const int32_t* seqused,
    int batch, int max_pages, int stride_bt, int n_kv, int n_blocks,
    cudaStream_t stream) {
  dim3 grid(batch, n_kv);
  ivcore_pack_bt_asym_kernel<<<grid, 128, 0, stream>>>(
      k_hd, v_bndh, ku8, vu8, ksc, vsc, block_table, seqused,
      batch, max_pages, stride_bt, n_kv, n_blocks, /*last_page_only=*/1);
}
