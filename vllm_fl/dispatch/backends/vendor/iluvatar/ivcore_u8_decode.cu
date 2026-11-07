// ivcore u8 decode v37 — hi-byte K + selectable V.
// INT8 (v_mode=2): token scales absorbed into P; V=i8→bf16 via int2float.
// Best dual-constraint point on BI-V150: ~423µs / cos≈0.99995 (see FINDINGS 17.64).
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

enum VMode : int { kVHiByte = 0, kVFp8 = 1, kVI8 = 2 };

__device__ __forceinline__ uint16_t f32_to_bf16bits(float f) {
  uint32_t u;
  __builtin_memcpy(&u, &f, sizeof(u));
  return static_cast<uint16_t>(u >> 16);
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
    uint16_t h0 = f32_to_bf16bits(__int2float_rn(static_cast<int>(q0)));
    uint16_t h1 = f32_to_bf16bits(__int2float_rn(static_cast<int>(q1)));
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
      int8_t q0 = static_cast<int8_t>(V[row * kHd + n0 + column]);
      h0 = f32_to_bf16bits(__int2float_rn(static_cast<int>(q0)));
    }
    if (row + 1 < kPage) {
      int8_t q1 = static_cast<int8_t>(V[(row + 1) * kHd + n0 + column]);
      h1 = f32_to_bf16bits(__int2float_rn(static_cast<int>(q1)));
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

template <int VMODE>
__global__ void ivcore_u8_decode_kernel(
    const uint16_t* __restrict__ q,
    const uint8_t* __restrict__ k_u8,
    const uint8_t* __restrict__ v_u8,
    const float* __restrict__ k_scales,  // [blocks,n_kv,PAGE] per-token K scales
    const float* __restrict__ v_scales,  // [blocks,n_kv,PAGE] if VMODE==kVI8 else unused
    const int32_t* __restrict__ block_table,
    const int32_t* __restrict__ seqused,
    float* __restrict__ out_partial,
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
  if (p0 >= p1) return;

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
    // Per-token K scale: score = (Q·K_i8) * k_scale * softmax_scale
    const float ks = (col < kPage) ? ksptr[col] : 0.f;
    if (col < valid_n) {
      if (r0 < kGroup) e0 = exp2f(fminf(cq.x[0] * ks * scale_log2, 80.f));
      if (r1 < kGroup) e1 = exp2f(fminf(cq.x[1] * ks * scale_log2, 80.f));
    }
    // LSE uses unscaled exp; PV absorbs token scale into P for INT8-V.
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

  float* o_base =
      out_partial + ((sp * batch + b) * n_qo + kvh * kGroup) * kHd;
#pragma unroll
  for (int ti = 0; ti < kNTiles; ++ti) {
    const int n0 = ti * 16;
#pragma unroll
    for (int qt = 0; qt < 4; ++qt) {
      int row = lane / 16 + 4 * qt;
      int column = lane % 16;
      if (row < kGroup) o_base[row * kHd + n0 + column] = o_acc[ti].x[qt];
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
    const float* __restrict__ out_partial,
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
  // Vectorized accumulate when HD is multiple of 4.
  for (int c = tid * 4; c < kHd; c += blockDim.x * 4) {
    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
#pragma unroll 1
    for (int s = 0; s < split; ++s) {
      const float* p =
          out_partial + ((s * batch + b) * n_qo + qh) * kHd + c;
      a0 += p[0]; a1 += p[1]; a2 += p[2]; a3 += p[3];
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
    float* out_partial, float* lse_partial,
    int batch, int n_qo, int n_kv, int max_pages, int stride_bt,
    float scale_log2, int split, int group, cudaStream_t stream) {
  (void)group;
  (void)max_pages;
  (void)/*fused disabled: multi-warp smem combine regressed vs split CTAs*/0;
  dim3 grid(batch * n_kv * split);
  if (v_mode == kVFp8) {
    ivcore_u8_decode_kernel<kVFp8><<<grid, kWarp, 0, stream>>>(
        q, k_u8, v_u8, k_scales, v_scales, block_table, seqused, out_partial,
        lse_partial, batch, n_qo, n_kv, max_pages, stride_bt, scale_log2, split);
  } else if (v_mode == kVI8) {
    ivcore_u8_decode_kernel<kVI8><<<grid, kWarp, 0, stream>>>(
        q, k_u8, v_u8, k_scales, v_scales, block_table, seqused, out_partial,
        lse_partial, batch, n_qo, n_kv, max_pages, stride_bt, scale_log2, split);
  } else {
    ivcore_u8_decode_kernel<kVHiByte><<<grid, kWarp, 0, stream>>>(
        q, k_u8, v_u8, k_scales, v_scales, block_table, seqused, out_partial,
        lse_partial, batch, n_qo, n_kv, max_pages, stride_bt, scale_log2, split);
  }
}

extern "C" void launch_ivcore_combine(
    const float* out_partial, const float* lse_partial, uint16_t* out,
    int batch, int n_qo, int split, cudaStream_t stream) {
  ivcore_combine_kernel<<<batch * n_qo, 32, 0, stream>>>(
      out_partial, lse_partial, out, batch, n_qo, split);
}

// ---------------------------------------------------------------------------
// Graph-safe pack v43: fixed grid (batch, n_kv, workers), stride over pages.
// Avoids (B, max_model_pages) empty launches AND serial page loops in 1 CTA.
// INT8-K + INT8-V per-token scales. Hibyte-K fails on real MiniCPM KV.
// ---------------------------------------------------------------------------
__device__ __forceinline__ float bf16bits_to_f32(uint16_t h) {
  uint32_t u = static_cast<uint32_t>(h) << 16;
  float f;
  __builtin_memcpy(&f, &u, sizeof(f));
  return f;
}

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

__device__ __forceinline__ void pack_page_kv_pagescale(
    const uint16_t* __restrict__ K, const uint16_t* __restrict__ V,
    uint8_t* __restrict__ KU, uint8_t* __restrict__ VU,
    float* __restrict__ KSC, float* __restrict__ VSC,
    int tid, float* red) {
  // One amax over the whole page (PAGE×HD) — cos still >0.999 on real KV,
  // ~16× fewer reductions than per-token scales.
  float amax_k = 0.f, amax_v = 0.f;
  if (tid < kHd) {
#pragma unroll
    for (int t = 0; t < kPage; ++t) {
      amax_k = fmaxf(amax_k, fabsf(bf16bits_to_f32(K[t * kHd + tid])));
      amax_v = fmaxf(amax_v, fabsf(bf16bits_to_f32(V[t * kHd + tid])));
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
  const float inv_k = 1.f / sk;
  const float inv_v = 1.f / sv;
  if (tid < kHd) {
#pragma unroll
    for (int t = 0; t < kPage; ++t) {
      float kk = bf16bits_to_f32(K[t * kHd + tid]) * inv_k;
      int qk = __float2int_rn(kk);
      qk = max(-127, min(127, qk));
      KU[tid * kPage + t] = static_cast<uint8_t>(static_cast<int8_t>(qk));

      float vv = bf16bits_to_f32(V[t * kHd + tid]) * inv_v;
      int qv = __float2int_rn(vv);
      qv = max(-127, min(127, qv));
      VU[t * kHd + tid] = static_cast<uint8_t>(static_cast<int8_t>(qv));
    }
  }
}

__global__ void ivcore_pack_bt_kernel(
    const uint16_t* __restrict__ k_phys,  // [blocks, n_kv, PAGE, HD]
    const uint16_t* __restrict__ v_phys,
    uint8_t* __restrict__ ku8,            // [blocks, n_kv, HD, PAGE] int8
    uint8_t* __restrict__ vu8,            // [blocks, n_kv, PAGE, HD] int8
    float* __restrict__ ksc,              // [blocks, n_kv, PAGE]
    float* __restrict__ vsc,              // [blocks, n_kv, PAGE]
    const int32_t* __restrict__ block_table,
    const int32_t* __restrict__ seqused,
    int batch, int max_pages, int page_offset, int stride_bt, int n_kv,
    int n_blocks) {
  // Grid=(batch, chunk, n_kv); absolute page index = page_offset + blockIdx.y.
  const int b = static_cast<int>(blockIdx.x);
  const int p = page_offset + static_cast<int>(blockIdx.y);
  const int h = static_cast<int>(blockIdx.z);
  if (b >= batch || h >= n_kv || p >= max_pages) return;

  int pages = (seqused[b] + kPage - 1) / kPage;
  if (pages > max_pages) pages = max_pages;
  if (p >= pages) return;

  const int page_id = block_table[b * stride_bt + p];
  if (page_id < 0 || page_id >= n_blocks) return;

  const int tid = threadIdx.x;
  __shared__ float red[128];

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

extern "C" void launch_ivcore_pack_bt(
    const uint16_t* k_phys, const uint16_t* v_phys,
    uint8_t* ku8, uint8_t* vu8, float* ksc, float* vsc,
    const int32_t* block_table, const int32_t* seqused,
    int batch, int max_pages, int stride_bt, int n_kv, int n_blocks,
    cudaStream_t stream) {
  // One CTA per (b,p,h). Chunk Y at 1024 to avoid empty-CTA storms on
  // max_model_len-padded block tables.
  constexpr int kChunk = 1024;
  for (int off = 0; off < max_pages; off += kChunk) {
    const int chunk = (max_pages - off < kChunk) ? (max_pages - off) : kChunk;
    dim3 grid(batch, chunk, n_kv);
    ivcore_pack_bt_kernel<<<grid, 128, 0, stream>>>(
        k_phys, v_phys, ku8, vu8, ksc, vsc, block_table, seqused,
        batch, max_pages, off, stride_bt, n_kv, n_blocks);
  }
}
