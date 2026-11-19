// Host launcher for the native split-KV decode attention kernel.
//
// The kernel itself is experiments/attn_kernel/decode_attn.h -- the same file the
// tuning harness compiles -- so this file only picks the geometry and the grid.
#include <cstdint>

#include "decode_attn.h"

namespace {

template <int W>
void launch_typed(const bf_t *q, const bf_t *k, const bf_t *v, bf_t *out, float *Op,
                  float *Mp, float *Lp, const int32_t *block_table,
                  const int32_t *seqused, const int32_t *qrow, int nunit, int S,
                  int maxblk, int NH, int NKVH, float log2scale, cudaStream_t stream) {
  const int ctas = (nunit * S + W - 1) / W;
  decode_split<W><<<ctas, W * 64, 0, stream>>>(q, k, v, Op, Mp, Lp, block_table, seqused,
                                               qrow, nunit, S, maxblk, NH, NKVH, log2scale);
  decode_merge<bf_t><<<nunit, 128, 0, stream>>>(Op, Mp, Lp, out, qrow, S, NH, NKVH);
}

}  // namespace

extern "C" void launch_native_decode(const uint16_t *q, const uint16_t *k,
                                     const uint16_t *v, uint16_t *out, float *Op,
                                     float *Mp, float *Lp, const int32_t *block_table,
                                     const int32_t *seqused, const int32_t *qrow,
                                     int nunit, int S, int maxblk, int NH, int NKVH,
                                     int W, float log2scale, cudaStream_t stream) {
  const bf_t *qb = reinterpret_cast<const bf_t *>(q);
  const bf_t *kb = reinterpret_cast<const bf_t *>(k);
  const bf_t *vb = reinterpret_cast<const bf_t *>(v);
  bf_t *ob = reinterpret_cast<bf_t *>(out);
  if (W == 4) {
    launch_typed<4>(qb, kb, vb, ob, Op, Mp, Lp, block_table, seqused, qrow, nunit, S, maxblk,
                    NH, NKVH, log2scale, stream);
  } else if (W == 16) {
    launch_typed<16>(qb, kb, vb, ob, Op, Mp, Lp, block_table, seqused, qrow, nunit, S, maxblk,
                     NH, NKVH, log2scale, stream);
  } else {
    launch_typed<8>(qb, kb, vb, ob, Op, Mp, Lp, block_table, seqused, qrow, nunit, S, maxblk,
                    NH, NKVH, log2scale, stream);
  }
}
