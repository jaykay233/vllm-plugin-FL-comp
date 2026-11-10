#include <mc_runtime.h>
#include <cstdint>
typedef __NATIVE_VECTOR__(2, uint32_t) v2u;
// Emit three candidate lane-expansions of a 4-byte e4m3 word so the selector
// byte order can be settled by inspection instead of guesswork.
__global__ void perm_probe(const uint32_t* in, uint32_t* out) {
  uint32_t bq = in[0];
  out[0] = __builtin_mxc_byte_perm(bq, 0u, 0x4140);
  out[1] = __builtin_mxc_byte_perm(bq, 0u, 0x0414);
  out[2] = __builtin_mxc_byte_perm(bq, 0u, 0x0011);
  out[3] = __builtin_mxc_byte_perm(bq, 0u, 0x1100);
}
extern "C" int launch_perm_probe(const uint32_t* in, uint32_t* out, void* s) {
  perm_probe<<<1, 32, 0, (mcStream_t)s>>>(in, out);
  return 0;
}
