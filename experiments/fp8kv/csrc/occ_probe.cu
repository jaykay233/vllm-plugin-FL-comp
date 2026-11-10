// How many of our 1-warp CTAs are actually resident per SM?
//
// Every guess about "more warps will help" hinges on this, and the indirect
// evidence is contradictory: split-K=32 launches 4096 CTAs without getting any
// faster, yet copy_bw.cu shows the same engine reaching 1640+ GB/s once eight
// warps per SM are in flight.  So measure the residency directly: each CTA
// bumps a live counter on entry, records the peak and drops it on exit.
//
// Run with the kernel's real shared-memory footprint (kSmem bytes) so the
// answer applies to the actual kernel.

#include <mc_runtime.h>
#include <cstdint>

extern "C" __global__ void occ_probe(int smem_bytes, int* live, int* peak) {
  // Only touch dynamic shared memory when it was actually requested; with a
  // zero-byte allocation the array has no storage and writes trap.
  if (smem_bytes > 0) {
    extern __shared__ uint8_t smem[];
    const int lane = threadIdx.x & 63;
    smem[lane * 16] = (uint8_t)lane;
  }

  const int now = atomicAdd(live, 1) + 1;
  atomicMax(peak, now);

  // Burn a little time so the CTAs really overlap rather than being pipelined
  // through one at a time.
  long long t0 = clock64();
  while (clock64() - t0 < 20000) {
  }

  atomicSub(live, 1);
}

extern "C" int launch_occ(int smem_bytes, int ctas, int warps, void* live,
                          void* peak, void* stream) {
  if (smem_bytes > 0)
    occ_probe<<<ctas, 64 * warps, smem_bytes, (mcStream_t)stream>>>(
        smem_bytes, (int*)live, (int*)peak);
  else
    occ_probe<<<ctas, 64 * warps, 0, (mcStream_t)stream>>>(
        smem_bytes, (int*)live, (int*)peak);
  return 0;
}
