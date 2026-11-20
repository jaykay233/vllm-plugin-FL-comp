"""int8-KV port of the tuned decode kernel, driven through the known-good harness.

Imports fp8_decode_run.py (which works) and swaps in libint8_decode.so, int8 KV
tensors, and the folded per-tensor KV scale.
"""
import ctypes
import importlib.util
import torch

BASE = "/workspace/vllm-plugin-FL/experiments/fp8kv/csrc/"
spec = importlib.util.spec_from_file_location("h", BASE + "fp8_decode_run.py")
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)

lib = ctypes.CDLL(BASE + "libint8_decode.so")
lib.launch_fp8_decode.argtypes = h.lib.launch_fp8_decode.argtypes
lib.launch_fp8_decode.restype = ctypes.c_int
h.lib = lib

D, HKV, H, B, SEQ, P = 128, 2, 16, 64, 4096, 64
sm = D ** -0.5
torch.manual_seed(0)

q = torch.randn(B, H, D, device="cuda", dtype=torch.bfloat16)
npg = (SEQ + P - 1) // P
nb = B * npg
kbf = torch.randn(nb, P, HKV, D, device="cuda", dtype=torch.bfloat16) * 0.5
vbf = torch.randn(nb, P, HKV, D, device="cuda", dtype=torch.bfloat16) * 0.5
bt = torch.arange(nb, device="cuda", dtype=torch.int32).reshape(B, npg)
lens = torch.full((B,), SEQ, device="cuda", dtype=torch.int32)

s = kbf.abs().amax().item() / 127.0
ki = (kbf.float() / s).round().clamp(-127, 127).to(torch.int8).contiguous()
vi = (vbf.float() / s).round().clamp(-127, 127).to(torch.int8).contiguous()
kdeq = (ki.float() * s)
vdeq = (vi.float() * s)
PAYLOAD = 2 * ki.numel()
print("int8 KV: %.1f MB/layer, per-tensor scale %.6f" % (PAYLOAD / 1e6, s))


class R8(h.Runner):
    def __init__(self, *a, ks=1.0, vs=1.0, **kw):
        super().__init__(*a, **kw)
        self.ks, self.vs = ks, vs

    def __call__(self):
        h.lib.launch_fp8_decode(
            self.q.data_ptr(), self.k.data_ptr(), self.v.data_ptr(),
            self.bt.data_ptr(), self.lens.data_ptr(), self.out.data_ptr(),
            self.mid_o.data_ptr(), self.mid_m.data_ptr(), self.mid_l.data_ptr(),
            self.B, self.H, self.HKV, self.P, self.bt.stride(0), self.S,
            self.sm, self.ks, self.vs, self.mode, None)
        return self.out


ref = h.reference(q[:8], kdeq, vdeq, bt[:8], lens[:8], sm)

# correctness at a few splits
for S in (8, 16):
    out = R8(q, ki, vi, bt, lens, S, 0, sm, ks=s, vs=s)()
    torch.cuda.synchronize()
    err = (out[:8].float() - ref.float()).abs().max().item()
    cos = torch.nn.functional.cosine_similarity(
        out[:8].float().flatten(), ref.float().flatten(), dim=0).item()
    print("  S=%2d correctness vs dequantised reference: maxabs=%.5f cos=%.6f"
          % (S, err, cos))

print("\nsplit-K sweep, int8 KV (mode 0 = bf16 tensor-core kernel):")
best = (1e9, 0)
for S in (1, 2, 4, 8, 16, 32):
    t = h.bench(R8(q, ki, vi, bt, lens, S, 0, sm, ks=s, vs=s))
    if t < best[0]:
        best = (t, S)
    print("  S=%2d  %8.4f ms/layer  %6.0f GB/s  x42=%6.2f ms/step"
          % (S, t, PAYLOAD / (t * 1e6), t * 42))
print("BEST int8: %.4f ms/layer at S=%d -> x42 = %.2f ms/step"
      % (best[0], best[1], best[0] * 42))

# fp8 reference (same tuned kernel, e4m3 KV) for a like-for-like comparison
libf = ctypes.CDLL(BASE + "libfp8_decode.so")
libf.launch_fp8_decode.argtypes = h.lib.launch_fp8_decode.argtypes
h.lib = libf
kf = kbf.to(torch.float8_e4m3fn)
vf = vbf.to(torch.float8_e4m3fn)
bestf = (1e9, 0)
for S in (4, 8, 16):
    t = h.bench(h.Runner(q, kf, vf, bt, lens, S, 0, sm))
    if t < bestf[0]:
        bestf = (t, S)
print("BEST fp8 KV (same kernel) : %.4f ms/layer at S=%d" % bestf)

from vllm_fl.dispatch.backends.vendor.metax.impl.attention.utils.fa_utils import (
    flash_attn_with_kvcache)
kc, vc, qq = kbf.contiguous(), vbf.contiguous(), q.unsqueeze(1)
tv = h.bench(lambda: flash_attn_with_kvcache(
    q=qq, k_cache=kc, v_cache=vc, block_table=bt, cache_seqlens=lens,
    softmax_scale=sm, causal=True, num_splits=0), iters=30)
print("vendor bf16 FA2           : %.4f ms/layer  x42 = %.2f ms/step" % (tv, tv * 42))
print("RATIO int8 / vendor       : %.2fx  (>1 means slower)" % (best[0] / tv))
