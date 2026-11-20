"""Decode-attention kernel benchmark: fp8-KV baseline vs int8-KV port.

Usage: python3 kv_run.py [fp8|int8]
Real shape: B=64, seq=4096, HKV=2, H=16, D=128, page=64.
"""
import ctypes
import sys
import torch

KIND = sys.argv[1] if len(sys.argv) > 1 else "int8"
BASE = "/workspace/vllm-plugin-FL/experiments/fp8kv/csrc/"
LIB = BASE + ("libint8_decode.so" if KIND == "int8" else "libfp8_decode.so")
lib = ctypes.CDLL(LIB)
lib.launch_fp8_decode.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_float,
    ctypes.c_int, ctypes.c_void_p]
lib.launch_fp8_decode.restype = ctypes.c_int

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

if KIND == "int8":
    scale = kbf.abs().amax().item() / 127.0
    ki = (kbf.float() / scale).round().clamp(-127, 127).to(torch.int8).contiguous()
    vi = (vbf.float() / scale).round().clamp(-127, 127).to(torch.int8).contiguous()
    kdeq, vdeq = ki.float() * scale, vi.float() * scale
    ks, vs = scale, scale
else:
    ki, vi = kbf.to(torch.float8_e4m3fn), vbf.to(torch.float8_e4m3fn)
    kdeq, vdeq = ki.float(), vi.float()
    ks, vs = 1.0, 1.0

PAYLOAD = 2 * ki.numel()          # K+V bytes per layer (1 byte/element)
print("%s KV: payload %.1f MB/layer, scale %.6f" % (KIND, PAYLOAD / 1e6, ks))


def reference(rows=4):
    g = H // HKV
    out = torch.empty(rows, H, D, device="cuda", dtype=torch.float32)
    for b in range(rows):
        n = int(lens[b])
        pages = bt[b, :(n + P - 1) // P].long()
        kk = kdeq[pages].reshape(-1, HKV, D)[:n]
        vv = vdeq[pages].reshape(-1, HKV, D)[:n]
        for kvh in range(HKV):
            qg = q[b, kvh * g:(kvh + 1) * g].float()
            p = torch.softmax((qg @ kk[:, kvh, :].t()) * sm, dim=-1)
            out[b, kvh * g:(kvh + 1) * g] = p @ vv[:, kvh, :]
    return out


def bench(fn, iters=30, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


best = (1e9, 0)
for S in (4, 8, 16, 32):
    mid_o = torch.empty((S, B, H, D), device="cuda", dtype=torch.float32)
    mid_m = torch.empty((S, B, H), device="cuda", dtype=torch.float32)
    mid_l = torch.empty((S, B, H), device="cuda", dtype=torch.float32)
    out = torch.empty_like(q)

    def call():
        lib.launch_fp8_decode(
            q.data_ptr(), ki.data_ptr(), vi.data_ptr(), bt.data_ptr(),
            lens.data_ptr(), out.data_ptr(), mid_o.data_ptr(),
            mid_m.data_ptr(), mid_l.data_ptr(), B, H, HKV, P, bt.stride(0),
            SEQ, sm, ks, vs, 0, None)
        return out

    call()
    torch.cuda.synchronize()
    ref = reference()
    err = (out[:4].float() - ref).abs().max().item()
    cos = torch.nn.functional.cosine_similarity(
        out[:4].float().flatten(), ref.flatten(), dim=0).item()
    ms = bench(call)
    if ms < best[0]:
        best = (ms, S)
    print("  splits=%2d  %7.1f us/layer  %6.0f GB/s  x42 = %5.2f ms/step  "
          "maxabs=%.5f cos=%.6f"
          % (S, ms * 1000, PAYLOAD / ms / 1e9 * 1e3, ms * 42, err, cos))

print("BEST %s: %.4f ms/layer at splits=%d -> x42 = %.2f ms/step"
      % (KIND, best[0], best[1], best[0] * 42))

from vllm_fl.dispatch.backends.vendor.metax.impl.attention.utils.fa_utils import (
    flash_attn_with_kvcache)
kc, vc, qq = kbf.contiguous(), vbf.contiguous(), q.unsqueeze(1)
t = bench(lambda: flash_attn_with_kvcache(
    q=qq, k_cache=kc, v_cache=vc, block_table=bt, cache_seqlens=lens,
    softmax_scale=sm, causal=True, num_splits=0))
print("vendor bf16: %.4f ms/layer  x42 = %.2f ms/step  ratio best/vendor = %.2fx"
      % (t, t * 42, best[0] / t))
