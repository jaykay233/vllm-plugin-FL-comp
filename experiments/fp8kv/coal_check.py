import torch, sys
sys.path.insert(0, "/workspace/vllm-plugin-FL-comp/experiments/fp8kv")
from fp8_gqa_decode import fp8_gqa_decode, _reference

dev = "cuda"
D, H, HKV, LAYERS = 128, 16, 2, 42

# correctness first (cheap)
Bc, SEQc, Pc = 4, 512, 64
q = torch.randn(Bc, H, D, device=dev, dtype=torch.bfloat16)
npg = (SEQc + Pc - 1) // Pc
tot = Bc * npg
kv = torch.randn(tot, Pc, HKV, D, device=dev, dtype=torch.bfloat16) * 0.5
bt = torch.arange(tot, device=dev, dtype=torch.int32).reshape(Bc, npg)
lens = torch.full((Bc,), SEQc, device=dev, dtype=torch.int32)
kf8 = kv.to(torch.float8_e4m3fn); vf8 = kf8.clone()
ref = _reference(q, kf8, vf8, bt, lens, D**-0.5, 1.0, 1.0)
got = fp8_gqa_decode(q, kf8, vf8, bt, lens, D**-0.5, num_kv_splits=8)
print("CORR fp8 vs ref maxabs=%.5f" % float((got.float()-ref.float()).abs().max()))

# one config for bf16 and fp8, at the shape that matters
B, SEQ, P = 64, 4096, 64
npg = SEQ // P
tot = B * npg
qb = torch.randn(B, H, D, device=dev, dtype=torch.bfloat16)
kv = torch.randn(tot, P, HKV, D, device=dev, dtype=torch.bfloat16) * 0.5
vb = torch.randn(tot, P, HKV, D, device=dev, dtype=torch.bfloat16) * 0.5
btb = torch.arange(tot, device=dev, dtype=torch.int32).reshape(B, npg)
lb = torch.full((B,), SEQ, device=dev, dtype=torch.int32)

def bench(fn, it=20):
    for _ in range(3): fn()
    torch.cuda.synchronize()
    s,e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(it): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e)/it

kb1 = B*SEQ*HKV*D*2
for tag, K, V in (("bf16", kv, vb), ("fp8 ", kv.to(torch.float8_e4m3fn), vb.to(torch.float8_e4m3fn))):
    for w, sp in ((1,16),(2,16),(4,8)):
        try:
            t = bench(lambda K_=K,V_=V,w_=w,s_=sp: fp8_gqa_decode(qb, K_, V_, btb, lb, D**-0.5, num_kv_splits=s_, num_warps=w_))
            print("COAL %s w=%d sp=%-2d %7.3f ms/layer %6.0f GB/s x42=%6.2f ms"
                  % (tag, w, sp, t, kb1/(t*1e6), t*LAYERS))
        except Exception as exc:
            print("COAL %s w=%d sp=%-2d FAIL %s" % (tag, w, sp, type(exc).__name__))

from flash_attn import flash_attn_with_kvcache
qd = qb.unsqueeze(1)
t = bench(lambda: flash_attn_with_kvcache(qd, kv, kv, block_table=btb, cache_seqlens=lb, causal=True))
print("COAL vendor FA bf16 %7.3f ms/layer %6.0f GB/s x42=%6.2f ms" % (t, (kb1*2)/(t*1e6), t*LAYERS))
