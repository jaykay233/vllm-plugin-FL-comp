import torch

dev = "cuda"
torch.manual_seed(0)


def bench(fn, iters=30, warmup=8):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(True)
    e = torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1000  # us


# (M, K, N, label)
SHAPES = [
    (64, 2048, 12288, "decode gate_up"),
    (64, 2048, 2560, "decode qkv"),
    (64, 2048, 2048, "decode o_proj"),
    (64, 6144, 2048, "decode down"),
    (64, 2048, 130560, "decode lm_head"),
    (16384, 2048, 12288, "prefill gate_up"),
    (16384, 2048, 2560, "prefill qkv"),
    (16384, 6144, 2048, "prefill down"),
]

print("%-18s %8s %8s %7s %9s %9s %7s" % (
    "shape", "bf16_us", "i8_us", "speedup", "bf16_TF", "i8_TF", "i8_GB/s"))
for M, K, N, label in SHAPES:
    flops = 2.0 * M * K * N
    a = torch.randn(M, K, device=dev, dtype=torch.bfloat16) * 0.5
    b = torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.05
    t_bf = bench(lambda: torch.nn.functional.linear(a, b))

    # int8: per-token act quant + per-channel weight quant (offline here)
    sa = a.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 127.0
    ai = (a / sa).round().clamp(-127, 127).to(torch.int8)
    sw = b.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 127.0
    bi = (b / sw).round().clamp(-127, 127).to(torch.int8)
    bt = bi.t().contiguous()

    def i8():
        o = torch._int_mm(ai, bt)
        return o

    t_i8 = bench(i8)
    # sanity: accuracy of the quantised product vs bf16
    o32 = torch._int_mm(ai, bt).to(torch.float32)
    ref = torch.nn.functional.linear(a, b).to(torch.float32)
    approx = o32 * (sa * sw.t())
    err = (approx - ref).abs().max().item() / (ref.abs().max().item() + 1e-8)
    w_bytes = N * K  # int8 weights
    print("%-18s %8.1f %8.1f %7.2fx %9.1f %9.1f %7.0f   relerr=%.4f"
          % (label, t_bf, t_i8, t_bf / t_i8, flops / t_bf / 1e6,
             flops / t_i8 / 1e6, w_bytes / t_i8 / 1e3, err))
