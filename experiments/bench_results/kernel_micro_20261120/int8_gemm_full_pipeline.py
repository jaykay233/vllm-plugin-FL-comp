import torch
import torch.nn.functional as F

dev = "cuda"
torch.manual_seed(0)


def bench(fn, iters=20, warmup=5):
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


def prep_int8(b):
    sw = b.abs().amax(dim=1).clamp(min=1e-8).float() / 127.0
    bq = (b.float() / sw[:, None]).round().clamp(-127, 127).to(torch.int8)
    return bq.t().contiguous(), sw


SHAPES = [
    (64, 2048, 130560, "decode lm_head"),
    (16384, 2048, 12288, "prefill gate_up"),
    (16384, 2048, 2560, "prefill qkv"),
    (16384, 6144, 2048, "prefill down"),
]

print("%-18s %9s %9s %8s   breakdown of int8 (us): quant/gemm/dequant" % (
    "shape", "bf16_us", "i8_all", "speedup"))
for M, K, N, label in SHAPES:
    a = torch.randn(M, K, device=dev, dtype=torch.bfloat16) * 0.5
    b = torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.05
    wt, ws = prep_int8(b)

    t_bf = bench(lambda: F.linear(a, b))

    def q():
        xs = a.abs().amax(dim=1, keepdim=True).clamp(min=1e-8).float() / 127.0
        xi = (a.float() / xs).round().clamp(-127, 127).to(torch.int8)
        return xi, xs

    xi, xs = q()

    def gemm_only():
        return torch._int_mm(xi, wt)

    def deq():
        o32 = torch._int_mm(xi, wt)
        return (o32.float() * xs).mul_(ws).to(torch.bfloat16)

    t_q = bench(q)
    t_g = bench(gemm_only)
    t_all = bench(deq)
    t_deq = t_all - t_g
    print("%-18s %9.1f %9.1f %8.2fx   %.0f / %.0f / %.0f" % (
        label, t_bf, t_all, t_bf / t_all, t_q, t_g, t_deq))
