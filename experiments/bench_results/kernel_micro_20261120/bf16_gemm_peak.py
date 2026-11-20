"""What is C500's actual BF16 GEMM peak, and is mcblas at it?

Prefill projections measured 211-225 TFLOPS at M=16384.  If a much larger,
purely compute-bound GEMM also tops out near there, 244 TFLOPS in the real
trace is the hardware limit and prefill has no headroom.  If it goes
substantially higher, mcblas is leaving prefill throughput on the table.
"""
import torch
import torch.nn.functional as F

dev = "cuda"


def bench(fn, iters=10, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1000  # us


print("%-26s %10s %10s" % ("shape (M,N,K)", "us", "TFLOPS"))
for M, N, K in [(4096, 4096, 4096), (8192, 8192, 8192), (16384, 16384, 8192),
                (16384, 12288, 2048), (65536, 12288, 2048)]:
    a = torch.randn(M, K, device=dev, dtype=torch.bfloat16) * 0.1
    b = torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.05
    t = bench(lambda: F.linear(a, b))
    print("%-26s %10.0f %10.1f" % ("%dx%dx%d" % (M, N, K), t,
                                   2.0 * M * N * K / t / 1e6))
