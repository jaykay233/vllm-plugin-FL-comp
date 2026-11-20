import torch
import triton
import triton.language as tl

dev = "cuda"
# Per-layer KV payload at batch 64, ctx 4096, 2 kv heads, head_dim 128, K and V
NEL_BF16 = 2 * 64 * 4096 * 2 * 128          # elements
BYTES_BF16 = NEL_BF16 * 2
BYTES_INT8 = NEL_BF16 * 1


@triton.jit
def read_bf16(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    m = off < n
    v = tl.load(x_ptr + off, mask=m, other=0.0)
    s = tl.sum(v.to(tl.float32))
    tl.store(out_ptr + pid, s)


@triton.jit
def read_i8(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    m = off < n
    v = tl.load(x_ptr + off, mask=m, other=0)
    s = tl.sum(v.to(tl.float32))
    tl.store(out_ptr + pid, s)


def bench(fn, iters=30, warmup=10):
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
    return s.elapsed_time(e) / iters * 1000


BLOCK = 8192
for name, dt, kern, nbytes in (
        ("bf16 KV", torch.bfloat16, read_bf16, BYTES_BF16),
        ("int8 KV", torch.int8, read_i8, BYTES_INT8)):
    x = torch.randint(-127, 127, (NEL_BF16,), device=dev, dtype=torch.int8) if dt is torch.int8 \
        else torch.randn(NEL_BF16, device=dev, dtype=dt)
    n = x.numel()
    grid = (triton.cdiv(n, BLOCK),)
    out = torch.empty(grid[0], device=dev, dtype=torch.float32)
    fn = lambda: kern[grid](x, out, n, BLOCK=BLOCK)
    t = bench(fn)
    print("%-8s %7.1f us  payload %6.1f MB  -> %7.1f GB/s   (x42 layers = %6.2f ms/step)"
          % (name, t, nbytes / 1e6, nbytes / t / 1e3, t * 42 / 1000))
