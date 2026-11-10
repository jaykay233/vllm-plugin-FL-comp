"""Is fp8 slow because of the dtype, the strided head layout, or the widen?"""

import torch, triton, triton.language as tl

@triton.jit
def probe(SRC, Out, n_elems, stride_row,
          BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
          MODE: tl.constexpr):
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    ptr = SRC + offs_n[:, None] * stride_row + offs_d[None, :]
    x = tl.load(ptr, mask=offs_n[:, None] < n_elems, other=0.0)
    if MODE == 0:      # widen to fp32
        y = tl.sum(x.to(tl.float32), 1)
    elif MODE == 1:    # widen to bf16
        y = tl.sum(x.to(tl.bfloat16).to(tl.float32), 1)
    else:              # bitcast, no numeric conversion at all
        y = tl.sum(tl.cast(x.to(tl.uint8, bitcast=True), tl.float32), 1)
    tl.store(Out + offs_n, y, mask=offs_n < n_elems)


def run(dtype, tag, stride_row, BLOCK_N=64, BLOCK_D=128, MODE=0, warps=4, stages=2):
    n = 64 * 4096            # rows
    src = torch.randn(n, stride_row, device='cuda', dtype=torch.bfloat16)
    src = src[:, :BLOCK_D].contiguous() if stride_row == BLOCK_D else src[:, :BLOCK_D]
    if dtype == torch.float8_e4m3fn:
        src = src.to(torch.float8_e4m3fn)
    elif dtype == torch.bfloat16:
        src = src.to(torch.bfloat16)
    out = torch.empty(n, device='cuda', dtype=torch.float32)
    grid = (triton.cdiv(n, BLOCK_N),)
    fn = lambda: probe[grid](src, out, n, src.stride(0), BLOCK_N=BLOCK_N,
                             BLOCK_D=BLOCK_D, MODE=MODE, num_warps=warps, num_stages=stages)
    try:
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        for _ in range(20):
            fn()
        e.record()
        torch.cuda.synchronize()
        t = s.elapsed_time(e) / 20
        nb = n * BLOCK_D * src.element_size()
        print("PROBE %-28s %7.0f GB/s  %.3f ms" % (tag, nb / (t * 1e6), t))
    except Exception as exc:
        print("PROBE %-28s FAIL %s" % (tag, type(exc).__name__))


def main():
    D = 128
    run(torch.bfloat16,      "bf16 contiguous widenf32", D, MODE=0)
    run(torch.float8_e4m3fn, "fp8  contiguous widenf32", D, MODE=0)
    run(torch.float8_e4m3fn, "fp8  contiguous widenbf16", D, MODE=1)
    run(torch.float8_e4m3fn, "fp8  contiguous bitcast  ", D, MODE=2)
    run(torch.bfloat16,      "bf16 strided256 widenf32", 256, MODE=0)
    run(torch.float8_e4m3fn, "fp8  strided256 widenf32", 256, MODE=0)
    run(torch.float8_e4m3fn, "fp8  strided256 bitcast  ", 256, MODE=2)
    run(torch.float8_e4m3fn, "fp8  contig w=1 st=4     ", D, MODE=0, warps=1, stages=4)
    run(torch.float8_e4m3fn, "fp8  contig BN=16        ", D, MODE=0, BLOCK_N=16)
    run(torch.bfloat16,      "bf16 contig BN=16        ", D, MODE=0, BLOCK_N=16)


if __name__ == "__main__":
    main()
