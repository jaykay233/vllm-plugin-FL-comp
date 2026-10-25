#!/usr/bin/env python3
"""Can the small-N projections be steered onto the fast tile by transposing?

Measured in the live decode step (per call, bf16, M=64):

    gate_up   N=12288  reads 50.3 MB   41.8 us   1205 GB/s  85% of peak
    down      N=2048   reads 25.2 MB   26.4 us    953 GB/s  67%   (+9.0 us reduction)
    qkv       N=2560   reads 10.5 MB   17.2 us   ~610 GB/s  43%
    o         N=2048   reads  8.4 MB   17.2 us   ~488 GB/s  34%

mcblas emits statically-tiled kernels (`Mck_bf16gemm_tn_<Mtile>x<Ntile>x<Ktile>`)
and picks 64x32 for the small-N shapes but 128x64 for gate_up. There is no tuning
database to override; libmcblasLt only exposes a heuristic cache, and torch goes
through the non-Lt path with fixed tiles.

So the only lever left is the *shape* we hand it. The same GEMM can be posed as
    C   = x @ W^T          [64, N]   -> tile 64x32   (what we do now)
    C^T = W @ x^T          [N, 64]   -> M=N, N=64, different tile selection
The second form transposes the output, which costs a [64, 2048] copy, so it only
wins if the GEMM itself gets much faster. Measure both, verify identical results,
and report the net after the transpose.

Also tries `out=` variants and a fused transposed+add, since the transposed result
has to be laid back out as [64, 2048] for the following norm.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

BATCH, HID, INTER, VOCAB = 64, 2048, 6144, 130560
REPEATS = 200


def bench(fn, iters=1):
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(REPEATS):
        for _ in range(iters):
            fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / (REPEATS * iters) * 1000  # us


def main() -> None:
    print(f"M={BATCH}, bf16, timings in us per call\n")
    cases = [
        ("qkv", HID, 2560),
        ("o_proj", HID, HID),
        ("down", INTER, HID),
        ("gate_up", HID, 2 * INTER),
    ]
    for label, fin, fout in cases:
        x = torch.randn(BATCH, fin, dtype=torch.bfloat16, device="cuda")
        w = torch.randn(fout, fin, dtype=torch.bfloat16, device="cuda")
        ref = F.linear(x, w)
        out = torch.empty_like(ref)
        wbytes = fout * fin * 2

        # form 1: what the serving stack does today
        t1 = bench(lambda: F.linear(x, w))
        # form 2: same, preallocated destination
        t2 = bench(lambda: torch.mm(x, w.t(), out=out))
        # form 3: transpose the problem -> M=fout, N=BATCH
        xt = x.t().contiguous()
        ct = torch.empty(fout, BATCH, dtype=torch.bfloat16, device="cuda")
        t3 = bench(lambda: torch.mm(w, xt, out=ct))
        # form 4: transposed GEMM + transpose back into the caller's buffer
        def t_and_back():
            torch.mm(w, xt, out=ct)
            out.copy_(ct.t())
        t4 = bench(t_and_back)
        # form 5: transposed GEMM + elementwise write-back (no memcpy node)
        def t_and_add():
            torch.mm(w, xt, out=ct)
            torch.add(ct.t().reshape(-1), 0.0, out=out.reshape(-1))
        t5 = bench(t_and_add)

        # correctness of the transposed route
        ct.zero_()
        torch.mm(w, xt, out=ct)
        ok = torch.allclose(ct.t(), ref, atol=2e-2, rtol=2e-2)

        print(f"{label}  [{BATCH}x{fin}] @ [{fin}x{fout}]  weights {wbytes / 1e6:.1f} MB")
        for name, t in [("F.linear", t1), ("mm(out=)", t2), ("mm transposed", t3),
                        ("transposed+copy_back", t4), ("transposed+add_back", t5)]:
            print(f"   {name:<24} {t:7.1f} us   {wbytes / t / 1e3:6.0f} GB/s "
                  f"({wbytes / t / 1e3 / 1416 * 100:3.0f}%)")
        print(f"   transposed result correct: {ok}")
        best = min(t2, t4, t5)
        print(f"   best vs today: {best - t1:+.1f} us/layer "
              f"-> {(best - t1) * 42 / 1000:+.3f} ms/step\n")


if __name__ == "__main__":
    main()
