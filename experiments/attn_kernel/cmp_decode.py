"""Numeric gate for the split-KV decode kernel.

The reference is a plain fp32 softmax over the gathered pages, computed one sequence
at a time.  That is stricter than comparing against the vendor kernel: it isolates
this kernel's own arithmetic, so a failure here is a bug in the kernel rather than a
difference in how two kernels round.

Page tables are shuffled and the pool is padded with garbage, so a kernel that walks
the cache linearly instead of following block_table gets a wrong answer.  Lengths are
ragged and deliberately not multiples of the 16-position page, which exercises the
tail masking.

Usage:  python cmp_decode.py            # all shapes
        python cmp_decode.py 30 16384   # one shape
"""
import os
import subprocess
import sys

import numpy as np
import torch

D, NH, NKVH, PAGE = 128, 16, 2, 16
BIN = os.environ.get("BIN", "/tmp/fdec")
S = os.environ.get("S", "8")
W = os.environ.get("W", "8")


def f2bf(x):
    """Round a float32 tensor to bfloat16 and back, to hold the reference on the
    same precision as the kernel's inputs."""
    return (x.to(torch.bfloat16)).float()


def build(n, ctx, seed, ragged=True):
    """One decode step: n sequences, each with one query row."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    if ragged:
        lens = ctx - torch.randint(0, max(1, ctx // 8), (n,), generator=g)
        lens[0] = ctx
    else:
        lens = torch.full((n,), ctx)
    lens = lens.to(torch.int32)

    nb_max = int((int(lens.max()) + PAGE - 1) // PAGE)
    pool = n * nb_max + 64  # slack, so the table is neither dense nor contiguous
    q = (torch.randn(n, NH, D, generator=g) * 2.0).to(torch.bfloat16)
    k = (torch.randn(pool, NKVH, PAGE, D, generator=g) * 2.0).to(torch.bfloat16)
    v = (torch.randn(pool, NKVH, PAGE, D, generator=g) * 2.0).to(torch.bfloat16)
    perm = torch.randperm(pool, generator=g)[: n * nb_max]
    bt = perm.reshape(n, nb_max).to(torch.int32)
    return q, k, v, bt, lens, pool


def reference(q, k, v, bt, lens):
    """fp32 softmax over the gathered pages, per sequence.

    k/v are HND, so a page is [kvh][pos][dim]; the positions of a sequence are the
    page list, then the position inside the page.  Permuting before the reshape is
    what puts the kv-head axis where the repeat_interleave expects it -- getting this
    backwards silently compares against a transposed cache.
    """
    scale = 1.0 / (D ** 0.5)
    out = torch.empty(q.shape, dtype=torch.float32)
    rep = NH // NKVH
    for i in range(q.shape[0]):
        L = int(lens[i])
        nb = (L + PAGE - 1) // PAGE
        pages = bt[i, :nb].long()
        kk = k[pages].permute(0, 2, 1, 3).reshape(-1, NKVH, D)[:L].float().repeat_interleave(rep, dim=1)
        vv = v[pages].permute(0, 2, 1, 3).reshape(-1, NKVH, D)[:L].float().repeat_interleave(rep, dim=1)
        p = torch.softmax(torch.einsum("hd,lhd->hl", q[i].float(), kk) * scale, dim=-1)
        out[i] = torch.einsum("hl,lhd->hd", p, vv)
    return out


def run(name, n, ctx, seed=0, ragged=True):
    q, k, v, bt, lens, pool = build(n, ctx, seed, ragged)
    pre = f"/tmp/dc_{name}"
    # the kernel reads raw little-endian bytes; bf16 has no numpy view, so go via uint8
    for t, nm in ((q, "Q"), (k, "K"), (v, "V")):
        with open(f"{pre}.{nm}.bin", "wb") as f:
            f.write(t.contiguous().view(torch.uint8).numpy().tobytes())
    with open(f"{pre}.bt.bin", "wb") as f:
        f.write(bt.contiguous().numpy().tobytes())
    with open(f"{pre}.case", "w") as f:
        f.write(f"{n} {NH} {NKVH} {bt.shape[1]}\n")
        for i in range(n):
            f.write(f"{int(lens[i])}\n")
        f.write(f"{pre}.Q.bin {pre}.K.bin {pre}.V.bin {pre}.bt.bin {pre}.out.bin\n")

    r = subprocess.run([BIN, f"{pre}.case", S, W], capture_output=True, text=True)
    if r.returncode != 0:
        print(f"{name}: KERNEL FAILED\n{r.stdout}\n{r.stderr}")
        return None
    out = np.fromfile(f"{pre}.out.bin", dtype=np.float32).reshape(n, NH, D)
    got = torch.from_numpy(out.copy())
    ref = reference(q, k, v, bt, lens)

    diff = (got - ref).abs()
    # relative to the row's own magnitude, since attention output is an average of v
    denom = ref.abs().amax(dim=-1, keepdim=True).clamp_min(1e-3)
    rel = (diff / denom).amax().item()
    mx = diff.max().item()
    bad = int((diff > 0.05 * denom).sum().item())
    tot = n * NH * D
    tail = [
        ln.strip() for ln in r.stdout.strip().splitlines()
        if ln.startswith(("split", "merge", "total"))
    ]
    print(f"{name:<22} maxabs {mx:.3e}  rel {rel:.3e}  bad {bad}/{tot}  "
          f"{'OK' if rel < 2e-2 and bad == 0 else 'FAIL'}   | {tail[-1] if tail else ''}")
    return rel


def main():
    if len(sys.argv) == 3:
        cases = [(f"{sys.argv[1]}x{sys.argv[2]}", int(sys.argv[1]), int(sys.argv[2]))]
    else:
        cases = [
            ("1x256", 1, 256),
            ("1x4096", 1, 4096),
            ("4x1024", 4, 1024),
            ("8x1000", 8, 1000),      # length not a multiple of 16
            ("30x16384", 30, 16384),  # the scoring shape
            ("32x4096", 32, 4096),
            ("64x8192", 64, 8192),
            ("3x17", 3, 17),          # shorter than one tile
            ("130x2048", 130, 2048),  # more units than splits x CTAs
        ]
    worst = 0.0
    for name, n, ctx in cases:
        r = run(name, n, ctx, seed=n * 7919 + ctx)
        if r is not None:
            worst = max(worst, r)
    print(f"\nworst relative error over all shapes: {worst:.3e}")
    print("pass criterion: rel < 2e-2 and no row with error over 5% of its own peak")


if __name__ == "__main__":
    main()
