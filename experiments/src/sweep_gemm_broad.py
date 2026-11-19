import itertools

import torch

from flag_gems.runtime.backend._iluvatar.ops.mm import _launch_mm, _launch_mm_fixed

dev = "cuda"
REPS = 42


def timed(launch, reps=REPS):
    for _ in range(3):
        launch()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps):
            launch()
    torch.cuda.synchronize()
    s = torch.cuda.Event(True)
    e = torch.cuda.Event(True)
    s.record()
    for _ in range(10):
        g.replay()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / 10 / reps * 1000.0


CASES = [
    ("qkv M64", 64, 2560, 2048),
    ("qkv M32", 32, 2560, 2048),
    ("down M32", 32, 2048, 6144),
]

for label, M, N, K in CASES:
    a = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
    b = torch.randn(K, N, device=dev, dtype=torch.bfloat16)
    c = torch.empty(M, N, device=dev, dtype=torch.bfloat16)
    wbytes = K * N * 2
    t_prod = timed(lambda: _launch_mm(a, b, c, M, N, K))
    print(f"\n=== {label} N={N} K={K} {wbytes/1e6:.1f}MB ===")
    print(f"production {t_prod:7.1f} us  {wbytes/t_prod/1e3:6.1f} GB/s")
    best = (t_prod, "prod")
    for bm, bn, bk, w, st in itertools.product(
        (M,), (128, 256, 512), (64, 128), (4, 8, 16), (1, 2)
    ):
        cfg = (bm, bn, bk, 1, w, st)
        try:
            t = timed(lambda cfg=cfg: _launch_mm_fixed(a, b, c, M, N, K, cfg))
        except Exception:
            continue
        if t < best[0]:
            best = (t, cfg)
    if best[1] != "prod":
        print(f"best       {best[0]:7.1f} us  {wbytes/best[0]/1e3:6.1f} GB/s  "
              f"{(t_prod-best[0])/t_prod*100:+5.1f}%  cfg={best[1]}")
    else:
        print("best = production")
