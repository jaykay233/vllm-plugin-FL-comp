import torch

from flag_gems.runtime.backend._iluvatar.ops.mm import (
    _launch_mm,
    _launch_mm_fixed,
)

dev = "cuda"
REPS = 42


def timed_graph(launch, reps=REPS):
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


# (label, M, N, K, base_cfg_used_by_production)
CASES = [
    ("qkv M64", 64, 2560, 2048, (64, 128, 64, 1, 8, 2)),
    ("o   M64", 64, 2048, 2048, (64, 64, 128, 1, 8, 2)),
    ("qkv M32", 32, 2560, 2048, (64, 128, 64, 1, 8, 2)),
    ("o   M32", 32, 2048, 2048, (32, 64, 128, 1, 4, 2)),
    ("gu  M32", 32, 12288, 2048, (32, 256, 64, 1, 8, 1)),
    ("dn  M32", 32, 2048, 6144, (32, 64, 128, 1, 8, 2)),
]

for label, M, N, K, base in CASES:
    a = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
    b = torch.randn(K, N, device=dev, dtype=torch.bfloat16)
    c = torch.empty(M, N, device=dev, dtype=torch.bfloat16)
    wbytes = K * N * 2

    t_prod = timed_graph(lambda: _launch_mm(a, b, c, M, N, K))
    bm, bn, bk, _, w, st = base
    print(f"\n=== {label} N={N} K={K} weight={wbytes/1e6:.1f}MB ===")
    print(f"production: {t_prod:8.1f} us  {wbytes/t_prod/1e3:6.1f} GB/s  cfg={base}")
    for sp in (2, 4):
        if K % (bk * sp) != 0:
            continue
        cfg = (bm, bn, bk, sp, w, st)
        try:
            t = timed_graph(
                lambda cfg=cfg: _launch_mm_fixed(a, b, c, M, N, K, cfg))
        except Exception as exc:
            print(f"  cfg={cfg} failed: {exc!r}")
            continue
        print(f"  cfg={cfg} {t:8.1f} us  {wbytes/t/1e3:6.1f} GB/s  "
              f"{(t_prod-t)/t_prod*100:+5.1f}%")
