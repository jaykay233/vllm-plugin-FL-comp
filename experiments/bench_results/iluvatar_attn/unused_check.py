import time, torch, flag_gems
flag_gems.enable()
print("excluded:", sorted(flag_gems.current_work_registrar.get_unused_ops()))
x = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
i = torch.randint(0, 2048, (64,), device="cuda")
n = torch.arange(64, device="cuda", dtype=torch.int32)
cases = {
    "index": lambda: x[i], "copy_": lambda: x.copy_(x), "to_copy": lambda: n.to(torch.int64),
    "sub": lambda: n - n, "eq_scalar": lambda: n == 1, "floor_divide": lambda: n // 16,
    "remainder": lambda: n % 16, "where": lambda: torch.where(n > 3, n, n), "cumsum": lambda: n.cumsum(0),
    "full": lambda: torch.full((64,), 1, device="cuda"), "zeros": lambda: torch.zeros(64, device="cuda"),
    "fill_": lambda: n.fill_(1), "add": lambda: n + n, "mul(kept)": lambda: n * n,
}
from torch.profiler import profile, ProfilerActivity
for k, f in cases.items():
    for _ in range(5): f()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(200): f()
    h = (time.perf_counter() - t) / 200 * 1e6; torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        f(); torch.cuda.synchronize()
    ks = sorted({e.name for e in p.events() if e.device_type.name == "CUDA"})
    print(f"{k:12s} host {h:7.1f}us kernels {[s[:40] for s in ks][:3]}")
