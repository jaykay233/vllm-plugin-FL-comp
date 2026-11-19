import time

import torch

GB = 1 << 30
n = GB // 2  # bf16 elements -> 1 GiB tensor
x = torch.randn(n, device="cuda", dtype=torch.bfloat16)
y = torch.empty_like(x)

# warm
for _ in range(3):
    y.copy_(x)
torch.cuda.synchronize()

print("window  copy GB/s  read GB/s")
t0 = time.time()
win_start = t0
fx, fy = x, y
while True:
    iters = 50
    for _ in range(iters):
        fy.copy_(fx)
    torch.cuda.synchronize()
    now = time.time()
    dt = now - win_start
    gbs = 2 * GB * iters / dt / 1e9
    print(f"{now-t0:6.1f}s  {gbs:8.1f}")
    win_start = now
    if now - t0 > 20.0:
        break
