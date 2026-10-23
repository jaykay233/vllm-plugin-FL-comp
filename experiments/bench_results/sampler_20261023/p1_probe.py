import sys, torch
sys.path.insert(0, "/workspace/FlagGems-comp/src")
import flag_gems
from torch.profiler import ProfilerActivity, profile
ROWS, VOCAB = 64, 130560
def us(fn, w=10, it=50):
    for _ in range(w): fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(it): fn()
        torch.cuda.synchronize()
    return sum(e.self_device_time_total or 0 for e in p.key_averages())/it
logits = torch.randn(ROWS, VOCAB, device="cuda", dtype=torch.float32)
p1 = torch.full((ROWS,), 1.0, device="cuda")
print(f"top_p_threshold(p=1.0)      {us(lambda: flag_gems.top_p_threshold(logits, p1)):7.1f} us")
# cost of the host-side decision we would add to skip it
def decide():
    return bool((p1 >= 1.0).all().item())
print(f"host check all(p>=1)         {us(decide):7.1f} us  value={decide()}")
print(f"identity (baseline)          {us(lambda: logits):7.1f} us")
