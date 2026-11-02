#!/usr/bin/env python3
"""Is mcblas::DevDstElementwiseAdd an epilogue of the projection GEMM itself?

The decode trace has 42 of these per step, 1 per layer, sitting between down_proj
and the next layer's fused_add_rms_norm, costing 0.377 ms/step (2.2%). FINDINGS
18.5 left its identity open.

Two hypotheses:
  (a) it is a separate residual/bias add issued by model code -- in which case it
      can be fused away into the following rmsnorm or removed outright;
  (b) it is mcblas's own epilogue for a GEMM shape, in which case plugin code
      cannot remove it.

The model has no bias tensors at all (checked) and vLLM's llama.py returns
(hidden_states, residual) separately so the next layer's fused_add_rms_norm does
the add, which argues against (a) on code inspection. Test (b) directly: run each
projection shape through torch profiler in isolation and see which emit the
kernel. No server needed.
"""
from __future__ import annotations

import torch
from torch.profiler import ProfilerActivity, profile

HID, INTER, BATCH = 2048, 6144, 64
SHAPES = [
    ("qkv  fuse 2048->2560", 2048, 2560),
    ("qkv  fuse 2048->5120", 2048, 5120),   # 0.20 GEMM 64x32 tile variants
    ("o_proj 2048->2048", 2048, 2048),
    ("gate_up 2048->12288", 2048, 2 * INTER),
    ("down_proj 6144->2048", INTER, HID),
]


def kernels(prof):
    out = []
    for e in prof.key_averages():
        n = str(e.key)
        if e.device_time_total > 0 and (
            "gemm" in n.lower() or "Add" in n or "add" in n or "Mck" in n
        ):
            out.append((n[:70], e.count, e.device_time_total / max(e.count, 1)))
    return out


def main() -> None:
    torch.manual_seed(0)
    print(f"batch M={BATCH}, bf16\n")
    for label, fin, fout in SHAPES:
        w = torch.randn(fout, fin, dtype=torch.bfloat16, device="cuda")
        x = torch.randn(BATCH, fin, dtype=torch.bfloat16, device="cuda")
        lin = torch.nn.functional.linear
        for _ in range(5):
            lin(x, w)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(10):
                lin(x, w)
            torch.cuda.synchronize()
        ks = kernels(prof)
        hit = [k for k in ks if "DevDst" in k[0] or "ElementwiseAdd" in k[0]]
        print(f"{label:<24} kernels={len(ks):2d}  "
              f"{'DevDstElementwiseAdd: YES' if hit else 'DevDstElementwiseAdd: no'}")
        for n, c, us in ks[:4]:
            print(f"    {c / 10:5.1f}x {us:8.1f} us  {n}")

    # control: does a plain add on the same shape emit it?
    a = torch.randn(BATCH, HID, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(BATCH, HID, dtype=torch.bfloat16, device="cuda")
    for _ in range(5):
        torch.add(a, b)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(10):
            torch.add(a, b)
        torch.cuda.synchronize()
    ks = kernels(prof)
    hit = [k for k in ks if "DevDst" in k[0]]
    print(f"\n{'torch.add [64,2048]':<24} kernels={len(ks):2d}  "
          f"{'DevDstElementwiseAdd: YES' if hit else 'DevDstElementwiseAdd: no'}")
    for n, c, us in ks[:4]:
        print(f"    {c / 10:5.1f}x {us:8.1f} us  {n}")


if __name__ == "__main__":
    main()
