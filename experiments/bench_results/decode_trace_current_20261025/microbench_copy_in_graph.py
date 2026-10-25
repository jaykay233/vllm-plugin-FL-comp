#!/usr/bin/env python3
"""Does the 35 us only appear once the copy is captured in a CUDA graph?

Eager, a 256 KB bf16 device copy costs 6-8 us on this card (33 GB/s), and every
alternative writeback path is within 0.1 ms/step -- useless.  But the decode
trace, where these copies are graph nodes, attributes ~35 us to each of the 42
per-step copies, i.e. 1.48 ms/step.

So the hypothesis is that the cost is specific to graph-executed memcpy nodes
(for example a cross-engine sync the graph has to materialise), not to the copy
itself.  Test it by capturing the identical 42-copy sequence into a graph and
timing the replay, against two controls:

  * 42x copy        -- the current code path
  * 42x add_zero    -- elementwise kernel writing the same bytes into dst
  * 42x none        -- same graph shape with no writeback, isolates graph overhead

Each is interleaved with a real GEMM to mimic the layer's compute neighbours,
because a bare chain of copies may pipeline differently.
"""
from __future__ import annotations

import torch

N_TOK, HID = 64, 2048
LAYERS = 42
REPEATS = 100


def make_graph(writeback: str):
    src = torch.randn(N_TOK, HID, dtype=torch.bfloat16, device="cuda")
    dst = torch.empty_like(src)
    w = torch.randn(HID, HID, dtype=torch.bfloat16, device="cuda")
    x = torch.randn(N_TOK, HID, dtype=torch.bfloat16, device="cuda")
    out = torch.empty_like(x)

    def step():
        for _ in range(LAYERS):
            out = torch.mm(x, w.t())
            if writeback == "copy":
                dst.copy_(out)
            elif writeback == "add":
                torch.add(out, 0.0, out=dst)
            # "none": skip the writeback entirely
        return dst

    for _ in range(3):
        step()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        step()
    return g


def timeit(g, label):
    for _ in range(10):
        g.replay()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(REPEATS):
        g.replay()
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / REPEATS
    print(f"  {label:<12} {ms:7.3f} ms/replay   ({ms / LAYERS * 1000:6.2f} us/layer)")
    return ms


def main() -> None:
    print(f"graph of {LAYERS} layers: gemm 64x2048x2048 + writeback\n")
    t_none = timeit(make_graph("none"), "no writeback")
    t_copy = timeit(make_graph("copy"), "dst.copy_")
    t_add = timeit(make_graph("add"), "add(0,out=)")

    print("\nper-step deltas vs no writeback")
    print(f"  copy_   +{(t_copy - t_none):6.3f} ms")
    print(f"  add     +{(t_add - t_none):6.3f} ms")
    print(f"\nswitch to add(0,out=) would save {(t_copy - t_add):.3f} ms/step "

          f"({(t_copy - t_add) / 19.17 * 100:.1f}% of the 19.17 ms decode step)")


if __name__ == "__main__":
    main()
