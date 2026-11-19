"""Quantify the per-node overhead of a CUDA graph on this card.

The decode step has ~336 kernels; if the graph charges ~9 us per node, the
~3 ms gap between the summed kernel times and the replay wall is boundary
overhead that only a fused/persistent kernel could remove.  If it is ~1 us,
the gap is kernel inefficiency and fusion cannot buy it.
"""
import torch

N = 336
x = torch.zeros(1, device="cuda")


def bench(fn, it=200):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(True)
    e = torch.cuda.Event(True)
    s.record()
    for _ in range(it):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / it * 1000.0


# eager: N tiny kernels
def eager():
    for _ in range(N):
        x.add_(1.0)


# graph: same N kernels captured once
g = torch.cuda.CUDAGraph()
torch.cuda.synchronize()
with torch.cuda.graph(g):
    for _ in range(N):
        x.add_(1.0)
torch.cuda.synchronize()


def replay():
    g.replay()


t_e = bench(eager)
t_g = bench(replay)
print(f"N={N} eager={t_e:8.1f}us ({t_e/N:5.2f}us/kernel)  "
      f"graph={t_g:8.1f}us ({t_g/N:5.2f}us/node)  "
      f"saved={t_e-t_g:7.1f}us")
print(f"=> 336-node graph boundary cost if it were {t_g/N:.2f}us/node: "
      f"{t_g:.1f}us total")
