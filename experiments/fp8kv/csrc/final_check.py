import torch, fp8_decode_run as R, val_fast as V
torch.manual_seed(0)
sm = 128 ** -0.5
q, k, v, bt, lens = R.make(2, 300, 2, 128, 32)
ref = V.reference_fast(q, k, v, bt, lens, sm)
o = R.Runner(q, k, v, bt, lens, 4, 4096, sm)()
print("  correctness maxabs =", round((o.float()-ref).abs().max().item(), 5),
      flush=True)
q, k, v, bt, lens = R.make(64, 4096, 2, 128, 64)
t = min(R.bench(R.Runner(q, k, v, bt, lens, 16, 4096, sm), iters=25) for _ in range(2))
print(f"  latency = {t:.4f} ms", flush=True)
