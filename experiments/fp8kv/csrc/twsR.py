import ctypes, torch, fp8_decode_run as R
torch.manual_seed(0)
sm = 128 ** -0.5
base = R.lib
at = base.launch_fp8_decode.argtypes
w = ctypes.CDLL('./libws.wip.so'); w.launch_fp8_decode.argtypes = at
q, k, v, bt, lens = R.make(1, 64, 2, 128, 32)   # ntiles = 2
cases = (("full     ", 0), ("const P  ", 8), ("const exp", 16),
         ("no PV    ", 1), ("no QK    ", 2))
print("  ntiles=2, |old - ws| per mode")
for tag, extra in cases:
    R.lib = base
    a = R.Runner(q, k, v, bt, lens, 1, 4096 | extra, sm)().float().clone()
    R.lib = w
    b = R.Runner(q, k, v, bt, lens, 1, 4096 | extra | 131072, sm)().float().clone()
    print(f"    {tag}  {(a-b).abs().max().item():.6f}", flush=True)
R.lib = base
