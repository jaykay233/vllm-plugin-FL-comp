import ctypes, torch, fp8_decode_run as R, val_fast as V
torch.manual_seed(0)
sm = 128 ** -0.5
base = R.lib
at = base.launch_fp8_decode.argtypes
def load(p):
    l = ctypes.CDLL(p); l.launch_fp8_decode.argtypes = at; return l
for SEQ in (32, 64):
    q, k, v, bt, lens = R.make(2, SEQ, 2, 128, 32)
    ref = V.reference_fast(q, k, v, bt, lens, sm)
    for tag, path in (("double-slot", "./libws.wip.so"),
                      ("single-slot", "./libwssingle.so")):
        R.lib = base if path is None else load(path)
        o = R.Runner(q, k, v, bt, lens, 1, 4096 | 131072, sm)()
        print(f"  ntiles={SEQ//32} {tag:<12s} maxabs="
              f"{(o.float()-ref).abs().max().item():.5f}", flush=True)
R.lib = base
