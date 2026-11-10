import ctypes
import os
import subprocess

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
LIB = os.path.join(HERE, "libstage3.so")
r = subprocess.run(["/opt/maca/mxgpu_llvm/bin/mxcc", "-shared", "-fPIC", "-O3",
                    "--offload-arch=xcore1000", "-I/opt/maca/include",
                    os.path.join(HERE, "stage3.cu"), "-o", LIB],
                   capture_output=True, text=True)
ok = r.returncode == 0 and "error:" not in r.stderr
print("BUILD", "OK" if ok else "FAIL")
if not ok:
    print(r.stderr[-2500:])
    raise SystemExit(1)

lib = ctypes.CDLL(LIB)
lib.launch_stage3.argtypes = [ctypes.c_void_p] * 2 + [ctypes.c_void_p] + [ctypes.c_int] * 3 + [ctypes.c_void_p]

dev = "cuda"
st = torch.cuda.current_stream().cuda_stream

CFG = {
    0: ("memcpy16 WPB4 per-chunk", 4, 2, 128),
    1: ("ldg128   WPB4 per-chunk", 4, 2, 128),
    2: ("ldg128   WPB4 one-barrier", 4, 2, 128),
    3: ("ldg128   WPB8 one-barrier", 8, 2, 128),
    4: ("ldg128   WPB16 one-barrier", 16, 1, 128),
    5: ("ldg128   WPB4 per-chunk b", 4, 2, 128),
    6: ("ldg128   WPB4 KC256", 4, 2, 256),
    7: ("ldg128   WPB4 NT4", 4, 4, 128),
    8: ("ldg128   WPB2 per-chunk", 2, 2, 128),
    9: ("ldg128   WPB4 per-chunk c", 4, 2, 128),
    10: ("memcpy16 WPB4 one-barrier", 4, 2, 128),
    11: ("memcpy16 WPB8 one-barrier", 8, 2, 128),
}


def bench(fn, iters=30, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(iters):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / iters


N, K = 12288, 2048
a8 = torch.randint(0, 100, (64, K), device=dev, dtype=torch.uint8).contiguous()
b8 = torch.randint(0, 100, (N, K), device=dev, dtype=torch.uint8).contiguous()
out = torch.zeros(4096, device=dev, dtype=torch.float32)
print("N=%d K=%d  fp8 weight bytes = %.1f MB" % (N, K, N * K / 1e6))
for cfg, (name, wpb, nt, kc) in CFG.items():
    try:
        t = bench(lambda c=cfg: lib.launch_stage3(a8.data_ptr(), b8.data_ptr(),
                                                 out.data_ptr(), N, K, c, st))
    except Exception as e:
        print("  %-28s FAIL %s" % (name, type(e).__name__))
        continue
    per = wpb * nt * 16
    tiles = (N + per - 1) // per
    byt = (64 * K + per * K) * tiles
    print("  %-28s %8.5f ms  %6.0f GB/s  (blocks=%d warps=%d)" %
          (name, t, byt / (t * 1e6), tiles, wpb))
