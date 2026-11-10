import ctypes, os, subprocess, sys, torch

HERE = os.path.dirname(os.path.abspath(__file__))
MX = "/opt/maca/mxgpu_llvm/bin/mxcc"
SRC = os.path.join(HERE, "mma_cmp.cu")
SO = os.path.join(HERE, "libmmacmp.so")

if not os.path.exists(SO) or os.path.getmtime(SRC) > os.path.getmtime(SO):
    cmd = [MX, "-shared", "-fPIC", "-std=c++17", "-O2",
           "--offload-arch=xcore1000", "-I/opt/maca/include",
           "-I/opt/maca/include/mcr", "-I/opt/maca/include/common",
           "-L/opt/maca/lib", "-lmxc-runtime64", "-o", SO, SRC]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)

lib = ctypes.CDLL(SO)
lib.launch_mma_cmp.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
                               ctypes.c_int, ctypes.c_int, ctypes.c_int,
                               ctypes.c_void_p]
lib.launch_mma_cmp.restype = ctypes.c_int

ITERS = 2000

def bench(which, warps=1):
    out = torch.zeros(104, dtype=torch.float32, device="cuda")
    cyc = torch.zeros(104, dtype=torch.int64, device="cuda")
    def go():
        lib.launch_mma_cmp(which, out.data_ptr(), cyc.data_ptr(), ITERS,
                           warps, 104, None)
    go(); torch.cuda.synchronize()
    best = None
    for _ in range(10):
        go(); torch.cuda.synchronize()
        c = cyc.min().item() / (ITERS * 8.0)
        best = c if best is None else min(best, c)
    return best

print("  per-MMA cycles, 8 independent accumulator chains, operands rebuilt "
      "from GPRs every instruction\n")
rows = []
for tag, fn, flop in (("f8   m16n16k32 (no dequant)", 0, 16384.0),
                      ("bf16 m16n16k16 (no dequant)", 1, 16384.0),
                      ("bf16 + LUT dequant, K=32", 2, 16384.0),
                      ("bf16 + ALU dequant, K=32", 3, 16384.0)):
    c = bench(fn)
    rows.append((tag, c, flop))
    print(f"  {tag}   {c:7.2f} cyc/mma   {flop / c:7.1f} flop/cyc")

f8, bf, lut, alu = rows[0][1], rows[1][1], rows[2][1], rows[3][1]
print(f"\n  equal-work comparison (K=32 worth of MMA per chain step):")
print(f"    f8  direct          {f8:7.2f} cyc")
print(f"    bf16 + LUT dequant  {lut:7.2f} cyc   "
      f"-> {'WINS' if lut < f8 else 'LOSES'} ({f8 / lut:.2f}x)")
