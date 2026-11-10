import ctypes, os, subprocess, torch

HERE = os.path.dirname(os.path.abspath(__file__))
MX = "/opt/maca/mxgpu_llvm/bin/mxcc"
SRC = os.path.join(HERE, "copy_bw.cu")
SO = os.path.join(HERE, "libcopybw.so")

if not os.path.exists(SO) or os.path.getmtime(SRC) > os.path.getmtime(SO):
    subprocess.run([MX, "-shared", "-fPIC", "-std=c++17", "-O2",
                    "--offload-arch=xcore1000", "-I/opt/maca/include",
                    "-I/opt/maca/include/mcr", "-I/opt/maca/include/common",
                    "-L/opt/maca/lib", "-lmxc-runtime64", "-o", SO, SRC],
                   check=True)

lib = ctypes.CDLL(SO)
lib.launch_copy_bw.argtypes = [ctypes.c_void_p, ctypes.c_longlong,
                               ctypes.c_int, ctypes.c_int, ctypes.c_int,
                               ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
                               ctypes.c_void_p]

SM = torch.cuda.get_device_properties(0).multi_processor_count
CALLS = 256
BYTES_PER_CALL = 1024
CLK = 1.5e9  # approximate, used only for the GB/s figure


def run(warps, depth, ctas):
    nwarps = ctas * warps
    total = nwarps * CALLS * BYTES_PER_CALL
    buf = torch.empty(total, dtype=torch.uint8, device="cuda")
    sink = torch.zeros(ctas, dtype=torch.float32, device="cuda")
    cyc = torch.zeros(ctas, dtype=torch.int64, device="cuda")
    stride = CALLS * BYTES_PER_CALL

    def go():
        lib.launch_copy_bw(buf.data_ptr(), stride, CALLS, depth, warps, ctas,
                           sink.data_ptr(), cyc.data_ptr(), None)
    go()
    torch.cuda.synchronize()
    best = None
    for _ in range(5):
        go()
        torch.cuda.synchronize()
        t = cyc.float().mean().item() / CLK
        best = t if best is None else min(best, t)
    return total / (best * 1e9)


print(f"MetaX C500  SMs={SM}   {CALLS} x 1 KiB requests per warp\n")
print("  warps/CTA  ctas   depth   GB/s")
for ctas in (SM, SM * 4):
    for warps in (1, 2, 4, 8):
        for depth in (0, 8, 4, 2, 1):
            gb = run(warps, depth, ctas)
            d = "inf" if depth == 0 else str(depth)
            print(f"  {warps:9d}  {ctas:5d}   {d:>4s}   {gb:7.0f}")
        print()
