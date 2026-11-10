import ctypes, os, subprocess, torch
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "mma_peak.cu"); LIB = os.path.join(HERE, "libmma_peak.so")
r = subprocess.run(["/opt/maca/mxgpu_llvm/bin/mxcc", "-shared", "-fPIC", "-O3",
                    "--offload-arch=xcore1000", "-I/opt/maca/include", SRC, "-o", LIB],
                   capture_output=True, text=True)
print("BUILD", "OK" if r.returncode == 0 else "FAIL")
if r.returncode:
    print(r.stderr[-3000:]); raise SystemExit(1)

lib = ctypes.CDLL(LIB)
lib.launch_mma_peak.argtypes = [ctypes.c_void_p, ctypes.c_void_p] + [ctypes.c_int]*3 + [ctypes.c_void_p]
dev = "cuda"; st = torch.cuda.current_stream().cuda_stream
src = torch.randint(0, 100, (64 * 16 * 32,), device=dev, dtype=torch.uint8)
src = (src.float() * 0.01).clamp(-4, 4).to(torch.float32).view(torch.uint8)[:64*16*32].contiguous()
OUT = torch.zeros(4096, device=dev, dtype=torch.float32)

ITER = 256
print("\n%-4s %-8s %10s %10s %12s %12s" % ("NCH", "blocks", "ms", "TFLOPS", "GB/s@M64eq", "cyc/MMA/warp"))
for blocks in (104, 208, 416):
    for nch in (1, 2, 4, 8, 16):
        for _ in range(3):
            lib.launch_mma_peak(src.data_ptr(), OUT.data_ptr(), blocks, nch, ITER, st)
        torch.cuda.synchronize()
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record()
        for _ in range(10):
            lib.launch_mma_peak(src.data_ptr(), OUT.data_ptr(), blocks, nch, ITER, st)
        b.record(); torch.cuda.synchronize()
        t = a.elapsed_time(b) / 10
        warps = blocks * 4
        mmas = warps * nch * ITER
        flops = mmas * 2 * 16 * 16 * 32
        tflops = flops / (t * 1e9)
        eq_gbs = tflops * 1000 / (2 * 64)   # BW a M=64 GEMM could reach at this rate
        cyc = (t * 1e-3) * 1.5e9 / (nch * ITER)   # per warp-iteration, 1.5 GHz
        print("%-4d %-8d %10.4f %10.1f %12.0f %12.1f" % (nch, blocks, t, tflops, eq_gbs, cyc))
