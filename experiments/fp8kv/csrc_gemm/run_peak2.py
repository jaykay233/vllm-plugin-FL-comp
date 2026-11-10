import ctypes, os, subprocess, torch
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "mma_peak2.cu"); LIB = os.path.join(HERE, "libpeak2.so")
r = subprocess.run(["/opt/maca/mxgpu_llvm/bin/mxcc", "-shared", "-fPIC", "-O3",
                    "--offload-arch=xcore1000", "-I/opt/maca/include", SRC, "-o", LIB],
                   capture_output=True, text=True)
print("BUILD", "OK" if r.returncode == 0 else "FAIL")
if r.returncode:
    print(r.stderr[-3000:]); raise SystemExit(1)
lib = ctypes.CDLL(LIB)
lib.launch_peak.argtypes = [ctypes.c_void_p, ctypes.c_void_p] + [ctypes.c_int]*3 + [ctypes.c_void_p]
dev = "cuda"; st = torch.cuda.current_stream().cuda_stream
src = (torch.randint(0, 40, (64*16*32,), device=dev, dtype=torch.int32)).to(torch.uint8).contiguous()
OUT = torch.zeros(8192, device=dev, dtype=torch.float32)
KIND = {0: "bf16 16x16x16", 1: "f16  16x16x16",
        3: "fp8  16x16x32", 4: "bf16 2x16x16 (k=32)"}
print("%-22s %-5s %10s %10s %14s" % ("primitive", "NCH", "ms", "TFLOPS", "max GB/s@M64"))
for kind, name in KIND.items():
    for nch in (1, 8, 16):
        blocks = 416
        try:
            for _ in range(3):
                lib.launch_peak(src.data_ptr(), OUT.data_ptr(), blocks, nch, kind, st)
            torch.cuda.synchronize()
            a, b = torch.cuda.Event(True), torch.cuda.Event(True)
            a.record()
            for _ in range(10):
                lib.launch_peak(src.data_ptr(), OUT.data_ptr(), blocks, nch, kind, st)
            b.record(); torch.cuda.synchronize()
            t = a.elapsed_time(b)/10
        except Exception as e:
            print("%-22s %-5d FAIL %s: %s" % (name, nch, type(e).__name__, str(e)[:120])); continue
        kdim = 32 if kind in (3, 4) else 16
        flops = blocks*4*nch*256*2*16*16*kdim
        tf = flops/(t*1e9)
        print("%-22s %-5d %10.4f %10.1f %14.0f" % (name, nch, t, tf, tf*1000/(2*64)))
