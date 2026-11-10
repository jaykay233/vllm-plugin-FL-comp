import ctypes, os, subprocess, torch, struct

HERE = os.path.dirname(os.path.abspath(__file__))
MX = "/opt/maca/mxgpu_llvm/bin/mxcc"
SRC = os.path.join(HERE, "mma_bf16_probe.cu")
SO = os.path.join(HERE, "libbfprobe.so")

if not os.path.exists(SO) or os.path.getmtime(SRC) > os.path.getmtime(SO):
    subprocess.run([MX, "-shared", "-fPIC", "-std=c++17", "-O2",
                    "--offload-arch=xcore1000", "-I/opt/maca/include",
                    "-I/opt/maca/include/mcr", "-I/opt/maca/include/common",
                    "-L/opt/maca/lib", "-lmxc-runtime64", "-o", SO, SRC],
                   check=True)

lib = ctypes.CDLL(SO)
lib.launch_probe_ones.argtypes = [ctypes.c_uint32, ctypes.c_uint32,
                                  ctypes.c_void_p, ctypes.c_void_p]
lib.launch_probe_ones.restype = ctypes.c_int


def bf16_bits(x):
    """float -> bf16 bit pattern, round to nearest even."""
    u = struct.unpack("<I", struct.pack("<f", x))[0]
    u += 0x7FFF + ((u >> 16) & 1)
    return u >> 16


def fp16_bits(x):
    return struct.unpack("<H", struct.pack("<e", x))[0]


def run(ba, bb, label):
    out = torch.zeros(64 * 4, dtype=torch.float32, device="cuda")
    lib.launch_probe_ones(ba, bb, out.data_ptr(), None)
    torch.cuda.synchronize()
    o = out.cpu()
    print(f"  {label:<34s} total={o.sum().item():12.4f} "
          f"min={o.min().item():9.4f} max={o.max().item():9.4f}")
    return o


print("expectation: both-ones -> every C element 16, total 4096\n")
o = run(bf16_bits(1.0), bf16_bits(1.0), "bf16 bits for 1.0 in both")
run(bf16_bits(1.0), 0, "A=bf16(1.0), B=0")
run(fp16_bits(1.0), fp16_bits(1.0), "fp16 bits for 1.0 in both")
print()
print(f"  bf16(1.0) = 0x{bf16_bits(1.0):04X}   fp16(1.0) = 0x{fp16_bits(1.0):04X}")

# How many distinct values appear, and where?  If the layout is the natural
# one, every lane holds four equal values.
u = o.unique().tolist()
print(f"  distinct values: {u}")
