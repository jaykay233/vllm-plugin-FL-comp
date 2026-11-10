import ctypes, os, subprocess, torch

HERE = os.path.dirname(os.path.abspath(__file__))
MX = "/opt/maca/mxgpu_llvm/bin/mxcc"
SRC = os.path.join(HERE, "mma_bf16_bump.cu")
SO = os.path.join(HERE, "libbfbump.so")

if not os.path.exists(SO) or os.path.getmtime(SRC) > os.path.getmtime(SO):
    subprocess.run([MX, "-shared", "-fPIC", "-std=c++17", "-O2",
                    "--offload-arch=xcore1000", "-I/opt/maca/include",
                    "-I/opt/maca/include/mcr", "-I/opt/maca/include/common",
                    "-L/opt/maca/lib", "-lmxc-runtime64", "-o", SO, SRC],
                   check=True)

lib = ctypes.CDLL(SO)
lib.launch_probe_bump.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                  ctypes.c_void_p, ctypes.c_void_p]
lib.launch_probe_bump.restype = ctypes.c_int

buf = torch.zeros(256, dtype=torch.float32, device="cuda")
flat = None


def run(which, lane=0, slot=0):
    global flat
    lib.launch_probe_bump(which, lane, slot, buf.data_ptr(), None)
    torch.cuda.synchronize()
    flat = buf.cpu().clone()
    return flat


base = run(2)
assert (base == 16.0).all(), base.unique()
print(f"baseline: all {base.unique().tolist()}  (ok)\n")


def changed():
    idx = (flat > 16.5).nonzero().flatten().tolist()
    lanes = sorted({i // 4 for i in idx})
    slots = sorted({i % 4 for i in idx})
    return idx, lanes, slots


print("=== A map: bumping A(lane,slot) lights up a whole C row ===")
print("  A elem        -> #C     C lanes                        C slots")
amap = {}
for L in range(64):
    for S in range(4):
        run(0, L, S)
        idx, lanes, slots = changed()
        amap[(L, S)] = (tuple(lanes), tuple(slots))
        if S == 0 or L < 4:
            print(f"  L={L:<2d} S={S}  -> {len(idx):3d}   "
                  f"{str(lanes)[:34]:<34s} {slots}")

print("\n=== B map: bumping B(lane,slot) lights up a whole C column ===")
bmap = {}
for L in range(64):
    for S in range(4):
        run(1, L, S)
        idx, lanes, slots = changed()
        bmap[(L, S)] = (tuple(lanes), tuple(slots))
        if S == 0 or L < 4:
            print(f"  L={L:<2d} S={S}  -> {len(idx):3d}   "
                  f"{str(lanes)[:34]:<34s} {slots}")

# Group A elements by the C row they belong to, and B by the C column.
def group(m, key):
    g = {}
    for k, v in m.items():
        g.setdefault(v, []).append(k)
    return g

arows = group(amap, 0)
bcols = group(bmap, 0)
print(f"\nA: {len(arows)} distinct rows, sizes {sorted({len(v) for v in arows.values()})}")
print(f"B: {len(bcols)} distinct cols, sizes {sorted({len(v) for v in bcols.values()})}")

# Which C positions belong to one row?  Each row should contain 16 C positions.
rowpos = sorted(arows.keys())
print("\n  row signature -> its 4 A elements (lane,slot):")
for r in rowpos[:6]:
    print(f"    {str(r)[:44]:<44s} {arows[r]}")
print("\n  col signature -> its 4 B elements (lane,slot):")
for r in sorted(bcols.keys())[:6]:
    print(f"    {str(r)[:44]:<44s} {bcols[r]}")
