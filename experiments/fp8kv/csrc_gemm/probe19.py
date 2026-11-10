import ctypes, os, torch
lib=ctypes.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)),"libgemm_v19.so"))
lib.v19_probe.argtypes=[ctypes.c_int,ctypes.c_int]; lib.v19_probe.restype=ctypes.c_int
torch.zeros(1,device="cuda")
TC={20:"KC64 NT4 256col (current)",50:"KC128 NT2 128col A1",51:"KC128 NT2 64col A2",
    52:"KC128 NT2 64col A1",53:"KC128 NT3 96col A1",54:"KC64 NT2 64col A2",
    55:"KC64 NT2 128col A2"}
print("%-28s %7s %5s %6s"%("config","smem","regs","blk/SM"))
for tc in sorted(TC):
    print("%-28s %7d %5d %6d"%(TC[tc],lib.v19_probe(tc,0),lib.v19_probe(tc,1),lib.v19_probe(tc,3)))
