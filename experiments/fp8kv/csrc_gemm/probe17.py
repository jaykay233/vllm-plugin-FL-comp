import ctypes, os, torch
lib=ctypes.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)),"libgemm_v17.so"))
lib.v17_probe.argtypes=[ctypes.c_int,ctypes.c_int]; lib.v17_probe.restype=ctypes.c_int
torch.zeros(1,device="cuda")
TC={20:"NT4 KC64 v13best",21:"NT4 KC64 PERSIST",22:"NT8 KC32 wide",
    23:"NT8 KC32 wide PERSIST",24:"NT6 KC32 wide",25:"NT4 KC32 (KC32 base)",
    26:"NT4 KC32 PERSIST",27:"NT2 KC64 PERSIST",28:"NT6 KC32 wide PERSIST"}
print("%-26s %7s %5s %7s %6s"%("config","smem","regs","spill","blk/SM"))
for tc in sorted(TC):
    print("%-26s %7d %5d %7d %6d"%(TC[tc],lib.v17_probe(tc,0),lib.v17_probe(tc,1),
          lib.v17_probe(tc,2),lib.v17_probe(tc,3)))
