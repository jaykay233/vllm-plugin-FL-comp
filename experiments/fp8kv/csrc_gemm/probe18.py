import ctypes, os, torch
lib=ctypes.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)),"libgemm_v18.so"))
lib.v18_probe.argtypes=[ctypes.c_int,ctypes.c_int]; lib.v18_probe.restype=ctypes.c_int
torch.zeros(1,device="cuda")
TC={20:"WPB2 NT2 KC64 (current)",40:"WPB1 NT1 KC64",41:"WPB1 NT2 KC64",
    42:"WPB1 NT4 KC64",43:"WPB1 NT2 KC32",44:"WPB1 NT1 KC128",
    45:"WPB2 NT1 KC64",46:"WPB2 NT1 KC128",47:"WPB2 NT2 KC128"}
print("%-26s %7s %5s %6s %8s"%("config","smem","regs","blk/SM","tiles@2048"))
for tc in sorted(TC):
    print("%-26s %7d %5d %6d"%(TC[tc],lib.v18_probe(tc,0),
          lib.v18_probe(tc,1),lib.v18_probe(tc,3)))
