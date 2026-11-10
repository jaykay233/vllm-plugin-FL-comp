import ctypes, os, torch
lib=ctypes.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)),"libgemm_v16.so"))
lib.v16_probe.argtypes=[ctypes.c_int,ctypes.c_int]; lib.v16_probe.restype=ctypes.c_int
torch.zeros(1,device="cuda")
TC={20:"WPB4 NT4 buffered A (v13)",21:"WPB4 NT4 STREAM A",22:"WPB2 NT2 STREAM A",
    23:"WPB2 NT2 buffered A",24:"WPB4 NT2 STREAM A",25:"WPB8 NT2 STREAM A",
    26:"WPB4 NT4 STREAM A padA48",27:"WPB2 NT4 STREAM A"}
print("%-30s %7s %5s %7s %6s"%("config","smem","regs","spill","blk/SM"))
for tc in sorted(TC):
    print("%-30s %7d %5d %7d %6d"%(TC[tc],lib.v16_probe(tc,0),lib.v16_probe(tc,1),
          lib.v16_probe(tc,2),lib.v16_probe(tc,3)))
