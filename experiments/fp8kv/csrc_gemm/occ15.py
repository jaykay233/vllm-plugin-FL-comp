import ctypes, os, torch
lib=ctypes.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)),"libgemm_v15.so"))
lib.v15_smem_bytes.restype=ctypes.c_int; lib.v15_occ_blocks.restype=ctypes.c_int
torch.zeros(1,device="cuda")
TC={20:"WPB4 NT4 KC64 padA16 padF16 AS2",30:"WPB4 NT4 KC32 padA16 padF16 AS2",
    31:"WPB4 NT4 KC32 padA16 padF16 AS1",32:"WPB4 NT4 KC32 padA16 padF0  AS2",
    33:"WPB4 NT4 KC32 padA16 padF0  AS1",34:"WPB4 NT2 KC32 padA16 padF16 AS1",
    35:"WPB2 NT2 KC64 padA16 padF16 AS1",36:"WPB4 NT4 KC32 padA16 padF16 AS1 (dup)",
    37:"WPB2 NT4 KC32 padA16 padF16 AS1"}
print("%-38s %8s %6s"%("config","smem B","blk/SM"))
for tc in sorted(TC):
    print("%-38s %8d %6d"%(TC[tc],lib.v15_smem_bytes(tc),lib.v15_occ_blocks(tc)))
