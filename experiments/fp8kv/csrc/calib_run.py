import ctypes, itertools, torch

lib = ctypes.CDLL("/workspace/vllm-plugin-FL-comp/experiments/fp8kv/csrc/libcalib.so")
lib.launch_calib.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                             ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
lib.launch_calib.restype = ctypes.c_int

torch.manual_seed(0)
# small integers representable exactly in e4m3
Af = (torch.randint(1, 8, (16, 32)).float())
Bf = (torch.randint(1, 8, (16, 32)).float())  # Bn[n][k]
A8 = Af.to(torch.float8_e4m3fn).cuda()
B8 = Bf.to(torch.float8_e4m3fn).cuda()
# reference: C = A @ B^T  (A:[m][k], Bn:[n][k])
ref = (A8.float().cpu() @ B8.float().cpu().t())

outs = {}
for va, vb in itertools.product([0, 1], [0, 1, 2, 3]):
    Cout = torch.full((256,), -999.0, device="cuda")
    lib.launch_calib(A8.data_ptr(), B8.data_ptr(), Cout.data_ptr(), va, vb, None)
    torch.cuda.synchronize()
    C = Cout.cpu().reshape(16, 16)
    err = (C - ref).abs().max().item()
    outs[(va, vb)] = (err, C)

for k, (err, C) in sorted(outs.items(), key=lambda kv: kv[1][0]):
    print(f"VA={k[0]} VB={k[1]}  maxerr={err:.4f}")

best = min(outs.items(), key=lambda kv: kv[1][0])
print("\nBEST", best[0], "err", best[1][0])
print("ref[0,:8]", ref[0, :8].tolist())
print("got[0,:8]", best[1][1][0, :8].tolist())
print("ref[1,:8]", ref[1, :8].tolist())
print("got[1,:8]", best[1][1][1, :8].tolist())
