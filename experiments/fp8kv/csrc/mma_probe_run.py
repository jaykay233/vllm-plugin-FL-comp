import ctypes, torch

LIB = "/workspace/vllm-plugin-FL-comp/experiments/fp8kv/csrc/libmma.so"
lib = ctypes.CDLL(LIB)
lib.launch_mma_probe.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
                                 ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                 ctypes.c_void_p]
lib.launch_mma_probe.restype = ctypes.c_int

NAMES = {0: "mma_thru (4 indep chains)", 1: "mma_lat  (1 dep chain)",
         2: "fma_thru"}


def run(which, ctas, warps, iters=4096, ops_per_iter=4):
    out = torch.zeros(ctas, device="cuda", dtype=torch.float32)
    cyc = torch.zeros(ctas, device="cuda", dtype=torch.int64)
    lib.launch_mma_probe(which, out.data_ptr(), cyc.data_ptr(), iters, warps,
                         ctas, None)
    torch.cuda.synchronize()
    c = cyc.float().mean().item()
    per = c / (iters * ops_per_iter)
    return per, c


def main():
    p = torch.cuda.get_device_properties(0)
    print(f"{torch.cuda.get_device_name(0)}  SMs={p.multi_processor_count}")

    print("\n-- MMA instruction cost, cycles per instruction per warp --")
    for which in (0, 1):
        for warps, ctas in ((1, 104), (2, 104), (4, 104), (8, 104), (8, 416)):
            per, tot = run(which, ctas, warps)
            print(f"  {NAMES[which]:<26s} ctas={ctas:<4d} warps/cta={warps} "
                  f"-> {per:8.2f} cyc/mma   (total {tot/1e6:7.3f} Mcyc)")

    print("\n-- fp32 FMA reference --")
    per, _ = run(2, 104, 8)
    print(f"  fma_thru warps/cta=8 -> {per:8.3f} cyc/fma")

    # effective MMA rate assuming a ~1.5 GHz part
    per, _ = run(0, 104, 8)
    warps_tot = 104 * 8
    ops = 16 * 16 * 32 * 2
    rate = warps_tot / per            # mma per cycle for the whole GPU
    print(f"\nThroughput: {rate:.1f} mma/cycle/GPU  = "
          f"{rate*ops:.1f} FLOP/cycle  -> at 1.5GHz = "
          f"{rate*ops*1.5e9/1e12:.1f} TFLOP/s")


if __name__ == "__main__":
    main()
