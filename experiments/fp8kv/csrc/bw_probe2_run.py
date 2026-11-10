import ctypes, torch

LIB = "/workspace/vllm-plugin-FL-comp/experiments/fp8kv/csrc/libbw2.so"
lib = ctypes.CDLL(LIB)

lib.launch_stream_pages.argtypes = [
    ctypes.c_void_p] * 5 + [ctypes.c_int] * 7 + [ctypes.c_void_p]
lib.launch_stream_flat.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
for f in (lib.launch_stream_pages, lib.launch_stream_flat):
    f.restype = ctypes.c_int


def bench(fn, iters=30):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def probe_pages(B, SEQ, HKV, P, splits, W, tstep=32):
    D = 128
    npg = (SEQ + P - 1) // P
    nb = B * npg
    k = torch.randint(0, 255, (nb * P * HKV * D,), device="cuda", dtype=torch.uint8)
    v = torch.randint(0, 255, (nb * P * HKV * D,), device="cuda", dtype=torch.uint8)
    bt = torch.arange(nb, device="cuda", dtype=torch.int32).reshape(B, npg)
    lens = torch.full((B,), SEQ, device="cuda", dtype=torch.int32)
    out = torch.empty(B * HKV * splits, device="cuda", dtype=torch.float32)
    nw = B * HKV * splits
    ctas = (nw + W - 1) // W
    fn = lambda: lib.launch_stream_pages(
        k.data_ptr(), v.data_ptr(), bt.data_ptr(), lens.data_ptr(),
        out.data_ptr(), B, HKV, P, bt.stride(0), splits, W, tstep, None)
    t = bench(fn)
    nbytes = B * SEQ * HKV * D * 2
    return t, nbytes / (t * 1e6), ctas, nw


def probe_flat(MB, W):
    n4 = MB * 1024 * 1024 // 16
    src = torch.empty(n4 * 16, device="cuda", dtype=torch.uint8)
    out = torch.empty(2048 * W, device="cuda", dtype=torch.float32)
    fn = lambda: lib.launch_stream_flat(
        src.data_ptr(), out.data_ptr(), n4, W, None)
    t = bench(fn)
    return t, MB / 1000 / (t / 1000) / 1000


def main():
    B, SEQ, HKV, P = 64, 4096, 2, 64
    lines = []
    lines.append("== paged K+V stream, coalesced 16B/lane, no compute ==")
    for W in (1, 2, 4, 8):
        for S in (8, 16, 32):
            t, gbs, ctas, nw = probe_pages(B, SEQ, HKV, P, S, W)
            lines.append(f"W={W} S={S:<2d} ctas={ctas:<5d} warps={nw:<6d} "
                         f"{t:7.4f} ms  {gbs:7.0f} GB/s")
    lines.append("")
    lines.append("== flat contiguous read (machine peak) ==")
    for W in (1, 2, 4, 8):
        t, gbs = probe_flat(512, W)
        lines.append(f"W={W} {t:7.4f} ms  {gbs:7.0f} GB/s")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
