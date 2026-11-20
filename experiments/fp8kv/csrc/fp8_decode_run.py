"""Definitive benchmark for the hand-written MACA FP8 decode-attention kernel.

Reports, for MiniCPM5-2B decode shape (B=64, seq=4096, HKV=2, H=16, D=128, P=64):
  * correctness vs an unquantised torch reference (fp8 tolerance)
  * split-K sweep for the direct (gmem) and the shared-memory staged kernel
  * phase ablation (QK / PV / softmax / loop) to locate the cost
  * split vs combine kernel cost
  * vendor FlashAttention bf16 comparison

mode bits understood by launch_fp8_decode:
  1 skip PV | 2 skip QK | 4 smem-staged kernel | 8 skip softmax math
  16 fake exp2 | 32 fake fp8 pack
  default (0) = bf16 tensor-core kernel   <- fastest, 0.356 ms/layer
  64 fp8-MMA async kernel | 128 register-prefetch pipe1 | 256 double-buffered pipe
  4096 bf16 kernel (explicit) | 8192 bf16 + async-staged V (slower, kept for data)
  512 skip combine kernel | 1024 skip split kernel | 2048 legacy direct kernel
"""
import ctypes, torch

LIB = "/workspace/vllm-plugin-FL/experiments/fp8kv/csrc/libfp8_decode.so"
lib = ctypes.CDLL(LIB)
lib.launch_fp8_decode.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_float,
    ctypes.c_int, ctypes.c_void_p]
lib.launch_fp8_decode.restype = ctypes.c_int


def make(B, SEQ, HKV, D, P, fp8=True):
    H = HKV * 8
    q = torch.randn(B, H, D, device="cuda", dtype=torch.bfloat16)
    npg = (SEQ + P - 1) // P
    nb = B * npg
    kv = torch.randn(nb, P, HKV, D, device="cuda", dtype=torch.bfloat16) * 0.5
    bt = torch.arange(nb, device="cuda", dtype=torch.int32).reshape(B, npg)
    lens = torch.full((B,), SEQ, device="cuda", dtype=torch.int32)
    if fp8:
        kv = kv.to(torch.float8_e4m3fn)
    return q, kv, kv.clone(), bt, lens


class Runner:
    """All buffers preallocated: the timed region is a single ctypes call."""

    def __init__(self, q, k, v, bt, lens, S, mode=0, sm_scale=None):
        B, H, D = q.shape
        self.q, self.k, self.v, self.bt, self.lens = q, k, v, bt, lens
        self.B, self.H, self.HKV, self.P = B, H, k.shape[2], k.shape[1]
        self.mode, self.S = mode, S
        self.sm = D ** -0.5 if sm_scale is None else sm_scale
        self.mid_o = torch.empty((S, B, H, D), device="cuda", dtype=torch.float32)
        self.mid_m = torch.empty((S, B, H), device="cuda", dtype=torch.float32)
        self.mid_l = torch.empty((S, B, H), device="cuda", dtype=torch.float32)
        self.out = torch.empty_like(q)

    def __call__(self):
        lib.launch_fp8_decode(
            self.q.data_ptr(), self.k.data_ptr(), self.v.data_ptr(),
            self.bt.data_ptr(), self.lens.data_ptr(), self.out.data_ptr(),
            self.mid_o.data_ptr(), self.mid_m.data_ptr(), self.mid_l.data_ptr(),
            self.B, self.H, self.HKV, self.P, self.bt.stride(0), self.S,
            self.sm, 1.0, 1.0, self.mode, None)
        return self.out


@torch.no_grad()
def reference(q, k, v, bt, lens, sm_scale):
    B, H, D = q.shape
    HKV, P = k.shape[2], k.shape[1]
    g = H // HKV
    kd, vd = k.float(), v.float()
    out = torch.empty(B, H, D, device=q.device, dtype=torch.float32)
    for b in range(B):
        n = int(lens[b])
        pages = bt[b, :(n + P - 1) // P].long()
        kk = kd[pages].reshape(-1, HKV, D)[:n]
        vv = vd[pages].reshape(-1, HKV, D)[:n]
        for kvh in range(HKV):
            qg = q[b, kvh * g:(kvh + 1) * g].float()
            p = torch.softmax((qg @ kk[:, kvh, :].t()) * sm_scale, dim=-1)
            out[b, kvh * g:(kvh + 1) * g] = p @ vv[:, kvh, :]
    return out.to(torch.bfloat16)


def bench(fn, iters=50):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def main():
    torch.manual_seed(0)
    D, HKV = 128, 2
    sm = D ** -0.5
    p = torch.cuda.get_device_properties(0)
    print(f"{torch.cuda.get_device_name(0)}  SMs={p.multi_processor_count}")

    # ---------------- correctness ----------------
    print("\n=== correctness vs torch reference (unquantised) ===")
    for P in (16, 32, 64):
        q, k, v, bt, lens = make(4, 256, HKV, D, P)
        ref = reference(q, k, v, bt, lens, sm)
        for S in (1, 3, 8):
            row = []
            for mode, tag in ((0, "bf16"), (64, "fp8"), (2048, "gmem")):
                out = Runner(q, k, v, bt, lens, S, mode, sm)()
                err = (out.float() - ref.float()).abs().max().item()
                cos = torch.nn.functional.cosine_similarity(
                    out.float().flatten(), ref.float().flatten(), dim=0).item()
                row.append(f"{tag} maxabs={err:.5f} cos={cos:.6f}")
            print(f"  P={P:<2d} S={S:<2d}  " + "   ".join(row))

    # ---------------- benchmark ----------------
    B, SEQ, P = 64, 4096, 64
    q, k, v, bt, lens = make(B, SEQ, HKV, D, P)
    _, kb, vb, _, _ = make(B, SEQ, HKV, D, P, fp8=False)
    nbytes = B * SEQ * HKV * D * 2          # K+V, 1 byte/elt
    LAYERS = 42
    print(f"\n=== benchmark: {nbytes/1e6:.1f} MB K+V per layer, "
          f"B={B} seq={SEQ} fp8 ===")

    print("  split-K sweep (bf16 / fp8 / gmem / smem / pipe1):")
    best = {}
    for tag, mode in (("bf16", 0), ("fp8", 64), ("gmem", 2048), ("smem", 4), ("pipe1", 128)):
        for S in (1, 2, 4, 8, 16, 32):
            t = bench(Runner(q, k, v, bt, lens, S, mode, sm))
            best[tag] = min(best.get(tag, (1e9, 0)), (t, S))
            print(f"    {tag:<5s} S={S:<2d} {t:8.4f} ms/layer "
                  f"{nbytes/(t*1e6):6.0f} GB/s  x42={t*LAYERS:6.2f} ms")

    print("\n  phase ablation at S=16 (bf16):")
    base = bench(Runner(q, k, v, bt, lens, 16, 3, sm))          # no QK, no PV
    qk = bench(Runner(q, k, v, bt, lens, 16, 1, sm))            # QK only
    pv = bench(Runner(q, k, v, bt, lens, 16, 2, sm))            # PV only
    full = bench(Runner(q, k, v, bt, lens, 16, 0, sm))
    nsm = bench(Runner(q, k, v, bt, lens, 16, 8, sm))           # no softmax
    for name, t in (("loop + softmax", base), ("+ QK  (K loads + mma)", qk),
                    ("+ PV  (V loads + mma)", pv), ("QK+PV, no softmax", nsm),
                    ("full", full)):
        print(f"    {name:<24s} {t:8.4f} ms  {nbytes/(t*1e6):6.0f} GB/s")

    print("\n  kernel split (S=16):")
    both = bench(Runner(q, k, v, bt, lens, 16, 0, sm))
    split_only = bench(Runner(q, k, v, bt, lens, 16, 512, sm))
    comb_only = bench(Runner(q, k, v, bt, lens, 16, 1024, sm))
    print(f"    split kernel  {split_only:8.4f} ms")
    print(f"    combine kernel{comb_only:8.4f} ms   (both {both:.4f} ms)")

    try:
        from flash_attn import flash_attn_with_kvcache
        qd = q.unsqueeze(1)
        t = bench(lambda: flash_attn_with_kvcache(
            qd, kb, vb, block_table=bt, cache_seqlens=lens, causal=True))
        print(f"\n  vendor FA bf16   {t:8.4f} ms/layer "
              f"{2*nbytes/(t*1e6):6.0f} GB/s  x42={t*LAYERS:6.2f} ms")
        print(f"  RATIO best kernel / FA = {best['bf16'][0]/t:.2f}x")
    except Exception as exc:
        print("  vendor FA skipped:", type(exc).__name__, str(exc)[:100])


if __name__ == "__main__":
    main()
