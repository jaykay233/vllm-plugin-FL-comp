"""Is the split path host-bound rather than GPU-bound?

Splitting a mixed batch wins 3.1-3.4x on GPU time per layer but lost 2.4% end to
end.  The suspicion is host cost: the split issues two kernels per layer instead
of one, and rebuilds flashinfer page metadata every call, on a device where an
earlier profile put GPU utilisation at only 12-20% of the step.

So measure the two sides separately.  "issue" is wall time to enqueue the work
without synchronising, i.e. what the CPU must pay per layer.  "gpu" is CUDA
event time for the same work.  A step runs 42 layers, so whichever side is
larger sets the floor for the step.
"""

from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("VLLM_PLUGINS", "fl")
sys.path.insert(0, "/root/bench_results")

import torch  # noqa: E402

N_QH, N_KVH, HD, BLOCK = 16, 2, 128, 16
DT, DEV = torch.bfloat16, "cuda"
N_LAYERS = 42


def gpu_ms(fn, warm=5, reps=20):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(reps):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / reps


def issue_ms(fn, warm=5, reps=20):
    """CPU time to enqueue, measured without waiting for the GPU.

    The queue is drained first so the measurement is not distorted by work
    already in flight, and nothing is synchronised inside the timed region.
    """
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    t1 = time.perf_counter()
    torch.cuda.synchronize()
    return (t1 - t0) * 1e3 / reps


def build(q_lens, ctx):
    nb = ctx // BLOCK
    n = len(q_lens)
    num_blocks = n * nb
    k_hnd = torch.randn(num_blocks, N_KVH, BLOCK, HD, dtype=DT, device=DEV) * 0.5
    v_hnd = torch.randn(num_blocks, N_KVH, BLOCK, HD, dtype=DT, device=DEV) * 0.5
    kc, vc = k_hnd.permute(0, 2, 1, 3), v_hnd.permute(0, 2, 1, 3)
    q = torch.randn(int(sum(q_lens)), N_QH, HD, dtype=DT, device=DEV) * 0.5
    bt = torch.randperm(num_blocks, device=DEV).to(torch.int32).reshape(n, nb)
    lens = torch.tensor(q_lens, dtype=torch.int32, device=DEV)
    cu = torch.zeros(n + 1, dtype=torch.int32, device=DEV)
    cu[1:] = torch.cumsum(lens, 0)
    sk = torch.full((n,), ctx, dtype=torch.int32, device=DEV)
    return q, kc, vc, bt, cu, sk


def count_launches(fn, label):
    from torch.profiler import ProfilerActivity, profile

    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
    evs = prof.key_averages()
    kernels = sum(
        e.count for e in evs
        if e.device_type.name == "CUDA" and e.self_device_time_total > 0
    )
    cpu_us = sum(e.self_cpu_time_total for e in evs)
    return kernels / 5.0, cpu_us / 5.0 / 1e3


def main() -> int:
    import vllm_fl.dispatch.backends.vendor.iluvatar  # noqa: F401
    import vllm.v1.attention.ops.triton_unified_attention as _uam
    import vllm.v1.attention.backends.triton_attn as _ta
    from flag_gems.runtime.backend._iluvatar.fused.paged_attention_decode import (
        paged_attention_decode,
    )
    from flag_gems.runtime.backend._iluvatar.fused.flash_attn_prefill import (
        flash_attn_prefill,
    )

    from flag_gems.runtime.backend._iluvatar.fused import (
        clear_batch_split,
        set_batch_split,
    )

    wrapper = _ta.unified_attention
    triton_ua = None
    for cell in wrapper.__closure__ or ():
        c = cell.cell_contents
        if callable(c) and getattr(c, "__name__", "") == "unified_attention":
            triton_ua = c
    if triton_ua is None or wrapper.__name__ != "_unified_attention_ixformer":
        print("could not find the raw Triton entry behind the plugin wrapper")
        return 1
    scale = 1.0 / (HD ** 0.5)
    lines = []

    def log(s=""):
        print(s, flush=True)
        lines.append(s)

    CTX = 4096
    cases = [
        ("63 decode + 1 chunk q=2048", 63, [2048]),
        ("32 decode + 4 chunks q=512", 32, [512] * 4),
    ]

    log("Per attention layer.  issue = CPU time to enqueue (no sync);")
    log("gpu = CUDA event time.  A decode step runs 42 such layers.")
    log("triton/split call the kernels directly; eng-tri/eng-spl go through the")
    log("plugin wrapper and its dispatch gate, which is what the engine pays.")
    log()

    for name, n_dec, chunks in cases:
        torch.manual_seed(0)
        q_lens = [1] * n_dec + list(chunks)
        n = len(q_lens)
        q, kc, vc, bt, cu, sk = build(q_lens, CTX)
        max_q = int(max(q_lens))
        out_t, out_s = torch.empty_like(q), torch.empty_like(q)

        ua_kwargs = dict(
            q=q, k=kc, v=vc, out=out_t, cu_seqlens_q=cu, max_seqlen_q=max_q,
            seqused_k=sk, max_seqlen_k=CTX, softmax_scale=scale, causal=True,
            window_size=(-1, -1), block_table=bt, softcap=0.0,
            q_descale=None, k_descale=None, v_descale=None,
        )

        def run_triton():
            triton_ua(**ua_kwargs)

        def run_eng_triton():
            clear_batch_split()
            wrapper(**ua_kwargs)

        def run_eng_split():
            set_batch_split(n_dec, n_dec, int(q.shape[0]))
            wrapper(**ua_kwargs)

        def run_split():
            paged_attention_decode(
                out=out_s[:n_dec], query=q[:n_dec], key_cache=kc, value_cache=vc,
                block_table=bt[:n_dec], seqused_k=sk[:n_dec],
                softmax_scale=scale, max_seqlen_k=CTX,
            )
            flash_attn_prefill(
                out=out_s[n_dec:], query=q[n_dec:], key_cache=kc, value_cache=vc,
                cu_seqlens_q=cu[n_dec:] - cu[n_dec], max_seqlen_q=max_q,
                seqused_k=sk[n_dec:], max_seqlen_k=CTX,
                block_table=bt[n_dec:], softmax_scale=scale,
            )

        log(f"=== {name} ===")
        log(f"{'path':>8} {'issue ms':>10} {'gpu ms':>9} {'kernels':>9} "
            f"{'42x issue':>11} {'42x gpu':>9} {'bound by':>10}")
        for label, fn in (
            ("triton", run_triton),
            ("split", run_split),
            ("eng-tri", run_eng_triton),
            ("eng-spl", run_eng_split),
        ):
            i = issue_ms(fn)
            g = gpu_ms(fn)
            k, _ = count_launches(fn, label)
            bound = "host" if i > g else "gpu"
            log(f"{label:>8} {i:10.3f} {g:9.3f} {k:9.1f} "
                f"{i * N_LAYERS:11.1f} {g * N_LAYERS:9.1f} {bound:>10}")
        log()

    log("If 42x issue for split exceeds 42x gpu for triton, the split cannot")
    log("pay off however fast its kernels are.")

    with open("/root/bench_results/split_host_overhead.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
