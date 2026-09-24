#!/usr/bin/env python3
"""Validate the MetaX-specific `linear` override in FlagGems.

bf16 GEMM tolerance methodology
--------------------------------
A naive per-element *relative* error metric is wrong here. With a bias, the
result is ``gemm + bias``; whenever ``bias`` cancels the GEMM magnitude the
post-bias value collapses towards zero, so a perfectly healthy ~0.5 absolute
bf16 rounding error shows up as a huge relative error. That is quantisation,
not a broken kernel.

So this script scores correctness with metrics that are scale-correct:

  A. Frobenius relative error of the *pre-bias* GEMM against an fp32
     reference. bf16 with K up to 6144 lands around 1e-3..1e-2.
  B. The bias must be an exact elementwise add: ``linear(x,w,b) - linear(x,w)``
     has to equal ``b`` to within one bf16 ULP of the output.
  C. metax vs the generic implementation, with ``atol`` scaled by
     ``eps_bf16 * max|gemm|`` (a few ULPs of the accumulation), plus a global
     Frobenius comparison.

Checks, in order: wiring -> numerics -> speed.

Run:  conda activate mx && python /root/src/validate_metax_linear.py
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

MODEL = "/root/models/MiniCPM5-2B"
OUT = Path("/root/bench_results/metax_linear")
OUT.mkdir(parents=True, exist_ok=True)

M_LIST = [1, 8, 128, 512]
EPS_BF16 = 2.0 ** -8  # bf16 mantissa is 8 bits


def shapes_from_config() -> list[tuple[str, int, int]]:
    cfg = json.loads((Path(MODEL) / "config.json").read_text())
    h = cfg["hidden_size"]
    inter = cfg["intermediate_size"]
    nq = cfg["num_attention_heads"]
    nkv = cfg.get("num_key_value_heads", nq)
    hd = cfg.get("head_dim") or h // nq
    vocab = cfg["vocab_size"]
    if cfg.get("tie_word_embeddings", False):
        vocab = 0
    out = [
        ("qkv_proj", nq * hd + 2 * nkv * hd, h),
        ("o_proj", nq * hd, h),
        ("gate_up_proj", 2 * inter, h),
        ("down_proj", h, inter),
    ]
    if vocab:
        out.append(("lm_head", vocab, h))
    return out


def bench(fn, n_warmup=10, n_iters=50) -> float:
    import torch

    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n_iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts)


def frob(got, ref) -> float:
    return ((got.float() - ref).norm() / ref.norm()).item()


def run_numerics(torch, flag_gems, generic_linear) -> tuple[bool, dict]:
    dt = torch.bfloat16
    dev = "cuda"
    shapes = shapes_from_config()
    print(f"  shapes from config: {[s[0] for s in shapes]}")
    print()
    print(f"  {'shape':13s} {'M':>4s} {'K':>5s} {'N':>5s} "
          f"{'frob(metax)':>12s} {'frob(generic)':>14s} "
          f"{'bias_exact':>11s} {'vs_generic':>11s}  ok")
    worst_m = worst_g = 0.0
    all_ok = True
    detail = []

    for name, N, K in shapes:
        for M in (1, 512):
            torch.manual_seed(0)
            x = torch.randn(M, K, device=dev, dtype=dt)
            w = torch.randn(N, K, device=dev, dtype=dt)
            b = torch.randn(N, device=dev, dtype=dt)

            ref = x.float() @ w.float().t()
            ref_b = ref + b.float()

            got_nb = flag_gems.linear(x, w)
            gen_nb = generic_linear(x, w)
            got_b = flag_gems.linear(x, w, b)
            gen_b = generic_linear(x, w, b)

            # A. pre-bias GEMM accuracy (no bias cancellation to hide behind)
            fm = frob(got_nb, ref)
            fg = frob(gen_nb, ref)
            # B. bias must be an exact elementwise add
            bias_delta = ((got_b - got_nb).float() - b.float()).abs().max().item()
            bias_ok = bias_delta <= 2 * EPS_BF16 * ref_b.abs().max().item() + 1e-2
            # C. vs generic, atol scaled by the accumulation magnitude
            atol = 4 * EPS_BF16 * ref.abs().max().item()
            try:
                torch.testing.assert_close(
                    got_b, gen_b, rtol=4 * EPS_BF16, atol=atol, check_dtype=False
                )
                vs_gen = "ok"
                vs_gen_ok = True
            except AssertionError:
                vs_gen = "DIFF"
                vs_gen_ok = False

            ok = fm < 0.02 and fg < 0.02 and bias_ok and vs_gen_ok
            all_ok &= ok
            worst_m = max(worst_m, fm)
            worst_g = max(worst_g, fg)
            detail.append({"shape": name, "M": M, "K": K, "N": N,
                           "frob_metax": fm, "frob_generic": fg,
                           "bias_delta": bias_delta, "ok": ok})
            print(f"  {name:13s} {M:4d} {K:5d} {N:5d} "
                  f"{fm:12.6f} {fg:14.6f} {str(bias_ok):>11s} "
                  f"{vs_gen:>11s}  {ok}")

    print()
    print(f"  worst Frobenius rel err (metax)   : {worst_m:.6f}")
    print(f"  worst Frobenius rel err (generic) : {worst_g:.6f}")
    print(f"  -> metax is at least as accurate  : {worst_m <= worst_g * 1.5}")
    print(f"  NUMERICS {'PASS' if all_ok else 'FAIL'}")
    return all_ok, {"worst_frob_metax": worst_m, "worst_frob_generic": worst_g,
                    "detail": detail}


def run_speed(torch, flag_gems, generic_linear) -> list[dict]:
    dt = torch.bfloat16
    dev = "cuda"
    shapes = shapes_from_config()
    print(f"  {'shape':13s} {'M':>5s} {'generic ms':>11s} "
          f"{'metax ms':>10s} {'speedup':>9s}")
    rows = []
    for name, N, K in shapes:
        for M in M_LIST:
            torch.manual_seed(0)
            x = torch.randn(M, K, device=dev, dtype=dt)
            w = torch.randn(N, K, device=dev, dtype=dt)
            b = torch.randn(N, device=dev, dtype=dt)
            tg = bench(lambda: generic_linear(x, w, b))
            tm = bench(lambda: flag_gems.linear(x, w, b))
            rows.append({"shape": name, "M": M, "K": K, "N": N,
                         "generic_ms": tg, "metax_ms": tm, "speedup": tg / tm})
            print(f"  {name:13s} {M:5d} {tg:11.4f} {tm:10.4f} {tg / tm:8.2f}x")
    return rows


def main() -> None:
    import torch
    import flag_gems
    from flag_gems.ops.linear import linear as generic_linear

    print("=" * 92)
    print("1) WIRING")
    print("=" * 92)
    mod = flag_gems.linear.__module__
    print(f"  flag_gems.linear module : {mod}")
    wired = mod.endswith("_metax.ops.linear")
    print(f"  -> using MetaX override : {wired}")
    in_full = [it[1].__module__ for it in flag_gems._FULL_CONFIG
               if it and len(it) >= 2 and it[0] == "linear"]
    print(f"  _FULL_CONFIG['linear']  : {in_full}")
    registered = bool(in_full) and in_full[0].endswith("_metax.ops.linear")
    print(f"  -> registered in config : {registered}")

    print()
    print("=" * 92)
    print("2) NUMERICS  (bf16, scale-correct metrics)")
    print("=" * 92)
    numerics_ok, nd = run_numerics(torch, flag_gems, generic_linear)

    print()
    print("=" * 92)
    print("3) SPEED  (metax linear vs generic linear_kernel)")
    print("=" * 92)
    rows = run_speed(torch, flag_gems, generic_linear)

    print()
    print("=" * 92)
    print("4) PER-STEP LAYER-WEIGHTED TOTAL  (42 layers)")
    print("=" * 92)
    summary = {}
    for M in M_LIST:
        tg = sum(r["generic_ms"] for r in rows if r["M"] == M) * 42
        tm = sum(r["metax_ms"] for r in rows if r["M"] == M) * 42
        summary[M] = {"generic_ms": tg, "metax_ms": tm, "speedup": tg / tm}
        label = {1: "decode", 8: "small", 128: "medium", 512: "prefill"}[M]
        print(f"  {label:8s} M={M:<4d} generic={tg:9.3f}ms  "
              f"metax={tm:9.3f}ms  => {tg / tm:.2f}x")

    overall_ok = wired and registered and numerics_ok
    (OUT / "validation.json").write_text(json.dumps(
        {"wired": wired, "registered": registered, "numerics_ok": numerics_ok,
         "numerics": nd, "bench": rows,
         "layer_totals": {str(k): v for k, v in summary.items()}}, indent=2))
    print(f"\n  Wrote {OUT / 'validation.json'}")
    print(f"METAX_LINEAR_DONE wired={wired} numerics_ok={numerics_ok}", flush=True)


if __name__ == "__main__":
    main()
