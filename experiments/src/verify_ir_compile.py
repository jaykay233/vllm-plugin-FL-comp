#!/usr/bin/env python3
"""End-to-end check of the FL IR-op provider path under torch.compile.

Run twice, with the feature on and off, and compare the greedy token ids:

    cd /workspace/vllm-plugin-FL && PYTHONPATH=$PWD \
        IR=1 python /root/src/verify_ir_compile.py     # FL providers active
    cd /workspace/vllm-plugin-FL && PYTHONPATH=$PWD \
        IR=0 python /root/src/verify_ir_compile.py     # stock vLLM (priority=native)

Both must produce identical token ids (same math, different kernels) while the
IR=1 run shows FlagGems kernels for fused_add_rms_norm and a dispatch record for
`rms_norm`.
"""

from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from pathlib import Path

IR = os.environ.get("IR", "1") == "1"
os.environ["VLLM_FL_IR_KERNELS"] = "1" if IR else "0"
os.environ.setdefault("VLLM_FL_DISPATCH_DEBUG", "1")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

MODEL = "/root/models/MiniCPM5-2B"
OUT = Path("/root/bench_results/ir_compile")
OUT.mkdir(parents=True, exist_ok=True)
TAG = "ir_on" if IR else "ir_off"

GEMS_KERNELS = {
    "fused_add_rms_norm_kernel": "rms_norm(FlagGems)",
    "rms_norm_kernel": "rms_norm(FlagGems)",
    "silu_and_mul_kernel": "silu(FlagGems)",
    "apply_rotary_pos_emb_inplace_kernel": "rope(FlagGems)",
    "mm_kernel": "GEMM(FlagGems)",
    "linear_kernel": "GEMM(FlagGems linear)",
}
INDUCTOR_PREFIX = "triton_red_fused_fused_add_rms_norm"


def family(name: str) -> str:
    low = name.lower()
    for k, v in GEMS_KERNELS.items():
        if k in low:
            return v
    if INDUCTOR_PREFIX in low:
        return "rms_norm(inductor)"
    if low.startswith("triton_"):
        return "inductor-fused(other)"
    return "other"


def main() -> None:
    import torch
    from vllm import LLM, SamplingParams

    print(f"\n########## {TAG}: VLLM_FL_IR_KERNELS={os.environ['VLLM_FL_IR_KERNELS']} ##########",
          flush=True)

    t0 = time.perf_counter()
    llm = LLM(model=MODEL, dtype="bfloat16", trust_remote_code=True,
              max_model_len=2048, gpu_memory_utilization=0.85)
    init_s = time.perf_counter() - t0

    # ---- what did the platform publish? ----
    try:
        import vllm.ir.ops  # noqa: F401
        from vllm.ir.op import IrOp
        from vllm_fl.ops.ir_kernels import default_ir_op_priority
        print(f"  FL default_ir_op_priority(): {default_ir_op_priority()}", flush=True)
        for op_name in ("rms_norm", "fused_add_rms_norm"):
            op = IrOp.registry[op_name]
            print(f"  IR {op_name:22s} priority={op.get_priority()} "
                  f"providers={op.supported_providers()}", flush=True)
    except Exception as e:
        import traceback
        print(f"  IR introspection failed: {e}", flush=True)
        traceback.print_exc()

    sp = SamplingParams(max_tokens=24, temperature=0.0, ignore_eos=True)
    prompt = "The capital of France is"

    llm.generate([prompt], SamplingParams(max_tokens=4, temperature=0.0, ignore_eos=True))
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = llm.generate([prompt], sp)
    torch.cuda.synchronize()
    gen_s = time.perf_counter() - t0
    ids = list(out[0].outputs[0].token_ids)
    print(f"  init={init_s:.1f}s gen={gen_s:.2f}s tokens={ids}", flush=True)
    print(f"  text={out[0].outputs[0].text!r}", flush=True)

    # ---- which dispatch ops were actually used? ----
    from vllm_fl.dispatch import get_default_manager
    called = dict(get_default_manager()._called_ops)
    print(f"  FL dispatch ops used: {called}", flush=True)

    # ---- kernel profile ----
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        llm.generate([prompt], sp)
        torch.cuda.synchronize()

    fam_ms: dict[str, float] = defaultdict(float)
    fam_calls: dict[str, int] = defaultdict(int)
    for e in prof.key_averages():
        t = e.self_device_time_total
        if t <= 0:
            continue
        f = family(e.key)
        fam_ms[f] += t / 1e3
        fam_calls[f] += e.count

    print("  --- kernel families ---", flush=True)
    for f, ms in sorted(fam_ms.items(), key=lambda kv: -kv[1]):
        print(f"    {f:26s} {ms:9.3f}ms  x{fam_calls[f]}", flush=True)

    rec = {"tag": TAG, "ir": IR, "init_s": init_s, "gen_s": gen_s,
           "token_ids": ids, "text": out[0].outputs[0].text,
           "dispatch_ops": called,
           "kernel_families": {k: {"ms": v, "calls": fam_calls[k]}
                               for k, v in fam_ms.items()}}
    (OUT / f"{TAG}.json").write_text(json.dumps(rec, indent=2))
    print(f"\n{tag_upper()}_DONE", flush=True)


def tag_upper() -> str:
    return "IR_ON" if IR else "IR_OFF"


if __name__ == "__main__":
    main()
