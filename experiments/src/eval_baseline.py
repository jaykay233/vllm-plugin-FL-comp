#!/usr/bin/env python3
"""Competition baseline maths: thresholds, prefill/decode split, lever sizing.

Inputs are the organiser's published baselines (沐曦 / 天数) for the two scored
scenarios.  Everything here is arithmetic on those numbers plus the model
config; no GPU needed, so it can be re-run any time the baselines are updated.

Scenario shape (from /root/pingshen.md):
    4k  : [input_len, output_len, concurrency, num_prompts] = [4096, 1024, 64, 256]
    16k : [16384, 1024, 64, 128]

Metric: Total tok/s = (input_tokens + output_tokens) / duration.
Gates:  Total tok/s >= baseline - 1%   and   TTFT <= baseline + 1%.

Usage:
    /opt/conda/envs/mx/bin/python /root/src/eval_baseline.py
"""

from __future__ import annotations

import json

# name -> (in_len, out_len, conc, n_prompts, duration_s, out_tok_s, total_tok_s, ttft_ms)
META_X = {
    "4k": (4096, 1024, 64, 256, 257.525, 1017.93, 5089.645, 3199.435),
    "16k": (16384, 1024, 64, 128, 316.975, 413.51, 7029.675, 27197.135),
}
TIAN_SHU = {
    "4k": (4096, 1024, 64, 256, 646.31, 405.6, 2028.01, 11573.53),
    "16k": (16384, 1024, 64, 128, 2434.82, 53.83, 915.15, 599262.03),
}

MODEL_CONFIG = "/workspace/MiniCPM5-2B/config.json"


def rule(title: str) -> None:
    print(f"\n{'=' * 94}\n{title}\n{'=' * 94}")


def check_self_consistency(tag: str, T: dict) -> None:
    rule(f"A. 基线自洽性校验 —— {tag}（Total tok/s 是否 = (in+out)/duration）")
    for k, (i, o, c, n, dur, ots, tts, ttft) in T.items():
        tin, tout = n * i, n * o
        calc_total, calc_out = (tin + tout) / dur, tout / dur
        print(
            f"  {k:4s}: dur={dur:8.2f}s in={tin:>9,} out={tout:>8,} "
            f"| Total 实测={tts:9.2f} 反算={calc_total:9.2f} "
            f"差={abs(calc_total - tts) / tts * 100:5.2f}% "
            f"| Out 实测={ots:8.2f} 反算={calc_out:8.2f}"
        )


def solve_prefill_decode(T: dict) -> tuple[float, float]:
    """Solve  I1/P + O1/D = t1  and  I2/P + O2/D = t2  for (P, D)."""
    n1, n2 = T["4k"][3], T["16k"][3]
    I1, O1 = n1 * T["4k"][0], n1 * T["4k"][1]
    I2, O2 = n2 * T["16k"][0], n2 * T["16k"][1]
    t1, t2 = T["4k"][4], T["16k"][4]
    det = I1 * O2 - I2 * O1
    inv_p = (t1 * O2 - t2 * O1) / det
    inv_d = (I1 * t2 - I2 * t1) / det
    return 1 / inv_p, 1 / inv_d


def split_time(tag: str, T: dict) -> None:
    rule(f"B. prefill / decode 吞吐与耗时占比 —— {tag}")
    P, D = solve_prefill_decode(T)
    print(f"  prefill ≈ {P:8.1f} tok/s     decode ≈ {D:8.1f} tok/s")
    for k, (i, o, c, n, dur, *_rest) in T.items():
        tin, tout = n * i, n * o
        tp, td = tin / P, tout / D
        print(
            f"      {k:4s}: prefill {tp:7.1f}s ({tp / dur * 100:5.1f}%)  "
            f"decode {td:7.1f}s ({td / dur * 100:5.1f}%)  "
            f"合计 {tp + td:7.1f}s vs {dur:7.1f}s"
        )


def gates(tag: str, T: dict) -> None:
    rule(f"C. 硬门槛 —— {tag}（tok/s 下限 −1%，TTFT 上限 +1%）")
    for k, (i, o, c, n, dur, ots, tts, ttft) in T.items():
        print(f"  {k:4s}: Total tok/s >= {tts * 0.99:9.2f}  (基线 {tts:9.2f})")
        print(f"        TTFT        <= {ttft * 1.01:9.2f} ms (基线 {ttft:9.2f} ms)")


def cost_847(tag: str, T: dict, per_call_us: float = 471.0,
             layers: int = 42, chunk: int = 2048) -> None:
    """Size the blocking-D2H removal against the real eval shape.

    flash_attn.py:847 fires once per layer on every step that contains a
    prefill, so per-prefill-step cost is layers * per_call_us.  Number of
    prefill steps is governed by max_num_batched_tokens, which on this box
    defaults to 2048 (device memory 63.59 GiB < vLLM's 70 GiB tier).
    """
    rule(f"D. flash_attn.py:847 预估收益 —— {tag}（chunk={chunk}, {per_call_us} µs/次）")
    for k, (i, o, c, n, dur, *_rest) in T.items():
        steps = n * i / chunk
        saved = steps * layers * per_call_us * 1e-6
        gain = (dur / (dur - saved) - 1) * 100
        print(
            f"  {k:4s}: prefill 步数≈{steps:7.0f}  阻塞≈{saved:7.2f}s  "
            f"占 duration {saved / dur * 100:5.2f}%  -> 全消除: Total tok/s {gain:+5.2f}%"
        )


def decode_sensitivity(T: dict) -> None:
    """A 10% cut in decode/prefill time moves the metric by that phase's time share."""
    rule("E. 单项加速 10% 对指标的换算（等于该阶段时长占比）")
    P, D = solve_prefill_decode(T)
    for k, (i, o, c, n, dur, *_rest) in T.items():
        tp, td = n * i / P, n * o / D
        print(
            f"  {k:4s}: prefill 快 10% -> 指标 {(dur / (dur - tp * 0.1) - 1) * 100:+5.2f}%"
            f"   |   decode 快 10% -> 指标 {(dur / (dur - td * 0.1) - 1) * 100:+5.2f}%"
        )


def model_shape() -> None:
    rule("F. 模型结构（用于 FLOPs / 带宽估算）")
    try:
        c = json.load(open(MODEL_CONFIG))
    except OSError:
        print("  config.json 不可读，跳过")
        return
    layers = c["num_hidden_layers"]
    h = c["hidden_size"]
    ffn = c["intermediate_size"]
    kv_heads = c["num_key_value_heads"]
    hd = c["head_dim"]
    vocab = c["vocab_size"]
    per_layer = (
        h * h          # q_proj
        + h * kv_heads * hd * 2  # k_proj + v_proj
        + h * h        # o_proj
        + h * ffn * 3  # gate + up + down
    )
    total = per_layer * layers + vocab * h * (1 if not c.get("tie_word_embeddings") else 0) + vocab * h
    print(f"  layers={layers} hidden={h} ffn={ffn} kv_heads={kv_heads} head_dim={hd} vocab={vocab}")
    print(f"  参数量 ≈ {total / 1e9:.2f} B   （每层 {per_layer / 1e6:.1f} M）")
    print("  注：prefill FLOPs ≈ 2·N·tokens，可据此反推算力利用率。")


def main() -> None:
    check_self_consistency("沐曦", META_X)
    check_self_consistency("天数", TIAN_SHU)
    split_time("沐曦", META_X)
    split_time("天数（仅对照，反解会出现负值说明瓶颈不同）", TIAN_SHU)
    gates("沐曦", META_X)
    cost_847("沐曦", META_X)
    decode_sensitivity(META_X)
    model_shape()
    print("\nEVAL_BASELINE_DONE", flush=True)


if __name__ == "__main__":
    main()
