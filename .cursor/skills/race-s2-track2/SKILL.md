---
name: race-s2-track2
description: >-
  Brief for the FlagOS Open Compute Global Challenge Season 2, Track 2
  (赛道二：MiniCPM 模型推理吞吐性能优化) — what the track asks for, the two scored
  scenarios, the organiser's published baselines and hard gates for both the
  天数 BI-V150 and 沐曦 C500 cards, the 70/30 scoring formula, the allowed and
  forbidden optimisation techniques, the official serve/benchmark/eval commands,
  and the submission requirements. Use when asked what this competition requires,
  how it is scored, which number is the baseline or the gate, what techniques are
  banned, or for the organiser's official commands.
---

# FlagOS Race · Season 2 · Track 2 — MiniCPM Inference Throughput

## Source

- Race detail page: <https://flagos.net/race-detail-season2?id=539vlt2p&lang=cn>
- Prize pool ¥140,000 · window 2026-09-11 → 2026-11-20
- Hosts: 众智 FlagOS 社区、IEEE
- The page's two baseline tables are **images**, not text. The authoritative,
  self-checked transcription lives in
  `experiments/src/eval_baseline.py` — run it to re-verify (see below).

## The task

- **Model: MiniCPM5-2B — mandatory.** Using any other model voids the result.
- **Cards:** 天数 BI-V150 and 沐曦-曦云 C500 (64 GB).
- **Framework:** `vllm-plugin-FL`, branch `flagos-2026-s2`; operator library
  **FlagGems, tag `v5.3.5`**. Using the designated framework version is a
  validity requirement, not a suggestion.
- **Objective:** maximize **total tokens/s**. Accuracy is a constraint, not a
  trade.
- **Scored metric:** `total tok/s = (input_tokens + output_tokens) / duration`.
  TTFT is the second, gating metric.

### Allowed

显存管理（KV Cache 优化、动态批处理）、计算逻辑优化（算子融合、内核重写）、
自行开发的量化策略 — anything that requires **real development effort in the
framework**. Generic strategies are explicitly preferred.

### Forbidden (result invalid)

- changing `vllm serve` args away from the baseline
- enabling quantization or speculative decoding via hyperparameters
- tuning the benchmark script itself
- swapping operators with no actual operator optimisation, or deleting the
  framework's main operator-selection logic
- merging the upstream latest branch to inherit free performance
- modifying model behaviour so the eval is no longer comparable

The technical report's claimed strategies **must be present in the submitted
code and active in the final version**.

## Scored scenarios

| case | input_len | output_len | concurrency | num_prompts |
|---|---|---|---|---|
| 4k  | 4096  | 1024 | 64 | 256 |
| 16k | 16384 | 1024 | 64 | 128 |

## Baselines and hard gates

Gates are the organiser's tolerance: `total tok/s ≥ baseline × 0.99` and
`TTFT ≤ baseline × 1.01`.

### 天数 BI-V150 — **this box**

| case | duration | out tok/s | total tok/s | TTFT | gate: total ≥ | gate: TTFT ≤ |
|---|---|---|---|---|---|---|
| 4k  | 646.31 s  | 405.60 | **2028.01** | 11573.53 ms | 2007.73 | 11689.27 ms |
| 16k | 2434.82 s | 53.83  | **915.15**  | 599262.03 ms | 906.00 | 605254.65 ms |

### 沐曦-曦云 C500 — the *other* card

| case | duration | out tok/s | total tok/s | TTFT | gate: total ≥ | gate: TTFT ≤ |
|---|---|---|---|---|---|---|
| 4k  | 257.525 s | 1017.93 | **5089.645** | 3199.435 ms | 5038.75 | 3231.43 ms |
| 16k | 316.975 s | 413.51  | **7029.675** | 27197.135 ms | 6959.38 | 27469.11 ms |

> **Know which card you are on before quoting a baseline.** The two columns
> differ by ~2.5× on 4k and ~7.7× on 16k. The sibling skill
> `eval-gated-optimization` also carries both cards' gates (it was originally
> authored on a MetaX box — `/opt/conda/envs/mx`, which does not exist on the
> 天数 image — and its serve command still assumes that); `experiments/FINDINGS-mx.md`
> is likewise "MiniCPM5-2B on MetaX C500". On this box (`Iluvatar BI-V150`,
> `torch …+corex…`) the **天数** column governs. Do not compare a measurement
> taken here against a 沐曦 baseline.
>
> The 天数 16k pair is pathological: TTFT ≈ 599 s. Note that the prefill/decode
> split solves to a *negative* decode rate, i.e. the same two-phase model does
> not describe this card — treat 天数 numbers as measured facts, not as a model.

Re-verify the arithmetic (no GPU needed, ~0.3 s):

```bash
python3 experiments/src/eval_baseline.py
```

Every row prints its own `Total 实测 vs 反算` check; all four currently agree to
0.00%. The script also prints the prefill/decode split, the gate table, and the
per-phase sensitivity (what a 10% speedup of prefill vs decode is worth).

## Correctness (hard blocker)

| item | value |
|---|---|
| dataset | `math_500`, **Level 3** subset |
| accuracy baseline | **0.962** |
| required | **≥ 0.95** |

Accuracy below the gate — or below the same-session baseline — invalidates the
result. Additional invalidating conditions per the rules: an obvious drop in the
model's core capability, malformed output or an inability to complete the
benchmark, or a change that makes the eval incomparable.

## Scoring formula

```
总分 = 性能分(天数) + 性能分(沐曦) + 创新分

性能分  = 100 × 提升比例 × 70%
创新分  = 专家评分 × 30%
```

- A single scenario must improve by **>1%** to score at all — a deviation within
  1% is defined as normal fluctuation, not a gain.
- The performance score is the **average of the two scenarios** (4k and 16k).
- Innovation is graded from the technical report: 系统级调度、算子编译优化创新等
  原创优化思路. Ties in performance are broken by report innovation (organiser
  vote).
- Example from the rules: baseline 200 → 210 is +5%, giving `100 × 5% × 70%`.

Awards: 一等奖 ¥30,000 · 二等奖 ¥20,000 · 三等奖 ¥10,000.

## Official commands

Paths used by the organiser: model `/workspace/MiniCPM5-2B`, dataset
`/workspace/evalscope-datasets/math_500`, framework `/workspace/vllm-plugin-FL`,
FlagGems `/workspace/FlagGems`. Source updates go in with
`pip install --no-build-isolation`.

### Serve — 天数 (this box)

Note the `--compilation-config`: it is part of the baseline, so it must stay.

```bash
export VLLM_PLUGINS=fl
vllm serve /workspace/MiniCPM5-2B \
  --port 9031 \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --served-model-name minicpm \
  --gpu-memory-utilization 0.85 \
  --max-model-len 131072
```

### Serve — 沐曦

No `--compilation-config`; that is the difference between the two commands.

```bash
export VLLM_PLUGINS=fl
vllm serve /workspace/MiniCPM5-2B \
  --port 9031 \
  --served-model-name minicpm \
  --gpu-memory-utilization 0.85 \
  --max-model-len 131072
```

### Liveness check

```python
import requests
response = requests.post(
    'http://127.0.0.1:9031/v1/chat/completions',
    headers={'Content-Type': 'application/json'},
    json={
        'model': 'minicpm',
        'messages': [{'role': 'user', 'content': 'introduce llm'}],
        'max_tokens': 256,
    },
)
print(response.json())
```

### Correctness — `math_500` Level 3

```bash
evalscope eval \
  --model minicpm \
  --api-url http://127.0.0.1:9031/v1/chat/completions \
  --api-key EMPTY \
  --eval-type openai_api \
  --datasets math_500 \
  --dataset-args '{"math_500": {"dataset_id": "/workspace/evalscope-datasets/math_500", "subset_list": ["Level 3"]}}' \
  --eval-batch-size 8 \
  --timeout 3600 \
  --generation-config '{"temperature": 1.0, "top_p": 0.95, "max_tokens": 32768}' \
  --work-dir /workspace/evalscope-datasets/level3 \
  --ignore-errors
```

Read the score from the `Overall report table` or from
`<work-dir>/<timestamp>/reports/minicpm/math_500.json`.

### Throughput benchmark

Writes `benchmark_results/summary_xxxx.csv` into the **CWD**, so run it from
`/workspace`.

```bash
cd /workspace
python3 /workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py \
  --model /workspace/MiniCPM5-2B \
  --served-model-name minicpm \
  --port 9031 \
  --test-cases '[[4096,1024,64,256],[16384,1024,64,128]]'
```

Ranking is decided by the organiser's own reproduction, not by a submitted log —
these commands exist to steer optimisation, not to win by documentation.

Both cards' full loop — serve, benchmark, `math_500`, and the gate comparison —
is already scripted; prefer it over retyping the commands:

| card | script |
|---|---|
| 天数 BI-V150 | `experiments/src/run_official_eval_iluvatar.sh` (via `launch_official_eval_iluvatar.sh`) |
| 沐曦 C500 | `experiments/src/run_eval_all.sh` / `run_eval_whitelist.sh` |

The 天数 script differs from the 沐曦 one in exactly four ways: it uses
`/usr/local/bin` instead of `/opt/conda/envs/mx`, it passes
`--compilation-config`, it leaves `VLLM_FL_CUDAGRAPH_ONLY` alone (the plugin's
default-vendor list is empty, so 天数 has no CUDAGraph-only default to suppress),
and it gates against the 天数 baselines.

## Submission

Deadline **2026-11-20 23:59 (UTC+8)**, as `<队伍名>.zip`:

```
<队伍名>
├── FlagGems          # optional — only if operators changed
├── readme.md
├── report.pdf
└── vllm-plugin-FL
```

- `report.pdf`: 方案设计思路、算法原理、实现架构、优化策略、实验环境、测试结果与
  效果分析.
- `readme.md`: 软硬件环境、依赖版本、环境部署步骤、编译执行流程、数据集准备方式、
  运行命令与结果校验方法 — a reviewer must be able to reproduce from it alone.
- If optimisations exist for both cards, merge both into one submission.
- Winners must additionally open a PR against
  <https://github.com/flagos-ai/vllm-plugin-FL/tree/flagos-2026-s2> within
  **3 working days** of the review, named
  `【FlagOS开放计算全球挑战赛S2.赛题2】【xxx队伍】特性简要描述xxxx`.
  Reference PR: <https://github.com/flagos-ai/vllm-plugin-FL/pull/463>.
  Not submitting it forfeits the award.

### Timeline

| stage | window (UTC+8) |
|---|---|
| 报名 | 2026-07-19 → 2026-11-10 |
| 官方算力申请 | 2026-09-14 → 2026-11-10 |
| 算力评审及发放 | 2026-09-21 → 2026-11-15 |
| 开发与提交 | 2026-09-21 → 2026-11-20 |
| 评审 | 2026-12 初 |
| 结果发布 | 2026-12 中旬 |

Teams: at most 3 members; solo entries allowed.

## Related skills

- `eval-gated-optimization` — the commit gate: the official throughput
  benchmark plus the `math_500` correctness check. Its recorded gates are the
  **沐曦** ones; see the card warning above.
- `bench-progress-report` — every command here is a multi-minute job. Launch it
  backgrounded with a `notify_on_output` hook; never busy-wait.
- `self-directed-optimization` — derive the ranked plan and execute it rather
  than ending a turn with a question.
