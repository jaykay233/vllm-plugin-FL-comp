# 评测结果 —— 配置 B（赛事方默认口径）

- 时间：2026-09-23 22:10–23:14
- 仓库：`/workspace/vllm-plugin-FL`
- HEAD：`bb2c763 docs(readme): refresh the CUDAGraph-only numbers and document IR_FUSED_ADD`
- 启动命令（pingshen.md 沐曦口径，未传 `--compilation-config`）：
  `VLLM_PLUGINS=fl vllm serve /workspace/MiniCPM5-2B --port 9031 --served-model-name minicpm --gpu-memory-utilization 0.85 --max-model-len 131072`
- 实际生效配置：`CompilationMode.VLLM_COMPILE` + `CUDAGraphMode.FULL_AND_PIECEWISE`
  （`custom_ops=['none']`，`enable_prefix_caching=True`，`max_num_batched_tokens`=2048 默认）

## PART B：性能（benchmark_throughput_serve.py，RUNS=4 / SKIP_FIRST=1）

### 原始 4 轮

| case | run | dur_s | Total tok/s | TTFT_ms | TPOT_ms | out tok/s |
|---|---|---|---|---|---|---|
| 4096_1024_c64 | 1（跳过） | 276.24 | 4744.78 | 3772.12 | 63.37 | 948.96 |
| 4096_1024_c64 | 2 | 285.48 | 4591.36 | 3004.21 | 66.47 | 918.27 |
| 4096_1024_c64 | 3 | 247.25 | 5301.16 | 2806.10 | 57.32 | 1060.23 |
| 4096_1024_c64 | 4 | 286.13 | 4580.83 | 2983.84 | 66.65 | 916.17 |
| 16384_1024_c64 | 1（跳过） | 320.29 | 6956.89 | 24950.87 | 128.13 | 409.23 |
| 16384_1024_c64 | 2 | 300.45 | 7416.21 | 24791.52 | 118.57 | 436.25 |
| 16384_1024_c64 | 3 | 300.46 | 7416.10 | 24797.70 | 118.58 | 436.24 |
| 16384_1024_c64 | 4 | 300.48 | 7415.64 | 24797.59 | 118.59 | 436.21 |

### 有效轮均值（run 2/3/4）vs 基线

| case | Total tok/s | vs 基线 | TTFT_ms | vs 基线 | 门槛 | 判定 | run 抖动 |
|---|---|---|---|---|---|---|---|
| **16k** | **7415.98** | **+5.50%** | 24795.60 | **−8.83%** | ≥6959.38 / ≤27469.11 | ✅ PASS | **0.0%** |
| **4k** | 4824.45 | **−5.21%** | 2931.38 | −8.38% | ≥5038.75 / ≤3231.43 | ❌ tok/s FAIL | **14.9%** ⚠️ |

基线：4k `5089.65 / 3199.44 / 1017.93 out`；16k `7029.68 / 27197.14 / 413.51 out`

### 结论
- **16k 可信**：抖动 0.0%，Total tok/s +5.50%，与「修掉 flash_attn.py:847 阻塞 D2H 应得 +6.83%」的预测高度吻合。
- **4k 不可信**：抖动 14.9%，均值落在门槛之下，但单轮有 5301（过线）。需提高轮数重测。
- 4k/16k 分化未解释：两者同为 prefill 主导，16k 稳定增益而 4k 抖动。

## PART C：正确性（evalscope math_500 Level 3）

| 指标 | 结果 | 基线 | 门槛 | 判定 |
|---|---|---|---|---|
| Accuracy | **97.1%**（105 题） | 0.962 | ≥0.95 | ✅ PASS |

- 用时 15:46；Avg Lat 50.017 s，Avg Out 3256 tok，Avg Thpt 65.1 tok/s
- 输出：`/workspace/evalscope-datasets/level3/20260923_225851/`

## 待办

1. 4k 重测：轮数提到 8 轮以上，独占 GPU。
2. 可选决定性实验：HEAD vs 分支基线 `13eb9be` 的 4k 对照。

## 附：另测的两种启动配置（4k，均劣于默认）

| 配置 | Total tok/s | vs 基线 | 启动耗时 |
|---|---|---|---|
| A：`VLLM_FL_CUDAGRAPH_ONLY=1`（mode=NONE + FULL_DECODE_ONLY） | 4342 / 4569 | −10~15% | ~100 s |
| C：`--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'`（保留 compile，无 prefill 图） | 3328（仅 run1） | −23%+ | ~15.5 min |
