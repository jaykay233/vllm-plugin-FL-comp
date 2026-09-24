# 官方口径评测结果：FlagOS 白名单

配置：`VLLM_FL_FLAGOS_WHITELIST=silu_and_mul,rms_norm,rotary_embedding`
其余与评测口径一致（不传 `--compilation-config`、`gpu-memory-utilization 0.85`、
`max-model-len 131072`）。脚本：`experiments/src/run_eval_whitelist.sh`

## 性能（4k：256 × (4096+1024)，并发 64）

| 轮次 | Total tok/s |
|---|---|
| Run 1/4 | 8797.67 |
| Run 2/4 | 8529.41 |
| Run 3/4 | 按需求提前中止 |

对照（无白名单）：`5271.99` / `4793.31` → **首轮同口径 +67%**。
TTFT：Mean 1819.48 ms、Median 565.47 ms、P99 10873.52 ms。
TPOT：Mean 34.38 ms、Median 35.42 ms。

## 正确性（evalscope math_500 Level 3，105 题）

| 指标 | 本轮 | 无白名单基线 | 竞赛基线 | 门槛 | 判定 |
|---|---|---|---|---|---|
| Accuracy | **97.1%** | 97.1% | 0.962 | ≥0.95 | ✅ PASS |

**提速 67% 而 Accuracy 完全不变（97.1%），即未牺牲正确性。**

## 另两处已证实的收益

- **消除 20 s 停顿**：白名单使 `mm` 回落原生实现，FlagGems 的 autotune 不再被触发。
  判据：autotune DB 增量 `mm` 类 +0、全库 +0；看门狗停顿采样 0 次。
- 报告输出：`/workspace/evalscope-datasets/level3/20260924_111830/`
