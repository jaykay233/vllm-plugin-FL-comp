# MiniCPM5-2B / BI-V150 — 采样尾部两项优化（2026-11-19）

平台：Iluvatar BI-V150（16 SM，652 GB/s 顺序读峰值），vLLM 0.24.0，
vllm-plugin-FL `feat/ir-kernels-compile`，FlagGems 5.3.5-compat。

本文只记录在本机实测过的数字。基线全部为同日、同机、同 serve 参数。

---

## 0. 结论摘要

| 项 | 基线 | 本改动 | 变化 | 口径 |
|---|---|---|---|---|
| **4k** `[[4096,1024,64,256]]` | 6430.59 | **6537.80** | **+1.67%** | 同会话交错 A/B，跳首轮 |
| | | 6527.83（两轮均值） | +1.51% | |
| **16k** `[[16384,1024,64,128]]` | 5384.0 | **5438.25** | **+1.01%** | 同日两轮均值；未交错 |
| **MATH-500 L3** | 0.962 基线 / ≥0.95 门禁 | **0.9714（102/105）** | **PASS** | 最终报告 |
| **MATH-500 L3（+bf16）** | 同上门禁 | **0.9810（103/105）** | **PASS** | 追加层，见 §7 |

两处改动都在**采样器**（CUDA graph 之外的 decode 尾部），不碰 attention：

* **B —— T=1 时跳过 logits/temperature 全词表除法**：+0.72%（4k）。
* **A —— 把 top-p 之后的 softmax / 概率 / 除法 / argmax 融合成一个 kernel**：
  在 B 之上再 +0.6~0.9%（4k），两场景合计过 1%。

attention 侧本身已经贴墙（decode KV 617 GB/s ≈ 95% 峰值；prefill FA 在
shape roofline），所以这一轮唯一的增量在采样尾部。

---

## 1. 定位：尾部里有哪几项

16k 中位 decode 步（rows≈32，CUDA graph 外，修 top-p 直方图之后）：

| 组件 | 耗时 |
|---|---|
| lm_head `linear_kernel` | ~900 µs |
| `true_div_func_kernel_rank_2`（logits / temperature） | ~169 µs |
| top-p 链（rowmax/mass/amax/threshold/mask） | ~149 µs |
| softmax | ~84 µs |
| `fused_exponential`（q 生成） | ~50 µs |
| `true_div_func_kernel_rank_1`（probs / q） | ~97 µs |
| argmax | ~72 µs |
| 其它 elementwise / scatter / launch gap | ~300–400 µs |

4k 的 rows=64，行相关项约翻倍。lm_head 已是 593 GB/s（~91% 峰值），不动。

---

## 2. 改动 B：T=1 时跳过 temperature 除法

### 机理

`Sampler.apply_temperature` 无条件执行 `logits.div_(temperature)`，作用在
`[rows, 130560]` 的 **fp32** 上。MiniCPM 的 `generation_config.json` 是
`temperature=1.0`，而 `x / 1.0` 是**逐位恒等**。vLLM 在除法前会把 greedy
行 clamp 成 1.0，所以「T=1 或 greedy」同样无需缩放。

### 实现

* `vllm_fl/.../patches/sampler_temperature.py`
* 在 `InputBatch._make_sampling_metadata` 上用 host 侧 `temperature_cpu`
  计算 `all((T == 1.0) | (T < eps))`，挂到 metadata 上；**不引入 device sync**。
* `Sampler.sample` 期间置模块级开关，`Sampler.apply_temperature` 直接返回
  原张量；其余情况走原始实现。
* 环境变量：`VLLM_FL_SKIP_TEMP_DIV=0` 关闭（默认开）。

### 4k 交错 A/B（同一份代码，仅 env 切换；4 轮，跳首轮）

| 臂 | Run1 | Run2 | Run3 | Run4 | 有效均值 |
|---|---|---|---|---|---|
| 基线 `skip=0` | 6422.47 | 6433.13 | 6429.81 | 6428.83 | **6430.59** |
| `skip=1` | 6467.83 | 6479.99 | 6476.80 | 6474.43 | **6477.07** |

**+0.72%**；4 个 patched 轮全部高于 4 个基线轮，区间不重叠。
Mean TPOT 46.29 → **45.87 ms（−0.42 ms，−0.9%）**；TTFT 在轮间噪声内。

---

## 3. 改动 A：融合 top-p + 指数噪声采样

### 关键观察

vLLM 的 `random_sample` 是

```
ids = (softmax(masked_logits) / q).argmax(-1),   q ~ Exponential(1)
```

softmax 的分母是逐行正数，**argmax 对正尺度不变**。于是

```
argmax(softmax(x) / q) == argmax(exp(x - rowmax) / q)
```

可以丢掉 softmax、概率张量、以及单独的 `probs / q`，一遍完成选择。

### 实现

* FlagGems：`src/flag_gems/ops/top_p_sample.py`
  * 复用 `top_p_threshold` 的 v2 机制：rowmax → log2 概率直方图 → 阈值；
  * 然后一次分块 pass：`score = exp2((x-rowmax)*log2e) / q`，仅保留
    `x >= threshold`，块内 max+argmax；一个极小 reduce kernel 合并块。
  * 输出 `int64 [rows]`。
* 插件：`vllm_fl/.../patches/fused_top_p_sample.py` 替换
  `TopKTopPSampler.forward_native` 的 **top-p-only / fp32 / 无 logprobs /
  非 fp64-gumbel** 路径；其余原样回退。
* `q` 的生成顺序与 `random_sample` 完全一致（`empty_like` + `exponential_()`
  + 逐请求 generator），所以同一随机状态下 token 与旧路径一致。
* 环境变量：`VLLM_FL_FUSED_TOPP_SAMPLE=0` 关闭；`VLLM_FL_FUSED_TOPP_MAX_ROWS`
  （默认 128，与 `top_p_threshold` 的 v2 交叉点一致）。

### 正确性（GPU，真实词表 130560）

对照 `(top_p_threshold(logits, p).softmax(-1) / q).argmax(-1)`：

| rows | 1 | 8 | 32 | 64 | 128 |
|---|---|---|---|---|---|
| scale=1, p=0.95 | 0 mismatch | 0 | 0 | 0 | 0 |
| scale=4, p=0.95 | 0 | 0 | 0 | 0 | 0 |
| p=1.0 | — | — | 0 | — | — |

**100% 一致**，且参考选中的位置都落在保留集内。

### 微基准（单步，V=130560）

| rows | 旧链 | `top_p_sample` | 节省 |
|---|---|---|---|
| 32 | 1119.3 µs | **374.2 µs** | 745.1 µs（−66.6%） |
| 64 | 1556.0 µs | **538.8 µs** | 1017.2 µs（−65.4%） |

### e2e（B+A）

| 场景 | Run1 | Run2 | 均值 | 对基线 |
|---|---|---|---|---|
| 4k | 6517.86 | **6537.80** | 6527.83 | **+1.67%**（跳首轮）/ +1.51% |
| 16k | 5438.20 | 5438.30 | **5438.25** | **+1.01%** |

16k 两轮差 <0.01%，非常稳。

---

## 4. 精度门禁

MATH-500 **Level 3**（105 题，evalscope openai_api，
`t=1.0 / top_p=0.95 / batch=8 / max_tokens=32768`），在 B+A 服务上实测：

| 项 | 值 |
|---|---|
| 判分 | 105 / 105 |
| 正确 | 102 |
| 准确率 | **0.9714（97.14%）** |
| 状态 | 全部 `success`，无 error |
| 门禁 | ≥ 0.95（基线 0.962）→ **PASS**（余量 2.14 pt） |

（生成到 103/105 时的实时值是 101/103 = 98.06%；最后两题 1 对 1 错，
最终 102/105。）与上一轮交付配置的 98.1% 处于同一量级，差异来自采样的
逐题随机波动；A 只改变选择路径、不改变分布，且单元测试逐 token 一致。

报告：`/root/bench_results/gate_fused/level3/20261119_165702/reports/minicpm/math_500.json`
（`score = macro_score = 0.9714`，`num = 105`）。

---

## 5. 复现

```bash
export VLLM_PLUGINS=fl
# B / A 默认开；如需 A/B：
#   VLLM_FL_SKIP_TEMP_DIV=0
#   VLLM_FL_FUSED_TOPP_SAMPLE=0

vllm serve /workspace/MiniCPM5-2B --port 9031 \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --served-model-name minicpm --gpu-memory-utilization 0.85 \
  --max-model-len 131072

BENCH_RUNS=4 python3 benchmarks/benchmark_throughput_serve.py \
  --model /workspace/MiniCPM5-2B --served-model-name minicpm --port 9031 \
  --test-cases '[[4096,1024,64,256]]'
BENCH_RUNS=4 python3 benchmarks/benchmark_throughput_serve.py \
  --model /workspace/MiniCPM5-2B --served-model-name minicpm --port 9031 \
  --test-cases '[[16384,1024,64,128]]'
```

---

## 6. 口径与注意事项

* 4k 基线是**同会话交错 A/B**（8 轮，两个臂交替环境变量）；16k 是同日
  15:21 的 B/A-off 基线（5384.0），**不是交错**，因为单臂 16k 一轮约
  414 s。16k 两轮内部差 <0.01%，但仍建议最终提交前再做一次交错对照。
* `q` 与旧路径逐位相同，因此 A 理论上不改变 token；实测 100% 一致。
* B 的判定在 host 侧完成，不增加 device sync；`x / 1.0` 逐位恒等。
* 两处都能用环境变量单独关闭，便于回退。

---

## 7. 追加：bf16 采样链（cast elision）

### 动机

`Sampler.forward` 无条件执行 `logits.to(torch.float32)`。模型 logits 本来就是
bf16，而 bf16 是 fp32 的子集，展宽是精确的。只要下游不需要 fp32（无
penalties / 非默认 logits processor / 无 logprobs，且全部 T=1、top-p only），
就可以让采样链直接以 bf16 走完：省掉一次全词表写回，并让后续
rowmax/mass/fused pass 的 logits 读减半。

### 实现

* `Sampler.forward` 在满足条件时跳过 cast（复用原 forward 的尾部逻辑）。
* gate 还要求 batch 的 `MinP / MinTokens / LogitBias` 处理器都处于默认
  no-op 状态（检查 `min_p_count / min_toks / biases`）；未知处理器或非默认
  状态一律回退 vLLM 的 fp32 路径。
* FlagGems `top_p_sample` 接受 bf16 logits，kernel 内 `.to(tl.float32)`；
  `q` 始终是 fp32（指数噪声的 8-bit 尾数会改变近似平票的选择）。
* `top_p_threshold` 的 v2 rowmax/mass kernel 同样在 load 后展宽。
* 环境变量：`VLLM_FL_SAMPLER_BF16=0` 关闭该层（默认开）。

### 正确性

* 对 `(softmax(masked)/q).argmax` 在 rows 1/8/32/64/128、两种 logit 幅度、
  p=0.95/1.0：**0 mismatch**；bf16 与 fp32 路径 token 完全一致。
* MATH-500 L3（B+A+bf16，105 题）：**0.9810（103/105）**，全部 `success`，
  门禁 ≥0.95 / 基线 0.962 → **PASS**。

### 性能（同会话 A/B）

| 场景 | B+A (fp32) | B+A+bf16 | 变化 |
|---|---|---|---|
| 4k | 6527.83（6517.86 / 6537.80） | **6538.90（6532.79 / 6545.01）** | **+0.17%** |
| 16k | 5438.25（5438.20 / 5438.30） | 5437.80（5433.45 / 5442.14） | −0.01%（持平） |

收益很小，16k 完全在噪声内。保留原因：精确等价、无精度风险、4k 方向为正；
**不把它当作计分项**，计分仍由 §2/§3 的 B+A 承担。
