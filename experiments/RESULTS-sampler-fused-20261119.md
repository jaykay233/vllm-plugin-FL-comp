# MiniCPM5-2B / BI-V150 — 采样尾部与 decode GEMM 优化（2026-11-19）

平台：Iluvatar BI-V150（16 SM，16 MiB L2，持续读 ~586.5 GB/s），vLLM 0.24.0，
vllm-plugin-FL `feat/ir-kernels-compile`，FlagGems 5.3.5-compat。

本文只记录本机实测数字。基线为同日、同机、同 serve 参数。

---

## 0. 结论摘要

| 项 | 基线 | 最终 | 变化 | 口径 |
|---|---|---|---|---|
| **4k** `[[4096,1024,64,256]]` | 6430.59 | **6555.49**（best 6561.33） | **+1.94%** | 同会话 A/B，两轮 |
| **16k** `[[16384,1024,64,128]]` | 5384.0 | **5444.09**（5443.49 / 5444.68） | **+1.12%** | 最终配置两轮 |
| **MATH-500 L3** | 0.962 基线 / ≥0.95 门禁 | **0.9810（103/105）** | **PASS** | 最终报告 |

四项有效改动（全部默认开启，无额外环境变量）：

* **B —— T=1 时跳过 logits/temperature 全词表除法**（§2）：4k +0.72%。
* **A —— 把 top-p 之后的 softmax / 概率 / 除法 / argmax 融合成一个 kernel**
  （§3）：在 B 之上再 +0.6~0.9%，B+A 4k +1.67%。
* **bf16 采样链 —— 跳过 bf16→fp32 cast**（§7）：4k +0.17%（噪声），16k 持平。
* **down M64 split-K**（§9）：53.7→49.6 µs（+7.7%），4k +0.25%；16k 不受影响。

attention 侧本身已经贴墙（decode KV 594–621 GB/s，高于顺序读 586.5；prefill
FA 在 shape roofline），剩余增量全部来自采样尾部与一个 decode GEMM。

**7000 tok/s 目标**：低于本卡 decode 的裸 kernel 下限（§12），与调度器 /
PDL / persistent 无关，独立不可达。

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

---

## 8. 7000 tok/s 目标的可行性评估（2026-11-19 追加）

### 8.1 4k decode 归因（eager trace，graph 内 kernel 全暴露）

| 组件 | 每步 | 占比 | 实测带宽 |
|---|---:|---:|---:|
| `_paged_decode_kernel` ×42（KV attention） | 21.4 ms | 63% | 594 GB/s（约 91–95% 峰值） |
| `mm_kernel` ×168（权重 GEMM） | 8.7 ms | 26% | 456–546 GB/s |
| lm_head | 0.96 ms | 3% | 556 GB/s |
| rms/add/silu/rope/KV write/sampler 等 | ~3 ms | 9% | — |
| **kernel 合计** | **~34 ms** | | |
| graph-inner（含图内边界） | ~36–37 ms | | |
| 图外尾部（lm_head + sampler） | ~1.9 ms | | |
| **单步墙钟** | **38.9 ms** | | |

attention 是最大头（此前文档「decode attention 0.5%」是单层/单位口径错误）。
attention 已贴 HBM 峰值，KV 12.69 GB/步无法再省。

### 8.2 本轮实测证伪

| 尝试 | 结果 |
|---|---|
| decode split 1/2/4 × warps 1/2/4 | split=2/warps=1 最优（618 GB/s）；split=1 仅 365 GB/s |
| prefill FA paged vs dense | 差 0.6–0.9%，dense + gather 更慢 |
| bf16 采样链（跳过 fp32 cast） | 4k +0.17%（噪声），16k −0.01% |
| top-p 直方图 block 汇总原子 | 4k −0.03%，rows=64 微基准更慢 |
| OOT fused add+rms | 4k +0.04%（噪声），图内本就已摊薄 |

### 8.3 结论

理论上限 ~31 ms/步（attention 19.5 + GEMM 7.7 + 小算子 ~2 + 尾部 ~1.9）。
实测 38.9 ms，差 ~7.8 ms，来自 336 个 kernel 的逐层依赖/波次/图边界，
不是单个 kernel 能调出来的。再拿 2.9 ms（到 7000）只能：
- 把 KV 流量减半（需要 fp8/int8 paged attention 的硬件支持；ivcore11 没有，
  软件反量化已被证伪），或
- 减少 decode 步数（投机解码，比赛禁用）。

因此 **7000 tok/s 在当前硬件 + 精度约束 + 可用技术下不可达**；
B+A+bf16 稳定值约 6539（4k）/ 5438（16k），L3 复验 0.9714。

### 8.4 结构性杠杆的独立验证（2026-11-19）

| 杠杆 | 实测 / 验证 | 收益上限 |
|---|---|---|
| **图边界移除** | 336 个空 kernel：graph replay **570.5 µs（1.70 µs/node）**，eager 2145.5 µs（6.39 µs/kernel） | **≤ 0.57 ms/步** |
| **persistent megakernel（grid sync）** | `cooperativeLaunch=0`；corex 头文件无 `this_grid` / `grid_group` / `cudaLaunchCooperativeKernel`，`cg::this_grid()` 编译报错 | **不可实现** |
| **层内 GEMM 融合（qkv+gate_up）** | `LlamaDecoderLayer.forward`：`input_norm → self_attn → post_attn_norm → mlp`。qkv 与 gate_up 的输入不同，中间隔着 attention | **不可融合** |
| **silu epilogue** | 既有图内实测 4.87 µs/层（§17.47） | ≤ 0.20 ms/步 |
| **fused add+rms** | 本轮 4k A/B +0.04%（噪声） | ≈ 0 |

三类结构手段合计上限 **~0.8 ms/步**，而到 7000 需要 **2.9 ms/步**。
剩余 ~7.8 ms 是 kernel 自身的带宽/效率（attention 594 GB/s、GEMM 456–546 GB/s、
小算子），不是边界或层融合能拿回来的。**7000 不可达由此独立成立。**

复现（探针在 `experiments/src/`）：
```bash
# grid sync
/usr/local/corex/bin/clang -O3 -x ivcore --cuda-gpu-arch=ivcore11 \
  --cuda-path=/usr/local/corex -L/usr/local/corex/lib64 -lcudart \
  -o /tmp/probe_coop experiments/src/probe_coop.cu
# error: no member named 'this_grid' in namespace 'cooperative_groups'
/usr/local/corex/bin/clang -O3 -x ivcore --cuda-gpu-arch=ivcore11 \
  --cuda-path=/usr/local/corex -L/usr/local/corex/lib64 -lcudart \
  -o /tmp/probe_coop_attr experiments/src/probe_coop_attr.cu && /tmp/probe_coop_attr
# cooperativeLaunch=0
# graph node overhead
python3 experiments/src/probe_graph_boundary.py   # 1.70 us/node
```

---

## 9. decode GEMM split-K（2026-11-19 追加）

`tune_configs.yaml` 的 `mm` 里 `split_k` 只列了 1，所以 tuner 从未试过 split-K。
在图内（42 次 replay，生产同款 `_launch_mm_fixed`）对四个 decode GEMM 扫描：

| 矩阵 | 生产 | split-K 最佳 | 结论 |
|---|---|---|---|
| **down M64** (N=2048,K=6144) | 53.7 µs / 468 GB/s | **49.6 µs / 508 GB/s** `(64,128,64,S=4,8,1)` | **+7.7%，采纳** |
| gate_up M64 | 84.2 / 597 | 无更优 | 已贴峰值 |
| qkv M64 | 21.0 / 499 | 28.0 (−33%) | 更差 |
| o M64 | 14.4 / 581 | 18.9 (−31%) | 更差 |
| qkv/o/gate_up/down M32 | — | −9% ~ −67% | 全部更差 |

已把 down M64 的配置加入 `_SKINNY_MM_LE64`（FlagGems `ops/mm.py`）。

e2e（4k，两轮）：

| 臂 | Run1 | Run2 | 均值 |
|---|---|---|---|
| B+A+bf16 | 6532.79 | 6545.01 | 6538.90 |
| **+ down split-K** | **6549.64** | **6561.33** | **6555.49（+0.25%）** |

L3：**0.981（103/105）PASS**（split-K 的 bf16 原子累加没有造成可测的精度损失）。
16k 不受影响：M≤32 仍走 `_SKINNY_MM_LE32`（未改）。

---

## 10. PDL 与 wave-quantization 的实测（2026-11-19）

corex 头文件里有 PDL 全套接口
（`ixLaunchAttributeProgrammaticStreamSerialization`、`cudaLaunchKernelEx`、
`__ivcorex_griddepcontrol_launch_dependents/wait`），探针可编译运行，但：

| 双 kernel 依赖对 | 时间 |
|---|---|
| serial | 0.739 ms |
| PDL | 0.740 ms |
| 差 | **−1.46 µs/对（无收益，略负）** |

CUDA graph 已经把 kernel 边界压到 1.70 µs/node（§8.4），PDL 能重叠的
drain/setup 已经很小；在图内再叠 PDL 也没有可回收量。

再测**部分波次**（grid=20 on 16 SM，主 kernel 带长 epilogue，理论上 PDL
应让依赖 kernel 填满 12 个空闲 SM）：

| 不均衡波次 | 时间 |
|---|---|
| serial | 34.084 ms |
| PDL | 34.087 ms |
| 差 | **ratio 1.0001（无收益）** |

即 corex 暴露了 PDL 属性和 `griddepcontrol` builtin，但 runtime 并不会
实际 co-schedule 依赖 kernel。PDL 这条也关闭。

wave quantization 由 §9 的 split-K 全形状扫描覆盖：只有 `down M=64` 有
+7.7%，其余 −9% ~ −67%。每 kernel 的实测带宽已经贴住可达上限：

| kernel | GB/s | 备注 |
|---|---:|---|
| decode attention | 618 | 高于顺序读 589.6（K/V 双流 + L2） |
| gate_up | 597 | ≈ 峰值 |
| o_proj | 581 | ≈ 峰值 |
| down（split-K） | 508 | 全配置扫描最优 |
| qkv | 499 | 全配置扫描最优 |

至此，剩余 ~7.8 ms/步 已被逐项归到「kernel 在可达带宽上的固有开销」，
没有单个可调项能再提供 7000 所需的 ~3.8 ms。

---

## 11. 持续带宽与「理论下限」的口径修正（2026-11-19）

20 秒持续 `copy_`（1 GiB tensor），每 0.2 s 采样一次：
**586.2–586.7 GB/s，全程无下降** —— 这台卡的持续带宽就是 **586.5 GB/s**，
不存在 boost 与持续负载的差异，也不是热节流。

这修正了 §8.3 的「~31 ms 理论下限」：那个数用的是微基准速率
（attention 618、gate_up 597、qkv 499），而 attention 微基准的 618 含
**K/V 双流 + 跨迭代 L2 复用**（KV 302 MB，约 5% 可留 L2），真实 in-step
attention 是 **594 GB/s**。按真实 in-step 速率重算：

| 项 | 真实速率 | 每步 |
|---|---:|---:|
| attention | 594 GB/s | 21.4 ms |
| GEMM ×4 | 456–558 GB/s | 7.1–8.7 ms |
| 小算子 | — | ~2–3 ms |
| 尾部 | — | 1.9 ms |
| **下限** | | **~34 ms** |

7000 需要 ~34.9 ms/步，也就是必须把全部调度/依赖开销压到 **~1 ms**。
而 PDL 实测 ratio 1.0001、persistent 不可实现、融合被吸收（add+rms 0），
因此这个 ~1 ms 的窗口无法进入。**7000 的口径修正后依然不可达。**

### 11.1 无 L2 复用的真实 kernel 下限（2026-11-19 追加）

为避免微基准跨迭代 L2 复用，按真实 step 构造 **42 层不同权重 + 42 层不同
KV cache**（合计 16.7 GB）的图内序列：

| 序列 | 时间 | 带宽 |
|---|---:|---:|
| GEMM（42 层 × 4 矩阵，3.96 GB） | **7.84 ms** | 505 GB/s |
| attention（42 层 × 独立 KV，12.68 GB） | **20.42 ms** | 621 GB/s |
| **kernel 下限** | **28.27 ms** | |

attention 在 42 层不同 KV 下仍是 621 GB/s，所以 L2 复用不是差异来源。
加上小算子 ~1.5 ms、尾部 1.9 ms ≈ **31.7 ms**；实测单步 38.9 ms，
即 **~7.2 ms 是图内调度/依赖开销**。

7000 需要 34.9 ms，也就是要从这 7.2 ms 里回收 **~3.2 ms**。能回收它的
机制已逐一实测：PDL（两组 ratio 1.0001）、persistent（`cooperativeLaunch=0`）、
图边界（0.57 ms）、融合（add+rms +0.04%、mass −0.03%）。
**窗口存在，但没有任何可用机制能进入。**

---

## 12. 7000 tok/s 的独立算术证明（2026-11-19 最终）

§8/§11 的「~7.2 ms 图内缺口」用的是**最大上下文的中位步长 38.9 ms**。
按整个 benchmark 的阶段分解重算，平均步长其实是 31.5 ms：

| 量 | 值 |
|---|---:|
| 4k 当前吞吐 | 6555.49 tok/s |
| 总时长 = 1,310,720 / 6555.49 | **199.94 s** |
| prefill（§9.1 实测） | 71.0 s |
| decode = 199.94 − 71.0 | **128.94 s** |
| decode 步数 = 262,144 输出 token / 64 并发 | **4096** |
| **平均 decode 步长** | **31.48 ms** |

无 L2 复用的裸 kernel 下限（§11.1，42 层不同权重 + 42 层不同 KV）：

| 组件 | ms |
|---|---:|
| GEMM ×168 | 7.84 |
| attention ×42 | 20.42 |
| **GEMM + attention** | **28.27** |
| lm_head | 0.96 |
| sampler（已优化） | ~0.6 |
| **仅这两项再加尾部** | **≥ 29.83** |
| 42 层小算子（rms/rope/silu/KV write） | ~1.5 |

当前平均步长 31.48 ms ≈ 28.27 + 1.5 + 1.8 ≈ 31.6 ms，**系统已贴在物理下限上**。

7000 tok/s 需要：

```
总时长 = 1,310,720 / 7000 = 187.25 s
decode = 187.25 − 71.0    = 116.25 s
步长   = 116.25 s / 4096  = 28.38 ms
```

**28.38 ms < 29.83 ms**（仅 GEMM+attention 加 lm_head+sampler，不含 42 层
任何小算子与调度）。留给小算子/调度的预算 = 28.38 − 28.27 − 1.56 =
**−1.45 ms**。

即使把 prefill 打到其可测下限（GEMM 92% 峰值、FA 形状 roofline、小算子
83–94%，乐观 ~67 s），7000 仍要求 decode 步长 ≤ 29.2 ms < 29.83 ms。

⇒ **7000 tok/s 低于本卡 decode 的裸 kernel 下限，与调度器 / PDL /
persistent 无关，独立不可达。**

### 12.1 GEMM tile 空间的最终扫描（split=1 全组合）

| 矩阵 | 生产 | 最优 | 结论 |
|---|---|---|---|
| qkv M64 | 21.0 µs / 499 GB/s | 21.0 / 499 | 无更优 |
| qkv M32 | 21.0 / 500 | **20.3 / 517（+3.3%）** | 仅 0.7 µs/层 = **0.03 ms/步**（噪声） |
| down M32 | 47.9 / 525 | = 生产 | 无更优 |

结合 §9 的 split-K 扫描（仅 down M64 +7.7%），四个 GEMM 在 M=32/64 的
tile 与 split-K 空间已全部覆盖；GEMM 下限 7.84 ms 无法再降，§12 的
28.27 ms 是最终值。qkv M32 的 +3.3% 折合 0.03 ms/步，低于测量噪声，不采纳。

### 12.2 16k 最终复测（含 down split-K）

| 16k | tok/s |
|---|---:|
| Run1 | 5443.49 |
| Run2 | 5444.68 |
| **均值** | **5444.09** |

对原始基线 5384.0 → **+1.12%**；对 down split-K 之前的 5437.80 → +0.12%
（噪声内）。16k 的 M≤32 走 `_SKINNY_MM_LE32`，down M64 的 split-K 不生效，
符合预期。

---

## 13. 原生 ixBLAS GEMM 路线（2026-11-20 扫描）

图内 42 层**不同权重**的干净口径（一次 42 层序列，不是复用同一 buffer）：

| 形状 | FlagGems | 原生 ixBLAS | fg/nat |
|---|---:|---:|---:|
| M64 qkv | 1.23 ms | 1.31 ms | 0.94 |
| M64 o | 0.81 | 1.02 | 0.79 |
| M64 gate_up | 3.54 | 4.58 | 0.77 |
| M64 down | 2.09 | 2.44 | 0.86 |
| **M32 qkv** | 1.27 | **0.93** | **1.37** |
| M32 其它 | — | — | 0.78–0.91 |

- **4k（M=64）无收益**：FlagGems Triton 全面更快。
- 16k 的 qkv M32 原生快 1.37×（约 0.34 ms/步），但 **serving 里不可路由**：
  FlagGems 用 `torch.library.Library("aten", "IMPL")` 覆盖 `aten::mm`，
  `_launch_mm` 内调 `torch.mm` 会重入自身，EngineCore 报
  `RecursionError: maximum recursion depth exceeded` 并启动失败。
  该分支已回退；原生 qkv 的 0.34 ms/步无法落地。

### 13.1 功能性 fused add+rms（2026-11-20）

IR 的 `fused_add_rms_norm` 默认走 split（`x+residual` 一个 add kernel，再
rms），prefill 每步该 add 为 7.89 ms。实现了非 in-place 的功能性融合 kernel
（读 x/residual，写新的 out + residual_out，无克隆、无独立 add），
`residual_out` 与 `x+r` 逐位相等，`out` 行相对误差 ~0.0032（bf16 量级）。

4k A/B：**6528.23 / 6565.49 → 6546.86** vs 基线 6555.49，**−0.13%（噪声内）**。
与之前 OOT in-place 版一样没有可测收益（图内 add 的时间被吸收）。已回退。

### 13.2 扫描汇总（2026-11-20）

| 方向 | 实测 | 结论 |
|---|---|---|
| 原生 ixBLAS GEMM | 4k 全面更慢；16k qkv M32 快 1.37× 但被 aten 覆盖挡住 | 不可用 |
| FP8-KV 自研 kernel（另一条线） | 0.312 vs 0.172 ms/层 | 更慢 |
| lm_head+sampler 图捕获（另一条线） | graph 903 vs eager 905 µs | 最多 ~0.4% |
| 功能性 fused add+rms | 6546.86 vs 6555.49 | 噪声，回退 |
| prefill silu→gate_up epilogue | GEMM compute-bound 89 TF；silu 仅 0.25 ms/层 | 双累加器预计净损，未实施 |
| **prefill GEMM（M=4160）全 tile 扫描** | qkv/gate_up = tuner；o +0.3%；down +1.3%（0.76 ms/步） | 噪声/可忽略 |
| **7000 tok/s** | 需 28.38 ms/步 < 裸 GEMM+attention+尾部 29.83 ms | **独立不可达** |

补充：prefill FA 的「1.304 vs 隔离 1.03 ms」差主要是 **ixsys 逐 kernel 插桩
开销**（无 ixsys 的 paged≈dense≈1.10 ms），不是真实 gap；`max_seq_len` 来自
实际 seq_lens，没有被传大。prefill GEMM 的 tuner 在 M=4160 已接近最优。

### 13.3 silu→gate_up epilogue 可行性（2026-11-20）

- 生产 gate_up GEMM 2.43 ms/层 + silu 0.25 ms/层 = **2.68 ms/层**（112.63 ms/42 层）。
- 朴素双累加器融合 Triton kernel（8 组 config，M=4160/K=2048/I=6144）：
  最佳 **315.14 ms/42 层，2.8× 慢**——缺 tuner 的 GROUP_M / split-K / swizzle，
  裸 `tl.dot` 远达不到 tuned GEMM 的效率。
- 落地需把 FlagGems tuned mm kernel 改成双 tile epilogue 并重新调参（数天级），
  上限 +1.3% e2e，且不改变 7000 的下限结论。未实施。

### 13.4 K HD-major 布局（2026-11-20）

默认关闭的 `VLLM_FL_ILUVATAR_KV_K_HD_MAJOR=1` 在最终配置上的 4k A/B：

| 轮 | tok/s |
|---|---:|
| Run1 | 6126.83 |
| Run2 | 6135.18 |
| 均值 | **6131.01**（vs 6555.49，**−6.5%**） |

decode 的 k_contig 微基准 ~5% 的收益赢不过 HD-major 布局带来的 prefill
`to_bndh_contig` 转置 / stride 路径代价。默认保持 BNDH，该 flag 不启用。





