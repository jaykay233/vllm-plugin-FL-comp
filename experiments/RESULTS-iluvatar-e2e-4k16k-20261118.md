# MiniCPM5-2B on Iluvatar BI-V150 — 4k/16k 吞吐标定与采样器瓶颈

日期：2026-11-18（第二轮，supersedes 第一轮的 4k 6223 / 16k 5321 结论）

平台：Iluvatar BI-V150（16 SM），vLLM 0.24.0，flagtree 0.7.0。

---

## 0. 结论摘要

| 项 | 基线 (stock) | 本配置 | 变化 |
|---|---|---|---|
| 4k `[[4096,1024,64,256]]` | 6211.86 tok/s | **6436.06 tok/s** | **+3.6%** |
| 16k `[[16384,1024,64,128]]` | 5348 tok/s | **5421.40 tok/s** | **+1.4%** |
| 4k Mean TPOT | — | 46.07 ms（原 48.01） | −4.0% |
| 16k Mean TPOT | — | 84.00 ms（原 85.75） | −2.0% |
| MATH-500 L3 精度 | 95.2% | **98.1%** | 门禁 PASS |

本轮的真实收益来自**一处采样器内核缺陷**（`_top_p_mass_v2_kernel` 的直方图单桶争用），
不是 decode/prefill 记忆体路径 —— 后者在本轮被证明已经贴到硬件上限。

第二轮针对「还能不能再优化」做了 5 项定向追查（§7.1–7.3），**全部证伪，吞吐无变化**。
关键一条是工具层面的：CUDA graph replay 对 `ixsys` 不透明，逐层带宽归因不可行，
因此改用 context 拟合替代，结论是 16k 的 83% 属于 256 CTA / 16 SM 的占用上限。

---

## 1. 交付配置

```bash
VLLM_PLUGINS=fl
VLLM_FL_ILUVATAR_KV_K_HD_MAJOR=0      # HD-major K 关闭
VLLM_FL_ILUVATAR_HD_PREFILL=0
VLLM_FL_ILUVATAR_IVCORE_DECODE=0      # ivcore u8 decode 默认关闭
VLLM_FL_ILUVATAR_DECODE_TRITON=1      # Triton FlashDecoding 解码
VLLM_FL_ILUVATAR_KV_INT8=0 / KV_FP8=0 # 量化 KV 关闭
# 其余走 vllm-plugin-fl 默认：HND KV layout、ixformer FA prefill、
# decode-first batch reorder + split metadata (IXFORMER_SPLIT=1)、
# FULL_DECODE_ONLY cudagraph、chunked prefill、async scheduling
```

启动：

```bash
vllm serve /workspace/MiniCPM5-2B --port 9071 \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --served-model-name minicpm --gpu-memory-utilization 0.85 \
  --max-model-len 131072
```

本轮代码改动共两处，见 §3、§4。

---

## 2. 精度门禁（硬性要求）

MATH-500 **Level 3**，105 题，evalscope openai_api，
`t=1.0 / top_p=0.95 / batch=8 / max_tokens=32768`。

| 指标 | 结果 | 门槛 | 判定 |
|---|---|---|---|
| Accuracy（evalscope 报告） | **0.9808**（102/104 有效样本） | ≥ 95.0% | **PASS** |
| 保守下界（把丢失样本计为错） | 97.14%（102/105） | ≥ 95.0% | **PASS** |

证据（`num=104`、`score=0.9808`、`macro_score=0.9808`，直接从报告读出）：

- `/workspace/evalscope-datasets/level3_split_gate/20261118_183154/reports/minicpm/math_500.json`
- `.../reviews/minicpm/math_500_Level 3.jsonl`（逐题 `sample_score`，102 正确 / 104 判分）
- 运行日志：`/tmp/gate_split/evalscope.log`

> 第一轮记录为 95.24%（100/105），余量仅 0.24%。本轮 `lo` 改动后得 98.08%，
> 余量扩到 3.1%，说明放宽直方图范围并未损伤采样质量。
>
> 关于 104 而非 105：第 105 题在 18:44:43 因**运行者手工 kill 服务**而报
> `Connection refused`，被 `--ignore-errors` 记为 errored。日志明确写着
> `score computed from 104/105 successful samples; 1 errored`。这不是模型答错，
> 也不是改动引入的偏差；按最保守口径（该题计为错）仍是 102/105 = 97.14% > 95%。

---

## 3. 吞吐证据

benchmark：`benchmarks/benchmark_throughput_serve.py`，`BENCH_RUNS=2`（取后一轮）。

| case | 轮次 | tok/s | Mean TTFT | Mean TPOT |
|---|---|---|---|---|
| 4096→1024, c=64 | run1 | 6386.82 | — | 46.05 |
| 4096→1024, c=64 | **run2** | **6436.06** | 3611.34 | 46.07 |
| 16384→1024, c=128 | run1 | 5419.51 | — | 83.91 |
| 16384→1024, c=128 | **run2** | **5421.40** | 93149.04 | 84.00 |

同日对照（同一脚本、同一机器，仅差本轮改动）：

| 配置 | 4k tok/s | 16k tok/s |
|---|---|---|
| 基线 | 6211.86 | 5348 |
| 仅 split 规则修复 | 6204.57 | 5317.86 |
| + 采样器修复（本轮交付） | **6436.06** | **5421.40** |

证据目录：`/tmp/bench_lofix/`（4k）、`/tmp/bench_lofix16k/`（16k）。

**独立的机理旁证** —— 16k 纯 decode 区间（`prompt tps≈0`、30 并发）的
generation 速率：

| | 中位数 | 均值 |
|---|---|---|
| 修复前 | 617.8 tok/s | 608.1 |
| 修复前（另一次） | 616.3 tok/s | 607.2 |
| **修复后** | **644.9 tok/s** | **639.5** |

decode 速率单独提升 **+4.4%**，与 e2e 的 TPOT −2.0% 方向一致
（16k 里 decode 约占总时长 39%，故 e2e 幅度小于 decode 幅度）。

---

## 4. 本轮改动一：采样器直方图单桶争用（主要收益）

### 4.1 发现路径

用 `ixsys`（本机唯一可用 profiler，产物是 sqlite）包住真实 16k 服务，
在稳态窗口抓 20 秒：**21558 个 kernel / 275 次 CUDA graph replay**。

decode 步的精确分解（每步 48.5 ms）：

| 组成 | 每步 | 说明 |
|---|---|---|
| CUDA graph replay（42 层前向） | 44.6 ms | 275 次 replay 长度 44.52–45.15 ms |
| 未入图：lm_head `linear_kernel` | 0.905 ms | 622 MB 权重 → 687 GB/s |
| 未入图：采样链 43 个 kernel | 2.98 ms | **其中 `_top_p_mass_v2_kernel` 独占 2.13–2.45 ms** |

同一个采样链上的兄弟 kernel 全部正常：

```
_top_p_rowmax_v2_kernel      19.3 us
_top_p_amax_v2_kernel         6.3 us
_top_p_threshold_v2_kernel   17.7 us
_top_p_mask_v2_kernel        56.7 us
_top_p_mass_v2_kernel      2450.0 us   <-- 100x outlier，全步最慢 kernel
```

### 4.2 根因

`_top_p_mass_v2_kernel` 对每个元素发一次 `tl.atomic_add` 到
`(rows, 8192)` 直方图。桶按 **log2 概率** 划分，范围 `lo=-40` nat
对 `hi=0`。真实 lm_head logits 的绝大多数元素比该行最大值低 **> 40 nat**，
因此全部被 `tl.maximum(bucket, 0.0)` 夹进 **bin 0**，形成
**130560 个 atomic 打同一个地址** 的串行化。

微基准定量（rows=32, vocab=130560）：

| 分布 | 耗时 | bin0 占比 | 用到的桶数 |
|---|---|---|---|
| gaussian（合成） | 203.0 us | 0.1% | 6992 |
| gaussian→bf16 | 1524.8 us | 0.1% | 2848 |
| **production-like（贴合真实）** | **3048.1 us** | **87.9%** | 6344 |
| 更宽 σ | 3453.7 us | 99.9% | 2352 |
| 尖峰分布 | 3351.2 us | 93.4% | 958 |

### 4.3 修复

`src/flag_gems/ops/top_p_threshold.py`：

```python
_DEFAULT_LO = -40.0   ->   -120.0
```

把直方图范围放宽到 120 nat，元素自然散开到真实桶，不再挤 bin 0。
切分精度代价：`120/8192 = 0.0101` nat/桶（原 `0.0034`），
远小于 top-p 边界处的 logit 间距。

消融（rows=32 / rows=64，同一 kernel 只改 `lo`）：

| 方案 | production-like | gaussian |
|---|---|---|
| A 原始 `lo=-40` | 2908.3 / 4121.9 us | 210.5 us |
| B bin0 块内聚合 | 1839.8 / 2576.7 us | 206.9 us |
| C 纯读地板（无 atomic） | 20.7 / 55.9 us | 20.7 us |
| **D `lo=-120`** | **208.1 / 401.8 us** | **175.0 us** |

D 在病态情形 **14.0× / 10.3×**，干净情形还略快。

### 4.4 等价性

同一 logits 分别用 `lo=-40` 与 `lo=-120` 跑完整 `top_p_threshold`：

| 分布 | 保留 token 数 | Jaccard | 保留概率质量 | 分布 TV 距离 |
|---|---|---|---|---|
| production-like r=32 | 214 / 214 | **1.0000** | 1.0000 / 1.0000 | **0** |
| production-like r=64 | 498 / 498 | **1.0000** | 1.0000 / 1.0000 | **0** |
| gaussian r=32 | 10919 / 10962 | 0.9961 | 1.0000 / 1.0000 | 1.76e-4 |
| gaussian r=64 | 26289 / 26362 | 0.9972 | 1.0000 / 1.0000 | 1.76e-4 |

production-like 分布下**逐位一致**；gaussian 下差异仅在 0.01 nat 边界附近，
最终由 §2 的精度门禁背书。

---

## 5. 本轮改动二：FlashDecoding split 规则（正确但计分场景上是 no-op）

`src/flag_gems/runtime/backend/_iluvatar/fused/paged_decode_triton.py`：
`_pick_split` 原本向上取整到 `_CTA_TARGET=256`，会多跳一级。改为
取「CTAs ≤ 256 的最大 2 的幂」。

扫描 14 个 (batch, ctx) 点，**12 个变快 3.4–5.7%**，合计 1.0342×：

```
n=27/28 ctx=16384   s8 -> s4   1.041x / 1.043x
n=30    ctx=16384   s8 -> s4   1.034x
n=24/25 ctx=16384   s8 -> s4   1.057x / 1.050x
n=40/48 ctx=16384   s4 -> s2   1.046x / 1.050x
n=80..102 ctx=4096  s2 -> s1   1.035x .. 1.044x
n=64    ctx=4096    s2 -> s2   (tie)
```

**但 e2e 无变化**，原因已查明：vLLM 的 CUDA graph 按 **padding 后的 batch**
捕获，尺寸为 `[1,2,4,8,16,24,32,40,48,56,64,72]`。16k 实际 batch 29–31 →
走 batch=32 的 graph，`_pick_split(32)` 在新旧规则下**都返回 s4**；
4k 的 61–64 → batch=64，两边都是 s2。故两个计分场景下规则同值。

保留该改动（非 graph 路径下确实更好），但**不计入吞吐收益**。
新增 `VLLM_FL_ILUVATAR_DECODE_SPLIT_DEBUG=1` 可打印实际选中的 split。

数值：新旧 split 输出 `cos 0.99999`、maxabs 1.6e-2（纯 bf16 舍入）。

---

## 6. 硬件上限标定（本轮新测得，用于判断剩余空间）

| 项 | 实测 | 说明 |
|---|---|---|
| HBM 带宽峰值 | 652 GB/s | `torch.sum` 顺序读 |
| **GEMM 峰值** | **96.6 TFLOPS** | 27 个方阵组合，@ (8192,8192,16384) |
| 模型层 GEMM 实际 | **92.9 TFLOPS = 96%** | 4 个层的真实 prefill M |
| prefill attention (Triton) | 68.8 TFLOPS | 10 组 tiling (BLOCK_M/TILE/warps/stages) 全一致，已饱和 |
| 16k decode KV 搬运 | 21.1 GB/步 | 30 seq × 16384 × 43008 B/token |
| decode 总搬运 | 25.1 GB/步 | KV 21.1 + 权重 4.0 |
| decode roofline | 38.5 ms/步 | 25.1 GB ÷ 652 GB/s |
| **decode 实测** | **44.6 ms/步 = 86–91%** | graph replay 部分 |
| decode skinny GEMM | 533 GB/s = 82% | flag_gems 路径；aten 仅 421 GB/s |
| **step 内 kernel 间隙** | **6.7 us，占 1.0%** | 7212 个间隙的实测均值 |

### 6.1 对上一轮结论的更正

第一轮文档称「两个 case 落在理论上限 93%，剩 7% 是调度气泡」。
本轮 profiler 实测**推翻了这个说法**：step 内 kernel 间隙只有 6.7 us、
占 1.0%，不存在 7% 的调度气泡可捡。上一轮把 gap 估大了约 9 倍。

同时这也解释了为什么此前几轮试图优化 decode/prefill 记忆体路径都无收益：
GEMM 96% 峰值、attention 68.8 TFLOPS 平台、decode KV 86–91% 峰值
——这三条路都已经贴死。真正的非饱和点是**采样器**，它不在任何一条
记忆体/算力曲线上，所以从未被 roofline 分析照到。

---

## 7. 已证伪的方向（累计）

| # | 实验 | 结论 | 机理 |
|---|---|---|---|
| 1 | INT8 KV `int8_per_token_head` | 池 522k→1013k，4k 掉到 **2372** tok/s | dtype 门禁 `if k.dtype not in (bf16,fp16): return False` 使全部 attention 退回带反量化的通用 Triton |
| 2 | FP8 KV `fp8_e4m3` | 池 522k→**1044k**，16128 次 attention 全 `unsupported` | 同上 |
| 3 | 量化 KV 的带宽收益 | fp8 字节减半只快 5% | KV 读受 page 间接寻址 / 128B 小行限制，非纯带宽 |
| 4 | KV page 16→32/64/128 | 慢 2.0× / 3.3× / 14× | split 粒度与 page 数绑定的内核结构 |
| 5 | `max_num_batched_tokens` 8192 | 5283（−0.7%） | 4096 已最优 |
| 6 | 双 stream decode‖prefill 重叠 | **0.99–1.00×** | BI-V150 不支持核函数并发 |
| 7 | prefill GEMM 优化 | 92.9 vs 峰值 96.6 TFLOPS | 已 96%，无空间 |
| 8 | prefill attention tiling | 10 组全 68.8 TFLOPS | 已饱和 |
| 9 | `--no-enable-chunked-prefill` | 启动失败 | 要求 `max_num_batched_tokens ≥ max_model_len` |
| 10 | ivcore u8 fullpack | 4k 2625 | 上一轮已证伪 |
| 11 | **async scheduling A/B** | 测试无效 | `VLLM_FL_ILUVATAR_ASYNC_SCHEDULING=0` 语义是「留给 auto」，auto 本身即 enabled，两组配置相同 |
| 12 | **graph padding 行导致 bin0 争用** | **证伪** | pad32 微基准 2109 us 曾看似吻合生产 2128 us，但打上「同桶块内聚合」补丁后生产仍 2450 us，说明 padding 行不是主因；真因是全部元素（含真实行）都低于 `lo` |
| 13 | **step 内调度气泡** | **证伪** | 实测间隙 6.7 us / 1.0%，非上一轮估计的 13% |
| 14 | **把 16k decode 逐层归因** | **工具不可行** | CUDA graph replay 对 ixsys 完全不透明（见 7.1） |
| 15 | **长上下文回退 ixformer 是瓶颈** | **证伪** | 强制 `DECODE_TRITON=1` 让 16k 也走 Triton：50.82 vs 49.33 ms，反而慢 3% |
| 16 | **CTA 数不足（`_CTA_TARGET` 256→512）** | **证伪** | 4k Total 6335.05 vs 6436.06（−1.6%），output 1267 vs 1389（−8.8%） |
| 17 | **`true_div` 比 aten 慢 2.7×** | **证伪（测量伪影）** | 同进程 A/B 为 ~127 us，手写版仅快 1.06×；349 us 是一次性缺页离群值 |
| 18 | **prefill attention 还有 68.8→74.3 TFLOPS** | **证伪** | 生产 prefill 已全走 ixformer FA；68.8 是 Triton 微基准，非活路径 |

### 7.1 为什么 16k decode 的逐层归因做不到

用每步必定执行的 `linear_kernel`（lm_head，在 CUDA graph 之外）当标尺，
数它之间的 kernel 个数：

```
linear_kernel 出现                 392 次
相邻两次之间窗口的 kernel 数        43 个 x 389 步
这 43 个的构成                     采样器链 + lm_head + slot_mapping/scatter/fill/arange
单步 wall                          48.21 ms
可见 kernel 时长之和                3.57 ms
=> CUDA graph replay             44.64 ms   ← 完全不可见
```

**42 层 transformer 对 `ixsys -t cuda_kernel` 是一整块黑盒**：graph 内部的
kernel 不会被逐个上报。所以「把 46.5 ms 拆成每层 attention / GEMM」在工具
层面不可行；已知图内耗时为 44.6 ms，图外为 3.57 ms。

### 7.2 替代方法：用 context 拟合反推现场带宽

固定 batch 24、输出 2048、取中位 TPOT（长输出使 decode 主导，避开 TTFT 期
的 prefill 争用）：

| context | 中位 TPOT | 相邻斜率 | 反推 KV 带宽 |
|---|---|---|---|
| 4096 | 21.00 ms | — | — |
| 8192 | 28.66 ms | 1.87 us / ctx-token | 552 GB/s（85%）|
| 16384 | 49.33 ms | 2.52 us / ctx-token | 409 GB/s（63%）|

斜率随上下文**变陡**，即现场 KV 流效率随长度下降；同形状隔离微基准是
617 GB/s，所以确有 ~30% 的现场损失。

但把 batch 也纳入后可以看到，这个损失主要来自 **CTA 占用**而非 kernel 选择：

```
batch 24 -> 48 base x split 4 = 192 CTAs -> 393 GB/s
batch 32 -> 64 base x split 4 = 256 CTAs -> 555 GB/s   ← 官方配置
```

16 个 SM 下 192 CTA 是 12 波（尾部损失 8%）、256 CTA 是 16 波（6%）。官方
benchmark 的 batch 30 在 graph 里 pad 到 32，**已经站在更优的占用点上**。
这解释了为什么 16k 的「83%」不是可回收的浪费，而是 256 CTA / 16 SM 的实际水平。

### 7.3 代码状态

`paged_decode_triton.py` 新增 `VLLM_FL_ILUVATAR_DECODE_CTA_TARGET`（默认
仍为 256，行为与已交付版本完全一致），并用注释记录了 512 的实测失败结果，
使这个旋钮不会被后人重复探索。经 7.1–7.2 与 #15/#16 验证，**本轮无性能提升**：
4k 保持 6436.06 tok/s、16k 保持 5421.40 tok/s。


---

## 8. 复现步骤

```bash
# 1) 代码改动
#    FlagGems-comp:  src/flag_gems/ops/top_p_threshold.py            (_DEFAULT_LO)
#                    src/flag_gems/runtime/backend/_iluvatar/fused/paged_decode_triton.py

# 2) 精度门禁（硬门槛 ≥95%）
bash /tmp/gate_split.sh

# 3) 吞吐计分
BENCH_RUNS=2 python3 benchmarks/benchmark_throughput_serve.py \
  --model /workspace/MiniCPM5-2B --served-model-name minicpm --port 9071 \
  --test-cases '[[4096,1024,64,256]]'
BENCH_RUNS=2 python3 benchmarks/benchmark_throughput_serve.py \
  --model /workspace/MiniCPM5-2B --served-model-name minicpm --port 9071 \
  --test-cases '[[16384,1024,64,128]]'

# 4) kernel 级归因
ixsys -o trace.ixrep -t cuda_kernel -d 170 -l 20 bash inner_serve.sh
python3 /tmp/trace16k/analyze2.py trace.ixrep    # 注意 ixsys 时间戳单位是 ns

#    注：graph 内部 kernel 不上报。用 lm_head 标尺确认方法见 §7.1：
python3 experiments/tools/trace_graphvis.py trace.ixrep

# 5) 替代归因：context 拟合（batch 固定，长输出让 decode 主导）
BENCH_RUNS=1 python3 benchmarks/benchmark_throughput_serve.py \
  --model /workspace/MiniCPM5-2B --served-model-name minicpm --port 9065 \
  --test-cases '[[4096,2048,24,24],[8192,2048,24,24],[16384,2048,24,24]]'
#    读 Median TPOT；斜率 1.87 -> 2.52 us/ctx-token 即 552 -> 409 GB/s
```
