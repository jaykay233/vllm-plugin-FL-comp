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

---

## 9. 第三轮：未触碰阶段的穷举扫描（4k/16k 剩余空间定位）

前三轮的工作全部集中在 **decode** 上。本节第一次把两个用例的墙钟按
**prefill / decode 两阶段**拆开实测，再逐阶段做上限判断。结论：剩余空间
**不在 decode 的 kernel 里**，而在两个从未被测量的地方——图外尾部和
prefill 阶段。

### 9.1 阶段分解（首次实测，此前一直缺这项）

BI-V150 不支持 kernel 并发（§7 #6），所以 prefill 与 decode 完全串行，
两阶段耗时可以直接相加。用 `output_len=4` 把用例压成纯 prefill，实测：

| 用例 | 纯 prefill 吞吐 | 该用例 prefill 量 | prefill 耗时 | 占墙钟 |
|---|---|---|---|---|
| 4k | **14,772 tok/s** | 1,048,576 tok | 71.0 s | **34.9%** |
| 16k | **10,291 tok/s** | 2,097,152 tok | 203.8 s | **49.6%** |

对照官方计分（4k 6436.06 tok/s / 203.7 s，16k 5421.40 tok/s / 411 s）：

| 用例 | 墙钟 | prefill | decode | 校验 |
|---|---|---|---|---|
| 4k | 203.7 s | 71.0 s (34.9%) | 132.7 s (65.1%) | 干净 |
| 16k | 411 s | 203.8 s (49.6%) | 207.2 s (50.4%) | 干净 |

**16k 有一半墙钟是 prefill，4k 超过三分之一。** 而 decode 侧已被证明贴死
硬件上限（§9.3）。所以「还能优化哪里」的答案必须落在 prefill 和图外尾部。

### 9.2 prefill 阶段可以完整归因（与 decode 相反）

prefill 不进 CUDA graph，所以 `ixsys` 能看见全部 kernel。抓一次纯 prefill
trace（`-d 70 -l 40`，108 个 mixed step），取中位步 281.15 ms：

| 组件 | 耗时 | 占比 | 效率 | 判断 |
|---|---|---|---|---|
| GEMM `mm_kernel` ×168 | 185.03 ms | 65.8% | **88.7 TFLOPS = 92% 峰值** | 无空间 |
| `compute_flash_attn` ×42 | 54.78 ms | 19.5% | **52.7 TFLOPS = 55% 峰值** | 见下 |
| silu/rmsnorm/elementwise/kv-write | 35.72 ms | 12.7% | 83–94% 峰值 | 约 2–3 ms 可榨 |
| launch 间隙 | 4.76 ms | 1.7% | — | 可忽略 |

GEMM 按 M=4160（1 个 4096 prefill chunk + 64 decode token 拼成同一 GEMM）
计 16.49 TFLOP，实测 185 ms → 88.7 TFLOPS，**92% 峰值，确认无空间**。

prefill attention 的 52.7 TFLOPS 看似有空间（microbench 峰值 71–74），但
同 shape 的 microbench 直接否掉了替代方案：

```
1x4096 q / kv=4096   FA=1.114 ms   Triton=1.314 ms   FA/Triton=1.18x
```

**FA 在这个 shape 上本来就比 Triton 快 18%**，也就是说生产已经在用该 shape
的较优实现，换 Triton 会变慢。这一条独立复现了 §7 #18 的结论。

「其他算子」已全部是融合版本且跑到 83–94% 峰值（`triton_poi_fused_mul_silu_slice`
610 GB/s = 94%），只剩约 2–3 ms/步的理论空间。

### 9.3 decode 阶段：只剩「图外尾部」这一个未收割项

用一个中位 decode 步把结构钉死（16k，batch 30）：

```
步长 49.32 ms
  ├─ graph replay      45.65 ms   ← 图内 42 层，ixsys 不可见
  └─ 图外 43 个 kernel   3.67 ms   ← lm_head 900us + 采样链 1074us + 间隙/辅助
```

按模型真实规模（2.249 B 参数 → bf16 权重 4.50 GB，KV 43008 B/token）重算：

| 用例 | 每步字节 | 步长 | 整步带宽 | **graph 内** | 图外占比 |
|---|---|---|---|---|---|
| 4k (b64, ctx 5120) | 18.59 GB | 38.90 ms | 478 GB/s (73%) | **534 GB/s (82%)** | **10.5%** |
| 16k (b30, ctx 16896) | 26.30 GB | 49.32 ms | 533 GB/s (82%) | **576 GB/s (88%)** | **7.4%** |

4k 的步长 38.90 ms 是本轮新测（`[[4096,2048,64,64]]`，中位 TPOT），
此前只有被 prefill 污染的 mean TPOT。两个用例的 graph 内带宽 82% / 88%，
都已在 16-SM 硬件的实际可达区间内（256 CTA，§7.2）。

### 9.4 候选清单（按实测上限排序）

每一项都附「已测前提」，不含未验证的假设。上限 = 该阶段当前耗时 × 占墙钟。

| # | 候选 | 已测前提 | 理论上限 | 现实上限 | 是否已尝试 |
|---|---|---|---|---|---|
| **A** | **把 lm_head + 采样链纳入 CUDA graph** | 图外尾部 4k 4.07 ms/步(10.5%)、16k 3.67 ms/步(7.4%)；采样链 rows=64 实测 1.47 ms | 4k **+8.2%** / 16k **+3.9%** | 4k +4% / 16k +2%（假设能收一半） | **部分**：lm_head 本身试过 3 次（§17.19、qdraft W4），**把捕获边界挪到采样之后从未实施**（见 9.4.1） |
| **B** | **4k decode 的 graph 内带宽 82% → 88%** | 同机同内核 16k 达 88%，4k 仅 82%，6pt 差距已定量 | 4k **+4.9%**（步 38.90→36.47 ms） | +2~3% | **否**（机理未明：b64 有 64 张 page table 交错） |
| C | prefill「其他算子」83–94% → 95% | 35.72 ms/步 × 256(4k)/512(16k) 步 | 4k +4.5% / 16k +4.5%（归零为上限） | +0.2%（已在 83–94%） | 否 |
| D | prefill attention 52.7 TFLOPS | 同 shape FA 已优于 Triton 18% | 上限受该 shape 的 FA 实现限制 | ≈0 | **是（本轮二次确认）** |
| E | 采样链本身（top_p + softmax） | rows=64 527+253 us，已达 500+ GB/s | 4k +~1% | +0.3% | 部分（top_p 已修 14×） |

合计现实上限：**4k 约 +6~8%，16k 约 +2~3%**。

#### 9.4.1 为什么 A 和 B 当前都没做

这两个是仅有的有量级项，但都不属于「调 kernel」范畴，成本性质不同：

**A（把 lm_head + 采样链纳入 graph）——改的是 vLLM 内部结构。**

⚠️ **更正：这一项不是全新方向，之前已有结论，本节必须交代清楚。**

已试过的（都在 lm_head 本身上，不是挪捕获边界）：

| 尝试 | 出处 | 结果 |
|---|---|---|
| `linear()` 无 bias 且 (N<65k 或 M≤64) 转 `mm()` | `FINDINGS-iluvatar.md` §17.19.3 | 4k Run2 **+0.3%**，噪声级，已 revert |
| `CUSTOMIZED_UNUSED_OPS` 加 `"linear"` → ixBLAS | §17.19.4 | 4k Run2 **+0.3%**，已 revert |
| lm_head 改 W4（给投机 draft 用） | `bench_results/qdraft_lmhead_w4_e2e` | head **0.51×**（更慢），`go=false` |
| 与最后一层 RMSNorm 融合 | §17.19 表格 | 只省一个小 kernel，GEMV 仍在，未实施 |
| **融进 decode cudagraph** | §17.19 表格 | 判断为「难：vLLM 结构上 logits 在图外，硬拉进图要动 capture 边界」，**只做了判断，未实施** |

§17.19.5 的原结论是「lm_head theoretically ~2% TPOT，实测省 ~0.15 ms/step，
e2e 吃不出来；不再为 4k 做 lm_head 增量融合」。

**本轮的新信息只在于量级被修正，不在于可行性被改变**：当时按 lm_head 单算子
估 ~1.1 ms/步（~2%），而实测整条图外尾部是 **3.67–4.07 ms/步（7.4–10.5%）**，
差额来自采样链（1.47 ms）与 43 次 launch 间隙。所以「值不值」变了，但
「难不难」没变，阻塞理由与 §17.19 完全一致：

`FULL_DECODE_ONLY` 的捕获边界是 model forward，采样在 `model_runner` 里、
位于 graph replay 之后。要收进图必须把 sampler 移进被捕获函数，而采样链含
`argmax`/`torch.multinomial` 等带动态输出的算子，捕获后必须改成「固定形状 +
读出索引」的形式，且 `SamplingParams`（top_p/temperature）在运行期可变，
只能按固定值捕获、其余走 fallback。这会改动 vLLM 上游文件，超出本插件
（`vllm_fl`）的改动边界，属于需要单独评审的变更。**风险点**：捕获失败会
整体退化为 eager，可能倒亏。

**B（4k graph 内 82% → 88%）——机理还没定位，不能先改代码。**
已确认 16k(b30) 同内核同机达到 88%，4k(b64) 只有 82%，差 6pt。但两者的
结构差异同时有两个变量：序列数 64 vs 30、每序列 page 数 320 vs 1024。
现有证据无法区分是「64 张 page table 交错破坏 DRAM 局部性」还是
「每序列 page 太短、split 内游程不足」。**下一步应先做单变量消融**
（固定 ctx、扫 batch 32/48/64；再固定 batch、扫 ctx），把变量分离出来，
再决定动 `_pick_split` 的哪一维。在没有定位前改参数已被 §7 #16 证明会变慢。

> 说明：16k 的 prefill 单 chunk 成本随 kv 增长而上升（每个 prompt 的 4 个
> chunk 依次对应 kv=4096/8192/12288/16384），trace 中 342.7 ms 是其中一段，
> 平均约 398 ms；512 chunk × 398 ms ≈ 203.8 s，与实测吻合，不是异常。

### 9.5 本轮排除项（不重复挖掘）

| 项 | 排除依据 |
|---|---|
| 提高 `gpu_memory_utilization` 扩容 KV | **比赛规则禁止改该参数**。KV 上限锁死 522,576 token，16k 只能跑 30 条序列（名义 64），这是 16k 权重占比升到 17% 的直接原因 |
| 多卡张量并行 | `torch.cuda.device_count() == 1`，32 GiB / 16 SM 单卡 |
| MoE 专家分流 | 模型是 dense Llama，42 层全部命中 |
| KV 量化 / page size / 双 stream / decode GEMM / 投机解码(qdraft W4) / prefill tile 扫描 / SPLIT 规则 / CTA target | RESULTS §7 #1–#18 与 `experiments/bench_results/qdraft_*` 已有实测结论 |
| **lm_head 本身的提速**（GEMM 路由、ixBLAS、W4、与 RMSNorm 融合） | `FINDINGS-iluvatar.md` §17.19.3 / §17.19.4 / §17.19.5：两次尝试均 +0.3% 噪声级已 revert；W4 见 qdraft。**唯一未做的是挪捕获边界**，见 §9.4.1 |
| `argmax` 是异常慢算子 | **本轮新测并排除**：微基准 553 us 看似 30 GB/s，但生产 trace 实测 71.7 us，微基准与生产路径不同，不构成异常 |
| GPU 空闲浪费 | **本轮新测并排除**：prefill 态空闲 3.5%、decode 态 0%（45 ms「洞」即不可见的 graph replay） |

### 9.6 结论

两个用例的剩余空间**结构相同**：decode 已贴死硬件（graph 内 82–88% 峰值，
且 4k 的 34.9% / 16k 的 49.6% 墙钟其实是 prefill），因此唯一有量级的
未收割项是 **A（图外尾部）**，其次是 **B（4k 带宽 6pt 差距）**。

按现实上限，把 A 和 B 都打满约能拿到 **4k +6~8% / 16k +2~3%**。这与前两轮
的 +3.6% / +1.4% 同一量级——**不存在两位数百分比的剩余空间**，除非改变
算法本身（投机解码，其 W4 变体已在 `qdraft_lmhead_w4_e2e` 证伪）。

### 9.7 本轮新增可复现步骤

```bash
# 阶段分解：把 output_len 压到 4 即得纯 prefill 吞吐
BENCH_RUNS=1 python3 benchmarks/benchmark_throughput_serve.py \
  --model /workspace/MiniCPM5-2B --served-model-name minicpm --port 9065 \
  --test-cases '[[4096,4,64,256],[16384,4,30,128]]'

# 4k 纯 decode 步长（batch 64，长输出使 decode 主导）
BENCH_RUNS=1 python3 benchmarks/benchmark_throughput_serve.py \
  --model /workspace/MiniCPM5-2B --served-model-name minicpm --port 9065 \
  --test-cases '[[4096,2048,64,64]]'          # 读 Median TPOT = 38.90 ms

# prefill 逐 kernel 归因（prefill 不入图，可完整归因）
ixsys -o /tmp/pf.ixrep -t cuda_kernel -d 70 -l 40 bash experiments/tools/driver_prefill_trace.sh
python3 experiments/tools/trace_prefill_analyze.py /tmp/pf.ixrep
```
