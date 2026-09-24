# MiniCPM5-2B on MetaX C500：优化收益与排查经验

本文件汇总本次竞赛优化中**已量化的收益**与**排查过程得到的经验**。所有数字都注明
出处，原始日志与脚本见 `experiments/`（`src/` 与 `bench_results/`）。

环境：MetaX C500 单卡 · bf16 · `vllm-plugin-FL` + FlagGems 5.3.5 · vLLM 0.24.0
模型：MiniCPM5-2B（`LlamaForCausalLM`，42 层，hidden=2048，16 q-heads / 2 kv-heads，head_dim 128）

---

## 1. 摘要

优化围绕三条线：**消除阻塞式 D2H**、**按 workload 选对 attention 分片**、
**消除纯搬运的 kernel 启动开销**。共同特征是——它们都不改数值结果，
属于「白拿」的收益。

| 排名 | 优化 | 收益 | 状态 |
|---|---|---|---|
| 1 | FlagOS 融合算子白名单 | TTFT **−57.6%**、TPOT **−29.8%** | 配置级，未进代码 |
| 2 | 去掉 `flash_attn` 的阻塞 D2H | TTFT A/B **−32.9%**；16k 实测 **+5.50%** | 已提交 `2fc71f9` |
| 3 | decode attention `num_splits` 自适应 | TPOT **−7.3%** / 吞吐 **+7.8%** | 已提交 `4899f91` |
| 4 | `fused_add_rms_norm` 去掉 168 次 clone | 吞吐 **+2.6%** | 已提交 `dc27f60` |
| 5 | M==1 `mm` 路由到 GEMV | 每层 **1.12x**，预期 TPOT ≈ **−6.5%** | FlagGems `a68024d21`，e2e 待验 |

最可靠的端到端结果：**16k 场景 +5.50%（轮间抖动 0.0%）**，与预测的 +6.83% 同向且量级吻合。

---

## 2. 评测口径与基线

指标：`Total tok/s = (input_tokens + output_tokens) / duration`。
官方脚本 `RUNS=4`、跳过第 1 轮取均值，且要求 `successful_requests == num_prompts`，
否则整例作废。

> **读日志时别踩这个坑**：一次 `benchmark_throughput_serve.py` 调用内部就跑 4 轮，
> 于是输出里会出现 4 条 `Total token throughput`，**它们是 4 个单轮值，不是 4 次实验**；
> 真正该记录的是末尾 `Summary` 里的
> `Prefill=4096 Decode=1024 ... Total tok/s=5192.98 TTFT=2737.79ms`。
> 曾把 4 条单轮值误当成 4 次独立实验来算均值，结论偏差约 2 个百分点。

| 场景 | 配置 | 基线 Total tok/s | 门槛（−1%） | 基线 Mean TTFT |
|---|---|---|---|---|
| 4k | 256 × (4096+1024), 并发 64 | 5089.65 | ≥5038.75 | 3199.44 ms |
| 16k | 128 × (16384+1024), 并发 64 | 7029.68 | ≥6959.38 | 27197.14 ms |

「偏差 1% 以内视为正常波动」是硬信息 —— **任何小于 1% 的优化都不赋分**，
必须盯 2% 以上的动作。

### 2.0 加白名单后的官方口径结果（已用修好的 harness 复现）

`VLLM_FL_FLAGOS_WHITELIST=silu_and_mul,rms_norm,rotary_embedding`，
其余与评测口径完全一致（不传 `--compilation-config`、`gpu-memory-utilization 0.85`、
`max-model-len 131072`）。评测脚本见 `experiments/src/run_eval_whitelist.sh`。

**性能（4k）**：`Run 1 = 8797.67`、`Run 2 = 8529.41`（第 3 轮按需求提前中止）。
对比无白名单的 `5271.99 / 4793.31`，**首轮同口径 +67%**。

**正确性（evalscope math_500 Level 3, 105 题）**：

| 指标 | 本轮（白名单） | 无白名单基线 | 竞赛基线 | 门槛 | 判定 |
|---|---|---|---|---|---|
| Accuracy | **97.1%** | **97.1%** | 0.962 | ≥0.95 | ✅ PASS |

**结论：+67% 吞吐 / −45% TPOT 的同时，Accuracy 与基线完全一致（97.1%），
即提速没有牺牲正确性。** 这是该配置可安全用于提交的关键一关。

> 这轮在「单实例锁 + 偏移就绪检测」修好之后跑，`bench.log` 未出现交错
> （`Run 1/4 → Run 2/4 → Run 3/4` 顺序正常），因此结果可信。详见 §6 的 harness 教训。

### 2.1 本机落进了 vLLM 的「小 batch」默认档

vLLM 的 `get_batch_defaults()` 按显存分档，`device_memory >= 70 GiB` 才给大默认值。
MetaX C500 实测总显存 **63.59 GiB**，差 6.4 GiB 掉到 else 分支，于是
`max_num_batched_tokens=2048`、`max_num_seqs=256`（H100 档本应 8192 / 1024）。

后果：评测不是几次大 prefill，而是**大量 2048-token 的小步**（4k 约 512 步、
16k 约 1024 步）；这也放大了每步固定开销类问题的影响。

### 2.2 反解：时间维度上 decode 占 4k 场景的一半

用两场景联立 `T = in_tok/P + out_tok/D` 反解：

| 场景 | prefill | 占比 | decode | 占比 |
|---|---|---|---|---|
| 4k | 125.5 s | 48.7% | 132.0 s | **51.3%** |
| 16k | 251.0 s | 79.2% | 66.0 s | 20.8% |

token 维度确实是 80%/94% 是输入，但**时间维度**上 decode 占 4k 场景的 51.3%。
换算：decode 快 10% ≈ 4k 提升 5.1%、16k 提升 2.3%；prefill 快 10% ≈ 4k 提升 5.4%、
16k 提升 8.8%。两者量级相当，不能只押一边。

---

## 3. 收益清单

### 3.1 已验证的收益

#### (1) FlagOS 融合算子白名单 —— 最大单笔收益

`VLLM_FL_FLAGOS_WHITELIST=silu_and_mul,rms_norm,rotary_embedding`
（配合 `VLLM_FL_PREFER=flagos`、`USE_FLAGGEMS=1`）

shape：`input_len=512, output_len=128`，两种并发。
出处：`bench_results/flagos_sweep/summary.json`

| 用例 | 指标 | 默认 FlagOS | 白名单 | 变化 |
|---|---|---|---|---|
| 并发 1 | Mean TTFT | 91.11 ms | **38.62 ms** | **−57.6%** |
| 并发 1 | Mean TPOT | 10.14 ms | **7.12 ms** | **−29.8%** |
| 并发 1 | Total tok/s | 464.09 | **678.59** | **+46.2%** |
| 并发 8 | Mean TTFT | 351.62 ms | **90.40 ms** | **−74.3%** |
| 并发 8 | Mean TPOT | 12.34 ms | **9.96 ms** | **−19.3%** |
| 并发 8 | Total tok/s | 2659.76 | **3764.19** | **+41.5%** |

对照臂（`flagos_default` / `oot_off` / `per_op_fused_flagos` / `strict`）全部在
464–466 tok/s 附近，说明**默认配置实际上让 FlagGems 的融合算子没被用上**。
把 `rotary_embedding` 交给 FlagGems 尤其关键：默认路径下 RoPE 由 inductor
生成 `triton_poi_fused_cat_*`，一次 decode 步要 128 次、合计 515 µs（占 7.5%）。

> ⚠️ **这是配置级收益，未进代码**，而官方评测的启动命令没有带这个 env var。
> 也就是说 **§3.1(1) 的收益目前并没有体现在 §2 的实测结果里**，是最大的一处未收割项。

#### (2) 去掉 `flash_attn` 前向里的阻塞 D2H（`2fc71f9`）

原代码在 split 路径上这样构造前缀和：

```python
[0] + attn_metadata.prefill_seq_lens.tolist()   # 阻塞 D2H
# 再 torch.tensor(..., device=...) 搬回 GPU（多余的 H2D）
```

为搬 **4 字节**做了一次阻塞 D2H + 一次冗余 H2D，而且这行在 attention forward 里，
**每个 prefill step 每层各一次 = 42 次/step**。实测 **471 µs/次，19.79 ms/step**
（p50 465 / p90 507，稳定；总搬运量 200 字节）。

改为在 device 上直接构造（`new_zeros(1)` 保持 dtype/device，`cumsum` 数值等价）：

```python
torch.cat([prefill_seq_lens.new_zeros(1), prefill_seq_lens]).cumsum(0, dtype=torch.int32)
```

验证（batch=1，prompt ≈498 tok，`torch.compile`）：

| | Mean TTFT | 变化 |
|---|---|---|
| baseline | 59.53 ms | — |
| patched | **39.97 ms** | **−32.9%** |

32-token 生成结果**逐位一致**；独立 probe 测得的 19.79 ms/step 与实测差值
19.57 ms 吻合到 **1.1%**。

折算到评测形态：4k 省 ~10.1 s（占 duration 3.93%）→ **+4.09%**；
16k 省 ~20.3 s（6.39%）→ **+6.83%**。**16k 的官方口径实测为 +5.50%**（见 §3.1(4)）。

#### (3) decode attention `num_splits` 自适应（`4899f91`）

MetaX 的 `flash_attn_with_kvcache()` 在 decode 时不传 `num_splits`，
默认 `0` = 由 MACA 启发式决定，而该启发式**对真实序列长度不敏感**：
固定切 8 片（模板参数里的常量 `8`），于是每步都要付一次跨片归约：

```
flash_fwd_splitkv_kernel           ~351 µs/step  (42 层)
flash_fwd_splitkv_combine_kernel  ~1007 µs/step  (42 层)   <-- 纯开销
```

那 ~1007 µs 在 13 个 KV token 和 1201 个 KV token 时几乎相同 —— 这就是破绽：
它是**固定成本**，不是与上下文成正比的工作量。

工作负载对齐评测集（`math_500` 形态：短 prompt + 长 CoT，batch=1）：

| num_splits | TPOT 均值 | combine µs/step | 对比 |
|---|---|---|---|
| 0（启发式） | 7522.6 µs | 1007.1 | — |
| **16** | **6975.8 µs** | **399.8** | TPOT **−7.27%** / 吞吐 **+7.84%** |
| 32 | 6980.2 µs | 474.0 | −7.21% / +7.77% |

贪心 token id 全部 5 次运行**完全一致** → 是数值等价，不是精度换性能。

关键约束：**attention 在 FULL decode CUDA graph 内执行**（实测 16-token 生成只触发
42 次调用、`max_seqlen` 冻结在 9，说明只在捕获期调用），所以 `num_splits` 是
**捕获期常量**，只能在捕获时决定。因此交付的不是常量而是自适应规则
（`FL_METAX_ATTN_ADAPTIVE_BATCH` 以下用 16，否则回落到启发式）。

最优值随形态漂移，无单一常量通行：

| batch | seqlen | 启发式 | 最优 ns | 收益 |
|---|---|---|---|---|
| 1 | 32 | 13.5 µs | 1 | 37.1% |
| 1 | 2048 | 31.0 µs | 32 | 28.1% |
| 32 | 2048 | 68.7 µs | 0 | 0.0% |

#### (4) `fused_add_rms_norm` 去掉 168 次 clone（`dc27f60`）

`vllm_fl/ops/ir_kernels.py` 每次调用都 clone 两个 activation，只为满足
`torch.library.custom_op` 的 no-aliasing 规则。decode 时那是 **168 次启动/step、
~477 µs**，即 **8.6% 的 TPOT**，全部是 4 KB 张量的启动开销。

拆开操作（残差用普通 add，再调已存在的 functional `rms_norm`）后仍然逐位等价：

| | tok/s | µs/token | copy 启动/step | copy µs/step |
|---|---|---|---|---|
| `fused`（前） | 144.2 / 143.2 | 6937 / 6982 | 214.7 | 622.0 |
| **`split`（后）** | **147.7 / 147.5** | **6773 / 6779** | **45.3** | **143.8** |

**+2.6% 吞吐 / −2.6% TPOT**，交错运行复现性 0.14%。

### 3.2 已落地、微基准已验证、端到端待跑

#### (5) M==1 `mm` 路由到 GEMV（FlagGems `a68024d21`）

M=1 时四条投影是纯 memory-bound GEMV：权重流过一次、毫无复用，只关乎实测带宽；
而 dense/split-k 的分块是为 M>1 设计的，M=1 时机器大半闲置。

| 形状 | M | N | K | 权重 | 当前路径 | 实测带宽 |
|---|---|---|---|---|---|---|
| qkv | 1 | 2560 | 2048 | 10.5 MB | `mm_kernel_nt` | **699 GB/s** |
| gate_up | 1 | 12288 | 2048 | 50.3 MB | `mm_kernel_nt` | 1275 GB/s |
| o_proj | 1 | 2048 | 2048 | 8.4 MB | `mm_kernel_splitk` | **625 GB/s** |
| down | 1 | 2048 | 6144 | 25.2 MB | `mm_kernel_splitk` | **862 GB/s** |
| lm_head | 1 | 130560 | 2048 | 534.8 MB | `_gemv_kernel` | **1607 GB/s** |

A/B（profiler `device_time`）：

| 形状 | `mm` | GEMV | 加速 | max rel err |
|---|---|---|---|---|
| qkv | 14.99 µs | 13.24 µs | 1.13x | 4.2e-04 |
| gate_up | 39.48 | 39.04 | 1.01x | 1.4e-03 |
| o_proj | 13.43 | 11.36 | 1.18x | 5.4e-03 |
| down | 29.20 | 22.98 | **1.27x** | 7.5e-03 |
| **每层合计** | **97.10 µs** | **86.62 µs** | **1.12x** | — |

预期 **≈444 µs/step ≈ 6.5% TPOT**。开关 `FLAG_GEMS_METAX_MM_GEMV`（默认 1）。
命中条件严格（`M==1`、`a.stride()==(K,1)`、`b.stride()==(1,K)`、dtype 一致），
不命中即回退原路径。

### 3.3 负结果（同样重要：避免重走）

| 尝试 | 结果 | 结论 |
|---|---|---|
| `VLLM_FL_CUDAGRAPH_ONLY=1`（mode=NONE + FULL_DECODE_ONLY） | 4k **−10~15%** | 丢掉 torch.compile 等于丢掉 piecewise prefill 图 |
| `--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'`（保留 compile） | 4k **−23%+** | 证明**是 prefill 图而非 kernel 选择**支撑了 prefill 主导场景 |
| 关闭 `splitk_mm_scenario` | 每层 **+3.9 µs（更慢）** | split-k 在该启发式下是合理的，不要动 |
| 把 `clone` 归因于 `register_impl(inplace=True)` | A/B 证伪（两者都是 214.7 次） | 真凶是 custom op 自己的 clone |

`VLLM_FL_CUDAGRAPH_ONLY` 的价值在启动耗时（~100 s vs ~34 min），
但这是**用吞吐换启动**，默认必须关闭。

---

## 4. 4k 场景的双峰现象与 20 秒停顿

### 4.1 现象：不是一个均值加噪声，而是两个确定状态

4k 基准的历次结果落在两个精确的模式上，而不是连续分布：

| 模式 | Total tok/s | 轮间复现精度 |
|---|---|---|
| 低峰 | 4925.87 / 4928.25 / 4901.67 | 相互差 **0.05%** |
| 高峰 | 5303.77 / 5316.18 / 5314.03 / 5314.53 | 相互差 **0.23%** |

**16k 完全没有这个问题**（抖动 0.0%），所以这不是通用噪声。

### 4.2 定位：低峰 = 高峰 + 一次 20.0 秒停顿

用 `--enable-logging-iteration-details` 打开逐迭代日志后，4 个整轮共 16966 次迭代里，
**只有一处异常停顿**：

```
20.0 s  iter14912 -> iter14913  @08:13:11   前一步 ctx=1986 gen=62 27.4ms
```

算术完全闭合：

| | 迭代耗时合计 | 非迭代时间 | 客户端 duration |
|---|---|---|---|
| 高峰轮 | 148.9 s | 98 s | 246.6 s |
| 低峰轮 | 148.9 s | **118 s**（98 + **20**） | 267.4 s |

模型计算量完全相同，低峰纯粹是多出 20 秒空转。

### 4.3 已被数据排除的原因

| 假设 | 结论 | 依据 |
|---|---|---|
| Python GC 停顿 | **排除** | 110 次 GC，median 0.12 ms、**max 8.52 ms**、合计 0.04 s；只有 gen0/gen1，**无 gen2** |
| 内存压力 / swap | **排除** | `Swap: 0`，空闲内存 108 GB |
| 运行期编译 / 图重捕获 | **排除** | 编译日志全在启动期；`Capturing` 期间无时间戳且已完成 |
| 客户端断流 | **排除** | 引擎自报 `Running=64, Waiting=0`，是引擎自己不走步 |
| 调度脚本自身 I/O | **排除** | 拷贝 ctime 08:15:43，晚于停顿 08:13:11 |
| 上一轮诊断采样器残留 | **排除** | 最后写入 00:46，且已无残留 |
| 输出队列背压 | **排除** | `output_queue = queue.Queue()` 无 maxsize，`put_nowait` 不会阻塞 |
| 缺异步调度 | **排除** | 已是 `Asynchronous scheduling is enabled.` 默认开启 |
| `VLLM_GC_DEBUG=1` 调试器 | **排除** | 关掉后（`共 0 次 GC`）停顿照样复现，见 §4.5 |
| 输入队列 drain | **排除** | 该次 `slow loop 20.126s (input 0.000s, ...)`，输入侧为 0 |
| 被 `iteration elapsed time` 计时的区间 | **排除** | 全程 **无一条迭代 >1000 ms**（最大仅 141.66 ms），20 秒不在计时区内 |
| ~20s 超时常量 | **未找到** | `v1/` 下无 20s 量级的超时；最接近的是 `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=300` |

GC 之所以被排除得很干净，是因为 vLLM 启动时主动调 `freeze_gc_heap()`
把静态对象推到最老代并冻结，所以每次 GC 都是「collected 0/N」、gen2 根本不触发。

现场特征（08:13:08–08:13:38 引擎自报）：

```
08:13:08  prompt 13925.7 tok/s, gen 425.7, Running 62, Waiting 1
08:13:18  prompt  4505.6 tok/s, gen 135.9, Running 64, Waiting 0
08:13:28  prompt     0.0 tok/s, gen   0.0, Running 64, Waiting 0   <-- 冻结
08:13:38  prompt   409.5 tok/s, gen 870.1, Running 64, Waiting 0
```

而停顿跨越的那次迭代只量到 **0.05 ms** —— 那 20 秒**不落在被计时的区间里**。

### 4.4 残余候选与影响量化

剩下两个候选：**「外部设备/驱动事件」**与**「运行时/主机侧事件」**。
`20.0` 这个整齐的整数更像某个超时值，而不是随机调度抖动。

按官方口径（跳第 1 轮、其余 3 轮取均值）：单次停顿使该轮 **−7.5%**；
若每轮命中概率约 1/4，则 3 轮里至少命中一轮的概率 **≈58%**，
折算到最终均值的**期望损失 ≈1.9%** —— 与 847 修复的收益同量级，且它还会**抬高方差**，在取均值的评分口径下双重不利。

实测印证：某次 4 轮为 `5274.92, 5314.03, 5314.53, 4901.67`。
按官方口径跳首轮，后 3 轮均值 **5176.74（+1.71%）**；
而其中两个干净轮（`5314.03, 5314.53`）均值 **5314.28（+4.41%）**。
一次 20 秒停顿吃掉约 **2.70 个百分点**。

这也解释了 4k 数据的整体不可信：历次 4k 结果在 4581–5316 之间摆动
（另一轮为 `4591.36, 5301.16, 4580.83`，均值 4824.45 即 **−5.21%**、抖动 14.9%），
同一份代码既能跑出 +4.41% 也能跑出 −5.21%，差别全在停顿次数。
**16k 则完全稳定**（`7416.21, 7416.10, 7415.64`，均值 7415.98 即 **+5.50%**、抖动 <0.01%）。

最近一轮（`VLLM_ITER_STAGE_PROFILE=1`、`VLLM_GC_DEBUG=0`）再次命中停顿，把这笔损失
量得更准：4 轮为 `5274.65, 5323.40, 5321.46, 4934.08`。

| 口径 | Total tok/s | vs 基线 5089.65 |
|---|---|---|
| 官方（跳首轮，后 3 轮平均） | **5192.98** | **+2.03%** |
| 若第 4 轮也干净 | 5322.43 | +4.57% |
| 第 4 轮实测（含停顿） | 4934.08 | −3.06% |

同一次停顿还把 TTFT P99 顶到 **15772.71 ms**（Mean 2737.79 / Median 864.12）。
即：**一次停顿同时吃掉约 2.5 个百分点的吞吐、并污染 TTFT 长尾**，
而它命中与否不受代码控制 —— 这是当前**收益最高、也最该先解决的单项**。

### 4.5 把 20 秒逼进 `step_with_batch_queue` 的未计时区

上一轮埋点把「被计时的区间」也排除掉了：`--enable-logging-iteration-details` 的
`iteration elapsed time` 最大值仅 **141.66 ms**，16956 条迭代里**没有一条 >1000 ms**。
而那 20 秒是引擎自报的单次循环 `slow loop 20.126s`，且 `input 0.000s`。
两条合起来说明：**20 秒完全落在 `log_iteration_details` 覆盖的那一小段之外。**

读 vLLM 源码后确认上一版埋点埋错了函数：开异步调度时走的是
`step_with_batch_queue`，而 `iteration elapsed time` 只包住它的 `future.result()`；
下面的调用全部不在计时内：

```
scheduler.schedule()            <- 未计时
model_executor.execute_model()  <- 未计时
scheduler.get_grammar_bitmask() <- 未计时
model_executor.sample_tokens()  <- 未计时
scheduler.update_from_output()  <- 未计时（上一轮）
```

（上一版埋点埋在 `step()` 里，那是 `batch_queue is None` 才走的死路径 ——
所以 `sched/exe/upd` 长期恒为 0，这也是个自己骗自己的坑。）

现在埋点已移到 `step_with_batch_queue`，并在慢循环发生时打印**该次迭代**各阶段
的增量（`sched / submit / wait / update`），直接点名是哪一次调用吃掉了 20 秒。
开关是 `VLLM_ITER_STAGE_PROFILE=1`（见 §6.4：补丁本身不随仓库提交）。

现场形状（停顿紧跟在一个 prefill chunk 之后，`ctx_tokens≈2048` 正好是调度上限）：

```
08:54:16  Iteration(12840): 2 context reqs, 1986 context tokens, 62 gen   34 ms
08:54:30  stats: prompt 0, gen 0, Running 64, Waiting 0      <-- 冻结 10s
08:54:36  Iteration(12841): 1 context req, 1985 context tokens, 63 gen
08:54:36  slow loop 20.126s (input 0.000s, step 20.126s)     <-- 单次循环
```

### 4.6 双峰已量化到阶段级：`submit` 退化 13~15 倍

修正埋点后，`step_with_batch_queue` 的各阶段逐窗统计出现了**极干净的二分**：

修正埋点后（`unmarked ≈ 0.01 s`，归因闭合），双峰呈现出**完全相反的两套时间去向**：

| 窗口类型 | n | `submit` | `wait` | 主要去向 |
|---|---|---|---|---|
| **纯 decode**（`ctx=0`） | 422~431 | 2.9 s（**6.8 ms/it**） | 16.6 s（38.6 ms/it） | **等 GPU（83%）** |
| **含 prefill** | 149~200 | 15~17 s（**80~108 ms/it**） | 3.6~4.6 s | **提交侧（80%）** |

这与「每 20s 迭代数」的独立统计吻合（纯 decode 窗 n≈426，含 prefill 窗 n≈150~310）。

- **纯 decode 是健康流水线**：提交很快（6.8 ms），然后阻塞等 GPU（38.6 ms）—— GPU 是瓶颈。
- **含 prefill 时提交侧暴涨到 ~100 ms/it**，`wait` 反而降到 18~26 ms —— **CPU 成了瓶颈**。

即双峰不是「一次 20 秒停顿」，而是**每步开销随「该步是否含 prefill」跳变**：
含 prefill 的步被 CPU 侧拖慢约 13~15 倍，于是同样 20 秒内只跑完约 1/3 的迭代。

**必须记下的一个语义陷阱**：`non_block=True` 并不代表非阻塞。
`UniProcExecutor.collective_rpc` 是这么写的：

```python
if not non_block:
    return run_method(self.driver_worker, method, args, kwargs)
try:
    result = run_method(self.driver_worker, method, args, kwargs)   # 同步执行
    if isinstance(result, AsyncModelRunnerOutput):
        return AsyncOutputFuture(result, single_value)              # 只把「取结果」延后
    future = Future[Any]()
    future.set_result(...)
```

也就是说 `non_block=True` **只把返回值包成 Future**，`run_method`（即 worker 侧的
`execute_model` / `sample_tokens`）仍在该调用里同步跑。所以 `submit` 里装的是
**真实的 worker 侧工作**，不是一次消息发送 —— 不能按「发送延迟」去解释。

因此下一步是把 `submit` 再拆成 `execute_model` / `get_grammar_bitmask` / `sample_tokens`
三段（埋点已改为 `sched/exec/gram/sample/wait/update` 六段，`submit` 为派生值），
以确定那 ~100 ms 落在哪一段。

（20 秒停顿本轮未复现；`unmarked` 字段就是为它复现时点名而加的。）

### 4.7 停顿被锁定为「单次 `submit` 调用」，且归因闭合

下一次运行（闭合埋点已生效）**复现了停顿，且一次就抓到三条**：

```
19.853s | step_total=19.853s | this iter: sched=0.001s submit=19.851s wait=0.000s update=0.001s
26.222s | step_total=26.222s | this iter: sched=0.001s submit=26.220s wait=0.000s update=0.001s
19.335s | step_total=19.335s | this iter: sched=0.001s submit=19.333s wait=0.000s update=0.000s
```

注意 `unmarked=0.000s` —— 这一次各段之和**精确等于** `step_total`，所以归因可信。
对应窗口同样印证：

```
win 41.6s n=108 | submit=39.225s(363.20ms/it) wait=2.172s | step_total=41.625s unmarked=0.003s
win 24.7s n=66  | submit=22.619s(342.72ms/it) wait=1.979s | step_total=24.706s unmarked=0.002s
win 29.5s n=155 | submit=23.681s(152.78ms/it) wait=5.590s | step_total=29.494s unmarked=0.005s
```

**结论：那 20 秒是「单次迭代里、单次 `submit` 调用」被阻塞**（`sched≈0`、`wait≈0`、
`input≈0`）。因为 `submit` 由 `execute_model` / `get_grammar_bitmask` / `sample_tokens`
组成，且 `non_block=True` 在这条路径上并不真正非阻塞（见 §4.6），
所以下一步就是把 `submit` 拆成这三段定位到具体哪一次调用。

时长在 19.3~26.2 s 间波动、并不固定为整 20 s —— 这更像**某次阻塞在等待一个
外部事件**（驱动/设备/锁），而不是一个写死的超时常量。

同时这也解释了低峰：本轮第 4 个 4k 轮次连续吃到多次 19~26 s 停顿，
所以该轮 duration 被显著拉长。**停顿命中与否不受代码控制，是方差的来源。**

### 4.8 根因：FlagGems 在推理中现场 autotune（`do_bench_cudagraph`）

把 `submit` 拆成 `exec / gram / sample` 的三段埋点后，下一次运行立刻给出答案：

```
20.914s | step_total=20.914s | this iter:
         sched=0.001s  exec=20.905s  gram=0.000s  sample=0.006s  wait=0.000s  update=0.001s
```

`gram = 0.000s`（与读源码一致：无结构化输出时 `get_grammar_bitmask` 立即返回 `None`），
`sample` 也几乎为零 —— **20 秒全在 `exec`，即 `execute_model` 里。**

随后在停顿瞬间用标准库 `faulthandler` 抓到了**主线程**完整栈（无需 py-spy/gdb，
本机都没有；看门狗在 `step` 运行超过 5 s 时 dump 全线程栈）：

```
run_busy_loop → _process_engine_step → step_with_batch_queue → execute_model
  → vllm_fl/worker/worker.py:938 → worker/model_runner.py:4401 → :3862 _model_forward
  → llama.py:392 forward（inductor 编译图）
  → flag_gems/runtime/backend/_metax/ops/mm.py:1304 mm_out
  → mm.py:1092 general_mm_nt
  → triton/runtime/jit.py:346 <lambda>
  → flag_gems/utils/libentry.py:1701 run → 1035 run → 737 policy → 1427 default_policy
  → libentry.py:999 bench
  → triton/runtime/autotuner.py:135 _bench
  → triton/testing.py:76 do_bench_cudagraph
  → torch/cuda/__init__.py:1159 synchronize          <-- 卡在这里 ~20 s
```

**即：FlagGems 的 MetaX `mm` kernel 带 Triton autotune。运行中一旦遇到缓存里没有的
`(M, N, K, ...)`，就在推理热路径上现场跑基准（`do_bench_cudagraph`）挑配置，
一次约 20 秒，把整个 EngineCore 卡死。**

细节（`flag_gems/utils/libentry.py`）：

- autotune 的 key 含 `M`（`key=["M","N","K", ...]`，见 `_metax/ops/mm.py` 各处 `@libtuner`），
  而 `M` 就是该步的 token 数，随调度在变。
- 结果缓存在 **SQLite** 里（本机 `/root/.flaggems/config_cache/TunedConfig_metax_triton_3_0.db`，
  5.8 MB，41 张表；`mm_kernel_nt` 一张表就有 82 个不同 `M`）。缓存命中则**跳过基准**
  （`libentry.py:986`：`if bypass_config_cache or config_key not in self.cache:`）。
- 默认基准模式是 `BenchmarkMode.REPLAY`，即**为每个配置做 CUDA graph 捕获+重放**
  （`libentry.py:147`）。这是这 20 秒昂贵的主因。
- 该缓存**是持久化的**，所以同一 shape 只会付一次代价；但新的 `M` 会不断出现
  （本轮停顿时刻 DB 正在被写入），所以代价会以「每次首遇」的形式反复出现。

**这解释了全部现象**：4k 双峰（低峰 = 该轮吃到若干次 autotune 停顿）、
`iteration elapsed time` 测不到（autotune 在 worker 侧、而它发生在被计入 `exec` 的
`execute_model` 里）、`prompt/gen 吞吐归零而 `Running` 不变`、以及停顿紧跟 prefill
（prefill 才会产生新的 `M`）。

**候选修法（按稳健性）**：

1. **预热缓存**：在计分前用一遍 warmup 把用到的 `M` 填进 SQLite；或用官方预调优 CLI
   `flag_gems.flagtune.cli.pretune`，并可用 `FLAGGEMS_DB_URL` / `FLAGGEMS_CACHE_DIR`
   指向随提交一起带上的预调优库。最稳，但要保证评测环境的缓存不是冷的。
2. **把 `mm` 的 `benchmark_mode` 从 `REPLAY` 改为 `EVENT`**：去掉每配置的 CUDA graph
   捕获，20 s 量级可显著缩短（代价是选出的配置可能略次）。
3. **让 `M` 有界**：把 prefill 的 `M` 归一到少量桶，使 key 空间固定、缓存填满后不再 miss。
4. **把 `mm` 移出 FlagGems 路径**（回落到原生 `mm`）：彻底绕开 autotune —— 见 §4.9。
   注意 `prefer: flagos` 会把所有算子默认指向 FlagGems，需用白名单/黑名单调整。

### 4.9 prefill 的 `M` 是连续值 ⇒ 预热缓存不可行

查 SQLite 里 `mm_kernel_nt` 实际出现过的 `M`：

```
1,2,4,8,16,24,...,512（步长 8）           <- decode 的 padding 桶，有界
1308,1415,1417,1554,1568,1677,1789,1827,1828,1909,1911,
1921,1948,1965,1967,1973,1977,1979,1981,1987,2024,2030,
2032,2034,2048                             <- prefill 的 chunk 大小，**近乎连续**
```

**prefill 的 `M` 是 1..`max_num_batched_tokens`(=2048) 内的任意值**，
所以「预热缓存」对 prefill 无效：连续空间填不满，每遇到新 chunk 大小就再付一次 ~20 s。

而 `mm` 为什么会走 FlagGems，是关键：`IR_OP_TO_FL_OP`
（`vllm_fl/ops/ir_kernels.py:52`）只登记了 `rms_norm` / `fused_add_rms_norm`，
**不含 `mm`**。本 run 里 `mm` 走 FlagGems，是因为 `metax.yaml` 的
`prefer: flagos` 把**所有**算子默认指向 FlagGems，而本 run **没有设白名单**
（`run_stage_prof.sh` 只设了 `VLLM_ITER_STAGE_PROFILE` 等）。

对照 §3.1(1) 的最优配置：
`VLLM_FL_FLAGOS_WHITELIST=silu_and_mul,rms_norm,rotary_embedding` —— **不含 `mm`**。
白名单下未列入的算子回落到 reference（原生 PyTorch/MACA）实现，
**即那份最优配置本身就绕开了 FlagGems 的 `mm`，从而顺带消除了这 20 秒停顿。**
这可能正是白名单「收益异常大」的一部分原因。**待实验验证（§4.10）。**

### 4.10 A/B 验证：白名单同时消除停顿与 `mm` 的 autotune

唯一变量是是否设
`VLLM_FL_FLAGOS_WHITELIST=silu_and_mul,rms_norm,rotary_embedding`
（两臂其余配置完全一致：同样开了 `VLLM_ITER_STAGE_PROFILE=1`、`VLLM_PLUGINS=fl`，
都没设 `VLLM_FL_CUDAGRAPH_ONLY` / `FL_METAX_ATTN_NUM_SPLITS`）。

| | 无白名单（`stage_prof_nowl`） | 有白名单（`stage_prof_wl`） |
|---|---|---|
| 各轮 Total tok/s | **5271.99** / **4793.31** | 8854.49 / 8871.60 / 8855.08 / 8693.59 |
| 官方口径（丢首轮、后 3 轮均值） | 仅跑完 2 轮（被中止） | **8806.76** |
| 20 s 级停顿 | 多次（§4.7 抓到 3 次） | **零** |
| autotune DB 增量 | 增长（停顿时刻正在写入） | **`mm` 类 +0，全库 +0** |
| Mean TTFT / Mean TPOT | 2789.92 ms / 63.74 ms | **1814.53 ms / 34.82 ms** |
| P99 TTFT | 16013.47 ms | **10538.74 ms** |

**首轮同口径对比：5271.99 → 8854.49，即 +68%。**
（对照臂只跑完 2 轮即被中止，无法给出官方口径均值；其第 2 轮 `4793.31` 正是被停顿
命中后的低峰。白名单臂 4 轮波动仅 ±1%，且第 2~4 轮均值 `8806.76` 与官方 Summary
完全吻合 —— 顺带验证了对 `benchmark_throughput_serve.py` 口径的理解。）

三条证据**相互独立**且互相印证：

1. **autotune DB 零增长**（`mm` 类 19527→19527、全库 22314→22314、表数 41→41）
   —— 说明白名单下 FlagGems `mm` 的**调优根本没有被触发**，即 `mm` 确实已回落到
   原生实现（reference）。这是最硬的判据：它不依赖吞吐数字，直接观测副作用。
2. **零停顿** —— 白名单臂的 `slow loop` 全是启动期等请求
   （`input=47~65s` 而 `step_total≈0.05s`），另有一次真实 `wait=2.315s`，
   **没有任何 `exec` 主导的长停顿**；看门狗文件只有表头（从未触发）。
   与 §4.8 的根因（`mm` 的 `do_bench_cudagraph`）一致。
3. **吞吐大幅提升**（首轮同口径 +68%）—— 幅度**远超**停顿本身能解释的量级
   （§4.4 估过一次停顿约 2.5 个百分点）。所以这是**两笔收益叠加**：
   消除 autotune 长尾 ＋ **FlagGems 的 Triton `mm` 在 MetaX 上本身就是负优化**，
   换成原生 MACA `mm` 后大幅变快。TTFT/TPOT 同步改善（TPOT −45%）也支持这一点：
   若只是消除偶发长停顿，TPOT 中位数不该动这么多。

**这一步的重要含义**：白名单不只是「选几个融合算子」，它同时**决定了 `mm`
落在哪条实现路径上**；而 `metax.yaml` 的 `prefer: flagos` 默认让**所有**算子
（含 `mm`）都走 FlagGems —— 这正是评测环境最可能踩到的默认状态。

### 4.11 根治：把 `M` 的 autotune key 加上界（`align32_geometric`）

§4.8/§4.9 的结论是「prefill 的 `M` 近乎连续 ⇒ 缓存永远预热不完」。但这是
**key 设计问题**，不是宿命 —— 只要让相邻的 `M` 落进同一个桶，连续空间就被
折叠成有界的桶集。

**缺陷定位**：`flag_gems/runtime/common.py::DEFAULT_STRATEGIES` 已声明
`mm` / `mm_nt` / `mm_splitk` 应为 `align32`（把 key 向上取整到 32 的倍数），
但该表只在 `TuningMode.EXPANDED` 下被消费，而 MetaX 的 decorator 既没传
`strategy=` 又以 `TuningMode.DEFAULT` 运行，于是退化成**恒等策略**、
按原始 `M` 建 key。所以最早那版 `align32` 补丁方向是对的，
**问题只在于 `align32` 本身没有上界**：

```
align32(M) = ceil(M/32) * 32          # M=1..2048 -> 69 个 key
```

69 个 key 意味着最多 69 次完整 autotune。实测本机单次 `mm` autotune 均值
**5.36 s**（348.63 s / 65 次），于是最坏 ~370 s 的 tuning 全压在推理热路径上
—— 这正是 §8.2 里「零算子选择」那一臂启动 8 分 47 秒仍未就绪的原因。

**修法**：`_metax/tuner_strategies.py` 新增 `align32_geometric`，
在 **128 以下保留 `align32` 的 32 粒度**（那正是 vLLM 真正捕获 CUDA graph
的 batch 区间，decode 的 tile 选择在此最敏感），之上才转几何分桶：

| | `align32` | `align32_geometric` |
|---|---|---|
| 桶集（M=1..2048） | 69 | **13** |
| 桶集内容 | 每 32 一个 | `1 2 4 8 16 32 64 96 128 256 512 1024 2048` |
| 最坏 tuning 时间 | ~370 s | **~70 s** |

只对 **`M` 维**生效；`N` / `K` / `stride` 保持 `align32`，它们是静态形状，
`DEFAULT_STRATEGIES` 的选择本来就是对的。

**A/B 实测**（`run_m_bucket_ab.sh`，同一 M 范围 1..2048、步长 8、各自独立空 DB，
唯一变量是 `mm.py` 里的策略，共覆盖 7 个 `libtuner` decorator）：

| | `align32` | `align32_geometric` | 改善 |
|---|---|---|---|
| tuning 次数 | **65** | **9** | **7.2×** |
| tuning 总耗时 | 348.63 s | **55.04 s** | **6.3×** |
| 端到端 wall time | 349.9 s | **56.2 s** | **6.2×** |
| 命中桶数（理论） | 69 | 13 | 5.3× |

两臂的命中次数都紧贴各自桶集（65 vs 69、9 vs 13），差值是步长 8 未命中的
若干边界桶 —— 说明**分桶行为与设计完全一致**，不是碰巧。

**这条修复的合规含义（接 §8.3 第 2 条）**：它不涉及任何算子派发选择，
是纯粹的算子编译优化，且方向与赛事评判维度（第 116 行「算子编译优化」）一致。
更重要的是它让「把 `mm` 也交给 FlagGems」重新成为**可用选项**：
§10.6（`mm_metax_rootcause.md`）已验证白名单加入 `linear` + `mm` 后
4k 吞吐 8849.86 vs 8836.39、三项指标持平或略优、autotune DB 零增长。

**尚未覆盖**：几何桶仍随 `M` 增长，只是增长很慢（2048→11 个桶）。
若将来 `max_num_batched_tokens` 提到远大于 2048，还需要一层**显式上界**
（超过某值一律并入最大桶）。当前 MetaX C500 因显存低于 vLLM 的 70 GiB 门槛，
`max_num_batched_tokens` 落在 2048，13 个桶已经够用。

---

## 5. 排查经验

### 5.1 测量方法（最容易自欺的地方）

1. **CUDA event 测 ~30 µs 的 kernel，测的是 launch overhead，不是 GPU 时间。**
   早期一次快速 sweep 与端到端差了约 30 倍；改用 profiler 的 `device_time` 才一致。
   同一坑在更早的 `rms_norm` 调查里踩过一次 —— 这是复发型的坑。
2. **MACA profiler 不记录 CUDA kernel 的 python stack**（`with_stack=True` 返回
   `<no stack>`，`cpu_parent` 为空）。归因只能靠 `TorchDispatchMode` / chrome trace
   重建调用链，再用每步 CPU op 计数对齐（`clone` == `mm` == 168/step 是关键证据）。
3. **单个 TPOT 样本噪声足以骗人。** 某次 ns=16 报 7750 µs，而 profiler 显示 GPU
   时间明明下降了；同一配置三次重复给出 6988 µs。**取重复最小值。**
4. **快扫会系统性低估**：CUDA graph 下 split workspace 按捕获期上界分配，
   而非实时序列长度，所以快扫的 `combine`（~14 µs）低于真值（~24 µs）。
   **只有端到端数字是权威的。**
5. **秒级时间戳的量化陷阱。** 引擎日志时间戳只有秒精度，用它算相邻迭代间隔会出现
   大量负值与「恰好 1000 ms」的假象（本次统计出 4001 个负值、247 个整 1000 ms）。
   要做间隔分析必须拿亚秒级计时，或干脆读源码确认计时区间。
6. **插桩后必须验证「各标记之和能闭合墙钟时间」，否则会得到可信但错误的归因。**
   本次把 `wait` 累加写在了 `with log_iteration_details(...)` **之前**，而
   `future.result()`（即 `iteration elapsed time`）就在那个 `with` 块里 ——
   于是整个阻塞等待落了空，`wait` 恒为 0，反而把一个**次要**的 `submit`
   抬成了头号嫌疑。若非先做了一次闭合检查（`step_total` 与各标记之差），
   这条错误归因会被直接写进结论。
   判据：`step_total - (sched + submit + wait + update) ≈ 0`。
   本条已固化为埋点里的 `unmarked=` 字段，**每窗必看**。
   推广：任何分段计时都要先证明「段之和 = 总时间」，再去解释哪一段最大。

### 5.2 一个此前被忽略的计时盲区

`iteration elapsed time` **只包住 `execute_model` + `sample_tokens`**：

```python
future = self.model_executor.execute_model(scheduler_output, non_block=True)
grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)
with (self.log_error_detail(...), self.log_iteration_details(...)):
    model_output = future.result()          # <-- 计时区间
    if model_output is None:
        model_output = self.model_executor.sample_tokens(grammar_output)
self._process_aborts_queue()                 # <-- 不计时
engine_core_outputs = self.scheduler.update_from_output(...)   # <-- 不计时
```

**不含** `scheduler.schedule()`、`get_grammar_bitmask()`、`update_from_output()`、
以及 busy loop 里的输出入队与 `post_step`。实测占比：

| | 4k 单轮 |
|---|---|
| 引擎迭代耗时合计 | 148.9 s |
| 客户端 duration | 246.6 s |
| **占比** | **60.4%** |

客户端侧交叉验证吻合：**Mean ITL 57.2 ms vs 引擎 decode 步 38.6 ms = 1.48x**。

> 注：这个 40% 非迭代时间是既有行为，与是否开日志无关
> （未开日志的轮次 duration 246.6 s，与开了日志的完全一致）。
> 其中多少是可压缩的 host 开销，需实测拆解，**不可直接当作可回收收益**。

### 5.3 诊断工具本身可能是扰动源

昨晚开 `freeze_catcher.py` + `stall_sampler.py`（1 Hz 读 `/proc`，
每秒一次 `mx-smi --show-usage`）时抓到 7 次 10 秒冻结、2/4 轮低峰；
今天不跑采样器时 3/3 全高峰、零冻结。

**结论：测量工具会改变被测系统。** 报告「停顿」前先确认那不是采样器造成的。

### 5.4 一个真实的翻车：`VLLM_GC_DEBUG=1` 会打死引擎

为了验证 GC 假设，开了 vLLM 自带的 GC 调试器（`core.py:234` 已挂 callback）。
结果是基准第 1 轮 **208/256 请求失败**、`EngineDeadError: EngineCore encountered an
issue` —— EngineCore **信号级死亡**（无 Python traceback）：

```
08:32:53  EngineCore 最后活动
08:32:58  engine stats: Running 50, Waiting 14
08:32:59  [shutdown] MPClient: engine core exited unexpectedly
```

嫌疑明确：该调试器在 GC 回调里调 `gc.get_objects()`，而 vLLM 自己在
`utils/gc_utils.py` 里就注明这段代码
*"occasionally it would run into signals which kills the engine"*。

**教训：打开任何 debug 开关前，先读它自己的实现与注释。**
GC 结论已拿到，故此后改为 `VLLM_GC_DEBUG=0`。

### 5.5 失败结果会伪装成好结果

那次崩溃轮报出 `Total tok/s = 7816.88`，**远超**基线 5089 与我们的 5315，看着像巨大突破。
实际是**分子残缺**：只有 48/256 个请求成功（`Total input tokens 196560` ≈ 48 × 4096），
而分母是被提前结束的 25.45 s duration。同理 `Mean TTFT = 11470 ms` 也是假的 ——
客户端把自己约 25 s 的 tokenize 时间算进了「首 chunk 之前」。

**只看 `Total tok/s` 会被骗，必须同时核对 `Successful requests == num_prompts`。**

### 5.6 共享机器上的纪律

1. **不要 kill 不属于自己的进程。** 这台机器的 GPU 是物理共享的，
   `mx-smi --show-all-process` 可列出跨容器调用者，是判断「外部占用」的权威工具。
2. **显存 teardown race**：连着跑两次会撞
   `Free memory on device (...) is less than desired GPU memory utilization`。
   开跑前先查空闲显存，别浪费一个 block。
3. **残留后台进程会静默占显存**，并污染下一次测量的计时。
4. **共享 append-only 日志里 grep 成功标记，会匹配到上一次 run 的标记**，
   把失败报成 OK。要用 per-attempt marker 或 tail。

### 5.7 环境与脚本坑

1. **`PYTHONPATH` 残留会劫持 stdlib 模块。** 曾遗留指向 `/root/src/probe_site` 的
   `PYTHONPATH`，加上仓库里存在 `vllm_fl/platform.py`，两者合起来劫持了 stdlib 的
   `platform` 模块。测量前先清掉。
2. **`VLLM_ENABLE_V1_MULTIPROCESSING=0` 是进程内 profiling 的前提**；
   否则引擎跑在子进程，父进程 profiler 报告 0 个 kernel（早期一次跑出
   `COPY TOTAL: 0.0` 就是这原因）。
3. **Python 在非 tty 下缓冲 stdout**，日志文件为空不代表任务没在跑。
   判断进度要靠服务端自己的日志或进程状态。
4. **`pgrep`/`ps | grep` 会匹配到自己的命令行**，导致脚本自锁。
   本次踩了两次，最终把匹配串放进脚本文件规避。
5. **核验「改了代码」是否真的生效要看 inode / 内容**。
   本机 `site-packages/flag_gems` 与 `/workspace/FlagGems/src/flag_gems` 是
   **两份独立拷贝**（inode 不同），不是软链。改完必须 `diff` 确认两边一致，
   否则会对着未生效的代码调优。
6. **绝不要在 shell 脚本运行期间编辑该脚本。** bash 是**按字节偏移增量读取**
   脚本文件的，文件一变长，它会从旧偏移量读到错位的碎片并当成命令执行。
   本次后果：`line 61: ils: command not found`，而碎片里恰好含 `> "$LOG"`，
   把正在跑的这一轮的 `server.log` **截断成一行报错**，26 分钟白跑且毫无察觉。
   规避：跑之前 `cp` 到 `/tmp` 再执行（`src/launch_stage_prof.sh` 就是干这个的）。
7. **`kill` 掉 `vllm serve` 父进程不会带走 `EngineCore` 子进程。**
   子进程被 re-parent 到 init（`PPid=1`）后**继续占着约 55 GiB 显存**，
   于是下一次启动报 `Free memory on device (7.83/63.59 GiB) ... less than
   desired GPU memory utilization (0.85, 54.05 GiB)`。
   这个报错**极易被误读成「另一个 session 在占卡」而放弃重试** —— 本次就误判了一次。
判定方法：读 `/proc/<pid>/cmdline` 与 `/proc/<pid>/environ`，
`ENGINE` 进程的 `cmdline` 只有 `VLLM::EngineCore`，但环境变量会带上本次运行
特有的开关（如 `VLLM_ITER_STAGE_PROFILE=1`），据此可确认归属。
清理时务必以「专属环境变量」而非进程名作判据，避免误杀他人进程。
8. **最危险的一类坑：日志被共享，导致「已经死掉的实例被唤醒」。**
   一次评测中，第 1 个实例的 server 因显存不足启动失败，但它的**就绪轮询没有退出**，
   而轮询是 `grep "Application startup complete" server.log`，**grep 的是整个共享日志**。
   随后第 2 个实例的 server 把同一句话写进了**同一个文件** →
   第 1 个僵死实例被唤醒并启动了自己的 benchmark →
   **两个 client 打同一个 server**，`bench.log` 里出现 `Run 1/4` 与 `Run 2/4` 交错，
   两份结果全部作废。
   防范（缺一不可）：
   - **单实例锁**（`flock -n`）：拒绝与另一实例并存；
   - **就绪检测必须限定到本轮**：启动前记录日志字节偏移，只认之后写入的内容；
   - 加**存活检查**（`kill -0 $!`）：server 已死就立即失败，而不是空转满超时。
   教训推广：**任何「轮询共享文件/共享端口」来判定就绪的逻辑，都必须先证明
   「这段输出是本轮写的」**，否则它会把别人的成功当成自己的。
   实现见 `experiments/src/run_eval_whitelist.sh`。

### 5.8 两个 `vllm_fl` 拷贝：源码改了，服务却没变

本机同时存在**两份** `vllm_fl`，且由 `vllm` 可执行文件所在的 conda 环境决定加载哪一份：

| 加载方 | 安装方式 | 实际路径 | 是否含源码改动 |
|---|---|---|---|
| `/opt/conda/bin/vllm`（base env） | 静态拷贝（2026-09-08） | `/opt/conda/lib/python3.12/site-packages/vllm_fl` | **否** |
| `/opt/conda/envs/mx/bin/vllm` | editable | `/workspace/vllm-plugin-FL/vllm_fl` | 是 |

shell 里 `PATH` 把 `/opt/conda/bin` 排在前面，所以裸跑 `vllm serve` 走的是 **base 的静态旧拷贝**，
在 `/workspace` 里改 `vllm_fl/**` **不会生效**；而 `/opt/conda/envs/mx/bin/python` 因为
`.pth`/meta-path finder 指向 `/workspace`，跑的才是源码。于是出现「单测全过、服务行为照旧」的诡异现象
（本次新增 `metax.yaml` 的 `flagos_whitelist` 就踩了这个坑）。

**规避**：涉及源码改动的验证，一律用**官方对齐的 env**：
`/opt/conda/envs/mx/bin/vllm serve ...`（不是裸 `vllm serve`）。
判据：`head -1 $(which vllm)` 的 shebang 决定加载哪一份。

**但已跑的评测不受影响**：`run_eval_whitelist.sh` 第 8 行就
`export PATH=/opt/conda/envs/mx/bin:$PATH`，所以脚本里的 `vllm serve` 解析到 mx env、
加载的是**源码**——仓库里记录的官方数字（白名单 +67%、97.1%）**已包含**源码级优化
（`flash_attn` D2H、`num_splits`、`fused_add_rms_norm`、IR/GEMV），不是下界。

踩坑的是**手工排查**：裸 `vllm serve` 落进 base env，于是「config 白名单没生效」这个
假象出现了两次（`config` 改动明明在源码里）。**区别在于脚本显式改了 PATH，而手敲命令没有。**
因此结论是：*测之前先确认 `which vllm`*，而不是质疑已有评测。

### 5.9 一秒确认 FlagGems 白名单有没有生效

不用跑 benchmark、不用占 GPU：worker 在 rank 0 上会执行
`flag_gems.only_enable(include=<白名单>, record=True, path=$FLAGGEMS_ENABLE_OPLIST_PATH)`，
把**实际注册**的算子写进该文件。`only_enable` 只会注册 include 里的算子，所以：

```bash
cat "${FLAGGEMS_ENABLE_OPLIST_PATH:-/tmp/flaggems_enable_oplist.txt}" | wc -l
# 0 行  -> 白名单生效（只注册了白名单那几个，自然一行没记）
# 几十行 -> 白名单没生效（在跑 FlagGems 全套）
```

对照实测：env 白名单 `= silu_and_mul,rms_norm,rotary_embedding` → **0 行**；
config 白名单在**静态旧拷贝**上 → **36 行**（未生效）；在**源码 env** 上 → **0 行**（生效）。

### 5.10 长任务不要忙等

跑 benchmark / wait 时，应在**启动那一刻**同时做两件事：
后台化（稳定日志路径）+ 挂 `notify_on_output` 钩子，
然后用一次长 `AwaitShell` block 等待，而不是逐分钟轮询。
相关流程已固化进仓库 skill：`bench-progress-report`、
`eval-gated-optimization`、`self-directed-optimization`。

技术上的三个要点：钩子 pattern 要**只匹配结果行**；**静默 ≠ 挂死**
（冷 `torch.compile`/cudagraph 捕获安静约 20 分钟、客户端 tokenize 安静约 50 秒，
都是正常的）；判挂死前先看进程状态与日志字节增长。

---

## 6. 工具链与复现入口

### 6.1 硬件能力探测（已验证）

| 能力 | 结论 |
|---|---|
| CUDA Graph 捕获/重放 | 支持（含非默认 stream 上捕获） |
| CUDA 风格 stream / event | 支持：`current_stream`、`wait_stream`、`elapsed_time`、非阻塞 H2D/D2H |
| stream 优先级 | **不支持**（`priority_range()` 抛 `RuntimeError`） |
| stream 创建 | **很慢**（实测最慢 86 s，含重试） |
| attention 执行位置 | 在 FULL decode CUDA graph 内，非 eager |

### 6.2 关键脚本

| 脚本 | 用途 |
|---|---|
| `src/ab_attn_847.py` | 847 阻塞 D2H 的 A/B（含 token 逐位校验） |
| `src/probe_d2h_sites.py` | 定位阻塞式 D2H 站点（次数 / 字节 / 分位数） |
| `src/bench_attn_num_splits.py` | decode `num_splits` 端到端 A/B |
| `src/copy_by_mode.py` | 按配置统计 copy 预算与吞吐 |
| `src/mctracer_workload.py` + `mctrace_analyze.py` | mcTracer/Perfetto 归因 |
| `src/iter_detail_scan.py` | 扫逐迭代日志找长尾 |
| `src/iter_per_run.py` | 按轮次切分，算吞吐 / 非迭代时间 / 空档成因 |
| `src/stage_prof_scan.py` | 汇总 `VLLM_ITER_STAGE_PROFILE` 阶段埋点 |
| `src/run_stage_prof.sh` | 起带阶段埋点的 server + 跑 4k 并自动汇总 |
| `src/launch_stage_prof.sh` | 把上面的脚本复制到 `/tmp` 再跑（防编辑污染，见 §5.7） |
| `src/decode_kernel_budget.py` | 单步 kernel 预算 |
| `src/restart_server_iterlog.sh` | 带逐迭代日志重启 server |
| `src/stall_sampler.py` + `stall_correlate.py` | 1 s 采样 GPU/CPU，与引擎日志对齐判断停顿性质 |
| `src/freeze_catcher.py` | 检测进程冻结并快照各线程 `wchan` |
| `src/tail_stall_align.py` | 按轮次切分 server 日志，找深度融合停顿窗口 |
| `src/run_eval_whitelist.sh` | 官方口径全流程（性能 + 正确性），含 flock 单实例锁 |
| `src/run_m_bucket_ab.sh` + `m_bucket_measure.py` | `align32` vs `align32_geometric` 的 autotune 计数 A/B（§4.11） |
| `src/m_bucket_design.py` | 纯数学比较各分桶方案的桶数（秒级，无需 GPU） |
| `src/patch_m_bucket.py` | 给 `libtuner` 的 `M` 维换策略（幂等） |
| `src/mm_strategy_probe.py` | 打印 MetaX `mm` 族的 `LibTuner` 属性（定位 `strategy` 缺失） |
| `src/mm_tune_count.py` / `mm_tune_keys.py` | 统计 tuning 次数 / 逐次打 key |
| `src/mm_file_ab.py` | 换文件方式对 `mm.py` 做策略 A/B |
| `src/mm_host_vs_device.py` / `linear_host_vs_device.py` | 分离 host 开销与 device 时间 |
| `src/splitk_route.py` | 验证哪些 `(M,N,K)` 会走 `splitk` |
| `src/kill_tree.py` | 递归杀进程树（`vllm serve` 会忽略 TERM，见 §5.7） |
| `src/db_fam_snap.py` | 按 family 快照 autotune DB（行数而非表数） |
| `src/wl_probe.py` | 打印实际生效的算子派发清单 |
| `src/set_wl_mode.py` | 在 `metax.yaml` 里切换白名单开关 |
| `src/editable_smoke.py` | 确认 `flag_gems` 是 editable（`__file__` 指向 repo） |
| `src/probe_site/` | 临时 sitecustomize 注入（探测派发路径） |


### 6.3 关键环境变量

| 变量 | 作用 |
|---|---|
| `VLLM_FL_FLAGOS_WHITELIST` | 只启用指定 FlagGems 算子（收益最大，见 §3.1(1)） |
| `FL_METAX_ATTN_NUM_SPLITS` | 覆写 decode `num_splits`（0=启发式，N=固定） |
| `FL_METAX_ATTN_ADAPTIVE_BATCH` | 自适应规则阈值（默认 16） |
| `FLAG_GEMS_METAX_MM_GEMV` | M==1 `mm` → GEMV 路由（默认 1） |
| `VLLM_FL_IR_FUSED_ADD` | `split`（默认）/ `fused` |
| `VLLM_FL_CUDAGRAPH_ONLY` | 用吞吐换启动时间（**默认必须关**，见 §3.3） |
| `VLLM_ITER_STAGE_PROFILE` | EngineCore 阶段埋点（本机临时补丁） |
| `VLLM_GC_DEBUG` | **危险**，会打死引擎（见 §5.4），仅用于短时排查 |

### 6.4 归档范围与取舍

`experiments/src/` 是全部排查脚本，`experiments/bench_results/` 是产出日志（约 32 MB）。
刻意**不**入库的只有三类，均在 `experiments/.gitignore` 里按**显式路径**列出
（而非目录通配 —— 同一目录下还有值得保留的小 JSON）：

| 排除项 | 体量 | 原因 |
|---|---|---|
| `mctracer/**/tracer_out-*.json` × 4 | 1.2 GB | 单文件 226~457 MB 的 Perfetto trace |
| `zero_stack/trace.json` | 159 MB | 同上 |
| `flaggems_backup_*/` | 35 MB / 3484 文件 | 安装包的逐字拷贝，仓库里已有源码 |
| `*.pid` | — | 指向早已退出的进程，纯噪声 |
| `src/vllm_stage_prof.patch` + `src/_vllm_orig/` | 104 KB | **改的是上游 vLLM**，见下 |

保留的 `stage_prof*/server.log`、`server_CRASH.log`、`stall_stacks.txt` 是
§4.5~§4.8 停顿定位的**原始证据**，虽有几个 MB，但结论依赖它们，故入库。

最后一类需要单独说明：`VLLM_ITER_STAGE_PROFILE` 那套阶段埋点改的是
**上游 vLLM 的 `v1/engine/core.py`**，且只在 `site-packages` 里存在过。
按「提交物不应夹带上游 vLLM 的改动」的原则，补丁与原始备份**都不进仓库**
（规则写在根 `.gitignore`）。它们在磁盘上保留于 `/root/src/`，用于回滚本机环境。

需要注意的一个细节：埋点全部由 `_prof` 门控，**关闭时**唯一的非门控改动是
`step()` 里 `future` → `exec_future` 的**重命名**（语义等价），外加每步几次
`getattr`。所以不带 `VLLM_ITER_STAGE_PROFILE` 运行时，这份埋点对测量无影响。

---

## 7. 提交清单

### vllm-plugin-FL（`feat/ir-kernels-compile`）

相对 `upstream/main` 共 14 个提交（按时间正序）：

| commit | 内容 |
|---|---|
| `b528954` | 经 `vllm.ir` ops 在 `torch.compile` 下触达 FlagGems kernel |
| `4706cf8` | 引入 `VLLM_FL_CUDAGRAPH_ONLY`（负结果，见 §3.3） |
| `1fcf437` | decode attention num_splits knob |
| `4899f91` | decode num_splits 按捕获 batch 自适应 |
| `2fc71f9` | 去掉 prefill split attention 的阻塞 D2H |
| `c124d25` | skill: bench-progress-report |
| `dc27f60` | fused_add_rms_norm 不再 clone activation |
| `bb2c763` | 刷新 CUDAGraph-only 数据并记录 IR_FUSED_ADD |
| `e0f2d62` | 打印每个捕获 batch 的 decode num_splits |
| `875ba9a` | 说明 CUDAGraph-only 的取舍与 vendor 默认 |
| `854e466` | 长任务改为启动时挂钩子 |
| `639f232` | 封存 experiments（脚本与日志） |
| `1e16267` | 修 cudagraph-only 单测的 ruff format |
| `b52d1dd` | experiments 树排除 ruff/typos |

其中 `b528954`（`vllm.ir` 桥）是 §3.1(1) 与 §3.1(4) 的前提：
它让 `torch.compile` 下也能走到 FlagGems 的融合算子，而不是只走 inductor 生成物。

### FlagGems（`flagos-2026-s2/metax-gemv`）

| commit | 内容 |
|---|---|
| `a7a74aeed` | M=1 linear 的 GEMV 快速路径 |
| `a68024d21` | M==1 `mm` 路由到 GEMV |
| `4de92b5c8` | `align32_geometric`：给 `M` 的 autotune key 加上界（§4.11） |

---

## 8. 合规性分析：算子清单算不算「切换算子」

赛事规则里有一条（据参赛者转述，`pingshen.md` 本文档只到 4.3.3，未见原文）：

> 禁止没有算子优化的情况下切换算子，删掉框架主要算子选择逻辑等方式来进行模型性能优化

`pingshen.md` 里**能核实**的相关条款只有三条：

| 行 | 条款 |
|---|---|
| 122 | 必须使用 `vllm-plugin-FL` 和 `FlagGems` 作为开发工具；**经组委会代码合规性审核后**方可获得排名 |
| 124 | 组委会将对提交方案**复现验证**；无法复现视为成绩无效 |
| 126 | 未及时提交 **PR** 视为放弃获奖资格 |

因此逐字判定权在组委会。下面把我们的改动分类，并给出**实测**支撑。

### 8.1 两类改动的性质完全不同

| 改动 | 性质 | 是否涉及选择算子 |
|---|---|---|
| `_metax/ops/mm.py` 的 `strategy=["align32", ...]` | **纯算子优化** —— 修 FlagGems 算子自身的 autotune key 策略 | **否** |
| `dispatch/config/metax.yaml` 的 `flagos_whitelist` | **算子派发选择**（使用框架自带的配置键） | 是 |

### 8.2 关键实测：默认态在这台硬件上不可用

把 `flagos_whitelist` 置空（等价于「不做任何算子选择」，完全走框架默认
`prefer: flagos`），16 个算子全部落到 FlagGems：

| | 零算子选择（`whitelist: []`） | 出厂清单（3 个融合算子） |
|---|---|---|
| **启动到就绪** | **>8 分 47 秒仍未就绪** | **~2 分钟** |
| CUDA 图捕获单图耗时 | 3 s → **14 s**（持续恶化） | 正常 |
| `mm_kernel_nt` 行数 | 5170 → **5770**（+600） | 5170 → 5170（**0**） |
| `mm_kernel_splitk` | 288 行/5 表 → **360 行/7 表** | 288 → 288（**0**） |
| `mul`（新出现） | 无 → **108 行/6 表** | 不走 FlagGems |
| 4k 吞吐 | **未能产出** | **8836.39 tok/s** |

即：**框架默认配置在该平台上跑不到出结果**。给平台一个可用的默认配置，
属于必要的平台适配，而不是为了刷分去规避算子选择逻辑。

### 8.3 三条抗辩理由

1. **规则禁止的是「删掉框架主要算子选择逻辑」，我们是使用该逻辑。**
   `flagos_whitelist` 是新增的**配置键**，走的是框架既有的配置解析与派发路径
   （`vllm_fl/utils.py` 的优先级链），没有绕过或删除任何选择机制。
   若组委会不认这个键，只需把它清空即可退回默认态 —— 说明我们**没有移除**默认态。

2. **`align32` 修复不涉及任何选择**，是修 FlagGems 自身缺陷：
   `runtime/common.py` 的 `DEFAULT_STRATEGIES` 已声明 `mm`/`mm_nt`/`mm_splitk`
   应为 `align32`，但该表只在 `TuningMode.EXPANDED` 下被消费，而 MetaX 的
   decorator 既未传 `strategy=` 又以 `TuningMode.DEFAULT` 运行，于是退化成恒等策略、
   按原始 `M` 建 key → 每步 autotune。属「算子编译优化」（第 116 行明确列为
   创新性评判维度）。详见 §4.11：修好后 `M` 的 key 空间从 69 个桶降到 13 个，
   A/B 实测 tuning 次数 65→9、耗时 348.63 s→55.04 s。

3. **方向上是加分，不是减分。** 修复后能安全交给 FlagGems 的算子**变多了**：
   §10.6（见 `mm_metax_rootcause.md`）已验证白名单加入 `linear`、`mm` 后
   4k 吞吐 8849.86 vs 8836.39、三项指标持平或略优、autotune DB 零增长。
   这比「只让 3 个算子走 FlagGems」更符合第 122 行「必须使用 FlagGems 作为开发工具」。

### 8.4 复现要求下的真实风险

第 124 行的「复现验证」是**比合规更硬的风险**，我们踩到过一次：

`flag_gems` 在镜像里是**静态 pip 安装**（`flag_gems-5.3.5.dist-info`，无 `.pth`），
而 `vllm_fl` 是 editable。于是**改 `/workspace/FlagGems` 对运行时零影响**，
只有手改 `site-packages` 才生效。若只在那儿验证过，等于提交了一份
**从未在交付路径上跑通**的方案。

已处置：把 `flag_gems` 也改成 editable，使 repo 成为唯一真相源；并校验
`flag_gems.__file__` 指向 `/workspace/FlagGems/src/flag_gems/__init__.py`。
注意 `pip uninstall` 会**跳过内容被改过的文件**，残留文件会让包变成
namespace package（`__file__ is None`）并遮蔽 repo 版本，必须手工移走。

**结论**：`align32` 修复无合规问题；`flagos_whitelist` 是配置级的平台适配，
有三条实测依据支撑，且可一键退回默认态。最终解释权在组委会审核。

---

## 9. 未收割项（按价值排序）

1. **把 FlagOS 融合白名单带进官方启动命令** —— 现有最大的一笔收益，
且是纯配置，零代码风险（§3.1(1)）。**现在还要加上：它同时消除 §4.8 的
20 s autotune 停顿（§4.10 已 A/B 验证：零停顿 + autotune DB 零增长 + 首轮 +68%）。**
2. ~~**消除 FlagGems 运行时 autotune 停顿**~~ —— **已修复（§4.11）**。
根因是 `M` 维的 autotune key 无上界（`align32` 对 1..2048 产生 69 个 key）。
新增 `align32_geometric`（≤128 保留 32 粒度、之上几何分桶）后降到 13 个桶，
A/B 实测 tuning 次数 65→9、耗时 348.63 s→55.04 s。**注意：修好后白名单
不再需要靠「排除 `mm`」来规避停顿**，这反而使「用白名单」更没有「规避算子选择」
的嫌疑 —— 因为现在把 `mm` 放进白名单也是安全的（§4.11 末、§10.6）。
3. **`align32_geometric` 的显式上界** —— 当前 13 个桶在
`max_num_batched_tokens=2048` 下够用，但几何桶仍随 `M` 增长。若换到
`max_num_batched_tokens` 远大于 2048 的机器，需要再加一层「超过阈值并入最大桶」。
4. **M==1 GEMV 的端到端验证** —— 微基准 1.12x，预期 TPOT ≈−6.5%（§3.2(5)）。
5. **4k 重测** —— 现有 4k 数据因停顿不可信，需独占 GPU、提高轮数（§4.1）。
6. 拆解 40% 非迭代时间的可压缩比例（§5.2），**先量再动**。
