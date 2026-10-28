# MiniCPM5-2B on MetaX C500：优化收益与排查经验

本文件汇总本次竞赛优化中**已量化的收益**与**排查过程得到的经验**。所有数字都注明
出处，原始日志与脚本见 `experiments/`（`src/` 与 `bench_results/`）。

环境：MetaX C500 单卡 · bf16 · `vllm-plugin-FL` + FlagGems 5.3.5 · vLLM 0.24.0
模型：MiniCPM5-2B（`LlamaForCausalLM`，42 层，hidden=2048，16 q-heads / 2 kv-heads，head_dim 128）

> **本文件只讲 MetaX（沐曦）C500。** 天数 BI-V150 是另一张卡，两套基线相差
> 2.5×（4k）～7.7×（16k），本文件的收益数字与根因**不可直接移植**到天数。
> 天数的环境与已验证状态见 [`FINDINGS-iluvatar.md`](./FINDINGS-iluvatar.md)。

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
| 6 | P0：MBT 4096 + MLP silu fusion | 短测 4k **+29.2%** / 16k **+16.4%**；L3 **98.1%** | §12 |
| 7 | P1：cudagraph trim + 恢复 FlagOS 白名单默认 | 短测 4k **+58%** / 16k **+31%** vs 当日 bottleneck | §13 |
| 8 | P2：默认 MBT 8192（nopc 公平口径） | vs 4096：4k **+4.3%** / 16k **+2.9%** | §14 |
| 9 | P2 官方口径 4k/16k（RUNS=2，nopc） | Total **11957** / **11651**；vs 基线 **+135%** / **+66%** | §15（已被 §23 取代） |
| 10 | 当前栈官方口径 4k/16k | Total **12139** / **11852**（16k 两轮，抖动 0.08%） | §23 |

最可靠的端到端结果（历史）：**16k 场景 +5.50%（轮间抖动 0.0%）**，与预测的 +6.83% 同向且量级吻合。
**当前栈的 16k 参考值是 §23.2 的 11851.65**（4k 锚点 12138.58）；§15 因 silu
fusion 开着等问题不可作为当前基线，见 §23.3。

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

### 5.11 拿到引擎内部的原生栈（父子进程 + 容器限制）

现象：`vllm serve` 卡在启动，父进程的栈只显示 `wait_for_engine_startup`，
真正的卡点在 `EngineCore` 子进程里，看不到。而且容器里 `py-spy dump --pid` 直接
`Permission denied`。四个有效手法：

1. **把引擎拉进同进程**：`VLLM_ENABLE_V1_MULTIPROCESSING=0` 起
   `LLM(...)`，再用 `faulthandler.dump_traceback_later(60, exit=False)`
   定时 dump，一次就能命中到具体行（本次直接指出 `streams.py:37`）。
2. **`py-spy` 在容器里的正确用法**：宿主 `kernel.yama.ptrace_scope=1` 且容器
   无 `CAP_SYS_PTRACE` 时，attach 别人的进程必失败（`os error 13`）。
   改成**让 py-spy 自己启动目标**即可：
   `py-spy record --format raw --native -d 150 --rate 25 -o out.raw -- python target.py`。
   `--native` 是关键，否则只有 Python 帧，看不到 `mxr::HostQueue` 这类 C++ 卡点。
3. **`--nonblocking` 与 `--native` 互斥**，同时给会直接报
   `Can't get native stack traces with the --nonblocking option`。
4. **别用「最高频栈」判断主次**：本次 634 个样本里，572 个落在
   `torch.cuda.init()` 的 ZSTD 解压（21 s 的连续区间，**会结束**），
   真正致命的 `Stream()` 死锁只有 5 个样本。要按**是否返回**而不是**样本多少**
   分主次，否则会把「慢」误判成唯一问题。

→ 完整案例见 `maca_hostqueue_deadlock.md`。

---

## 6. 工具链与复现入口

### 6.1 硬件能力探测（已验证）

| 能力 | 结论 |
|---|---|
| CUDA Graph 捕获/重放 | 支持（含非默认 stream 上捕获） |
| CUDA 风格 stream / event | 支持：`current_stream`、`wait_stream`、`elapsed_time`、非阻塞 H2D/D2H |
| stream 优先级 | **不支持**（`priority_range()` 抛 `RuntimeError`） |
| stream 创建 | **很慢**（实测最慢 86 s，含重试）；2026-09-25 进一步恶化为**永久死锁**，见 `maca_hostqueue_deadlock.md` |
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
| `src/maca_queue_hang_repro.py` | 复现 MACA 命令队列死锁（`Stream()` 挂死），含底层 `mcStreamCreate*` 直调对照 |
| `src/gpu_health_check.py` + `src/run_after_recovery.sh` | 宿主 reset 后自检 `Stream()`，健康才起 profiler（避免再一次静默挂 20 分钟） |
| `maca_hostqueue_deadlock.md` + `stacks_maca_hostqueue.txt` | 2026-09-25 GPU 上下文泄漏死锁的定位过程与 `py-spy --native` 原生栈证据 |

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
| `d994520` | top-p 改走 `flag_gems.top_p_threshold`（§10） |
| `7d5c74b` | 覆盖统计只由真正采样过的进程上报 |
| `170657b` | gate_up_proj + SiluAndMul 融合 patch（默认关，§10.4） |
| `c7cd0fc` | 默认不再设置 flagos 算子白名单（§10.4） |
| `c25293c` | 归档 stage-profiling 工具与三份 4k 评测（§10.4） |

其中 `b528954`（`vllm.ir` 桥）是 §3.1(1) 与 §3.1(4) 的前提：
它让 `torch.compile` 下也能走到 FlagGems 的融合算子，而不是只走 inductor 生成物。

### FlagGems（`flagos-2026-s2/metax-gemv`）

| commit | 内容 |
|---|---|
| `a7a74aeed` | M=1 linear 的 GEMV 快速路径 |
| `a68024d21` | M==1 `mm` 路由到 GEMV |
| `4de92b5c8` | `align32_geometric`：给 `M` 的 autotune key 加上界（§4.11） |
| `7c7a983dc` | sort-free top-p：分桶阈值取代排序（§10） |
| `95d82e6f5` | top-p v2：去掉 int64 索引缓冲等四处源头修复，2.60x（§10） |
| `3d05dfc97` | MetaX mm 按阶段调 tile、M>1 走 tuned mm（§10.4） |
| `5f0bb89aa` | 融合 gate_up_proj + silu_and_mul 算子（§10.4） |

注：本次落到 `flagos-2026-s2/metax-mm-strategy`（上表早期提交来自
`flagos-2026-s2/metax-gemv`）。

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
4. **M==1 GEMV 的端到端验证** —— ~~微基准 1.12x，预期 TPOT ≈−6.5%（§3.2(5)）~~
   **已定性（2026-09-26）**：微探针确认 M==1 时五个投影形状（qkv/o_proj/
   gate_up/down/lm_head）**全部命中 `_gemv_kernel`**，命中条件逐项通过，路由
   无 bug。同时确认 **conc-64 的 decode 跑的是 M≈64**，按设计走
   `mm_kernel_nt` —— 所以那个「预期 −6.5%」在吞吐场景**不存在**，只对
   batch-1 长 CoT（即 MATH-500 评测那种）成立。别再按吞吐场景去追它。
5. ~~**4k 重测** —— 现有 4k 数据因停顿不可信，需独占 GPU、提高轮数（§4.1）。~~
   **已完成（§10.6）**：独占 GPU、既定 harness、两轮一致，Mean TPOT 29.5 ms /
   ~9985 tok/s（对比上版基线 30.80 / ~9601，−4.2% / +4.0%）。注意这次 delta
   是 top-p v2 与 mm 调优**叠加**，不可整体归给采样器。
6. ~~拆解 40% 非迭代时间的可压缩比例（§5.2），**先量再动**。~~
   **已量清（§16，2026-10-23）**：P2 上非迭代 ~**18%**（不再是 40%）；
   busy 内 wait(GPU) 82% / host 18%。可叠缝主要是 exec launch ~10%，
   已用 MACA light-launch 收 +1.16%（§16.2）；PDL 不可用。

## 10. 采样器：sort-free top-p v2（已落地）

### 10.1 结论

采样器从 decode 设备时间的 **41% 降到约 1.7%**。这是目前算子级改动里
最彻底的一笔，且**不依赖任何配置开关**。

### 10.2 背景

vLLM 的 top-p 在 MetaX 上被强制走 `apply_top_k_top_p_pytorch`（Triton 的
`topk_topp` 编不过来），它每步对完整词表排序。旧 decode profile 里
`aten::sort` 占 **54.69%**（4.85s / 11.86s），是 decode 的第一名。

`7c7a983dc` 的 sort-free `top_p_threshold`（分桶求阈值 + 两个线性 kernel）
先替掉排序：TPOT 从 31.5 ms 降到 11.6 ms（单流），采样器 rows=48 时 25x。
但 v1 的两个 kernel 是照「批量调用」的形状写的，四个阶段都没跑到带宽。

### 10.3 v2：四处源头修复（`95d82e6f5`）

rows=64、vocab 130560 的分阶段实测：

| 阶段 | 前 | 原因 | 改法 |
|---|---|---|---|
| mass | 275 us | 行读两遍；写 `e` 和**每元素 int64 桶索引** | 桶可由 `e` 反推（`bucket = (log2(e) - lo)/step`），索引缓冲整个删掉 |
| scatter_add | 308 us | 把 `e` 和 int64 索引读回来 | 并入 mass kernel 的原子加 |
| threshold | 183 us | 对 8192 个 bin 做 `flip(cumsum(flip))`，**搬 0 字节** | 改前向 exclusive cumsum + 计数（数学等价） |
| masked_fill | 282 us | torch 先物化 bool 掩码（10 B/元素） | 比较+选择融合进一个 kernel |
| **合计** | **1009 us** | | **388 us（2.60x）** |

最大的一笔是那个 int64 索引缓冲：每元素 8 B 写 + 8 B 读，纯冗余。
另外网格从「一行一个 program」改成 `(rows, col_blocks)`：v1 在 rows=1 时
把整行塞进一个 program，mass kernel 只有 **11 GB/s**（v2 后 142 us 全流程）。

**正确性**：与 v1 在**所有非平凡 p、4 种 logit 尺度**下保留集合逐行精确一致；
与 sort 参考 IoU ≥ 0.9979、保留质量 ≥ 0.9500、count slack ≤ 0.21%（与 v1 同）。
仅在 `p == 1.0` 时 v2 保留**更多**（不再比较两个 fp32 累加和是否相等）；
`lo = -40` 的十年界在两条路径上都会排除极深尾部（实测 rows=64 时 24/8.3M）。

**精度门禁**：MATH-500 Level 3（105 题）**97.1%**，与 sort 基线持平；
覆盖 `fast=65536 fallback=0`、`iou=1.00000`。v1 那次 98.1% 与本次差一题，
而两者数值已证等价 —— 是采样噪声，不是回归。

### 10.4 L2 约束（重要，决定要不要留回退）

v2 靠 `(rows, n_bins)` 的原子直方图驻留 L2。rows=64 时 2.1 MB（驻留），
rows=256 时 8.4 MB（溢出），逐元素原子加就输给原来的 `scatter_add_`。
实测交叉点：rows=128 时 2.30x、192 时 1.19x、**256 时 0.89x（变慢）**。
因此加了 `_V2_MAX_ROWS = 128`，超过就走原 kernel —— **任何行数都不会比之前慢**。
这条约束以后改 rows 上限或加并发时必须重新确认。

### 10.5 负结果：mass kernel 已到原子吞吐极限（不要再试）

v2 之后 mass kernel 是采样器里最大的一块（真实 profile 268 us/call）。它只读
33 MB（约 12 us 带宽），所以开销全在逐元素 `atomic_add`。第一嫌疑是 bin 0：
桶被 clamp 到 `[0, n_bins-1]`，所有低于 `lo` 的 token 都落进 bin 0，而长尾
lm_head 行里那应该是绝大多数。**实测：0.0% 的 token 落在 `lo = -40` 以下**
（高斯行与尖峰行都一样），把这些原子加 mask 掉只有 **1.01x，毫无收益**。

真相是它本就跑在原子吞吐上：8.36M 原子 / 289 us = 28.9 G 原子/s，
约 **6 个 L2 周期/原子**，已是硬件地板。唯一杠杆是减少原子数，而算法需要
逐元素一个原子 —— 除非整个分桶方案换掉（Triton 的 `tl.histogram` 只计数、
不累加权重，所以质量仍得另算）。

### 10.6 4k 重测（收掉 §9 的第 5 项）

用既定 harness（`benchmark_throughput_serve.py`，`[[4096,1024,64,256]]`，
RUNS=4 / SKIP_FIRST=1）、无白名单、独占 GPU：

| run | Mean TPOT | Total tok/s | Output tok/s | Median TTFT |
|---|---|---|---|---|
| run 1（预热，丢弃） | 29.43 | 9859.79 | 1971.96 | 682.58 |
| measured 2 | 29.67 | 9937.48 | 1987.50 | 684.74 |
| measured 3 | 29.35 | 10032.99 | 2006.60 | 697.06 |

两轮实测 TPOT 相差 1.1%、吞吐相差 1.0%，满足一致性。覆盖
`fast=4096 fallback=0`，确认是 sort-free 路径。对比上一版基线
（30.80 ms / ~9601 tok/s）：**TPOT −4.2%，吞吐 +4.0%**。

**但这次 delta 是两个改动叠加**：top-p v2 + 同时装在包里的 MetaX mm 调优
（`3d05dfc97`）。按算子级核算，v2 在 conc 64 下值约 0.6 ms/step，其余来自
mm。**不要把它整体归给采样器。** mm 调优自身的正确性本次未独立验证，
只验证了它对 benchmark 非回归。

### 10.7 同批落地但**默认关闭**的东西

- `170657b` + `5f0bb89aa`：gate_up_proj 与 SiluAndMul 融合
  （`VLLM_METAX_MLP_SILU_FUSION=1` 才开）。它省掉 (M, 2N) 中间张量的来回，
  decode 时占该 pair 的 21%；但两个 B tile 常驻会把每级共享内存翻倍，
  C500 的 64 KB 上限正好排除 prefill 想用的大 tile，所以算子自带按 M 的
  交叉点、之上回退两趟写法。**默认关，不影响上面任何数字。**
- `c7cd0fc`：默认不再设 flagos 白名单。因为 `align32_geometric`（§4.11）
  已经不靠「排除 `mm`」来规避 autotune 停顿，白名单的限制作用消失；
  同时让出厂默认与「无白名单」的实测配置一致。

---

## 11. MetaX：bench 客户端 `VLLM_PLUGINS=fl` 会 SIGKILL 已起的 serve

日期：2026-10-22。机器：MetaX C500，主机内存 64 GiB，
`flag_gems` `flagos-2026-s2/metax-mm-strategy` + `vllm-plugin-fl`
`feat/ir-kernels-compile`。

### 11.1 现象

官方起服：

```bash
export VLLM_PLUGINS=fl
vllm serve /workspace/MiniCPM5-2B \
  --port 9031 --served-model-name minicpm \
  --gpu-memory-utilization 0.85 --max-model-len 131072
```

`Application startup complete` 之后约 20s（或紧接着启动
`vllm bench serve` / 任何会再 import `fl` 的客户端）时，APIServer 消失。
日志几乎无 traceback，末尾常见
`multiprocessing.resource_tracker: ... leaked semaphore ... at shutdown`。
EngineCore 常变成孤儿（RSS 可达 ~23 GiB），GPU 进程列表清空或只剩孤儿。

supervisor / `waitpid` 记到的退出码：**`rc=-9`（SIGKILL）**。
不是 Python 异常，也不是 uvicorn 优雅 SIGTERM。

### 11.2 根因（对照实验）

| 步骤 | 结果 |
|---|---|
| serve READY 后空等 90s | **存活**（health OK） |
| 另起 `VLLM_PLUGINS=fl vllm bench serve ...`（哪怕 1 条短请求） | serve **立刻 SIGKILL** |
| 另起 `VLLM_PLUGINS= vllm bench serve ...`（客户端走 CPU platform） | **存活**，请求成功 |

结论：问题不在官方 `export VLLM_PLUGINS=fl` 的 **serve** 命令。
**第二个进程**再加载 `fl` / 初始化 MetaX CUDA 栈时，会把已占用 GPU 的
APIServer 直接 SIGKILL。同一 shell 里 `export VLLM_PLUGINS=fl` 后跑
`benchmark_throughput_serve.py`（子进程继承环境）必中招。

同类雷区：serve 已起来时，在另一进程里做 `torch.cuda` 健康检查 /
import 会触发 MetaX 初始化的代码，也会误杀。

### 11.3 修复

1. **serve**：继续官方方式，`VLLM_PLUGINS=fl`（不要改）。
2. **bench 客户端**：强制空插件。已改
   `benchmarks/benchmark_throughput_serve.py`：子进程 env
   `VLLM_PLUGINS=""`。
3. harness：`experiments/bench_results/eval_16k_bucket_nowl/launch_serve.py`
   （double-fork + 记录 signal）与 `run_16k_daemon.sh`
   （只在起服前做 CUDA health；bench 用空 `VLLM_PLUGINS`）。

官方流程建议：serve 与 bench **分终端**；bench 终端不要 export `fl`，
或先 `unset VLLM_PLUGINS`。用本仓库脚本则已自动隔离。

**官方吞吐脚本路径注意（2026-10-23）**：组委会常用

```bash
python3 /workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py \
  --model ... --served-model-name minicpm --port 9031 \
  --test-cases '[[4096,1024,64,256],[16384,1024,64,128]]'
```

该路径下的脚本**没有**强制清空 `VLLM_PLUGINS`（仍是上游原版）。
本仓库 `vllm-plugin-FL-comp/benchmarks/benchmark_throughput_serve.py`
在 `bb9ad49` 已对子进程 `env["VLLM_PLUGINS"]=""`。
若评测机跑的是 **无补丁的** `vllm-plugin-FL` 拷贝，同一 shell 里
`export VLLM_PLUGINS=fl` 后再跑官方脚本 → **仍会 SIGKILL serve**。
要么分终端且 bench 侧不 export `fl`，要么把 FL-comp 那几行同步过去。

### 11.4 修复后 16k 吞吐基线（无白名单）

配置：`[[16384,1024,64,128]]`，`RUNS=4` / `SKIP_FIRST=1`，Failed=0。
CSV：`/workspace/benchmark_results/summary_20261022_203415.csv`。

| run | Total tok/s | Output tok/s | 备注 |
|---|---|---|---|
| 1 | 7703.58 | 453.15 | 预热，丢弃 |
| 2 | 9886.91 | 581.58 | |
| 3 | 9894.03 | 582.00 | |
| 4 | 9894.40 | 582.02 | |
| **Summary（2–4 均）** | **9891.78** | **581.87** | Mean TTFT 21003.51 ms |

对比本文件 §2 旧基线 16k **7029.68**：同口径提升显著；本轮软件栈为
comp 分支 + `align32_geometric`，与早期基线不完全同版本，引用时注明日期。

---

## 12. P0 优化版（MBT 4096 + MLP silu fusion，2026-10-22）

在 §11 的 16k 基线之上开一版可落地的默认档，并用短测 + Level 3 正确性验收。

### 12.1 改动

| 项 | 开关 / 代码 | 说明 |
|---|---|---|
| OpenAI-API 默认 `max_num_batched_tokens` 2048→4096 | `vendor/metax/patches/mbt_defaults.py`（`VLLM_FL_METAX_MAX_NUM_BATCHED_TOKENS`，默认 4096；`0` 关闭） | C500 报 ~63.6 GiB，掉进 vLLM &lt;70 GiB 小 batch 档；4k 预填从 2 chunk 变 1 chunk |
| `gate_up_proj + SiluAndMul` 融合 | `VLLM_METAX_MLP_SILU_FUSION=1`（仍默认关） | 修了三处才能在 V1+torch.compile 下真正挂上：钩 `ModelRunnerFL.load_model`（原先找错类名）；用 `torch.library.custom_op` 包 FlagGems（logger.debug 不能进 dynamo）；去掉 forward 里对 `_STATS` 的写入（cudagraph 视作非法 buffer mutation） |
| M==1 GEMV | `FLAG_GEMS_METAX_MM_GEMV=1`（FlagGems 默认已开） | 无新增代码 |

Opt 起服脚本：`experiments/bench_results/opt_p0_20261022/launch_serve.py`
（显式 `--max-num-batched-tokens 4096` + 上述 env；`VLLM_PLUGINS=fl`）。

### 12.2 短测 A/B（同口径，非官方 RUNS=4）

对照：同日 bottleneck 基线 serve（MBT=2048、silu 未挂上）。
Client：`VLLM_PLUGINS=`。

| 场景 | baseline Total tok/s | MBT=4096 | MBT+silu | vs baseline |
|---|---|---|---|---|
| 4k in / 256 out，c64，n=64 | 12900.63 | 16072.30 | **16666.71** | **+29.2%** |
| 16k in / 64 out，c32，n=32 | 16284.97 | 18493.23 | **18950.13** | **+16.4%** |

主收益来自 MBT；silu 在 MBT 之上再约 **+3.7% / +2.5%**。日志确认
`MetaX MLP fusion enabled: fused 42 gated MLP module(s)`。

### 12.3 正确性：math_500 Level 3，105 题

配置与官方一致：`eval-batch-size 8`，`temperature=1.0`，`top_p=0.95`，
`max_tokens=32768`，P0 serve（MBT 4096 + silu fusion）。

| 指标 | 本轮（P0） | 门槛 | 判定 |
|---|---|---|---|
| Accuracy | **98.1%**（103/105） | ≥0.95 | ✅ PASS |

报告：
`experiments/bench_results/opt_p0_20261022/evalscope_level3/20261022_221204/reports/minicpm/math_500.json`。

### 12.4 新雷区：evalscope 客户端也会 SIGKILL serve

§11 已写 bench/`torch.cuda` 双初始化。本轮复现：`evalscope` 加载
`datasets` / `modelscope` 时会 `import torch` 且 `cuda.is_available()==True`，
在 serve 已占用 GPU 时同样把 EngineCore 打成「exited unexpectedly」、无 Python
traceback。

**客户端必须**：`VLLM_PLUGINS=` **且** `CUDA_VISIBLE_DEVICES=`（空），再跑
evalscope。本轮按此设置后 105 题跑完、serve 仍 health OK。

---

## 13. P1：cudagraph 裁剪 + 恢复 FlagOS 白名单（2026-10-22）

在 §12 P0 之上继续抠。

### 13.1 cudagraph capture sizes → ≤64 + spare 72

与天数侧同思路（官方 `--max-concurrency 64`）。默认 capture 列表到 **512**
（50+ 档），80..512 从不跑却占图内存估算。

| 项 | 值 |
|---|---|
| 代码 | `vendor/metax/patches/cudagraph_sizes.py` |
| 开关 | `VLLM_FL_METAX_CUDAGRAPH_SPARE`（默认 72；`0` 关闭） |
| 实测 | `…512] -> [1,2,4,8,16,24,32,40,48,56,64,72]`；启动更快 |
| KV 池 | **未变**：仍 1,206,272 tokens / 9.20×@131k（本机估算路径不把多余图算进 KV 扣减） |
| 短测 vs P0 | 4k **+1.1%**、16k **≈0%**（抖动内） |

结论：保留为干净默认（少捕图、省启动），**不是**本轮吞吐大头。

### 13.2 `align32_geometric` 硬顶 4096

FlagGems `_metax/tuner_strategies.py`：`FLAG_GEMS_METAX_ALIGN32_GEOMETRIC_CAP`
（默认 4096）。MBT 已到 4096，防止以后再抬 MBT 时桶数随 `2^ceil(log2 M)` 漂。

### 13.3 silu fusion 默认改为开

`VLLM_METAX_MLP_SILU_FUSION` 默认 `"1"`（`=0` 可关）。§12 已过 L3 98.1%。

> **已回滚（2026-10-23）**：当时 wrap 其实没挂上（§17.1），这个「默认开」从未真正生效，
> 也没被 e2e 验证过。公平 A/B 显示 conc=64 下 **−4.83%**（§17.2A），
> 默认已改回 `"0"`。

### 13.4 恢复 MetaX 默认 FlagOS 白名单（大头）

`metax.yaml` `flagos_whitelist` 从 `[]` 改回
`silu_and_mul,rms_norm,rotary_embedding`。

§10.7 曾因 `align32_geometric` 去掉默认白名单（不再需要靠排除 `mm` 躲
autotune）。本轮在 **P0/P1 齐全** 的栈上重测，白名单**仍然**明显更快：

| 场景 | P0 (MBT+silu) | P1 (trim) | **P1+白名单** | vs bottleneck 基线 |
|---|---|---|---|---|
| 4k/256 c64 | 16667 | 16855 | **20391** | **+58.1%** |
| 16k/64 c32 | 18950 | 18938 | **21276** | **+30.7%** |

相对无白名单 P1：4k **+21.0%**、16k **+12.3%**（TTFT/TPOT 同步下降）。
原因仍是：这三只 FlagGems 融合算子在 C500 上快于 native；`mm` 继续走
native + geometric，不进白名单。

可用 `VLLM_FL_FLAGOS_WHITELIST=` / `BLACKLIST` 覆盖。

### 13.5 当前推荐起服

```bash
export VLLM_PLUGINS=fl
# 白名单已在 metax.yaml；silu fusion / MBT / cudagraph spare 均为插件默认
vllm serve /workspace/MiniCPM5-2B --port 9031 --served-model-name minicpm \
  --gpu-memory-utilization 0.85 --max-model-len 131072
# （插件会把 OpenAI 默认 max_num_batched_tokens 提到 4096；也可显式传入）
```

Harness：`experiments/bench_results/opt_p1_wl_20261022/launch_serve.py`。

---

## 14. P2：默认 MBT 4096→8192（关 prefix cache 公平口径，2026-10-22）

### 14.1 关于 prefix cache

官方吞吐脚本是 **random 合成 prompt**，注释推荐起服带
`--no-enable-prefix-caching`。同 seed 反复打同一批 random 会把 TTFT/吞吐
虚高（曾测到 TTFT 666 ms / 49k tok/s）——那是测法污染，不是评测收益。
math_500 题干也不共享长前缀。后续 A/B **以 nopc 为准**。

### 14.2 MBT 8192 vs 4096（白名单+silu+cudagraph trim，seed=7，nopc）

| 场景 | MBT 4096 | MBT 8192 | Δ |
|---|---|---|---|
| 4k/256 c64 | 20400 | **21275** | **+4.3%** |
| 16k/64 c32 | 18446 | **18982** | **+2.9%** |

4k：MBT≥4096 时单条已是一刀；8192 主要让并发 prefill 一次打包更多序列。
16k：chunk 4→2。KV 池略降（约 1.192M vs 1.206M tokens），仍够官方 c64。

### 14.3 落地

- `mbt_defaults.py`：默认 `VLLM_FL_METAX_MAX_NUM_BATCHED_TOKENS=8192`
- FlagGems `ALIGN32_GEOMETRIC_CAP` 默认同步 **8192**

---

## 15. P2 官方口径复测（4k / 16k，各 RUNS=2，2026-10-22）

栈：默认白名单 + silu fusion + cudagraph ≤64+72 + **MBT 8192**，
起服 `--no-enable-prefix-caching`。客户端：`VLLM_PLUGINS=` 且
`CUDA_VISIBLE_DEVICES=`（空），规避 §11 / §12.4 双初始化 SIGKILL。
Harness：`experiments/bench_results/official_p2_20261022/`
（`BENCH_RUNS=2`，`SKIP_FIRST=0`，两轮均计入 Summary）。

### 15.1 结果

| Prefill | Decode | Conc | N | Run | Total tok/s | Output tok/s | Mean TTFT (ms) |
|---|---|---|---|---|---|---|---|
| 4096 | 1024 | 64 | 256 | 1 | 11959.01 | 2391.80 | 1956.09 |
| 4096 | 1024 | 64 | 256 | 2 | 11954.77 | 2390.95 | 1945.78 |
| 4096 | 1024 | 64 | 256 | **avg** | **11956.89** | ~2391 | **1950.93** |
| 16384 | 1024 | 64 | 128 | 1 | 11654.19 | 685.54 | 14913.55 |
| 16384 | 1024 | 64 | 128 | 2 | 11647.41 | 685.14 | 14895.40 |
| 16384 | 1024 | 64 | 128 | **avg** | **11650.80** | ~685 | **14904.47** |

两轮 Failed=0，轮间抖动 <0.1%。

### 15.2 对比

| 对比物 | 4k Total | 16k Total |
|---|---|---|
| §2 沐曦基线图 | 5089.65 | 7029.68 |
| 本轮 P2 | **11956.89** | **11650.80** |
| vs 基线 | **+134.9%** | **+65.7%** |
| §11 旧栈 16k Summary（无白名单） | — | 9891.78 |
| vs §11 | — | **+17.8%** |

16k Output tok/s（~685）远低于 4k（~2391）：同一 Decode=1024，但 prefill/KV
占 wall 时间更大，Total 分子里 input 占比更高，**Output 被稀释**，不是 decode
核突然变慢。Mean TTFT 4k ~2 s / 16k ~15 s，与 chunked prefill（MBT 8192）一致。

### 15.3 口径备注

官方脚本默认常为 `RUNS=4` 且跳过首轮；本轮按需求各跑 **2 轮且不跳首轮**。
若复现官方提交脚本，应以该脚本的 Summary 规则为准；本表数字可与短测 / §11
对照，勿与「跳过 Run1 的 3 轮均值」直接混比。

---

## 16. Host 空档量清 + MACA launch / decode64 热点（2026-10-23）

环境：P2（GEMV + silu fusion + MBT8192 + cg spare72 + FlagOS whitelist）+
`VLLM_METAX_LM_HEAD_GEMM=0`。产物目录：
`experiments/bench_results/hostgap_{measure,pdl,opt}_20261023/`、
`decode64_mcprof_20261023/`、`gemm_mcprof_20261023/`。

### 16.1 §5.2 非迭代时间：40% → ~18%

口径：官方 4k `[[4096,1024,64,256]]` ×2 + `--enable-logging-iteration-details`
+ `VLLM_ITER_STAGE_PROFILE`（site-packages 临时补丁，测完可 revert）。

| | client wall | Σ iteration | **non-iter** |
|---|---|---|---|
| run1 | 111.64 s | 91.1 s | **18.4%** |
| run2 | 111.63 s | 91.6 s | **17.9%** |

stage-prof（稳态 221 s，`unmarked≈0` 闭合）：

| 段 | 占 busy | ms/it |
|---|---|---|
| **wait** (`future.result`) | **82.1%** | 22.2 |
| exec 提交 | 10.1% | 2.7 |
| sample / sched / upd+resid | ~8% | — |

**判定**：GPU 受限；旧「40% host 空档」不能再当可回收头寸。wait **不是** D2H/D2H
（mcTracer 上 memcpy/copy ~1–2.5% GPU 时间）；主体是 GEMM + attention。

### 16.2 MACA launch A/B → light 包落地

| arm | 4k Total tok/s | Δ |
|---|---|---|
| baseline（无额外 MACA_*） | 11760.72 | — |
| **light**：`DIRECT_DISPATCH=1` + `CPU_THREAD_POLICY=-1` + `LAUNCH_MODE=4` | **11896.95** | **+1.16%** |
| light + `GRAPH_LAUNCH_MODE=3` | 9975.92 | **−15.2%**（拒） |
| `GRAPH_LAUNCH_MODE=2`（PDL） | — | cudagraph capture **挂死**（`metaxgpu-pdl` mxcc crash） |

落地：`vllm_fl/.../metax/patches/maca_launch_defaults.py`（`register()` 最早
setdefault；显式 env 优先；`VLLM_METAX_MACA_LAUNCH_DEFAULTS=0` 关闭）。
**仅 MetaX**；天数/其它 vendor 不加载该 patches 包。

### 16.3 conc-64 decode：GEMM + attention mcProfiler

Attention（`flash_attn_with_kvcache`，batch=64，seq∈{512,2048,4096,5120}）：
**所有长度上 `num_splits=0`（现自适应启发式）最优**；改 1/2/4/8/16 均更慢。
seq=4096 时 ~210 µs/层 → ×42 ≈ 8.8 ms/step。

GEMM M=64（device_time）：投影 **mcblas ≫ FlagGems**；仅 lm_head FlagGems
略快（620 vs 656 µs）。**不要**把 `mm` 加进 FlagOS 白名单。
`lm_head` 路由（`lm_head_gemm.py`）当时仍默认 off（先前 4k e2e「−1.8%」）——
**该结论已于 §17 推翻，现默认 on**。

C500 能力：`sm(8,0)`、smem **64 KB**、warp **64**。公开编程模型**无**
NVIDIA 式 TMA / TMEM / block cluster；Triton 大 tile 受 64 KB smem 限制，
打不过 mcblas 的 MMA 强度（见 `gemm_mcprof_20261023/RESULTS.md`）。

### 16.4 剩余优化空间（诚实排序）

1. 保证官方起服带齐 P2 + light 默认（防回退）——配置纪律。
2. `lm_head` FlagGems 已在 §17 结清（+0.80%，默认 on）。
3. 16k 点边角（注意力更重）——未本轮重扫。
4. 驱动侧双初始化共存 / PDL——依赖 MetaX，插件层做不了真修复（§11）。
5. **不值得再砸**：白名单加 `mm`、batch≥16 改 `num_splits`、MODE=2/3、megakernel、扫拷贝。
6. MLP silu fusion 在 conc=64 是 −4.8%（§17.2），已改默认 off；小 M 场景未重扫。

---

## 17. lm_head FlagGems 结清 + MLP silu fusion 否决（2026-10-23）

### 17.1 真因：wrap 从来没挂上过

先前「lm_head 4k −1.8%（11737 vs P2 11957）」**不是** FlagGems 造成的：

- `ModelRunnerFL` 在 APIServer 启动阶段被 import，此时 `current_platform` 还是
  `UnspecifiedPlatform`，`vllm_fl.compilation.graph.Graph` 抛
  `NotImplementedError: not support graph` → import 失败被吞 → 模块级 monkey-patch
  没执行；EngineCore 是 fork 子进程，继承了这个空壳，于是「开关打开」时实际仍是 mcblas。
- 那 11737 是**跨日/跨栈**对比（无 light 包 vs 带 light 包），噪声级差异。

修法：不再 import 时打 patch，改为显式钩子 —— `WorkerFL.load_model()` 末尾调
`_maybe_apply_metax_after_load()`，在 `self.model_runner.load_model()` 之后、平台已确定
时才 wrap。两条 patch 都改为导出 `maybe_apply_after_load(model)`。
顺带修掉 `cuda_wrapper.py` / `vllm_fl/__init__.py` 早期 import
`vllm.distributed.device_communicators` 导致的 EngineCore
`No implementation selected for op='attention_backend'`。

判断是否真挂上：日志里有
`MetaX lm_head GEMM: wrapped lm_head.apply weight=(130560, 2048) route M∈[8,64] N≥100000`
才算数。**以后所有「开了没收益」先查这一行。**

为什么便宜：`lm_head.weight.t()` 的 strides 是 `(1, K)`，
FlagGems `mm` 里的 `a.stride(0) > 1 and a.stride(1) > 1` 连续化检查不触发，
**没有**额外 ~1.9 ms 的 buffer copy；只影响 lm_head，不影响其它算子/其它 vendor。

### 17.2 公平 A/B（P2 + light 栈，官方 4k `[4096,1024]`，conc=64，每臂 2 轮）

**A. MLP silu fusion（2026-10-23 13:37–13:54，lm_head off）**

| arm | run1 | run2 | avg | Mean TTFT | Mean TPOT |
|---|---|---|---|---|---|
| silu=0 | 11690.61 | 11675.49 | **11683.05** | 1944 ms | 25.5 ms |
| silu=1 | 11127.26 | 11109.96 | **11118.61** | 2149 ms | 26.7 ms |
| Δ | | | **−4.83%** | +10.5% | +4.6% |

TTFT、TPOT **同时**变差 → 真实回归，不是纯 decode 抖动。
先前 P0「+3–4%」是在 wrap 没挂上的状态下量的（见 §17.1），无效。
**结论：`VLLM_METAX_MLP_SILU_FUSION` 默认改为 `0`。**

**B. lm_head FlagGems（2026-10-23 13:55–14:09，silu=0 干净基线）**

| arm | run1 | run2 | avg | Mean TTFT | Mean TPOT |
|---|---|---|---|---|---|
| off（mcblas） | 11678.40 | 11660.02 | **11669.21** | 1946/1936 ms | 25.47/25.53 ms |
| on（FlagGems mm） | 11769.92 | 11754.16 | **11762.04** | 1937/1927 ms | 25.27/25.32 ms |
| Δ | | | **+0.80%** | −0.5% | −0.8% |

臂内抖动 <0.2%，256/256 全成功；TTFT 与 TPOT 同向改善。
另在 silu=1 基线上测得 **+0.91%**（11118.61 → 11211.69），方向一致。
**结论：`VLLM_METAX_LM_HEAD_GEMM` 默认改为 `1`。**

### 17.3 基线口径提醒

同一套 P2 + light 栈，4k/conc64 的真实基线是 **~11670–11690 tok/s**
（silu off）。历史上出现过的 11110 是 silu fusion 拖累，11737 是窗口噪声，
两者都不能当基线用。

---

## 18. 瓶颈定位：decode 已到带宽屋顶，剩余空间在哪（2026-10-23）

环境：P2 + light 栈，`lm_head` on / `silu` off（§17），官方 4k 口径。
产物：`bench_results/decode_step_profile_{,4k_}20261023/`、
`decode64_mcprof_20261023/attn_fp8_kv.py`、`sampler_micro.json`。

### 18.1 先钉住硬件口径

`copy_` 1 GiB 实测 **1416 GB/s**（C500，104 SM，68 GB）。后面所有「% of peak」
都按这个数算，不用厂商标称值。

### 18.2 decode step 的字节账（batch 64，seq 4096，42 层）

| 项 | 每 step 流量 | 实测 | 带宽 | 占峰值 |
|---|---|---|---|---|
| 四个投影 GEMM | 3.96 GB（1.98 B 参数） | 4.85 ms | 816 GB/s | **58%** |
| decode attention | **11.27 GB** KV | 8.80 ms | 1280 GB/s | **90%** |
| lm_head（FlagGems） | 0.53 GB | 0.41 ms | 1290 GB/s | 91% |
| 其余 kernel（norm/silu/cat/add/kv_write） | ~0.2 GB | ~2.1 ms | — | — |
| **kernel 合计** | **16.0 GB** | **~16 ms** | 1010 GB/s | 71% |
| 实测 ITL p50（无 prefill 干扰，64 请求） | | **18.45 ms** | | |

**结论一：4k 下 attention 吃掉 70% 的 KV 带宽，且已经跑到 90% 峰值。**
GQA 8:1 已经用掉，`num_splits` 全长度都是 0 最优（§16.3），block/head 维度也没有
剩余空间 —— 这条线在 bf16 KV 下**没有可挖的**。

**结论二：真正还有 40% 空间的是投影 GEMM（58% 峰值）。**
M=64 是纯带宽场景，mcblas 选的 tile 是 `64x32x256`(qkv/o) 与 `128x64x256`
(gate_up/down)，对 M=64 的流式读权重不够友好；到 90% 就能省 1.75 ms/step
（≈7% 总吞吐）。这属于 mcblas 侧，插件层改不动（FlagGems `mm` 更差，§16.3）。

**结论三：ITL 18.45 与 kernel 16 ms 之间有约 2.5–3.5 ms/step 的非 kernel 时间**
（≈15%），含采样器 ~0.85 ms（不在 CUDA graph 内，串在关键路径上）。
注意 **async scheduling 本来就是开的**（vLLM 0.24 默认开，日志
`Asynchronous scheduling is enabled.`），所以这段不是「host 没跑在前面」，
需要更细的插桩才能定位，不要再当成可回收的 host 空档。

### 18.3 prefill 侧

4k 官方口径里 prefill 占约 1/6 的墙钟（1.05 M prompt tokens）：
按 M=2048 实测 mcblas 164–197 TFLOPS 折算，prefill 稳定在 **50–65% of soft peak**。
Triton 追不上（64 KB smem 卡死大 tile，§16.3），属平台侧。

### 18.4 剩余可做的，按价值排序

| # | 方向 | 预期 | 状态 |
|---|---|---|---|
| 1 | **fp8 KV cache** | attention 11.27→5.64 GB，省 ~4.0 ms/step ≈ **−25% decode** | **封死**：本机 flash_attn 要求 `q.dtype == k.dtype`（实测 `RuntimeError: query and key must have the same dtype`），而 vLLM 传 bf16 q；且 `MetaXPlatform.supports_fp8()` 返回 False |
| 2 | **fp8/w8a8 权重** | 4.49→2.25 GB，省 ~2.5 ms/step ≈ −16% decode，prefill 也受益 | **封死**：同上 `supports_fp8()`；仓里没有量化权重 |
| 3 | **weight-only 量化 + 反量化 GEMM（插件侧唯一能自己做的两位数收益）** | 读 2.2 GB 而非 4.4 GB，若 dequant kernel 能到 85% 峰值 → GEMM 4.85→2.7 ms，≈ **−13% decode** | 未做。风险：要自己写 Triton dequant+GEMM 并在 M=64 赢过 mcblas；这正是 FlagGems `mm` 输掉的那条线，但当时比的是 fp16 算力，不是带宽 |
| 4 | 投机解码（摊薄权重读） | 接受 k token ≈ k× 少读一遍权重 | 未接。随机数据集接受率为 0，**bench 上不会有收益**，只对真实有重复的流量有效；需要 draft/EAGLE 头或 ngram |
| 5 | 小 kernel 融合（插件侧） | 合计 ~1 ms/step ≈ **2–4%** | 见 18.5 |
| 6 | `block_size` 16→32/64 | 预期 <1% | **已测，无收益（§24）：+0.20%，在噪声内；保持 16** |

### 18.5 小 kernel 明细（decode，每 step）

trace（`decode_step_profile_20261023/`，CUPTI 在图内会丢约一半事件，
绝对值偏小，但**比例**可信）：

- 每层 2 个独立 residual `add` + 2 个 `rms_norm` → **fused_add_rms_norm 没被用上**。
  插件里其实已经注册了 flagos provider（`vllm_fl/ops/ir_kernels.py`，
  `metax.yaml` 的 `ir_op_priority` 也带 `fused_add_rms_norm: [flagos, vllm_c, native]`），
  说明是模型侧没走残差入参的调用形态。省 1 个 add/层。
  -> **已解决（§19）**：真因不是模型侧没传残差，而是 flagos provider 因 custom_op
  别名规则被迫 clone；改走 `vllm_c` 原地 kernel 后 **+3.1%**。
- 每层 2 个 inductor `triton_poi_fused_cat`（qkv / gate_up 的 cat），可换成
  预分配 buffer 直写。
  -> **已否决（§19.5）**：它就是 rope 的 `cat((rot, pass))` 融合形态，
  换成 FlagGems 单 kernel 反而慢 31%。
- 采样器 0.85 ms/step，`_top_p_mass_v2_kernel` 已到原子吞吐地板（§10.5），
  能省的只有 mask/softmax/random 那 ~0.4 ms。
- 每层还有 1 个 `mcblas::DevDstElementwiseAdd`（~0.37 ms/step），需确认是不是
  多余的 bias/residual 写回。

### 18.6 已经排除、别再试

`num_splits`（全长度 0 最优）、白名单加 `mm`、MODE=2/3、PDL、megakernel、
mlp silu fusion（−4.8%，§17）、fp8 KV（dtype 不允许）、大 MBT 换 prefill 效率
（prefill 已 50–65% 峰值，瓶颈是 mcblas 而非调度）。

**一句话**：这套栈（4k/conc64）已经跑在 HBM 屋顶的 71–90%，插件层剩余可挖
大约 **2–4%**；两位数收益只有 weight-only 量化（自己写 kernel）这一条路，
其余都卡在 MetaX 的 fp8 支持上。

---
## 19. fused_add_rms_norm 换到 vllm_c：+3.3%（2026-10-23）

接着 §18.5 的第一条做。产物：`bench_results/ir_norm_20261023/`、
`bench_results/norm_rope_20261023/norm_rope_micro.json`。

### 19.1 根因：flagos provider 因为 custom_op 别名规则被迫 clone

`vllm_fl/ops/ir_kernels.py` 给 `vllm.ir.ops.fused_add_rms_norm` 注册的 `flagos`
provider 是 `torch.library.custom_op`，而 torch 禁止 custom op 的输出与输入别名。
FlagGems 的 fused kernel 是**原地**写（`x + residual` 写进 `residual`，归一化结果写进
`x`），所以为满足规则必须 `x.clone()` / `x_residual.clone()`。

因此上一版默认的 `split` 模式（`residual = x + x_residual` + 函数式 rms_norm）
其实是**为了躲开 clone** 而主动放弃融合。三条路径在 decode 形状
（M=64, H=2048, bf16）实测：

| 路径 | 耗时 | kernel 数 |
|---|---|---|
| split（aten add + gems rms_norm）—— 旧默认 | 7.43 µs | 2 |
| **vllm_c 原地融合**（`FusedAddRMSNormKernelOpt`）—— 新默认 | **4.44 µs** | **1** |
| flagos 融合 + 2 次 clone | 27.92 µs | 3 |

clone 版比不融合还慢 3.8 倍 —— 所以「加个融合」这条路在 flagos 下是死的，
真正的解法是换 provider：`vllm_c` 注册时带 `inplace=True`，直接调
`torch.ops._C.fused_add_rms_norm`，不需要任何 clone。

> 注意：直接用 `torch.ops._C.fused_add_rms_norm` 在裸 Python 里会报
> `AttributeError`（mcoplib 的 op 要等 `import vllm.kernels` 之后才注册），
> 必须走 `vllm.ir` 的 provider 机制，不要绕过去。

### 19.2 改法

`vllm_fl/ops/ir_kernels.py`：新增 `_VENDOR_FIRST_OPS = ("fused_add_rms_norm",)`，
`default_ir_op_priority()` 对这些 op 把顺序翻成 `[vllm_c, flagos, native]`。
`rms_norm`（非残差形态）不受影响，仍是 `[flagos, vllm_c, native]` ——
flagos 的纯 rms_norm 没有 clone 问题且更快。

只改插件侧、只影响 MetaX（`default_ir_op_priority` 由 `vllm_fl/platform.py`
发布），天数（Iluvatar）的 `ir_op_priority` 不在这个包里。

### 19.3 同会话 A/B（4k 官方口径，conc=64，每臂 2 轮）

同一份代码、同一次会话，仅用 `--kernel-config` 翻 provider 顺序：

| arm | run1 | run2 | avg | Mean TPOT | Mean TTFT |
|---|---|---|---|---|---|
| flagos-first（旧默认） | 11789.46 | 11773.60 | **11781.53** | 25.25 ms | 1930 ms |
| **vllm_c-first（新默认）** | 12158.95 | 12138.94 | **12148.95** | **24.45 ms** | 1905 ms |
| Δ | | | **+3.12%** | −3.2% | −1.3% |

臂内抖动 0.14%，且旧默认复现了 §17 的 11762（差 0.17%），说明没有环境漂移。
TPOT 与吞吐同向，不是采样噪声。

**微基准只预估了 0.25 ms/step（1.4%），实测 3.1%** —— 当时归因到 launch：
decode 时每 step 84 次调用，少一个 kernel 就少 84 次 launch 的关键路径开销，
而 trace 显示 decode 每 step 有 ~2.5–3.5 ms 是非 kernel 时间（§18.2）。

> **该归因已被 §22 推翻，仅结论保留。** §22 在 CUDA graph 口径下测到 kernel
> 边界的固定成本只有 0.05–0.60 µs，不是当初假设的每边界 7–10 µs；这里的
> 3.12% 更可能来自 flagos split 路径本身的 device 时间（§22.4 测到
> +2.80 µs/call）被微基准的 eager 口径低估了。**不要把本节当作「launch 比
> kernel 贵」的证据引用。**

### 19.4 正确性

`vllm_c` 融合 kernel 与 flag_gems 版在 decode 形状逐元素比：
residual **完全一致**（max|diff| = 0.0），归一化输出 max|diff| = 0.0039，
相对量级 10.125 → **1 个 bf16 ulp**，属累加顺序差异。

### 19.5 负结果：rope 不要换

同一批测了把 rope 换成 `flag_gems.apply_rotary_pos_emb(inplace=True)`（1 kernel）：

| 路径 | 耗时 | kernel 数 |
|---|---|---|
| inductor 融合后的 `triton_poi_fused_cat_{0,1}`（现状） | **6.79 µs** | 2 |
| `apply_rotary_pos_emb_inplace_kernel` | 8.92 µs | 1 |

**反而慢 31%。** 而且 §18.5 里说的「每层 2 个 cat」正是这个
`triton_poi_fused_cat` —— 它已经是 inductor 把 `cat((rot, pass))` 融合后的形态，
不是可以再省的冗余算子。现状 0.285 ms/step 已经很低，**这条线关掉**。

（顺带：该微基准第一次跑的 rope 数值对比得到 NaN，原因是 benchmark 里带
`inplace=True` 的调用把输入 q0 改了，属测量脚本 bug，不影响上面的耗时结论。）

### 19.6 现在的位置

4k/conc64 从 **11781 → 12149 tok/s**，累计（含 §17 的 lm_head）：
`11762 → 12149`，**+3.3%**。silu 默认仍为 off（§17.2A）。

下一步若继续，按 §18.4 只剩 weight-only 量化这一条两位数路径；插件侧
小 kernel 还剩每层 1 个 `mcblas::DevDstElementwiseAdd`（~0.37 ms/step）
和采样器 mask/softmax（~0.4 ms/step），合计上限约 1%。

---

## 20. 投影 GEMM 占用率问题：诊断成立，修复不可达（2026-10-23）

### 20.1 重新拆开四个投影

§18 里说的「投影 58% 峰值」是四个平均，拆开后差异很大（M=64，batch=64）：

| shape | N | K | 权重 | 实测 | GB/s | %峰值 | block |
|---|---|---|---|---|---|---|---|
| qkv | 2560 | 2048 | 10.5 MB | 16.6 µs | 632 | **45%** | 20 |
| o_proj | 2048 | 2048 | 8.4 MB | 15.5 µs | 541 | **38%** | 16 |
| gate_up | 12288 | 2048 | 50.3 MB | 42.3 µs | 1190 | **84%** | 96 |
| down | 2048 | 6144 | 25.2 MB | 41.0 µs | 614 | **43%** | 16 |

C500 有 104 个 SM。固定 K=2048 扫 N 证明效率跟的是 **block 数**，不是形状体积：

| N | block | GB/s | %峰值 |
|---|---|---|---|
| 2048 | 16 | 530 | 37% |
| 4096 | 32 | 789 | 56% |
| 8192 | 64 | 902 | 64% |
| 12288 | 96 | 1187 | **84%** |
| 16384 | 128 | 930 | 66%（波次量化）|

只有 `gate_up`（96 block≈104 SM）吃满了带宽。

### 20.2 尝试修复：都输了

两条路都试了（`occupancy_20261023/proj_gemm_probe.py`）：

1. **缩小 BLOCK_N**（每 block 读完整 A，block 数 = N/BLOCK_N，单 kernel 无归约）
2. **split-K**（block 数 = N/BLOCK_N × SPLIT_K，fp32 atomic 归约）

结果全部劣于 mcblas，最好也只是 **0.71–0.72x**：

| shape | mcblas | 最好 Triton | 倍数 | 配置 |
|---|---|---|---|---|
| qkv | 16.6 µs | 23.2 µs | 0.71x | BN=32 BK=128 warps=8 |
| down | 41.0 µs | 56.7 µs | 0.72x | BN=32 BK=128 warps=8 |

`num_warps` 2/4/8 × `num_stages` 2/3/4 × BLOCK_N 32/64/128 × BLOCK_K 32/64/128
全扫过，没有组合能突破。假设的「42 层省 1.4 ms ≈ 5.7%」**拿不到**。

### 20.3 为什么拿不到（硬件约束）

```
shared_memory_per_multiprocessor = 65536   # 64 KB
L2_cache_size                    = 8388608  # 8 MB
multi_processor_count            = 104
```

- **64 KB smem/SM**（对比 Hopper 的 228 KB）让 Triton 的软件流水基本只能上
  2–3 级，BLOCK_N=128/BLOCK_K=128 直接 `OutOfResources: Required 98304, limit 65536`。
  隐藏不了访存延迟，所以 block 数加上去，单 block 效率掉得更多，净收益为负。
- **8 MB L2** 也吃不下 `down` 的 A（64×6144×2 = 786 KB）× 16 份 + 权重流。
- 缩小 BLOCK_N 还有个反向代价：每个 block 都重读一遍 A。qkv 用 BN=32 变 80
  block，A 流量 20×256 KB → 80×256 KB，把省下的带宽又吐回去。

结论：**mcblas 在 M=64 这个区间已经是手调过的，插件层没有余量**。
要动只能写 C++/mcblas 级的 split-K + 针对 64 KB smem 的流水，不属于插件范围。

诊断本身留档：`occupancy_20261023/note.txt`、`proj_gemm_probe.json`。

### 20.4 顺带确认（量化路径）

`torch.ops._C` 里 `dir()` 能看到 `awq_gemm` / `gptq_gemm` / `marlin_gemm` /
`cutlass_scaled_mm`，但**调用全部抛 `AttributeError`** —— schema 在、kernel 在这个
build 上不可用。与 `MetaXPlatform.supports_fp8() == False`、flash_attn 拒绝 fp8 KV
（要求 `q.dtype == k.dtype`）互相印证。vLLM 现成的量化 GEMM 在这台机器上不能复用。

### 20.5 位置

§19 的 12149 tok/s / TPOT 24.45 ms 不变。§19.6 的判断维持：插件侧只剩
`DevDstElementwiseAdd`（~0.37 ms/step）和采样器（~0.4 ms/step），上限约 1%；
二维数的路只剩自写 dequant GEMM（项目级）。**§20 这条关闭。**

## 21. decode 热路径上的 `.item()` 同步代价 10.8%（2026-10-23）

这条不是优化成果，是一个**否决性结论**，但它约束后续所有改动，单独成节。

### 21.1 起因：想跳过 p == 1.0 的 top-p 空转

§16 的 trace 显示采样链每步 ~1.02 ms（`_top_p_mass_v2` 412 µs、
`_top_p_zero` 等）。当时判断：`SamplingParams.top_p` 默认 1.0，而
`TopKTopPSampler.forward_native` 无条件调 `apply_top_k_top_p`，所以 p=1.0
时整条 bucketing 链是空转，可以提前返回。于是在
`vllm_fl/.../metax/patches/topk_topp_sampler.py` 加了：

```python
if _SKIP_P1 and k is None and p is not None:
    if bool((p >= 1.0).all().item()):   # <- 每次 decode 一次 D2H 同步
        return logits
```

### 21.2 事实一：这个分支是死代码

A/B 跑完查覆盖率，B 臂（默认开启）是 **`fast=4096 skip=0`** —— 4096 次采样，
跳过一次都没进。原因不在 bench，在模型自带的生成配置：

```json
// /workspace/MiniCPM5-2B/generation_config.json
{ "do_sample": true, "temperature": 1.0, "top_p": 0.95 }
```

`vllm bench serve` 的 `--top-p` 默认 None，payload 不发 top_p，server 回落到
模型的 `generation_config.top_p = 0.95`。于是：

- `gpu_input_batch.py:391` `if sampling_params.top_p < 1:` → 加入 `top_p_reqs`
- `no_top_p` 为 False → `top_p` 以 tensor 下发，每行 0.95
- `(0.95 >= 1.0).all()` 为 False → 永不跳过

更本质的一点：`no_top_p` 的定义是 `len(self.top_p_reqs) == 0`，即**只要
`p is not None`，就必然存在至少一个请求的 top_p < 1**。所以
`(p >= 1.0).all()` 在 vLLM 的语义下恒为 False —— 这个分支无论如何都不可能触发。

### 21.3 事实二：它仍然让 decode 慢了 10.8%

死代码没有提前返回，但**每次 decode 都执行了一次 `.item()`**。官方 4k
`[[4096,1024,64,256]]`，三组独立测量：

| 状态 | run1 | run2 | run3 | TPOT | TTFT |
|---|---|---|---|---|---|
| A 臂（`VLLM_FL_TOPP_SKIP_P1=0`，不检查） | 12165.1 | 12157.4 | — | 24.41–24.44 | 1901–1912 |
| **B 臂（默认，每次 `.item()`）** | **10847.9** | **10875.2** | — | **27.45–27.46** | **1992–2053** |
| 还原后（`git checkout`，无该逻辑） | 12193.2 | 12198.8 | 12186.2 | 24.35–24.41 | 1867–1910 |

- 三次独立测量（A 臂 ×2 + 还原后 ×3，共 5 个 run）都落在 **12150–12200**，
  组内一致性 0.1%；只有 B 臂那 2 个 run 在 **10850**，组内一致性 0.25%。
- 退化 **−10.8% 吞吐 / +3.03 ms TPOT**，可稳定复现，不是噪声。
- 两臂 workload 完全相同：`Successful requests 256`、`Total input tokens 1048576`、
  `Total generated tokens 262144`、`peak concurrency 73–75`。
- 两臂配置完全相同：`GPU KV cache size 1,191,968 tokens`、
  `max_num_batched_tokens=8192`、`Maximum concurrency 9.09x`、
  graph capture 12+12、`torch.compile` 10.53 vs 10.67 s。

### 21.4 机制：同步把 CPU/GPU 流水打穿

1.03 ms 的微基准（`sampler_20261023/`）说明**不能**用微基准估这个代价：
那里每次只排 40 个 matmul 就 sync，队列本来就浅，测出 0.05 ms/step。

真实 decode 每步要排 42 层 × 多个 kernel 的深队列。`.item()` 会把
D2H 排进队尾并阻塞 CPU，直到**整个队列**排空；于是 CPU 无法在上一步 GPU
还在跑时预排下一步，host 提交时间被完整暴露在关键路径上。

量化对上：§16.1 用 stage-prof 量到 **exec 提交 = 2.7 ms/it**，而这里 TPOT
增量是 **3.03 ms**。两者吻合，指向同一个量 —— 同步吃掉的正是原本被
CPU/GPU 重叠隐藏掉的 host 提交时间。

旁证：**TTFT 也退化 +7.4%（1911 → 2053 ms）**。TTFT 由 prefill 主导，
采样器只占其中极小一段，单独看采样器无法解释；但流水被打穿会让 chunked
prefill 的调度也串行化。这是"全局同步"而非"采样器变慢"的直接证据。

### 21.5 排除的其他解释

- **不是热降频**：`mx-smi --show-clk-tr` 全 `Not Active`（Over Power / Chip
  Over Temperature / VR / HBM 均未触发）；监控期温度 52–56 °C、功耗
  205–263 W（上限 350 W）；XCORE 稳定 1125 MHz、MC 1800 MHz。
- **不是带宽衰减**：实测 `copy_` 1 GiB = **1423.6 GB/s**（记录峰值 1416），
  256 MiB = 1343.6，`sum` 读 1 GiB = 1533.1。
- **不是 OOM**：B 臂开跑前 `oom_kill` 计数已是 8（A 臂期间的容器内存事故），
  跑的过程中无新增。
- **不是 preemption**：两臂 `preempt` / `recompute` 均为 0。
- 期间确实发生过一次容器 OOM（cgroup 64 GiB 触顶，8 次 kill），根因是
  **bench 客户端也加载了 FL 插件**——`export VLLM_PLUGINS=fl` 是全局的，
  客户端因此拉起 FlagGems + inductor 编译 worker（主进程 23.8 GB + 子进程
  15.6 GB），把 cgroup 占满后 OOM killer 杀掉 server 的 EngineCore。这是
  测量脚本的 bug，已修（bench 那行显式 `VLLM_PLUGINS=`），与 10.8% 无关，
  因为后续 A 臂与还原后的测量都在修好之后。

### 21.6 结论与约束

**否决**：不在 decode 热路径上加任何 `.item()` / `.cpu()` / `bool(tensor)` /
`.tolist()` 做分支判断。代价是**百分之十量级**，不是微秒量级。

若以后确实需要这类判断，只能：
1. 用已有的 kernel 产出（例如把条件塞进 Triton kernel 内部，让它自己
   `tl.where` 出恒等结果），不产生额外 D2H；
2. 或在请求入队时（非热路径）把 flag 算好，随 sampling metadata 一起下发。

**同时关闭**：top-p 采样器这条线的性能优化空间。它的 1.02 ms/step 不能靠
"少跑一次"省掉，因为这个判断本身就比它贵。要省只能改 kernel 内部
（`_top_p_mass_v2` 的 histogram 原子，§10.5 已判定到原子吞吐地板）。

代码已 `git checkout` 还原，`_SKIP_P1` / `_top_p_is_noop` 全部移除。

### 21.7 顺带修正 §18.5 的 memcpy 归因

`decode_step_profile_20261023/step_budget.py` 之前只统计 `kernel` 类事件，
漏了 `gpu_memcpy`。补上后 GPU 总计 13.43 → **14.96 ms/step**，多出一项：

| ms/step | 占比 | 次数/步 | 项目 |
|---|---|---|---|
| 4.897 | 32.7% | 42 | `flash_fwd_splitkv_kernel` |
| 1.758 | 11.8% | 42 | mcblas GEMM 128x64x256 |
| 1.437 | 9.6% | 84 | mcblas GEMM 64x32x256 |
| **1.430** | **9.6%** | **42** | **Memcpy D2D 256 KiB** |
| 1.111 | 7.4% | 42 | mcblas GEMM 128x64x256 |
| 0.587 | 3.9% | 42 | `flash_fwd_splitkv_combine` |

这 42 次 memcpy 每层恰好一次 **262144 B = 64 × 2048 × 2**，正是 residual 的
shape，即 §19 修掉的 flagos clone（§19 当时记的是"每层 1 个
`DevDstElementwiseAdd`"，实际还伴随一次整张量 D2D 拷贝）。256 KiB 搬 34 µs
只有 7.7 GB/s，纯固定开销。该 trace 采于 14:27，早于 15:12 的 `c4dadef`，
属**旧栈**；这也解释了 §19 那个 +3.12% 里有一部分来自它。

（这份 trace 是 512 输入、非官方 4k 口径，各项绝对时间不能直接和 TPOT
对比，只用来看 kernel 构成比例。当前栈未重采 —— 两次尝试都因采集窗口
落在请求结束之后而导出空 trace，`profiler_out` 里只有一次
`mcDeviceSynchronize`。）

---

## 22. 复查 fused_add_rms_norm：clone 已经没了，flagos 仍然更慢（2026-10-23）

§19 把 `fused_add_rms_norm` 切到 `vllm_c` 时，理由是「flagos 为了绕开
torch 的禁止别名规则必须 clone 两个 activation」。§21.7 的 trace 又在那条
路径上数到每层一次 256 KiB 的 D2D 拷贝。两个前提都值得复核一次：

1. 现在的栈里 flagos 还 clone 吗？（`ir_kernels.py` 后来改成
   `mutates_args` 契约 + `_FUSED_ADD_MODE=split` 默认，理论上已经不 clone 了）
2. 如果 clone 没了，flagos 会不会反超 `vllm_c`？

产物：`bench_results/fused_norm_verify_20261023/`（`verify_no_clone.py`、
`measure_boundary.py`、`measure_boundary_graph.py`）、
`bench_results/norm_provider_ab_20261023/`（三臂日志与 `SUMMARY.txt`）、
`experiments/src/` 下的 `run_arm_norm.sh` 与 `runs2_bench.py`
（B 臂当时用的是个一次性的 `run_arm_b_norm.sh`，已删；三条臂都可用
`run_arm_norm.sh <label> '<priority>'` 复现）。

### 22.1 事实一：clone 确实没了

`verify_no_clone.py` 在 decode 形状（M=64, H=2048, bf16）下逐个 provider
跑 30 次，统计 kernel 与 `Memcpy` 事件：

| provider | kernel 数 | memcpy/call | 构成 |
|---|---|---|---|
| `vllm_c` | 1 | **0** | `FusedAddRMSNormKernelOpt` |
| `flagos`（split） | 2 | **0** | `vectorized_elementwise_kernel` + `rms_norm_kernel` |
| `native` | 11 | 0 | eager 逐元素碎片 |

§21.7 里那 42 次/step 的 262144 B 拷贝在当前栈上**已经不存在** —— flagos
现在走 split（`residual = x + x_residual` + 函数式 rms_norm），两条路径都
不产生 clone。§21.7 归因到 flagos clone 是对的，但那是 `c4dadef` 之前的栈。

**这张表只用来数 clone。** 同一脚本给出的耗时（`vllm_c` 10.21 µs、
flagos 7.42 µs）当年被我用来判断「flagos 更快」，是错的，见 §22.2。

### 22.2 纠正：eager 下的耗时排序是假的，graph 里方向相反

按 §22.1 那张表（`vllm_c` 10.21 vs flagos 7.42），flagos 看起来该胜出。
我当时信了这个排序，还专门写了「微基准把两者排反了」。**那个判断是错的**，
错在测量工况。

decode 不是 eager 执行，vLLM 把整个 step 捕获成 CUDA graph 回放。eager 与
graph 回放对 4 KB 张量的行为差很远：eager 每次调用都是独立 launch，L2 状态
和 launch 时序都和回放不同。同一份代码在三种口径下的 `vllm_c`：

| 口径 | `vllm_c` 单次 |
|---|---|
| eager + profiler（`verify_no_clone.py`） | 10.21 µs |
| eager + `cuda.Event` 围 200 次连调 | 6.57 µs |
| **graph 回放 + profiler（`measure_boundary_graph.py`）** | **3.91 µs** |

同一个 kernel，三个数差到 2.6 倍。所以 `measure_boundary_graph.py` 把调用
捕获进图再测（20 次调用一张图，回放 50 次 = 1000 次调用），这才是 serving
的工况。两次独立运行：

| provider | wall µs | **device µs** | 边界 | kernel 数 |
|---|---|---|---|---|
| **`vllm_c`** | 4.51 / 4.51 | **3.92 / 3.91** | 0.60 | 1 |
| `flagos` | 6.78 / 6.77 | **6.71 / 6.72** | 0.05 | 2 |
| `native` | 31.66 | 34.27 | −2.60 | 11 |

复现性到 0.02 µs。**在 graph 里 `vllm_c` 的 device 时间本来就比 flagos
短 2.80 µs** —— 不存在「算子更快但整体更慢」的矛盾，flagos 是两头都输。

#### 附带：eager 下 flagos 会被 Python 派发拖死

`measure_boundary.py`（eager 版）测到 flagos wall **106.92 µs**，device
只有 7.40 µs。那 99.5 µs 不是 kernel 边界，是 `fused_add_rms_norm_op` 每次
调用都跑一遍 `_dispatch_rms_norm` → `call_op("rms_norm")` 的 Python
dispatcher 解析（日志里那行 `Op 'rms_norm' using 'default.flagos'`）。

graph 捕获把这层 host 代码整个绕开，所以 serving 只慢 1.5%，而不是 14 倍。
**这条路的性能只在 graph 里成立** —— 若哪天这个 op 掉出 graph（改 eager、
换后端、capture 失败回退），flagos 会立刻变成灾难。这是保留 `vllm_c` 优先
的第二个理由，比那 1.5% 更重要。

### 22.3 端到端：flagos 慢 1.5%，且不是环境漂移

官方口径 `[[4096,1024,64,256]]`，每臂 2 轮，仅用 `--kernel-config` 翻顺序：

| 臂 | 顺序 | run1 | run2 | 均值 | Mean TPOT | Mean TTFT |
|---|---|---|---|---|---|---|
| A | `vllm_c` 先 | 11973.65 | 12013.55 | 11993.60 | 24.79 ms | 1909 ms |
| **B** | **`flagos` 先** | 11796.80 | 11820.44 | **11808.62** | **25.14 ms** | 1982 ms |
| A2 | `vllm_c` 先 | 12005.97 | 12027.90 | 12016.93 | 24.74 ms | 1908 ms |

B 比 A 低 1.48%，比 A2 低 1.74%；臂内抖动 0.2%，两组完全不重叠。

A2 是**专门为排除漂移**加的：A 本身比 §21 的还原基线（12186–12199）低了
1.6%，正好和要测的效应同量级，不能直接采信。于是把 A 放到 B *之后*再跑一遍
（A2）。A2 回到 12017，与 A 只差 +0.27%，说明那段基线下移是**单调漂移之外的
一次阶跃**，之后 4 次独立起服务（A、B、A2 各自的 run1/run2）都稳定，B 的
落后因此是真实的。见 §22.5 的遗留问题。

### 22.4 机制：device 时间差已经解释掉大部分，没有"边界成本"这回事

先把三个独立测量并排放（每次调用，84 次/step）：

| 量 | 值 | 来源 |
|---|---|---|
| flagos − `vllm_c`，graph device 时间 | **+2.80 µs** | `measure_boundary_graph.py`，两次一致 |
| flagos − `vllm_c`，serving wall clock | **+4.11 µs** | 端到端 +0.345 ms/step ÷ 84 |
| graph 内 kernel 边界成本 | **0.05–0.60 µs** | 同左，wall − device |

方向一致、量级吻合，剩 1.3 µs 的缺口属于口径差：graph 里 20 次调用连跑，
L2 比真实 decode step（每层中间夹着 attention、GEMM）更热。

> **纠正**：本条早先写的是「多一次 launch 值 6.9 µs，比 kernel 本身还贵」，
> 更早还说 11 µs。两个数都是拿 **eager** 数据反推出来的，不成立。
> CUDA graph 回放时 kernel 边界的固定成本接近零（0.05–0.60 µs）——
> 这正是 graph 存在的意义。decode 里不存在「每个 kernel 边界付 7 µs」的税。

所以 flagos 落后的原因很朴素：**split 形态在两个 kernel 之间把 residual
写回显存再读回来**，`vllm_c` 的融合 kernel 把 add 和 normalize 放在同一趟、
中间不过内存。多出来的 2.80 µs device 时间基本就是这次往返的代价，
与 launch 次数无关。

#### 那 §19.3 的 +3.12% 怎么解释

§19.3 记录的是「少一个 kernel 拿到 +3.12%，微基准只预估 1.4%」，并归因到
launch 次数。按本轮结论，那个归因同样站不住（边界成本接近零）。更可能的
解释是当时一并切掉的 `flagos` split 路径本身就慢 —— 也就是本轮测到的
2.80 µs/call 量级，而这部分微基准的 eager 口径没有如实反映。

**这条待验证**：要证实得把 §19.3 那两臂在 graph 口径下重测。本轮没有做，
所以 §19.3 的机制段落按「结论（换 `vllm_c` 更快）成立、机制归因待修」对待，
不要引用它作为「launch 比 kernel 贵」的证据。

#### 由此得到的判据

不要用 eager 口径的 kernel 时间给 provider 排序 —— 三个口径差 2.6 倍，
足以把结论翻过来。判断这类小算子（每次几 µs、每 step 几十次）要
**捕获进 CUDA graph 后再测**，而且要看 device 时间差能不能撑起端到端的差。

本轮三者自洽（2.80 → 4.11 µs），因此结论可以直接落到「保留 `vllm_c`」，
不需要再引入边界成本这个假设。

### 22.5 遗留：相对 §21 的 1.6% 基线下移没有解释

§21 还原后（`1ac9f84`，工作树干净）测的是 12186–12199 / TPOT 24.35–24.41；
今天 17:00 之后 4 次独立起服务全是 11994–12017 / TPOT 24.72–24.83。代码没变
（`git status` 只有未跟踪的实验产物），GPU 也没有降频迹象（96% util、
258–267 W、56 °C、P9 1125 MHz），但整套口径稳定下移 1.6%。

不影响 §22 的结论（A/A2 一致且从两侧夹住 B），但**它意味着 §19/§21 的
绝对值不能直接和今天的数比**。若以后要跨会话比数字，必须每次重测一个
`vllm_c` 基线当锚点。这条留作开放项，没有继续追。

> **已结（2026-10-24，§23.5）：不是阶跃，是会话抖动。** 补测第 5 个会话后，
> 同一栈的 10 轮 4k 落在 11973.65–12198.80，带宽 1.88%（±0.94%），
> 2026-10-24 的点在中位。§21 抽到带顶、本节 A/A2 抽到带底，差额正好 1.6%。
> 「下移」撤回；但「跨会话必须带同会话锚点」这条规矩保留，理由见 §23.5。

### 22.6 结论与动作

**不改代码**：`_VENDOR_FIRST_OPS = ("fused_add_rms_norm",)` 保持原样，
`vllm_c` 仍是 `fused_add_rms_norm` 的第一优先。两个理由，第二个更硬：

1. 端到端快 1.5%（A/A2 与 B 不重叠）。
2. 它的优势只在 CUDA graph 里成立。flagos 每次调用要跑一遍
   `call_op("rms_norm")` 的 Python 派发，eager 下 wall 是 106.92 µs
   （§22.2 附带）。一旦这个 op 掉出 graph，flagos 会从「慢 1.5%」变成「慢 14 倍」。

`rms_norm`（非残差）不在例外表里，继续走 flagos，本轮没有动它。

`ir_kernels.py` 顶部的注释已按本轮结论重写：原来写的是「flagos 必须 clone
两个 activation，27.92 µs / 3 kernel」，现在记的是 graph 口径的 device 时间
（3.91 vs 6.72 µs）、eager 口径不可用于排序、以及 Python 派发这个隐患。

### 22.7 方法上的负结果（比结论本身更值得记）

同一个 `vllm_c` kernel，三种口径测出 10.21 / 6.57 / 3.91 µs，差 2.6 倍。
本轮先按 eager 的 10.21 vs 7.42 判断「flagos 更快」，又用它的差额反推出
「kernel 边界成本 6.9 µs」，两次都错，直到 graph 口径测完才纠正。

规矩：

- **判断每次几 µs、每 step 调用几十次的算子，必须捕获进 CUDA graph 后测。**
  eager 口径的绝对值和排序都不能用。
- **不要用「wall − device」在 eager 下反推 launch 成本** —— 那里面混着
  Python 派发（本例 99.5 µs）。
- 微基准只能定方向，最终以端到端为准；两者矛盾时先怀疑微基准的工况。

---

## 23. 16k 官方口径两轮 + 同会话 4k 锚点（2026-10-24）

§22 之后栈没有再改，本轮只为把 16k 点补齐。16k 一直是历史数据的薄弱处：
§15 那次是 2026-10-22 的栈，之后 §17/§19 又落了 lm_head、silu fusion、
`fused_add_rms_norm` 三处改动，16k 从未重测。

产物：`bench_results/official_16k_20261024/`（`server.log`、
`bench_16k.log`、`bench_4k_anchor.log`），脚本
`experiments/src/run_16k_two.sh`。

### 23.1 口径

官方 16k 形状 `[[16384,1024,64,128]]`，两轮，两轮都计入。客户端
`VLLM_PLUGINS=` 置空（§11 / §12.4 的双初始化 SIGKILL），服务端
`--no-enable-prefix-caching`，与 §15 一致。

**不改任何环境变量**，测的就是插件当前默认：MBT 8192、白名单
silu/rms/rope、lm_head on、MLP silu fusion off、`fused_add_rms_norm` →
`vllm_c`、MACA light launch、FlagGems `align32_geometric` cap 8192。

起服日志确认（不是按文档假设，是实读）：

```
patch_openai_api_mbt_for_metax: OpenAI-server default max_num_batched_tokens -> 8192
Final IR op priority: IrOpPriorityConfig(rms_norm=['flagos','vllm_c','native'],
                        fused_add_rms_norm=['vllm_c','flagos','native'])
```

同一服务会话末尾补跑 1 轮 4k `[[4096,1024,64,256]]` 作锚点 —— §22.5 已记明
跨会话比较必须有同会话锚点。

### 23.2 结果

| Prefill | Decode | Conc | N | Run | Total tok/s | Output tok/s | Mean TTFT | Mean TPOT |
|---|---|---|---|---|---|---|---|---|
| 16384 | 1024 | 64 | 128 | 1 | 11856.44 | 697.44 | 14857.93 ms | 76.67 ms |
| 16384 | 1024 | 64 | 128 | 2 | 11846.86 | 696.87 | 14839.35 ms | 76.76 ms |
| 16384 | 1024 | 64 | 128 | **avg** | **11851.65** | **697.15** | **14848.64 ms** | **76.72 ms** |
| 4096 | 1024 | 64 | 256 | 1（锚点） | 12138.58 | 2427.72 | 1899.66 ms | 24.48 ms |

Failed = 0 / 128 与 0 / 256；16k 两轮抖动 **0.081%**，比 4k 的臂内抖动更小。
**这两个数是当前栈的 16k 参考值**，以后改栈就拿它们当基线。

16k 的 Output（697）远低于 4k（2428）是口径所致，不是 decode 变慢：同一个
Decode=1024，但 16k 的 prefill/KV 占 wall 比例大得多，Total 分子里 input
占比高。TPOT 76.72 ms vs 24.48 ms 才是真实的 decode 差异（注意力随长度变重）。
这与 §15.2 的解释一致。

### 23.3 与 §15 比：名义 +1.72%，但不能算收益

| | 16k Total | 16k Output | 16k TTFT | 4k Total |
|---|---|---|---|---|
| §15（2026-10-22） | 11650.80 | 685.34 | 14904.47 ms | 11956.89 |
| 本轮（2026-10-24） | 11851.65 | 697.15 | 14848.64 ms | 12138.58 |
| Δ | **+1.72%** | +1.72% | −0.37% | **+1.52%** |

看起来 16k 涨了 1.72%，**但同会话 4k 也涨了 1.52%** —— 涨幅几乎与形状无关。

而且 §15 那次**不能当当前栈的基线**：它的 `launch_serve.py` 里显式设了

```python
env["VLLM_METAX_MLP_SILU_FUSION"] = "1"     # §17.2 实测 −4.83%
env["FLAG_GEMS_METAX_MM_GEMV"] = "1"
```

silu fusion 是 §17.2 判定要默认**关掉**的负优化，§15 却是开着的；同时 §15
还没有 light launch（§16.2，+1.16%）、没有 lm_head on（§17.2，+0.80%）、
没有 `fused_add_rms_norm` → `vllm_c`（§19，+3.12%）。四处差异单独 A/B
之和远超 +1.5%，实测却只有 +1.5%，说明这几处的历史 A/B 绝对值不能线性叠加
（各自的口径与漂移都不同）。

**结论：+1.72% 不归因于任何单一改动。§15 作废，23.2 的两个数取代它。**

### 23.4 唯一跨配置可比的量：16k/4k 比值

形状比可以消掉会话漂移和「整机快慢」这类共同因子：

| | 16k/4k 比值 |
|---|---|
| §15（2026-10-22） | 0.97440 |
| 本轮（2026-10-24） | 0.97636 |
| 差 | **+0.20%** |

**16k 相对 4k 只动了 +0.20%，等于没动。** 这与本轮预期一致：§22 之后改的都是
与形状无关的路径（norm provider），没有任何 16k 专属（注意力）的改动。

### 23.5 顺带结掉 §22.5：那是会话抖动，不是阶跃

§22.5 留了个开放项：10-23 的 4k 比 §21 低了 1.6%，当时无法判断是漂移还是
阶跃。今天多了 1 个点，把同一栈（`vllm_c` 默认）的 4k 实测按时间排开：

| 时间 | 来源 | 4k Total | 轮内抖动 |
|---|---|---|---|
| 10-23 ~15:14 | §19.3 | 12148.94 | 0.16% |
| 10-23 ~16:15 | §21 | 12192.73 | 0.10% |
| 10-23 ~16:52 | §22 A | 11993.60 | 0.33% |
| 10-23 ~17:11 | §22 A2 | 12016.93 | 0.18% |
| 10-24 ~22:58 | §23 | 12138.58 | —（1 轮） |

5 个会话 10 轮：min 11973.65 / max 12198.80，**带宽 1.88%，均值 12103.57，
±0.94%**。今天的 4k 距均值 +0.29%，**正落在带中间**。

> **§22.5 的「1.6% 阶跃」撤回。** 不存在单调漂移或永久下移：这是 ±0.9%
> 量级的会话间抖动，§21 抽到带顶、§22 的 A/A2 抽到带底，两者相差正好 1.6%。
> 当时只有 4 个点、且两个点连在一起，所以看起来像阶跃。

规矩不变、但理由更清楚：跨会话比数字**仍必须**带同会话锚点。不是因为有阶跃，
而是因为 ±0.94% 的抖动和很多待测效应同量级 —— §22 那 1.5% 的效应量级只有
抖动的 1.6 倍，没有锚点根本判不了。

### 23.6 对 §22 结论的复核

§22 判 B 臂（`flagos` 优先）慢，靠的是同一会话内的 A/B。用本条得到的抖动带
再检一遍：

- A/A2 = 11993.60 / 12016.93，B = 11808.62
- 抖动带下限（10 轮 min）= 11973.65
- **B 比带下限还低 1.38%**

也就是说 B 落在所有 10 次 `vllm_c` 测量之外，§22 的结论不受抖动影响，
比当时（只有 4 个点）更稳。

---

## 24. 负结果：KV `block_size` 16→32 无收益（2026-10-24）

§18.4 第 6 项「`block_size` 16→32，预期 <1%，未试，便宜」—— 便宜，所以先验掉。
结论：**保持 16，这个方向关闭。**

产物：`bench_results/blocksize_ab_20261024/`，脚本
`experiments/src/run_blocksize_arm.sh`。

### 24.1 为什么不抱期望，也还是测了

MetaX 把 `block_size` 解析成 16（`vllm_fl/platform.py:409`，CUDA 分支），
flash_attn 的 `get_supported_kernel_block_sizes` 要求 16 的倍数，所以 32 合法。
页更大意味着每次 attention 少走一半的页表项。但 4k 下 attention 已经跑到
**90% 带宽峰值**（§18.2），paging 元数据本来就不是瓶颈，预期只能是零点几。

### 24.2 结果

同会话连续两臂，官方 4k `[[4096,1024,64,256]]`，仅 `--block-size` 不同，
服务端 `--no-enable-prefix-caching`，客户端 `VLLM_PLUGINS=`。

| 臂 | run1 | run2 | avg | Mean TPOT | Mean TTFT |
|---|---|---|---|---|---|
| bs=32 | 11976.84 | 11966.10 | **11971.47** | 24.86 ms | 1896 ms |
| bs=16 | 11948.12 | —（终止） | **11948.12**（n=1） | 24.89 ms | 1916 ms |
| Δ | | | **+0.20%** | −0.035 ms | −20 ms |

两臂 KV 池都是 **1,191,968 tokens**，容量一致，所以不存在「页更大导致池子变小、
排队变多」的混淆项 —— 这个对照是干净的。

**n 不相等，必须写明**：bs=32 跑满 2 轮（轮内抖动 0.090%），bs=16 只拿到 1 轮
就被主动终止了。所以 +0.20% 是「2 轮均值 vs 1 轮」，不是配对比较。

### 24.3 判定：无收益，关闭

- +0.20% 远小于会话抖动带宽（±0.94%，§23.5），也小于 bs=32 自己的轮内抖动能
  分辨的程度。
- 方向甚至与直觉相反（更大的页没有更快），但幅度在噪声内，**不予采信**。
- TPOT −0.035 ms 同理，属噪声。

**动作：不改默认，保持 `block_size = 16`（平台解析值）。** 不为了 0.2% 引入
一个非默认值 —— 页变大还会影响 chunked prefill 的粒度对齐和 prefix cache 命中
边界（本口径关了 prefix cache，但真实线上是开的），风险大于收益。

### 24.4 这条对 §18.4 清单的影响

§18.4 第 6 项结清。剩余的项目按原排序仍然成立，其中最值得做的还是**采样尾巴
（图外 28 个 kernel / step，约 1.1–1.3 ms）**，但那个必须先在当前栈上重采一条
非空 trace —— §18.5 的明细来自旧栈，§22.7 已经立过规矩不能再拿旧口径下结论。
