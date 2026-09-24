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

| 场景 | 配置 | 基线 Total tok/s | 门槛（−1%） | 基线 Mean TTFT |
|---|---|---|---|---|
| 4k | 256 × (4096+1024), 并发 64 | 5089.65 | ≥5038.75 | 3199.44 ms |
| 16k | 128 × (16384+1024), 并发 64 | 7029.68 | ≥6959.38 | 27197.14 ms |

「偏差 1% 以内视为正常波动」是硬信息 —— **任何小于 1% 的优化都不赋分**，
必须盯 2% 以上的动作。

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

### 5.8 长任务不要忙等

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
| `src/decode_kernel_budget.py` | 单步 kernel 预算 |
| `src/restart_server_iterlog.sh` | 带逐迭代日志重启 server |

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

---

## 8. 未收割项（按价值排序）

1. **把 FlagOS 融合白名单带进官方启动命令** —— 现有最大的一笔收益，
   且是纯配置，零代码风险（§3.1(1)）。
2. **20 秒停顿** —— 期望回收 ≈1.9%，并显著降低方差（§4.4）。
3. **M==1 GEMV 的端到端验证** —— 微基准 1.12x，预期 TPOT ≈−6.5%（§3.2(5)）。
4. **4k 重测** —— 现有 4k 数据因停顿不可信，需独占 GPU、提高轮数（§4.1）。
5. 拆解 40% 非迭代时间的可压缩比例（§5.2），**先量再动**。
