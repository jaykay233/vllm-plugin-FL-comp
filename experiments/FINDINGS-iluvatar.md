# MiniCPM5-2B on 天数 BI-V150：环境与已验证状态

本文件是 [`FINDINGS-mx.md`](./FINDINGS-mx.md)（沐曦 MetaX C500）的**天数对应篇**。
区别很重要：两张卡的基线相差 **2.5×（4k）～7.7×（16k）**，沐曦篇里的收益数字、
根因结论、门槛都不能直接搬到这张卡上。

> **当前状态：已有本卡首批实测读数，但仍无已量化的优化收益。** §1~§6 是**环境**与
> **已验证的事实**；§7 是本卡 **4k 单轮**实测与瓶颈定位（**非提交成绩**，官方取
> 第 2~4 轮平均）。沐曦篇的收益结论在§5 一律标注为「尚未在本卡复测」，不要照搬。

---

## 1. 环境（本卡实测）

| 项 | 值 |
|---|---|
| 加速卡 | **Iluvatar BI-V150**，32768 MiB 显存 |
| 设备节点 | `/dev/iluvatar0` |
| 内核驱动 | `iluvatar` 4.5.0 |
| 用户态工具链 | `/usr/local/corex` 4.5.0（`IX-ML 4.4.0`） |
| 卡管理工具 | `ixsmi`（**注意不是 `nvidia-smi`**，已在本卡 PATH 中） |
| Python | 3.12.11（`/usr/local/bin/python3`，**本镜像无 conda**） |
| torch | `2.10.0+corex.4.5.0` |
| vLLM | `0.24.0+empty` |
| triton | 3.6.0 |
| flagtree | `0.6.2a3+iluvatar3.6` |

确认卡型号：

```bash
python -c "import torch; print(torch.cuda.get_device_name(0))"   # -> Iluvatar BI-V150
```

`COREX_VISIBLE_DEVICES=all` 相当于 NVIDIA 的 `CUDA_VISIBLE_DEVICES`。

与沐曦的差异，凡是脚本里都要处理：**没有 `/opt/conda/envs/mx`**，用系统
`/usr/local/bin/python3` 与 `/usr/local/bin/evalscope`；serve 口径也不同（见 §4）。

## 2. 基线（组委会公布，天数）

来源：`.cursor/skills/race-s2-track2/SKILL.md`；自洽性校验
`python3 experiments/src/eval_baseline.py`（四行 `Total 实测 vs 反算` 均为 0.00%）。

| case | in | out | conc | num_prompts | Duration (s) | Output tok/s | Total tok/s | Mean TTFT (ms) |
|---|---|---|---|---|---|---|---|---|
| 4k  | 4096  | 1024 | 64 | 256 | 646.31  | 405.60 | **2028.01** | 11573.53 |
| 16k | 16384 | 1024 | 64 | 128 | 2434.82 | 53.83  | **915.15**  | 599262.03 |

门槛为 ±1% 容差：4k `total ≥ 2007.73`、`TTFT ≤ 11689.27 ms`；
16k `total ≥ 906.00`、`TTFT ≤ 605254.65 ms`。

> **天数 16k 是病态读数。** TTFT ≈ 599 s，且用两阶段模型反解 prefill/decode 会得到
> **负的 decode 速率** —— 同一个两阶段模型不描述这张卡。把天数数字当**实测事实**，
> 不要用「阶段时长占比」去推断优化收益（沐曦篇 §2.2、§3 的做法在这里不成立）。

正确性口径与卡无关：`math_500` Level 3，基线 **0.962**，要求 **≥ 0.95**。

## 3. 已验证事实

### 3.1 端到端推理可用（非官方口径的冒烟测试）

在 `flagos-2026-s2/metax-mm-strategy` 的 FlagGems + `feat/ir-kernels-compile` 的
vllm-plugin-FL 组合下，vLLM 离线推理跑通，输出正确连贯：

```bash
VLLM_PLUGINS=fl python3 /tmp/smoke_vllm.py    # -> SMOKE_TEST_OK
```

参数：`max_model_len=2048`、`enforce_eager=True`、`gpu_memory_utilization=0.6`、
3 条 prompt、`max_tokens=32`。

> ⚠️ 这是**冒烟测试，不是 benchmark**。`load 43.6 s / gen 8.0 s / 12.04 tok/s` 与
> 官方 4k/16k 场景不同口径，**不得**拿来对比 §2 的基线。

### 3.2 FlagGems 确实在链路上（不是只装上没生效）

启动日志里 dispatch 走的是本卡配置，且四个算子都落到 flagos：

```
[vllm_fl.dispatch.policy] Using custom config from '.../vllm_fl/dispatch/config/iluvatar.yaml'
[vllm_fl.dispatch.manager] OpManager initialized: 11 ops with 23 implementations
Op 'attention_backend'  using 'default.flagos'
Op 'rms_norm'           using 'default.flagos'
Op 'rotary_embedding'   using 'default.flagos'
Op 'silu_and_mul'       using 'default.flagos'
```

### 3.3 FlagGems 5.4+ 在天数上起不来 —— 必须用 5.3.5（tag `v5.3.5` 系列）

`flag_gems` **5.4**（master）会让 engine 直接起不来：新增的 flagtune 自动调优路径
在加载 cost model 时无守卫地探测设备，而 `triton/flagtune` 未注册 Iluvatar 的
`corex` backend：

```
triton.flagtune.runtime.device.UnsupportedFlagTuneDeviceError:
FlagTune does not support Triton backend 'corex'; registered backends: cuda, hip, maca, musa
```

调用链：`flag_gems/flagtune/cost_model.py::load_model -> discover_gpu_metadata
-> probe_flagtune_device`。5.4 还引用了 `triton.flagtune.runtime.errors`，本机的
flagtree 里同样不存在。

- **正解：用组委会指定的 `v5.3.5` 系列**，它没有 `flag_gems/flagtune/cost_model.py`，
  开箱即用，不需要任何开关。
- 临时绕过 `5.4` 的办法是 `USE_FLAGTUNE=0`（让 `resolve_cost_model_intent` 返回
  `DISABLED`，退回默认 config space，只关掉调优搜索，不影响算子选择与正确性）。
  仅作应急，不要作为提交配置。

> 环境里的 `flagtree` 是 `0.6.2a3+iluvatar3.6`，而 FlagGems 的 iluvatar extra 期望
> `flagtree==0.6.1+iluvatar3.6`。这个偏差是 5.4 出问题的背景，但**不要去装那个 extra**：
> 它同时 pin 了 `torch==2.7.1+corex.4.4.0`，会把整个环境降级。

### 3.4 开 compile 必崩 `NameError: RMSNormQuantFusionPass` —— 已由插件修复

天数 serve 必须带 `--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'`
（§4），而**这条参数在出厂状态下必崩**：

```
NameError: name 'RMSNormQuantFusionPass' is not defined
```

根因是 vllm 自身**平台守卫不一致**，与插件业务代码无关：

```
vllm/compilation/passes/pass_manager.py:41   该类的 import 被 is_cuda_alike()/is_xpu() 门控
vllm/compilation/passes/pass_manager.py:167  使用处完全没有守卫
```

插件设了 `kernel_config.ir_op_priority.rms_norm[0] = 'flagos'`（≠ `'native'`），
使 `enable_norm_fusion()` 返回 `True`，于是 `pass_config.fuse_norm_quant` 默认开启，
必然走到那个坏分支；而 `PlatformFL.is_cuda_alike() == False`，名字就是未定义的。
**只要开 compile 就会命中，与是否量化无关。**

> **不是一个名字的问题。** 盘点 `configure()` 引用的 `*Pass` 名字后发现：本 build 里
> 有 **16 个**类是"导入被守卫、使用无守卫"，`RMSNormQuantFusionPass` 只是当前配置
> 恰好第一个踩到的。换 `--compilation-config` 的开关顺序（`enable_sp`、
> `fuse_act_quant`、`fuse_rope_kvcache` …）就会踩到别的名字。所以修法必须是**通用**的。

**不能靠放开 import 修**：`rms_quant_fusion.py` 在**模块作用域**就引用
`torch.ops._C.*_fp8*` 与 `current_platform.fp8_dtype()`，而本机连 `vllm._C` 都没有
（启动日志刷 `Failed to import from vllm._C`），只会把崩溃提前到导入期。

#### 修法：插件内注入惰性 stand-in（不需要改 vllm）

新增 `vllm_fl/patches/post_grad_passes.py`，在 `register()` 与 `register_model()`
两个入口挂载（幂等）。做法是：

1. 从 `PostGradPassManager.configure` 的**字节码**里取出它引用的所有全局 `*Pass` 名字
   （递归进 `co_consts`，覆盖嵌套 code object）——这样 vllm 以后增删/改名都能自适应；
2. 把其中**本 build 未导入**的名字，绑定为一个 `VllmInductorPass` 子类的 **stand-in**；
3. 该 stand-in 的 `is_applicable_for_range()` 恒返回 `False` → **永不执行**，
   语义上等价于 vllm 自己"该平台没导入这个 pass"的意图。

继承真实基类（而非 `object`）是必要的：这样 `isinstance` 检查、Inductor 代码缓存要用
的 `uuid()`、以及 `__call__` 里的 `dump_prefix` 计数都不会被破坏。

实测：本机注入 **16 个** stand-in（`RMSNormQuantFusionPass`、`SequenceParallelismPass`、
`ActivationQuantFusionPass`、`AllReduceFusionPass`、各 `RocmAiter*`、`RopeKVCacheFusionPass` …），
5 个本 build 真有的类（`NoOpEliminationPass` / `PostCleanupPass` / `FixFunctionalizationPass` /
`UnsafeCloneEliminationPass` / `VllmIRLoweringPass`）保持原样。

**验证（决定性）**：把 vllm 还原为**原版**（与备份逐字节一致），只留插件补丁，官方
serve 命令行**一字不改** → `Application startup complete`，CUDA graph 35/35 捕获完成，
`/v1/models` 正常响应。即**正式评测环境无需触碰 vllm**。

> 排查期间曾临时改过 `site-packages/vllm/.../pass_manager.py`（备份
> `pass_manager.py.bak.<ts>`，**已还原**）。那个改法在正式评测里一定会丢，**不要采用**；
> 正解是上面的插件补丁。

排查手记：`configure_post_pass()` 是在 `torch.compile` 的 AOT 阶段、
`_initialize_kv_caches() -> determine_available_memory() -> profile_run()` 里被调用的。
所以判据很好用 —— **只要 server 打印出 `GPU KV cache size`，就说明 `configure()` 已通过**。

### 3.5 本机安装态 `vllm_fl` 比两个仓库都新 —— 改完要"同步文件",不要重装

本机 `site-packages/vllm_fl`（`0.3.0rc1.post1+g0b8c5863c`）与
`/workspace/vllm-plugin-FL`（`main`）和 `/workspace/vllm-plugin-FL-comp`
（`feat/ir-kernels-compile`）**都不一致**，安装态多出这些文件：

```
dispatch/backends/vendor/{gcu,kunlunxin}/       dispatch/config/{enflame,kunlunxin}.yaml
patches/{arm_cpu_gdn,exponential_compat,triton_kernel}.py
quantization/arm_cpu_w4a8.py                    worker/common_attention_metadata.py
```

安装态与 comp 仓库的差异是**双向**的：仓库有 `ops/ir_kernels.py` 等安装态没有的，
安装态也有仓库没有的。

> ⚠️ **因此不要用 `pip install` 重装来"应用改动"** —— 会丢掉安装态独有的文件。
> 本次的做法是：改动落在**仓库**（`git` 可追踪、可提交），再**逐个 `cp` 同步**受影响的
> 文件进安装态。改动只涉及 `vllm_fl/__init__.py` 与新增的
> `vllm_fl/patches/post_grad_passes.py` 两个文件，同步是安全的。

## 4. 官方口径与脚本

天数与沐曦的 serve 命令行**不一样**，天数**带** `--compilation-config`：

```bash
export VLLM_PLUGINS=fl
vllm serve /workspace/MiniCPM5-2B --port 9031 \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --served-model-name minicpm \
  --gpu-memory-utilization 0.85 --max-model-len 131072
```

全套评测（serve + benchmark + `math_500` + 门槛判定）已脚本化：

| 用途 | 脚本 |
|---|---|
| 天数 BI-V150 | `experiments/src/run_official_eval_iluvatar.sh`（经 `launch_official_eval_iluvatar.sh` 调用） |
| 沐曦 C500 | `experiments/src/run_eval_all.sh` / `run_eval_whitelist.sh` |

天数脚本相对沐曦版的四处差异：`/usr/local/bin` 取代 conda 环境、serve 带
`--compilation-config`、不动 `VLLM_FL_CUDAGRAPH_ONLY`（插件默认厂商列表为空，天数
没有 CUDAGraph-only 默认需要压掉）、按天数门槛判定。

## 5. 尚未在天数复测（沐曦篇的结论都还只是假设）

沐曦篇（`FINDINGS-mx.md`）里这些已量化收益，**在天数上一概未经验证**，做优化前应
逐条在本卡重测，不要照搬：

| 沐曦篇结论 | 在天数的状态 |
|---|---|
| FlagOS 融合算子白名单（曾是最大单笔收益） | ❌ **规则禁止**，不采用（§7.5 注） |
| `M` 的 autotune key 加上界（`align32_geometric`） | ✅ **本卡已独立发现并修复**（§7.4），端口自沐曦 |
| `flash_attn` 阻塞 D2H 移除（`2fc71f9`） | 未测 |
| decode attention `num_splits` 自适应（`4899f91`） | 未测 |
| `fused_add_rms_norm` 去掉 clone（`dc27f60`） | 未测 |
| M==1 `mm` 路由到 GEMV | 未测 |
| 4k 双峰现象与 20 s 停顿（§4） | ✅ 同构现象已复现（§7.4，P99 82 s） |
| 采样器 sort-free top-p v2（§10） | 未测 |

> 白名单那一行值得单独说明：它是沐曦篇最早、也是当时最大的一笔收益，但**赛事
> Forbidden 段禁止删除框架的算子选择逻辑**，因此对两张卡都是死路。沐曦后来
> （`c7cd0fc`）也把默认白名单去掉了 —— 因为 §7.4 那种内核层修法已经在根上
> 解决了同一个问题，白名单就没必要了。**这也解释了为什么 §7.4 才是正解。**

另外本卡的 **flagtune/cost-model 路径根本不可用**（§3.3），所以沐曦篇里围绕
autotune 停顿（§4.8、§4.10）的结论在 5.3.5 上也不一定同构 —— 本卡走的是
`LibTuner` 的 key 策略路径（§7.4），不是 flagtune。

## 6. 测量纪律

沿用沐曦篇的经验（`FINDINGS-mx.md` §5）与 `eval-gated-optimization`
skill，关键几条：

- **只用同会话交错 A/B**，跨会话比较在本仓库已经产出过一次假 `+7.44%`。
- 跑满 **≥3 个有效轮次**（`RUNS=4, SKIP_FIRST=1`），把离散度与均值一起报。
- **先确认真正加载的是哪份代码**：`flag_gems` 是静态 pip 安装（改
  `/workspace/FlagGems` 源码对运行时**零影响**），`vllm_fl` 视安装方式而定；
  用 `python -c "import flag_gems; print(flag_gems.__file__)"` 落实。
- 长任务一律后台 + `notify_on_output` 钩子，不要忙等（见 `bench-progress-report`）。
- 本机是共享机器，**不要杀别的会话的 server / benchmark**。

---

## 7. 首批实测（4k，**单轮**，2026-09-27）

> ⚠️ **这不是提交成绩**：官方协议是 `RUNS=4 / SKIP_FIRST=1`，取第 2~4 轮平均。
> 这里只跑了**第 1 轮**（官方会丢掉的那一轮），含冷启动（首次 triton 编译、
> autotune cache 冷）。用途是**建立本卡的第一批可信读数**并定位瓶颈。

命令：官方 §4 的 serve 口径 + `vllm bench serve`，4k = `4096/1024/conc 64/256 prompts`，
`--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'`（补丁见 §3.4）。

| 指标 | 本次 | 基线 | 门槛 ±1% | 判定 |
|---|---|---|---|---|
| **Total tok/s** | **1690.29** | 2028.01 | ≥ 2007.73 | ❌ −15.8% |
| **Mean TTFT** | **14341.37 ms** | 11573.53 | ≤ 11689.27 | ❌ +22.7% |

原始读数（256/256 成功、0 失败）：

```
Benchmark duration (s)      775.44
Total input tokens        1,048,576
Total generated tokens      262,144
Output token throughput     338.06 tok/s   (Peak 759.00)
Mean TTFT                 14341.37 ms
Median TTFT                3491.03 ms
P99 TTFT                  82178.57 ms
Mean TPOT                   167.82 ms
```

### 7.1 瓶颈在 decode，不在 prefill

反解时间去向（两阶段模型在**本 case** 是自洽的；§2 的 16k 才病态）：

- 纯 decode 能力 = `64 × (1000 / 167.82 ms)` ≈ **381 out-tok/s**
- decode 需 `262144 / 381` ≈ **688 s**，占总时长 775 s 的 **88.8%**
- 余 ~87 s 处理 `1.048M` 输入 → prefill ≈ **12k tok/s**（不慢）

**结论：优化重心应放在 decode 路径。** 这与沐曦篇的收益结构（融合算子 + 消除
autotune 停顿）方向一致，但**不是同一个瓶颈**，不可照搬结论。

### 7.2 信号一：decode 步进慢得反常（结构性问题）

`381 tok/s ÷ 64 并发` = 每秒仅 **5.95 个 decode step**。2B 模型 bf16 权重约 4 GB，
每步读一遍 → 折算带宽仅 **~24 GB/s**，**远低于这张卡应有的水平**。

即：**不是显存带宽瓶颈，而是每步固定开销**（kernel launch / triton 单算子性能 /
CUDAGraph replay 是否真在 decode 生效）在主导。这是与冷启动无关的结构性损耗，
是当前最大的一块可挖空间。

### 7.3 信号二：TTFT 长尾拖垮 Mean（而门槛卡的正是 Mean）

`Median 3.5 s` 但 `Mean 14.3 s`、`P99 82.2 s` —— 均值被长尾拖垮。已确认官方脚本
汇总行 `TTFT=` 取的是 **`Mean TTFT`**（`benchmark_throughput_serve.py` 的
`f"TTFT={r['Mean TTFT (ms)']}ms"`），所以**必须压 Tail，压 Median 没用**。

启动日志中同时出现：

```
RuntimeWarning: active Triton backend does not provide a replay benchmarker; falling back to event timing
DeprecationWarning: use_cuda_graph is deprecated; use benchmark_mode='replay' or benchmark_mode='event'
```

来自 FlagGems `libentry` 的 autotuner。沐曦篇记录过同类问题（flag_gems autotune 在
推理热路径造成 **~20 s 停顿**）。P99 82 s 的形态与"新 shape 触发 autotune 停顿"
**吻合 —— 已在 §7.4 证实并修复**。

### 7.4 根因已定位并修复：`M` 的 autotune key 无界（`align32_geometric`）

原假设（FlagGems autotune 在热路径造成长尾）**已证实，并找到确切机制**。

**证据链**

1. 调优结果落在 sqlite：`~/.flaggems/config_cache/TunedConfig_iluvatar_triton_3_6.db`（7.2 MB）。
2. 其中 bf16 `mm` 的 key 表有 **134 个互不相同的 `M`**：
   `1, 2, 4, 8, 16, 24, 32, 40, 43, 44, 45, 48, 49, 51, 52, 53, 54 … 2030, 2032, 2034, 2048, 8192`
   —— 相邻、不对齐、不重复，正是 chunked prefill 的任意 chunk 边界（`key_0` 即 `M`）。
3. 机制：`DEFAULT_STRATEGIES` 的 `"mm": ["align32", ...]` **只在
   `TuningMode.EXPANDED` 下被消费**，而 `_iluvatar/ops/mm.py` 的 `@libtuner`
   **既没传 `strategy=` 又以 `DEFAULT` 运行** ⇒ 退化成**恒等策略、按原始 `M` 建 key**。
   （`linear.py` 传了 `align32`，但 `align32` **无上界**，1..2048 就有 64 个 key。）
4. 空 DB 回放那 134 个真实 `M`：**14 次 autotune**，每次 **350–550 ms**（均值 ~430 ms）。
   即修复前 ≈ **57.6 s** 调优压在推理热路径上 —— 正对应 P99 TTFT 82 s 与
   Mean/Median 3.4 s 的 4.2× 落差。

**修法**（纯算子编译优化，不涉及算子派发选择）

天数自建 `_iluvatar/tuner_strategies.py` 提供 `align32_geometric`：`≤128` 保留
`align32` 的 32 粒度（decode / 小 batch 区间，tile 选择最敏感），之上转几何分桶。
只对 **`M` 维**生效，`N`/`K` 是静态模型维度，保持 `align32`。

| | identity（修复前） | `align32` | **`align32_geometric`** |
|---|---|---|---|
| 134 个真实 `M` → key 数 | **134** | 64（1..2048） | **14** |
| 热路径调优耗时 | ≈57.6 s | ≈27 s | **≈6.0 s** |
| 桶集 | 逐值 | 每 32 | `1 2 4 8 16 32 64 96 128 256 512 1024 2048` |

> **只是 key 被粗化，kernel 拿到的仍是真实 `M`** —— 与 `align32` 在 32 宽窗口内的
> 做法一致，不改变数值结果、不改变算子选择。落在
> `FlagGems-comp@b7d869247`，**未触碰任何 `_metax` 文件**（两张卡各自持有独立策略
> 模块，互不依赖）。

**端到端效果尚未复测**（下一步）。预期主要落在 Mean/P99 TTFT，Total tok/s 受益较小
—— 调优以停顿形式出现，吃延时不直接吃吞吐。

### 7.5 待验证假设（按预期收益排序）

| # | 假设 | 状态 |
|---|---|---|
| 1 | FlagGems autotune 在热路径造成长尾停顿 | ✅ **已证实并修复**（§7.4） |
| 2 | CUDAGraph 未在 decode 真正 replay（每步走 eager） | 待验证 |
| 3 | 每步固定开销主导 → 提高并发 / batch 收益大 | 待验证 |
| 4 | 沐曦的 `flash_attn` D2H / `num_splits` / `fused_add_rms_norm` | 待在本卡复测 |
| 5 | 采样器 sort-free top-p v2 | 待在本卡复测 |

> **白名单已排除**：`VLLM_FL_FLAGOS_WHITELIST` 虽能压掉停顿，但赛事 Forbidden 段
> 明确禁止「swapping operators with no actual operator optimisation, or deleting
> the framework's main operator-selection logic」，用即成绩作废。沐曦也已在
> `c7cd0fc` 里去掉默认白名单，改走 §7.4 这类内核层修法。

### 7.6 复现

```bash
export FRAMEWORK=/workspace/vllm-plugin-FL-comp
export EVAL_OUT=/root/bench_results/eval_iluvatar
bash experiments/src/launch_official_eval_iluvatar.sh   # 全量 4k+16k+正确性
```

单轮快速复现（本次用的方式）：手动 `vllm serve`（§4 参数）+ 单次 `vllm bench serve`。
产物在 `/root/bench_results/quick4k/`（`server.log` / `bench4k.log` / `RESULT.txt`）。

