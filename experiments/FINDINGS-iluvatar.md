# MiniCPM5-2B on 天数 BI-V150：环境与已验证状态

本文件是 [`FINDINGS-mx.md`](./FINDINGS-mx.md)（沐曦 MetaX C500）的**天数对应篇**。
区别很重要：两张卡的基线相差 **2.5×（4k）～7.7×（16k）**，沐曦篇里的收益数字、
根因结论、门槛都不能直接搬到这张卡上。

> **当前状态（2026-09-27 15:50 更新）：本卡已有首批 4k 实测，并已落地一处可量化的
> 收益（sort-free top-p：暖 DB 单轮 `Mean TPOT −6.21%` / `duration −7.85%`，§9.11），
> 但仍是单轮、非提交成绩（官方取第 2~4 轮平均）。** §1~§6 是**环境**与**已验证的事实**；
> §7~§11 是**按时间累积**的测量与更正。沐曦篇的收益结论见 §5 的复测状态表（多数已在
> §10 复测），未复测项不要照搬。
>
> ⚠️ **本文件的瓶颈归因被反复推翻过，正文里仍留有已被推翻的旧结论。** 阅读瓶颈/收益时
> 一律以下面的 **§0 更正账本**为准；正文中带「❗已被推翻」的段落只作历史记录。

---

## 0. 更正账本（瓶颈归因的最终状态，2026-09-27 15:50）

本文件的瓶颈结论经历了三层反复（§7~§8 → §10 → §11）。§1~§6、§9、§10.1 的
**环境/事实/收益**类内容仍然有效；**瓶颈归因**一律以下表为准，正文冲突处只作历史。

| 结论（出处） | 状态 | 依据 |
|---|---|---|
| §7.1「瓶颈在 decode，不在 prefill」，两阶段模型给 decode 占 **88.8%** | ❗**已推翻** | 分母把 `Mean TPOT` 当单步；`Mean TPOT` 含爬坡/排空/prefill 饥饿（§8.2.1、§9.9）。且 prefill 吞吐随 ctx 10.9k→3.8k tok/s 明显下降（§11）。 |
| §7.2「decode 慢是**每步固定开销**主导，不是带宽」 | ✅**成立** | 曾被 §8.2.1/§10.3 否定，又被 §11 的墙钟 ctx 扫描重新证实：单步 86/100/139 ms（ctx 512/2048/4096），`step ≈ 75–80 ms + KV/≈185 GB/s`，截距与 ctx 无关。 |
| §7.3 / §7.4「P99 TTFT 82 s = autotune 停顿，已由 `align32_geometric` 修复」 | ❗**归因已推翻** | §7.4.1：三个 27 s 停顿修复前后逐点一致；是 4 个波次的边界排队（`wait=0`、KV usage 同步下降）。 |
| §7.4「57.6 s 调优压在热路径 → 正对应 P99 82 s」 | ❗**归因错误**（机制与修法本身有效） | `align32_geometric` 只在**冷 DB** 有效（key 134→12，调优 57.6 s→6.0 s；§7.4、§7.4.2）。 |
| §7.5 表「autotune 热路径长尾 ✅已证实并修复」 | ❗**错误** | §7.4.1。 |
| §7.5 表「decode 受 KV 带宽限制、仅 16% 天花板、当前最大收益点」 | ⚠️**数字作废；"KV 带宽受限"的定性也不成立** | 16% 来自 §8.2 的错误分母；§11 重算 ≈11–19%（"带宽远未用满"成立），但主导项是 **~80 ms/步、与 ctx 无关**的固定开销，而非 KV 带宽。 |
| §8.2「有效带宽 89 GB/s、4 倍空间」 | ❗**数字与"KV 带宽受限"定性均作废** | 用 `Mean TPOT=167.66 ms` 当单步；真正的限制是固定开销（§11）。 |
| §8.2.1「matmul 占单步 49.8%、仅 88 GB/s，是最大单项」 | ❗**已推翻** | 该 88 GB/s 来自含「每次重建权重」的 `decode_op_bench.py`（§10.2 更正一）；`mm_bw.py` 实测 matmul ≈8.89 ms、≈447 GB/s。 |
| §9.9「纯解码步 = 90.97 ms（由 e2e 峰值反推）」 | ✅**与 §11 一致** | §11 独立墙钟扫描得 86–139 ms/步。 |
| §9.9 的 Amdahl 外推（`6% × 48% ≈ 2.9%`） | ❗**已推翻** | §9.11 实测 `Mean TPOT −6.21%`（该外推被实测打脸，见 §9.11 自我更正）。 |
| §10.2 更正三「e2e 645–768 tok/s，离线纯 decode 3007 tok/s，e2e 因 prefill 交织慢 4–5×」 | ❗**已推翻** | §11：离线纯 decode 实测 459–743 tok/s，与 e2e 同量级；「3007 tok/s / 21 ms」无法复现。 |
| §10.3「attention 86%、506 GB/s ≈ 天花板 90%、decode kernel 已到硬件尽头」 | ❗**已推翻** | (1) KV 体量算错 2×（用了 5.64 GB；ctx=4096 应为 11.27 GB），按其口径重算得 802 GB/s > 564 天花板，**自相矛盾**；(2) profiler 的 A/B 相减漏掉了 graph replay 的 kernel（其 decode attention 16.36 ms 本身就需要 >690 GB/s 才可能）；(3) §11 墙钟为 86–139 ms/步、总有效带宽仅 ~11–19% 天花板。 |
| §10.4「attention(86%) 是下一步重点、decode kernel 已到尽头」 | ❗**已推翻** | 优先查那 **~80 ms/步、与 ctx 无关**的固定开销，其次才是 attention/KV 的有效带宽（§11）。 |

> 读法：§7~§10 正文里凡与本表冲突的句子都是历史记录；本表 + §11 是当前口径。

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

### 3.5 改完必须 `pip install` 重装（**本节早期结论已更正**）

#### 现状（2026-09-27 14:57 重装后）

两个包都是**非 editable 拷贝安装**，`site-packages` 只是源码树的一份复制：

| 包 | 安装来源（`dist-info/direct_url.json`） | 方式 |
|---|---|---|
| `vllm_fl` | `file:///workspace/vllm-plugin-FL-comp` | 拷贝（`dir_info: {}`，无 `.pth`） |
| `flag_gems` | `file:///workspace/FlagGems-comp` | 拷贝（`dir_info: {}`，无 `.pth`） |

**因此：改完仓库源码后必须重装，否则 site-packages 里跑的还是旧代码。**

```bash
pip install --no-deps --no-build-isolation --force-reinstall /workspace/vllm-plugin-FL-comp
pip install --no-deps --no-build-isolation --force-reinstall /workspace/FlagGems-comp
```

重装后版本会带 git 描述，可据此确认落到了哪个 commit：
`vllm-plugin-fl 0.3.0rc1.post1+g9c4ef72cb.d20260927`、
`flag_gems 5.3.5.post1.dev11+gb7d869247`（仍是 5.3.5 系列，符合 §3.3 的版本要求）。

#### 更正：早期"不要重装、只 cp 同步"的说法是错的

早期观察到安装态多出下列 9 个**两个仓库都没有**的文件，据此推断重装会丢失它们：

```
dispatch/backends/vendor/{gcu,kunlunxin}/       dispatch/config/{enflame,kunlunxin}.yaml
patches/{arm_cpu_gdn,exponential_compat,triton_kernel}.py
quantization/arm_cpu_w4a8.py                    worker/common_attention_metadata.py
```

实测（14:57，重装前后 `diff -rq` 全量对比）**这 9 个文件重装后全部仍在** ——
pip 的卸载只删 `RECORD` 里记录的文件，而这些文件不在其中。
重装与备份的差异只有两处，均无关：`_version.py`（版本号）与
`metax.yaml`（仓库版覆盖了安装版）。

而且即使它们真被删掉，对天数也无影响：`arm_cpu_gdn` / `arm_cpu_w4a8` /
`common_attention_metadata` **零引用**；`exponential_compat` 只被
`vendor/gcu/patch.py` 引用；`patches/__init__.py` 是空文件，不会自动 import 它们。

> **结论：重装是安全且非破坏性的，应当作为标准流程。**
> 手动 `cp` 同步只适用于"来不及重装、先验证一下"的临时手段，且容易漏文件
> （`pip install` 会连带更新 `RECORD`／`dist-info`，手动 cp 不会）。

#### 复核清单（每次重装后）

```bash
# 1) 改动是否随重装进去
ls /usr/local/lib/python3.12/site-packages/vllm_fl/dispatch/backends/vendor/iluvatar/patches/
grep -c "from . import patches" /usr/local/lib/python3.12/site-packages/vllm_fl/dispatch/backends/vendor/iluvatar/__init__.py
grep -c "align32_geometric" /usr/local/lib/python3.12/site-packages/flag_gems/runtime/backend/_iluvatar/ops/mm.py
# 2) 补丁是否真的挂上
python3 -c "import vllm_fl.dispatch.backends.vendor.iluvatar,vllm.v1.sample.ops.topk_topp_sampler as t;print(t.apply_top_k_top_p.__name__)"
# 3) 运行时印记：日志里应有 'Registering vendor backends for current platform: iluvatar'
```

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

## 5. 沐曦篇结论在天数的复测状态

沐曦篇（`FINDINGS-mx.md`）里这些已量化收益，**在天数上多数已复测**（详见 §10）；未复测前
不要照搬：

| 沐曦篇结论 | 在天数的状态 |
|---|---|
| FlagOS 融合算子白名单（曾是最大单笔收益） | ❌ **规则禁止**，不采用（§7.5 注） |
| `M` 的 autotune key 加上界（`align32_geometric`） | ✅ **本卡已独立发现并修复**（§7.4）；但仅在**冷 DB** 有收益（key 134→12，调优 57.6 s→6.0 s），**不是 P99 长尾的原因**（§7.4.1、§0） |
| `flash_attn` 阻塞 D2H 移除（`2fc71f9`） | ❌ **N/A**：天数走 `TRITON_ATTN`，不用 `vllm_flash_attn`（§10.1） |
| decode attention `num_splits` 自适应（`4899f91`） | ❌ **N/A**：天数 `kernel_unified_attention` 无 `num_splits` 旋钮（§10.1） |
| `fused_add_rms_norm` 去掉 clone（`dc27f60`） | 🟡 **wash（<1%）**：单步 91 ms 把 clone 开销稀释（§10.1.1） |
| M==1 `mm` 路由到 GEMV | ⚪ **收益不复现**：天数 mm 在 M=1 已 300–614 GB/s，自写 GEMV 反而更慢（§10.1.3） |
| 4k 双峰现象与 20 s 停顿（§4） | ⚠️ 现象复现但**归因不同**：P99 82 s 是 4 波边界排队，非 autotune（§7.4.1、§0） |
| 采样器 sort-free top-p v2（§10） | ✅ **已落地、使能并在 e2e 量化**：暖 DB 单轮 `Mean TPOT −6.21%` / `duration −7.85%`（§9.11；算子内 8.5x vs 沐曦 2.6x，§9.1） |

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

### 7.1 ❗瓶颈在 decode，不在 prefill（本节比例已作废，见 §0）

> ❗**已被推翻。** 下面把 `Mean TPOT` 当成了纯解码步长，而 `Mean TPOT` 含爬坡/排空/
> prefill 饥饿，比真实解码步长大得多（§8.2.1、§9.9）。"优化重心在 decode 侧"的方向
> 大致成立，但 **88.8% 这个比例不成立**，"不在 prefill"也不准确——prefill 吞吐随 ctx
> 从 10.9k 掉到 3.8k tok/s（§11）。

反解时间去向（两阶段模型在**本 case** 是自洽的；§2 的 16k 才病态）：

- 纯 decode 能力 = `64 × (1000 / 167.82 ms)` ≈ **381 out-tok/s**　← 分母用错
- decode 需 `262144 / 381` ≈ **688 s**，占总时长 775 s 的 **88.8%**
- 余 ~87 s 处理 `1.048M` 输入 → prefill ≈ **12k tok/s**（不慢）

**结论（已修正）：优化重心在 decode 侧，但必须用「波次峰值吞吐/单步时间」度量，
不能用 `Mean TPOT`（§9.9、§11）。**

### 7.2 信号一：decode 步进慢得反常（结构性问题）— ✅ 结论仍成立

> 注：本节结论曾被 §8.2.1 / §10.3 否定（"其实是带宽受限、已到硬件尽头"），最终由
> §11 的墙钟 ctx 扫描**重新证实**：单步 86→100→139 ms（ctx 512/2048/4096），存在
> `~75–80 ms` 与 ctx 无关的截距。见 §0。

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
推理热路径造成 **~20 s 停顿**）。当时以为 P99 82 s 与"新 shape 触发 autotune 停顿"
吻合 —— ❗**该归因已被 §7.4.1 推翻**：三个 27 s 停顿在修复前后逐点一致，是 4 个波次
的边界排队（同刻 KV usage 同步下降、`wait=0`）。见 §0。

### 7.4 ❗（P99 归因已推翻）`M` 的 autotune key 无界（`align32_geometric`）

> ❗**标题里的"根因"只对冷启动 DB 成立，对 P99 长尾不成立（§7.4.1）。** 下面的 key
> 爆炸机制与修法本身有效，但其收益只在**冷 DB** 出现（§7.4.2）。

原假设（FlagGems autotune 在热路径造成长尾）**找到确切机制**（但"它造成 P99 长尾"
的因果后被 §7.4.1 推翻；机制本身成立）。

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
   即修复前 ≈ **57.6 s** 调优。❗注意：这只发生在**空 DB 的冷启动**；原文接着说它
   "正对应 P99 TTFT 82 s"是错的（§7.4.1）。

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

#### 7.4.1 更正：那三个 27 s 停顿**不是** autotune 造成的

§7.4 曾把 P99 TTFT 82 s 归因于"57.6 s 调优压在热路径"。**该归因是错的**，实测反例：

| | 65/256 | 129/256 | 193/256 |
|---|---|---|---|
| 修复前（`bench4k.log`） | 28.3 s/it | 27.9 s/it | 27.5 s/it |
| 修复后（`e2e4k/bench.log`） | 27.8 s/it | 27.5 s/it | 27.5 s/it |

**逐点几乎一致。** 若真是 autotune 停顿，修复后应显著变小。

真实原因：256 请求 / `max_concurrency=64` = **4 个波次**，每波解码完 1024 token 后
整波释放 KV，下一波 64 × 4096 token 同时 prefill。证据是同刻的
`GPU KV cache usage` **同步下降**（61.8% → 52.9%）且 `wait=0`（无排队）。
这三个点正好是波次边界，与 autotune 无关。

**因此 P99 TTFT 82 s 的主因是排队**：第 4 波请求要等前 3 波（≈ 3 × 160 s）才能开始。
这解释了 Mean TTFT 14.3 s 与 Median 3.5 s 的 4.1× 落差 —— 是**长尾排队**，不是调优停顿。

**`align32_geometric` 的价值仍然成立，但要用正确的方式度量**：
其效果只在 **DB 冷启动**时显现（已单点验证：空 DB 回放 134 个真实 `M`，
调优 57.6 s → 6.0 s；见 §7.4.2 冷启动 key 数 134 → 12）。
**暖 DB 的复跑看不到它的收益**，因此对照实验必须 **冷启动 to 冷启动**。

#### 7.4.2 冷启动交叉验证（清空 DB，2026-09-27 14:33）

上一轮清库验证的 server 在 `torch.compile` 后中途死掉（日志止于 14:07:13，无
`Application startup complete`），因此**结论不完整**。重跑（GPU 空闲 68MiB/0%）：

```
Application startup complete ✓        GPU KV cache size: 520,144 tokens
TunedConfig_iluvatar_triton_3_6.db

mm_kernel_6f2aa3b7…-45eed72 : 11 个 distinct M
    [1, 2, 4, 8, 16, 32, 64, 96, 128, 256, 2048]
linear_kernel_eece4cf6…-0f0  :  1 个 distinct M  [256]

合计 12  （修复前 134）
```

这 11 个值**逐一对上 `align32_geometric` 的设计桶**：≤128 取 32 的倍数
（1,2,4,8,16,32,64,96,128），>128 取 2 的幂（256, 2048）。即修复是**按设计**
生效的，不是碰巧少建了几个 key。**134 → 12，压缩 11 倍。**（本小节原编号 §7.7，
误置于 §9，现归位到 §7.4。）

### 7.5 假设状态（已按 §0/§11 更正）

| # | 假设 | 状态 |
|---|---|---|
| 1 | FlagGems autotune 在热路径造成长尾停顿 | ❗ **已推翻**（§7.4.1）：停顿是波次排队，不是 autotune |
| 2 | CUDAGraph 未在 decode 真正 replay（每步走 eager） | ❌ **已证伪**（§8.3：成功捕获 35 图，FULL_DECODE_ONLY） |
| 3 | 每步固定开销主导 decode | ✅ **成立**（§7.2 原始判断 + §11：`~80 ms` 截距与 ctx 无关） |
| 4 | decode 受 KV-cache 带宽限制、有效带宽仅天花板 16% | ⚠️ **数字作废，"KV 带宽受限"也不成立**：16% 来自错误分母；§11 重算 ≈11–19%（带宽确实远未用满），但限制项是 `~80 ms` 固定开销 |
| 5 | 沐曦的 `flash_attn` D2H / `num_splits` / `fused_add_rms_norm` | 已复测：N/A 或无头寸（§10.1） |
| 6 | 采样器 sort-free top-p v2 | ✅ **已落地并在 e2e 量化**（§9.1/§9.11：Mean TPOT −6.21%） |

> **白名单已排除**：`VLLM_FL_FLAGOS_WHITELIST` 虽据当时观察能压掉"20 s 停顿"，但赛事
> Forbidden 段明确禁止「swapping operators with no actual operator optimisation, or
> deleting the framework's main operator-selection logic」，用即成绩作废。沐曦也已在
> `c7cd0fc` 里去掉默认白名单，改走 §7.4 这类内核层修法。
> （另注：本卡的 P99 82 s 长尾已确认是**波次排队**而非停顿，与白名单无关，见 §7.4.1。）

### 7.6 复现

```bash
export FRAMEWORK=/workspace/vllm-plugin-FL-comp
export EVAL_OUT=/root/bench_results/eval_iluvatar
bash experiments/src/launch_official_eval_iluvatar.sh   # 全量 4k+16k+正确性
```

单轮快速复现（本次用的方式）：手动 `vllm serve`（§4 参数）+ 单次 `vllm bench serve`。
产物在 `/root/bench_results/quick4k/`（`server.log` / `bench4k.log` / `RESULT.txt`）。


---

## 8. 解码态瓶颈的定量定位（2026-09-27，单点微基准）

§7 只知道「decode 慢」，本节把它拆成可归因的数字。**所有结论均来自单点测量，
不依赖长跑。**

### 8.1 硬件带宽天花板（先立标尺）

`/root/bench_results/bw_probe.py`，256MB bf16 张量，取 50 次中位数：

| pattern | ms | GB/s |
|---|---|---|
| `copy_` 256MB（读+写） | 0.9523 | **564** |
| `mul_` 标量（读+写） | 1.0289 | **522** |
| 只读 reduce | 0.8045 | 334 |

→ **本卡可持续流式带宽 ≈ 560 GB/s**。这是一个必须记住的标尺：任何"内存受限"
的 kernel，只要没跑到 500+ GB/s，就还有数倍空间。

### 8.2 解码单步的流量账

> ❗**本节数字与结论均已作废（推导保留备查）。** 这里用 `Mean TPOT=167.66 ms` 当单步
> 时间，而它不是单步长（§8.2.1 已自行更正分母；§8.2.1 的 matmul 排序又被 §10.2
> 更正一推翻）。更关键的是，"decode 是 KV 带宽受限、单步可降 4 倍"这一**定性结论**
> 被 §11 推翻：单步有一个 `~80 ms`、与 ctx 无关的截距，总带宽利用率仅 ~11–19%。
> 当前口径见 **§0 与 §11**。

MiniCPM5-2B 真实结构（`config.json`）：**42 层**、hidden 2048、inter 6144、
16 attn heads / **2 kv heads**、head_dim 128、vocab 130560、bf16。

| 项 | 单步流量 | 算法 |
|---|---|---|
| KV cache 读取 | **11.0 GB** | 64 seq × 4096 ctx × 42 层 × 1KB/token/层 |
| matmul 权重读取 | **3.97 GB** | 42 × (2048·2560 + 2048·2048 + 2048·12288 + 6144·2048) × 2B |
| **合计** | **≈ 15.0 GB** | |

按 §8.1 的 564 GB/s 跑完 = **26.6 ms/步**。
实测 `Mean TPOT = 167.66 ms`（§7）→ **实际有效带宽仅 ≈ 89 GB/s，是天花板的 16%**。

分组核算（`native` 微基准，M=64）：

| 组 | 实测 | 占 TPOT |
|---|---|---|
| 4 个 matmul × 42 层 | 1.078 × 42 = **45.3 ms** | 27% |
| 其余（attention / KV / 采样 / 框架） | **≈ 122 ms** | 73% |

**结论：decode 是 KV-cache 带宽受限，不是算子个数受限，也不是 FLOPs 受限。**
`122 ms` 的 attention/KV 部分若能从 89 GB/s 提到接近 560 GB/s，单步可降 4 倍以上。

#### 8.2.1 更正：上面那段的分母错了

> ❗**本节结论（matmul 占 49.8%、仅 88 GB/s、是最大单项）已被 §10.2 更正一推翻**：
> 88 GB/s 来自"在计时区间内重建权重"的 `decode_op_bench.py`。`mm_bw.py` 实测 matmul
> ≈8.89 ms、≈447 GB/s（天花板 80%）。且"decode 是带宽受限"的定性也被 §11 推翻
> （固定开销主导）。"90.97 ms 是纯解码步"这点后被 §10 否定、又被 §11 重新支持（见 §0）。

§8.2 用 `Mean TPOT = 167.66 ms` 当单步时间，推出"非 matmul ≈ 122 ms"。**这是错的**：
Mean TPOT 是**整轮平均**，包含 prefill 饥饿期、爬坡与排空，比真实解码步长大 ~1.84 倍。

正确的单步时间取自波次内的**峰值吞吐**（64 并发）：

| | 峰值 tok/s | 单步 = 64/峰值 |
|---|---|---|
| baseline | 703.5 | **90.97 ms** |
| + sort-free top-p | 748.3 | **85.53 ms** |

按 90.97 ms 重算（batch 64、ctx≈4096）：

| 组件 | 时间 | 占单步 | 有效带宽 | vs 564 GB/s 天花板 |
|---|---|---|---|---|
| matmul（42 层） | 45.28 ms | **49.8%** | 3.97 GB / 45.28 ms = **88 GB/s** | **16%** |
| attn / KV / other | 39.24 ms | **43.1%** | 11.0 GB / 39.24 ms = **281 GB/s** | **50%** |
| sampler（修复前 → 后） | 6.46 → 0.77 ms | 7.1% → 0.9% | — | — |

**结论不变（decode 是带宽受限），但瓶颈排序要改：matmul 才是单项最大的一块（50%），
且同样只有 88 GB/s（16% 天花板）。** 优先盯 matmul 的 tile/配置，而不是 attention。

（说明：KV 部分 281 GB/s 的算法是 `11.0 GB / 39.24 ms`，与 §8.2 的 89 GB/s 差在分母；
39.24 ms 是"单步减去 matmul 与 sampler"的余量，含 attention 计算与框架开销，
所以 281 GB/s 是**上界估计**——真实 KV 读取效率只会更低。）

### 8.3 关键更正：FlagGems 的 host 开销**不在** decode 热路径上

`server_patched.log` 证据：`cudagraph_mode = CUDAGraphMode.FULL_DECODE_ONLY`，
`Capturing CUDA graphs (decode, FULL)` 成功捕获 **35 个图**。
→ decode replay 阶段**完全不经过 Python**，FlagGems 的 host 开销被彻底跳过。

所以此前估算的「+149 ms/token 派发税」**只适用于 prefill（eager/未捕获段）**。
❗原文"这恰好解释 §7.3 的 TTFT 长尾"已被 §7.4.1 推翻（长尾是波次排队，不是派发税）；
派发税仍然**不能用来解释 TPOT**。

### 8.4 FlagGems 派发开销本身（用于指导 prefill 优化）

`/root/bench_results/decode_op_bench.py`，batch=64：

`t_gem = c + t_kernel` 反解出每次调用的固定 host 开销 **c ≈ 0.118 ms**（add/mul/sum/sin/copy 五点一致）：
`add` 0.1302−0.0125、`mul` 0.1307−0.0124、`sum` 0.1204−0.0163、`sin` 0.1358−0.0185。

纯 host enqueue（不 sync，2000 次）：native `add` **0.0071 ms** vs FlagGems `add` **0.1248 ms**
→ **17.6×**。绕过 aten override 直接调 `flag_gems.ops.add` 仍为 0.1224 ms，
说明开销在 `pointwise_dynamic` 内部，不在 aten shim。

cProfile（4000 次调用，profiler 放大约 2×）：

| 热点 | 说明 |
|---|---|
| `libentry.py:1624 run()` | LibEntry autotuner **每次调用**重算 key/dispatch |
| `pointwise_dynamic.py:1497 prepare_args()` | 每次调用重新准备参数 |
| `triton/spec/iluvatar/.../jit.py:1119 __call__` | JITFunction 被重入 3×/call |
| `shape_utils.py:199 heuristics_for_tile_size()` | tile size **每次重算** |
| `tensor_wrapper.py:64 __init__` | 3×/call |

全部是"每次调用"而非"每个 shape"，与 §7.4 的 autotune key 爆炸同源。

**扣掉这 0.118 ms 后，FlagGems 的 kernel 反而更好：**

> ❗下表四个 mm 的绝对值来自 `decode_op_bench.py`，**含"计时区间内重建权重"的污染**
> （§10.2 更正一），只能看相对结论；权威 mm 带宽见 §10.1.2 / `mm_bw.py`（447 GB/s）。

| op | native | FlagGems 净 kernel | 结论 |
|---|---|---|---|
| `mm_gate` | 0.5295 | **0.351** | FlagGems 快 **1.5×** |
| `mm_down` | 0.3083 | **0.274** | FlagGems 快 1.13× |
| `mm_qkv` | 0.1310 | **0.122** | FlagGems 略快 |
| `mm_o` | 0.1089 | 0.125 | 基本持平 |

→ **FlagGems 的 Triton kernel 在天数上是持平到更快的；账面变慢纯粹是 host 税。**
优化方向因此是「消 host 税（惠 prefill）」+「提 kernel 带宽利用率（惠 decode）」两条。

### 8.5 顺带发现的真实兼容性缺口

- `flag_gems.ops.rms_norm` 在 `weight=None` 时崩溃（`rms_norm.py:316`
  直接 `weight.contiguous()`），而 `torch.nn.functional.rms_norm(x, shape, None, eps)`
  合法。建议在插件里守卫，或补一个无 weight 分支。
- `flag_gems.fused.silu_and_mul(A, B)` 签名是**两个张量**，不是 vLLM 的
  `silu_and_mul(x)` 单张量切半形式；天数**没有**专用实现，走通用
  `@pointwise_dynamic` 版（`_kunlunxin`/`_cambricon`/`_enflame`/`_tsingmicro`/`_arm` 才有手写融合 kernel）。

### 8.6 复现

```bash
cd /root/bench_results
python3 decode_op_bench.py --batch=64 --iters=50        # §8.4
python3 bw_probe.py                                     # §8.1
```

---

## 9. 采样器 sort-free top-p（§10 的沐曦项，已在天数落地）

### 9.1 结论

**单点验证通过，端到端加速 8.5x（rows=64），且不需要任何配置开关。**

| rows | vLLM 原路径 (ms) | 快路径 (ms) | 加速 |
|---|---|---|---|
| 1 | 0.610 | 0.252 | 2.4x |
| 64 | 6.458 | 0.765 | **8.5x** |
| 128 | 12.848 | 1.297 | 9.9x |
| 256 | 25.785 | 5.080 | 5.1x |
| 512 | 51.216 | 10.049 | 5.1x |
| 1024 | 102.522 | 20.029 | 5.1x |

**比沐曦的 2.6x 更好**，原因见 §9.3。

### 9.2 天数与沐曦的路径不同（这是关键）

| | 沐曦 | 天数 |
|---|---|---|
| Triton `topk_topp` | **编不过**（`make_ttgir` 失败） | **能用** |
| 原始路径 | `apply_top_k_top_p_pytorch`（纯 sort） | `apply_top_k_top_p_triton`（radix sort） |
| 被替换 | 整个 `apply_top_k_top_p` → pytorch | **仅 p-only 分支**，top-k 保留 Triton |

所以 Iluvatar 补丁**不能照抄沐曦**：沐曦把所有非快路径都指向 `apply_top_k_top_p_pytorch`，
在天数上会把 top-k 采样从 Triton 拽回 sort，是**倒退**。本补丁的回退目标是
`_ORIGINAL`（未修改的原函数），已用「k-only 输出与原始实现逐元素一致」验证。

### 9.3 为什么这里比沐曦快得多

沐曦替换掉的是 pytorch sort（rows=48 时 21.48 ms），天数替换掉的是 Triton radix sort
（rows=48 时 4.87 ms）—— 基数排序本身比全排序快，但**都不必要**。所以天数的
相对收益反而更大（8.5x vs 2.6x）。

另外沐曦在 rows=256 出现 L2 溢出反向（0.89x），因此加了 `_V2_MAX_ROWS=128` 之外的
行数开关。**天数上这个交叉点不存在**：rows 从 192 到 1024 稳定在 5.1x，
因为被替换的基数排序够慢，即使 `top_p_threshold` 内部溢出（53 GB/s vs 128 行内的
103 GB/s）也仍然大幅领先。`flag_gems` 内部的 `_V2_MAX_ROWS = 128` 已经自动选了
较优的子路径。

### 9.4 正确性

与 `apply_top_k_top_p_pytorch` 在同一份 logits、单次应用下对比：

| 指标 | 值 |
|---|---|
| mask IoU | **0.999194** |
| 保留质量 ref / fast | 0.950001 / **0.950081** |
| 逐行 `mass_err_max` | **0.00e+00**（每行都不少留） |
| kept_ref / kept_fast | 3019254 / 3021689 |

分桶把阈值放在决定性桶的**下沿**，所以保留集合只会等于或略大于参考（≥ p），不会更小。

### 9.5 两个测量陷阱（记录以免重犯）

1. `apply_top_k_top_p_pytorch` 会**原地改写入参**（`logits.scatter_`，docstring 明写
   "may be updated in-place"）。在同一张量上跑 26 次基准 → 保留 token 数
   `130560 → 47329 → 577` 逐次塌缩。
2. 若用**同一个**张量既取 mask 又算分母，被 mask 成 `-inf` 的项在 `exp()` 后也是 0，
   于是 `分子 ≡ 分母`，mass 恒等于 **1.0**。第一版 harness 就是这样得出
   `mass_ref=1.00000` 的假象。**必须在 pristine logits 上算质量。**

> 冷启 buffer 假设已实测排除：新分配张量直接测就是 `mass=0.95000`，无需预热；
> `_get_scratch_v2` 的常驻 scratch 在代码里显式 `zero_()`，无残留。

### 9.6 落地物

| 文件 | 说明 |
|---|---|
| `vllm_fl/dispatch/backends/vendor/iluvatar/patches/topk_topp_sampler.py` | 新增 |
| `vllm_fl/dispatch/backends/vendor/iluvatar/patches/__init__.py` | 新增 |
| `vllm_fl/dispatch/backends/vendor/iluvatar/__init__.py` | `from . import patches` |

开关：`VLLM_FL_TOPP_FAST=0` 关闭；`VLLM_FL_TOPP_FAST_VERIFY=N` 校验前 N 次；
`VLLM_FL_TOPP_FAST_STATS=<path>` 记录实际走的路径（避免"静默回退让精度验证失去意义"）。

**预期 e2e**：rows=64 时省 5.7 ms/step，相对 TPOT 167.66 ms 约 **−3.4%**。
未跑 e2e 验证，需在下一轮全量测试中确认。

### 9.7 复现

```bash
cd /root/bench_results
python3 topp_repro2.py          # §9.1 分路径计时 + 正确性
python3 topp_scaling.py         # §9.1 行数扫描 / 门限选择
python3 topp_patch_validate.py  # §9.2 回退保真 + §9.4 正确性 + 覆盖率
python3 topp_massref_probe.py   # §9.5 陷阱复现
```

### 9.8 真实服务进程内的定论证据（2026-09-27 14:35）

单点基准只能证明"算子快"，不能证明"补丁在服务里生效"。为排除**静默回退**
（补丁的守卫条件若不满足会退回原路径，成绩就白算），带
`VLLM_FL_TOPP_FAST_STATS` + `VLLM_FL_TOPP_FAST_VERIFY=3` 重启并观测：

补丁被导入的直接证据（vendor 注册会先执行父包 `__init__.py`）：
```
[INFO] [vllm_fl.dispatch] Registering vendor backends for current platform: iluvatar
```

覆盖率（由 EngineCore 自己写出，不依赖 atexit —— 该进程退出时会被 SIGKILL）：
```
[fl] topp_fast coverage: fast=4 fallback=0 other=4 enabled=True op=True
```
`fast=4` 快路径命中、**`fallback=0` 零异常回退**、`other=4` 为 `p=None`
（贪心/prefill）、`op=True` 表示 `flag_gems.top_p_threshold` 可用。

**真实 serving logits 上的在线校验**（不是合成数据）：
```
iou=1.00000  mass_ref=0.999999  mass_fast=0.999999  mass_err_max=0.00e+00
iou=1.00000  mass_ref=0.998651  mass_fast=0.998651  mass_err_max=0.00e+00
iou=1.00000  mass_ref=0.979870  mass_fast=0.979870  mass_err_max=0.00e+00
```
IoU = 1.00000，保留质量 6 位小数一致，逐行误差 0 —— 优于合成测试（0.999）。

dtype 关卡已核实可通过：`vllm/v1/sample/sampler.py:96` 在调用采样器**之前**
执行 `logits = logits.to(torch.float32)`，因此传入的必为 fp32；
shape 为 2D `(tokens, vocab)`；rows=64 ≤ `_MAX_ROWS=2048`。

### 9.9 「8.5x」到 e2e 只剩 ~6% —— Amdahl 核算

> 注：本节的「纯解码步 = 90.97 ms」曾被 §10.2 更正三否定（称真实纯 decode 为 21 ms），
> 但 **§11 的独立墙钟扫描（86–139 ms/步）重新支持本节**；§10.2 更正三作废（§0）。

8.5x 是**算子内部**加速比；decode 关心的是它**占单步多少**。用波次峰值吞吐反推单步：

| | 峰值 tok/s（64 并发） | 单步 = 64/峰值 |
|---|---|---|
| baseline | 703.5 | **90.97 ms** |
| + sort-free top-p | 748.3 | **85.53 ms** |

* 实测节省 **5.45 ms/步**；微基准预测 `6.458 − 0.765 = 5.69 ms` → **吻合 95.7%**
* top-p 占单步 `6.458/90.97` = **7.1%**（Amdahl 上限）
* 实际拿到 `5.45/90.97` = **6.0%** —— 已贴近上限（干掉了它 88% 的耗时）

单步预算（batch 64）：

```
matmul          45.28 ms   49.8%
attn/KV/other   39.24 ms   43.1%
sampler          6.46 ms    7.1%   ->  0.77 ms   0.9%
```

#### 为什么 `Mean TPOT` 几乎不动（167.82 → 175.85）

**`Mean TPOT` 不是解码步长**，它比纯解码步长大 ~1.84 倍：

```
纯解码步长  =  91 ms
Mean TPOT  = 168 ms              (1.84x)
整轮平均    = 262144 tok / 774 s / 64 = 189 ms
```

整轮 774 s 里只有约 **373 s（48%）** 在纯解码（262144/703.5）；其余 52% 是
4 段 prefill 饥饿期（各 60–90 s，gen 掉到 ~99 tok/s）、首请求爬坡 187 s、排空 ~100 s。

⇒ 6% 的纯解码改善摊到 `Mean TPOT` 只剩 `6% × 48% ≈ 2.9%`，再叠加
§9.10 记录的冷启动 DB 污染（+38 s）就完全被掩盖。

> **教训**：评估 decode 侧优化要用**波次峰值吞吐/单步时间**，
> 不要用 `Mean TPOT` —— 后者被 prefill 饥饿与爬坡/排空稀释了近一倍。

### 9.10 作废一次测量：851.76 s 那轮是冷启动 DB 污染，不可用

`/root/bench_results/e2e4k/`（2026-09-27 14:54）得到 `duration 851.76 s`、
`Output 307.77 tok/s`、`Mean TPOT 175.85 ms`，看似是回归。**实测是测量污染：**

* 该轮启动前我为了别的实验**清空了 autotune DB**（运行中 DB 仅 68 KB，见 §7.4.2），
  于是整轮都在边跑边 autotune，重新长出 1123 行 key。
* 初始爬坡首请求耗时 **187.9 s**（暖 DB 对照轮为 175.5 s）。

按实际运转条件重算，污染的代价约 **+38 s**（851.76 − 774.44 ≈ 77 s，
其中还包括 4 个波次因为参数微差产生的抖动）。**可作为对照的只有暖 DB 的
774.44 s（`quick4k`,未含 top-p）与 `e2e4k_v2`（暖 DB + top-p）。**

> **纪律**：autotune DB 的冷/暖状态是 decode 侧测量的**一等变量**。
> 冷启动对照必须冷 to 冷，暖对照必须暖 to 暖，且要在记录里写明当时 DB 大小。

### 9.11 权威暖 DB 对照：774.44 s → 713.66 s（2026-09-27 15:12）

暖 DB、4k 单轮（MiniCPM5-2B，256 prompts，64 并发，4096 in / 1024 out）的**唯一
可比对**的两次 run —— 两次同为暖 DB、同一套脚本：

| 指标 | 基线 `quick4k`（无 top-p） | `e2e4k_v2`（sort-free top-p） | Δ |
|---|---|---|---|
| Benchmark duration | **774.44 s** | **713.66 s** | −60.78 s（**−7.85 %**） |
| Output token throughput | 338.49 tok/s | 367.33 tok/s | +28.84（**+8.52 %**） |
| Peak output throughput | — | 768.00 tok/s | — |
| Mean TTFT | 14377.76 ms | 12639.57 ms | −1738.19（−12.09 %） |
| Median TTFT | 3400.17 ms | 3379.14 ms | −21.03（−0.62 %） |
| P99 TTFT | 82290.09 ms | 75219.97 ms | −7070.12（−8.59 %） |
| Mean TPOT | 167.66 ms | 157.25 ms | −10.41 ms（**−6.21 %**） |
| Median TPOT | — | 161.63 ms | — |
| Failed requests | 0 | 0 | — |

**补丁全程生效的直接证据**（`topp_stats.txt`，覆盖率随 batch 增长 100 % 命中）：
```
[fl] topp_fast coverage: fast=256  fallback=0 other=4 enabled=True op=True
[fl] topp_fast coverage: fast=1024 fallback=0 other=4 enabled=True op=True
[fl] topp_fast coverage: fast=4096 fallback=0 other=4 enabled=True op=True
```
`fast` 随并发 8→64 线性增长（256→4096），**`fallback=0`** 说明没有任何一次因守卫
不满足而退回 Triton/PyTorch 排序；`enabled=True op=True` 确认补丁已装进
site-packages 且 `flag_gems.top_p_threshold` 可解析。

**核心结论：`Mean TPOT −6.21 %` 与 §9.9 的 Amdahl 预测（纯解码步 −6.0 %）吻合到
0.2 个百分点。** 这是 sort-free top-p 在真实 e2e decode 侧生效、且收益已贴近
Amdahl 上限（吃掉该阶段 88 % 耗时）的**直接验证**。

#### 对 §9.9 外推的修正（诚实记录）

§9.9 曾外推：「6 % 的纯解码改善摊到 `Mean TPOT` 只剩 `6 % × 48 % ≈ 2.9 %`」。
**实测 `Mean TPOT` 改善 6.21 %，是该外推的 2.1 倍，外推被推翻。**

原因：该外推隐含「prefill 饥饿/爬坡结构两轮完全相同」。但本轮 `Mean TTFT`
同时 −12.09 %、`P99 TTFT` −8.59 % —— 即 **v2 这轮的 TTFT 侧本身也更顺**。
若按「饥饿不变」假设，`ΔMean TPOT` 应 ≈ `Δ纯解码 = 5.45 ms = 3.25 %`；实测
`10.41 ms` 是它的 1.9 倍，多出的部分来自 TTFT/调度侧，**并非 top-p 的直接贡献**。

⇒ `duration −7.85 %` 应拆为两段：
* **top-p 真实收益 ≈ −6 %**（由 `Mean TPOT` 佐证，与理论闭环）；
* 其余 **≈ −1.8 pp** 是 TTFT 侧波动（`Median TTFT` 几乎不动 −0.62 %，但
  `Mean/P99 TTFT` 明显改善），含 **run-to-run variance**，单轮无法归因。

> **局限与下一步**：单轮 `duration` 级别的差异尚不足以把 top-p 收益与调度
> variance 严格分离。若要给出 `duration` 级的可信区间，应**同暖 DB 连跑 3 轮取
> 中位数**（本轮已证 `Mean TPOT` 是更干净的 decode 侧指标，波动小、与理论闭环）。

---

## 10. 沐曦优化项在天数的逐条复测（2026-09-27）

按用户要求「沐曦上做了但天数未复测的，逐条在天数验证」，不改 vLLM 源码、不碰
沐曦专属代码。结论先行：**5 项里 2 项 N/A（天数无对应路径）、3 项经实测无头寸**，
且复测过程**推翻了本卡此前的三处瓶颈归因**（§10.2）。

### 10.1 复测矩阵

| 沐曦项 | 天数是否有对应路径 | 复测结论 | 依据 |
|---|---|---|---|
| `2fc71f9` flash_attn 去阻塞 D2H | ❌ **无** | **N/A** | 天数 `attention_backend()` 返回 `TRITON_ATTN`，不用 `vllm_flash_attn`（`iluvatar.py:480`）。该修改长在 `vendor/metax/impl/attention/flash_attn.py`，属沐曦专属代码 |
| `4899f91` decode `num_splits` 自适应 | ❌ **无** | **N/A** | 同上：`num_splits` 是 flash_attn_with_kvcache 的参数；天数的 `kernel_unified_attention` 只暴露 `VLLM_TRITON_ATTN_USE_TD`（本卡必须为 0），无 num_splits 旋钮 |
| `dc27f60` `fused_add_rms_norm` 去 clone | ✅ **有** | 🟡 **wash（<1%）** | 见 §10.1.1 |
| `3d05dfc97` mm 按阶段调 tile | ✅ **有** | ⚪ **无头寸（0.66%）** | 见 §10.1.2 |
| `a68024d21`/`a7a74aeed` M==1 GEMV | ✅ **有** | ⚪ **收益不复现** | 见 §10.1.3 |
| `5f0bb89aa` gate_up+silu 融合 | ✅ **有** | ⚪ **无头寸（0.22%）** | inductor 已自动融合 `triton_poi_fused_mul_silu_slice_0`=0.042 ms/step；中间张量 1.57 MB≈5.6 µs，可忽略 |

#### 10.1.1 `fused_add_rms_norm`：两种模式都 <1%

`/root/bench_results/fab_ab2.py`（同进程交错 A/B，分离 host 与 device）：

| 模式 | host µs/call | device µs/call | host/step（prefill） | dev/step（decode） |
|---|---|---|---|---|
| **split**（当前默认） | 166.63 | **5.43** | 14.00 ms | **0.46 ms** |
| fused（沐曦改为 split 前） | **133.16** | 15.70 | **11.19 ms** | 1.32 ms |

* **decode（device）split 胜 −0.86 ms/step ≈ −0.95%**（单步 91 ms）
* prefill（host）fused 胜 −2.81 ms/step
* 两者均 <1%。**沐曦单步 7 ms，天数单步 91 ms，同样的 clone 开销被稀释 13 倍** ——
  这就是沐曦 −2.6% 到这里变 wash 的原因。**维持 split 默认。**

#### 10.1.2 mm 按阶段调 tile：M=64 已到天花板 79–91%

`/root/bench_results/mm_sweep.py`，144 组 tile 扫描（`mm_sweep.log`）：

| 形状 | 当前 FlagGems | 扫描最优 | 最优 config | 天花板占比 |
|---|---|---|---|---|
| qkv (N2560 K2048) | 332 GB/s | 455 | BM64/BN128/BK64 | 81% |
| o_proj (N2048 K2048) | 376 | 483 | BM64/BN64/BK128 | 86% |
| gate_up (N12288 K2048) | 531 | 511 | BM64/BN256/BK32 | 91% |
| down (N2048 K6144) | 466 | 444 | BM64/BN64/BK128 | 79% |

即便全换最优 tile，合计也仅省 **≈0.6 ms/step（0.66%）**。**M=64（4k 的实际形态）
本就在天花板 8 成，没有沐曦那种头寸。**

#### 10.1.3 M==1 GEMV：收益不复现

`/root/bench_results/gemv_m1.py`：自写 GEMV vs FlagGems mm（M=1）：

| 形状 | native | FlagGems mm | 自写 GEMV | GEMV 带宽 |
|---|---|---|---|---|
| qkv | 33.7 µs | 34.3 | 47.7 | 220 GB/s |
| o_proj | 27.4 | 26.6 | 47.0 | 179 |
| gate_up | 102.7 | 82.0 | 81.6 | 617 |
| down | 63.2 | 72.5 | 127.6 | 197 |
| **合计×42** | 9.53 ms | **9.05 ms** | 12.76 ms | — |

**GEMV 比 FlagGems mm 慢（0.709x）**，沐曦的 1.12x 在天数**不复现**。原因：天数的
mm 在 M=1 已能跑 300–614 GB/s，而朴素 GEMV 的 grid 只有 `cdiv(N, BLOCK_N)` 个
CTA（V150 的 SM 数远多于沐曦），**并行度不够**。要点是：M=1 时才有的头寸
（qkv 67%、o_proj 58% 天花板），而 **4k conc-64 是 M≈64，用不上**。

### 10.2 三处方法论更正（复测的副作用，但比复测本身更重要）

#### 更正一：`decode_op_bench.py` 的 mm 计时含「每次重新生成权重」

```python
("mm_gate", 1, *same(lambda: torch.mm(x, bf16(H, 2 * I))),  (H, 2 * I)),
#                                    ^^^^ 每次调用都做一次 50 MB GPU randn
```

四个 mm case 的 lambda 都在**计时区间内重新生成权重**（`bf16(2048,12288)` =
50.3 MB 的 `randn`），测的是「指令级任务」而非流式读权重。因此
`op_bench_b64.log` 里 `mm_qkv=44 GB/s`、`mm_gate=107 GB/s` 全部偏低。

**直接实测**（`/root/bench_results/mm_bw.py`，权重在循环外）：

| 形状 | native | FlagGems | FlagGems 带宽 |
|---|---|---|---|
| qkv | 41.2 µs | 33.4 | **332 GB/s** |
| o_proj | 33.0 | 23.7 | **376** |
| gate_up | 115.0 | 98.3 | **531** |
| down | 68.6 | 56.3 | **466** |
| **×42 层** | 10.83 ms | **8.89 ms** | ≈447 GB/s |

#### 更正二：`§8.2` / `§8.2.1` 的 matmul 瓶颈排序是错的

§8.2.1 称「matmul 占单步 49.8%、仅 88 GB/s」——该 88 GB/s 直接来自上面被污染的
`decode_op_bench.py`。真实 matmul 是 **8.89 ms、≈447 GB/s（天花板 80%）**。

#### 更正三：`§9.9` 的「纯解码步 = 90.97 ms」把 e2e 峰值当成了纯 decode

> ❗**本更正已被 §11 推翻。** §11 的离线纯 decode（conc 64）实测只有 **459–743 tok/s**
> （ctx 512/2048/4096），与 e2e 的 645–768 tok/s **同一量级**；文中的「3007 tok/s /
> 21 ms」无法复现，故"e2e 比纯 decode 慢 4–5×、主因是 prefill/decode 交织"作废。

§9.9 用 e2e 的波次峰值吞吐（703.5 tok/s）反推「纯解码步 = 90.97 ms」。但 e2e 峰值
**含 prefill 干扰**。实测纯 decode：

| 环境 | generation throughput | 单步 |
|---|---|---|
| 离线纯 decode（conc 64 / ctx 4096） | **3007 tok/s** | **21 ms** |
| e2e 4k 服务（conc 64） | **645–768 tok/s** | **83–99 ms** |

**同样的并发与上下文，e2e 比纯 decode 慢 4–5 倍。** ❗（该句已被 §11 推翻；见上）

4k 场景 256 请求连续到达、`max_concurrency=64`，于是 prefill 持续与 decode 交织
（prefill 1.05 M tok vs decode 262 k tok，4:1），decode 只能分到一部分 GPU。

⇒ ❗**本结论作废。** §11 显示离线纯 decode 与 e2e 同速，"4k 的主因是 prefill/decode
干扰"没有证据支持。

### 10.3 ❗真实 decode 单步预算（本节的 18.99 ms 是 profiler 漏计，已作废）

> ❗**本节核心数字全部作废**（方法论保留备查）：
> 1. **KV 体量算错 2×**：本节用 `5.64 GB`（ctx=4096、64 seq、42 层），漏了 2 个 KV head。
>    正确是 `2 kv_heads × 128 dim × 2 B × 2 (K,V) × 42 × 64 × 4096 = 11.27 GB`。按其自身
>    口径重算：`(11.27+3.97)/18.99 ms = 802 GB/s > 564` 天花板 —— **自相矛盾**。
> 2. **profiler 的 A/B 相减漏掉了 graph replay 的 kernel**：其 decode attention 被算成
>    16.36 ms，而读取 11.27 GB 需要 **>690 GB/s** 才可能，超过硬件天花板。
> 3. **墙钟反证（§11）**：同并发、ctx 512/2048/4096 实测 **86 / 100 / 139 ms/步**，是
>    18.99 ms 的 4.5–7×；ctx=4096 总流量 15.2 GB/139 ms ≈ 109 GB/s ≈ 天花板 19%。
> ⇒ "attention 86%、带宽 90%、decode kernel 已到硬件尽头"**不成立**。

`/root/bench_results/prof_decode2.py`：同进程引擎（`VLLM_ENABLE_V1_MULTIPROCESSING=0`，
否则 V1 的 EngineCore 子进程让 profiler 看到 0 个 kernel），**两次 profile 相减**
（A: max_tokens=1 只有 prefill；B: 多出 64 个 decode step）。conc=64、ctx=4096、42 层：

| kernel | ms/step | 占比 |
|---|---|---|
| **`kernel_unified_attention`** | **16.361** | **86.13%** |
| `linear_kernel` | 1.151 | 6.06% |
| `mm_kernel` | 1.081 | 5.69% |
| `_to_copy` / `argmax` / `add` / `fused_mul_silu` / `rms_norm` / `reshape_and_cache` 等 | ≈0.4 | ≈2% |
| **合计** | **18.99** | 100% |

粗分类：**attention/kv 86.19%、其余 12.26%、copy 0.77%、reduce/sampler 0.46%、
norm/act 0.32%。**

* **真瓶颈是 decode attention（86%），不是 matmul（合计 12%）。** ❗（86% 这个占比建立在漏计的 18.99 ms 上，作废；见本节顶部注）
* ~~KV 5.64 GB + 权重 3.97 GB = 9.61 GB / 18.99 ms = **506 GB/s ≈ 天花板 90%**~~
  ❗**算式有误**：KV 应为 11.27 GB，重算为 802 GB/s（> 564 天花板，不可能）。所谓
  "decode kernel 层面已接近硬件尽头"不成立；§11 实测总带宽仅 ~11–19% 天花板。
* ~~由此也推翻 §8.2.1 的「decode 是带宽受限、只有 16% 利用率、还有 4 倍空间」：
  那是分母（91 ms）错了，真实单步 19 ms 时带宽已到 90%。~~
  ❗**本条作废**：91 ms 的分母没错（§11 支持），19 ms 才是漏计值。

### 10.4 结论与下一步

1. **沐曦的 5 项在天数要么 N/A，要么无头寸**（§10.1）。原因统一：天数单步
   ~86–139 ms 远慢于沐曦的 7 ms，同样的固定开销被稀释。
2. ❗**本节的瓶颈归因（matmul 12% / attention 86% / 纯解码步 19–21 ms）已被 §11 推翻**，
   见 §0。当前口径：单步 ~86–139 ms（随 ctx），`~80 ms` 与 ctx 无关，总带宽利用率 ~11–19%。
3. **下一步优先级（已按 §11 修正）**：
   * **先拆那 `~80 ms/步`、与 ctx 无关的固定开销**（是 host 派发、哪几个 kernel、还是
     graph replay 没命中）——这是 4k 场景的最大单项。
   * 其次才是 attention/KV 的有效带宽（ctx=4096 时仅约 19% 天花板）。
   * ~~attention（86%）改后端~~、~~prefill/decode 交织来自调度~~ 均基于作废的
     18.99 ms，不要据此排期。
4. **复现入口**：`/root/bench_results/{fab_ab2.py, mm_bw.py, mm_sweep.py, gemv_m1.py,
   prof_decode2.py, ctx_scale.py}`，产物同目录。

---

## 11. ctx 扫描：decode 步长的墙钟反证（2026-09-27 15:47）

`/root/bench_results/ctx_scale.py`（产物 `ctx_scale.log`）。**动机**：§10.3 的 profiler
A/B 相减可能漏掉 graph replay 的 kernel，故改用**离线 `LLM.generate` 墙钟**独立测量。
conc=64，每点用 `max_tokens=257` 与 `max_tokens=1` 两次相减取 256 步；同一进程、同一
`cudagraph_mode=FULL_DECODE_ONLY` 配置、无 profiler。

| ctx | prefill s | prefill tok/s | **decode ms/step** | decode tok/s | KV GB | 总流量 GB | 有效带宽 GB/s | 天花板占比 | DRAM floor ms |
|---|---|---|---|---|---|---|---|---|---|
| 512  | 2.99  | 10969 | **86.15**  | 743 | 1.41  | 5.37  | 62  | 11% | 9.53  |
| 2048 | 21.46 | 6109  | **100.40** | 637 | 5.64  | 9.60  | 96  | 17% | 17.02 |
| 4096 | 69.60 | 3766  | **139.29** | 459 | 11.27 | 15.23 | 109 | 19% | 27.02 |

（总流量 = KV + 权重 3.96 GB；有效带宽 = 总流量/单步；天花板 564 GB/s 见 §8.1。
KV 按 `2 kv_heads × 128 dim × 2 B × 2 (K,V) × 42 layers × 64 seq × ctx` 计。）

**结论**

1. **单步时间随 ctx 上升，但有一个 ~75–80 ms 的截距**。三点线性拟合
   `step ≈ 75 ms + KV / ≈185 GB/s`。即 4k 场景下**过半的单步时间与上下文无关** ——
   §7.2 的原始判断（每步固定开销主导）成立，§10.3 的"attention 86%、已到硬件尽头"
   不成立。
2. **即便只看 ctx 相关部分，带宽也没跑满**：ctx 512→4096 的边际 KV 带宽只有
   ~185 GB/s；总有效带宽 62→109 GB/s，是 564 GB/s 天花板的 **11–19%**。
3. **e2e 并不比离线纯 decode 慢一个量级**：离线 459–743 tok/s，e2e 峰值 645–768 tok/s
   同一量级 ⇒ §10.2 更正三里"离线 3007 tok/s、e2e 慢 4–5×"作废。
4. prefill 吞吐随 ctx 从 10.9k 掉到 3.8k tok/s，所以 §7.1"不在 prefill"也要收回。

**方法学保留**：本测的墙钟包含 host/调度，而 §10.3 的 profiler 只覆盖被 CUPTI 记到的
device kernel。两者都不完美，但 §10.3 的 device 数字**自相矛盾**（会推出 >564 GB/s），
所以当前以本节的墙钟口径为准。

**下一步**：把 ctx=512 时的 86 ms 单步拆开（是否 host 派发、哪些 kernel、graph replay
是否命中），这比继续调 attention 更可能拿到 4k 的实际收益。

---

## 12. CPU+GPU 联合 profile：GPU 只占 12–20%，且发现「进程内 vs 多进程」的巨大分歧（2026-09-27 15:57）

`/root/bench_results/prof_cpu.py`（产物 `prof_cpu.txt`）。**动机**：直接回应 §11 的
「下一步」——把单步拆成 GPU / host 两侧。方法：**两次 `LLM.generate` 相减**
（A: `max_tokens=1` 只 prefill；B: `max_tokens=201`），并对 A、B **各做一次
CPU+CUDA 联合 profile**，再按同样方式相减，得到**纯 decode 的每步 CPU/GPU 分解**。
conc=64、ctx=1024、200 步。

| 指标 | 每步 |
|---|---|
| **wall** | **34.06 ms** |
| **GPU kernel（CUPTI device_time）** | **4.29 ms（12.6%）** |
| CPU self-time（profiler 放大约 2×，§8.4） | 59.59 ms |

纯 decode 的 per-step 热点：

| 函数 | 调用/step | CPU ms/step | GPU ms/step |
|---|---|---|---|
| **`cudaEventSynchronize`** | **2.0** | **23.9** | 0 |
| **`aten::index`** | **2.0** | **29.4** | 0.011 |
| `aten::mm` | 0.8 | 2.72 | 0.188 |
| `aten::linear` | 1.0 | 0.93 | 1.131 |
| `aten::copy_` | 12.1 | 0.94 | 0.037 |
| `vllm::unified_attention_with_output` | 0.2 | 0.071 | **0.413** |
| `cudaGraphLaunch` | 1.0 | 0.058 | — |

**结论**

1. **GPU 只占整步 12.6%（ctx=1024）**。结合 §11 的 wall 口径，`kernel_unified_attention`
   即便归零，收益上限也只有 **~12–16%**。**§10.3「attention 是 86% 瓶颈」的口径错误
   至此被独立复现（§11 由自相矛盾推出，本节由 CPU/GPU 分解直接量出）。**
2. **每步有 2 次强制 host 同步**（`cudaEventSynchronize` 2/step、23.9 ms/step）与
   **2 次高代价 `aten::index`（29.4 ms/step，疑似每步一次 D2H 索引）** —— 这两项
   合计就超过了整步 wall time，是「~80 ms 固定开销」的第一批候选。
3. **图外 kernel 仍多**：`cudaGraphLaunch` 仅 200 次（1/step），而 `cuLaunchKernel`
   在这一时段有 7919 次（≈39/step）——大量 kernel 仍走图外 eager 启动。

### 12.1 未决：本节的 34 ms/step 与 §11 的 ~93 ms/step（ctx=1024 插值）相差 2.7×

两次测量**用的是同一套墙钟减法**，差异只可能在运行形态。已知唯一显著区别：

* §11 `ctx_scale.py`：**默认多进程引擎**（V1 的 EngineCore 在子进程）
* 本节 `prof_cpu.py`：**`VLLM_ENABLE_V1_MULTIPROCESSING=0`**（进程内引擎，
  为了让 CUPTI 能看到 kernel —— §10 早期正是因多进程而拿到 0 个 kernel）

⇒ **最可能的解释：约 60 ms/步 的固定开销出在「多进程引擎的调度/IPC 路径」上**，
而进程内引擎没有这笔开销。若成立，这与 §11「~75–80 ms 与 ctx 无关的截距」指向同一
处，且**它是 4k 场景的最大单项**。

**这必须用一次 A/B 落实**（同一脚本、同一 ctx，只切
`VLLM_ENABLE_V1_MULTIPROCESSING=0/1`），因为：
* 若成立 ⇒ 优化目标应是**引擎的步间调度/IPC**，与 attention、matmul 都无关；
* 若不成立 ⇒ 说明是别的隐藏变量，需继续二分。

**在此之前，不要引用本节或 §11 的绝对 ms/step，只采信两条定性结论：**
**(a) decode 是 host/launch 受限，GPU 只占 12–20%；(b) 存在一个与 ctx 无关、
量级 ~60–80 ms/step 的固定开销。**

> **教训（第三、四条，同源）**：①「某 kernel 占 GPU 时间的比例」≠「占整步 wall 的比例」；
> ②同一份墙钟减法，**引擎运行形态（进程内/多进程）本身可能就是一个数倍因子**，
> 比较任何 decode 数字前必须先对齐它。

---

## 13. 定论：§11 的「~75–80 ms 固定开销」是冷启动 artifact；真瓶颈是 KV 带宽 200 GB/s（2026-09-27 16:20）

本节收束 §10–§12 的反复。三条独立证据指向同一个数字。

### 13.1 三条独立测量锁定 ctx=1024 的单步 ≈ 34 ms

| 方法 | 脚本 | ctx=1024 单步 |
|---|---|---|
| 引擎形态 A/B | `mp_ab.py` | **34.03**（3 轮 34.03/34.03/34.12） |
| CPU+GPU 联合 profile | `prof_cpu.py` | **34.06** |
| 修正版 ctx 扫描 | `ctx_scale2.py` | **33.36** |

三者吻合到 ±1 ms，**这是当前唯一可信的 decode 单步口径**。

### 13.2 §11 的 86–139 ms 是冷启动污染（已复现 artifact）

`ctx_scale.py` 的流程是「用 `max_tokens=1` 预热 → 计时 `max_tokens=1` → **计时
第一次 `max_tokens=N+1`**」，于是预热只覆盖了 prefill，**多步解码的首次运行（含
autotune）被整段计入**。`ctx_scale2.py` 同样输出 cold 与 warm 两列以暴露该 artifact：

| ctx | cold ms/step（§11 的口径） | **warm ms/step（真值）** | warm tok/s | KV GB | 有效带宽 GB/s | 天花板% |
|---|---|---|---|---|---|---|
| 512  | 125.74 | **24.80** | 2580 | 1.41  | 217 | 38% |
| 1024 | 115.36 | **33.36** | 1918 | 2.82  | 203 | 36% |
| 2048 | 256.89 | **49.73** | 1287 | 5.64  | 193 | 34% |
| 4096 | （§11: 139.29） | 待补 | — | 11.27 | — | — |

**冷/暖差 3.5–5.2 倍**。⇒ §11 的「~75–80 ms 与 ctx 无关的截距」**不成立**，
它与 ctx 无关恰恰是因为它是**一次性**的冷启动成本。

### 13.3 多进程引擎不是原因（假设已自我证伪）

`mp_ab.py` 同一脚本仅切 `VLLM_ENABLE_V1_MULTIPROCESSING`：

```
MP=0 (进程内)  : 34.03 / 34.03 / 34.12 ms/step
MP=1 (多进程)  : 33.73 / 33.90 / 33.94 ms/step
```

**无差异。** §12.1 的「~60 ms 出在多进程调度/IPC」假设**作废**。

### 13.4 正确的解码模型

三点线性拟合（warm）：

```
step(ctx) ≈ 16.5 ms (与 ctx 无关) + KV_bytes / ~200 GB/s
```

* ctx=4096 外推：`16.5 + 11.27 GB / 200 GB/s = 82.9 ms` ⇒ **772 tok/s**，
  与 e2e 4k 实测峰值 **768 tok/s** 精确吻合 ⇒ 模型闭合。
* §10.3 的 CUPTI device 时间（4.29 ms @ctx=1024，12.6%）与本节墙钟差 8 倍，
  说明 **CUPTI 在 graph replay 路径上系统性漏计**，device 占比不可用。
  **一律以墙钟为准。**

### 13.5 结论：唯一的大鱼是 KV 读取带宽（200 → 564 GB/s）

| 分量（ctx=4096） | 现状 | 跑满 564 GB/s | 可省 |
|---|---|---|---|
| KV 读取 | 56.4 ms（@200 GB/s） | 20.0 ms | **−36 ms** |
| 与 ctx 无关的固定部分 | 16.5 ms | 未知 | 待查 |
| **单步** | **≈83 ms** | ≈40 ms | **−52%** |

⇒ **优先级彻底改写**：
1. **KV/attention 的有效带宽 200 → 564 GB/s**（唯一 ≥40% 的头寸）。**这正是
   沐曦 `num_splits` 那条路径想解决的问题** —— 沐曦在天数 N/A，但**问题本身在天数
   真实存在**，只是位置不同：不是「固定归约开销」，而是 **KV 读取只有 35% 带宽**。
   做法应是在插件侧提供**并行度更高的 attention 实现**（16 SM、2 kv_heads 的
   grid 只有 ~128 CTA，且 `block_table` 是间接寻址），而不是改 vLLM。
2. **16.5 ms 的固定部分**（ctx=512 时它占 66%）—— 来源未定，待用「仅 decode、
   无反算」的最小 replay 复现来拆。
3. 沐曦其余各项（GEMV / tile / fused_add / flash_attn D2H / num_splits）**维持
   §10.1 结论：N/A 或无头寸**。

### 13.6 复现入口（本节）

`/root/bench_results/{ctx_scale2.py, mp_ab.py, prof_cpu.py}`；产物 `ctx_scale2.log`、
`mp_ab.log`、`prof_cpu.txt`。

> **教训（第五条，最终版）**：**任何 decode 数字都必须先预热多步解码路径再计时**。
> 本仓库因「冷启动未预热」已产生两次假结论（§9.10 的 851 s、§11 的 75–80 ms 截距），
> 且每次都会把优化方向带偏 —— 第一次指向 autotune，第二次指向「调度/IPC」，
> 而真因都是冷启动。

---

## 14. 两个定向问题的结论：sync 不可去、attention 默认值是错的（2026-09-27 16:47）

### 14.1 Q1「cuda 同步的开销能优化掉吗」→ **不能靠去掉同步，但要靠减少同步 + 保证 graph 覆盖**

**根因（源码级确证）。** 每步恰好 2 次 `cudaEventSynchronize`，实测 147 events /
64 steps = 2.3，与源码一一对应：

| 同步点 | 位置 | 触发条件 |
|---|---|---|
| `transfer_event.synchronize()` | `vllm/v1/worker/gpu_model_runner.py:7516`（`_to_list()`） | **始终** |
| `prepare_inputs_event.synchronize()` | `vllm/v1/worker/gpu_model_runner.py:3763`（`synchronize_input_prep()`） | 仅 `async_scheduling` |

`async_scheduling` 在 `vllm/config/vllm.py:1004` 默认解析为 `True`。

**为什么不能去掉。** ①D2H 载荷只有 `64 × int32 = 256 B` —— 是**纯延迟无带宽**，
去掉它等于让调度器不知道哪些序列已生成；②`prof_cpu` 的 `cudaEventSynchronize`
23.88 ms/step 是**会计假象**：同一次测量的 `unaccounted = wall − GPU − CPU =
−29.82 ms` 证明 CPU 自时间与 GPU 时间大幅重叠，`Synchronize` 只是**替 GPU 尾部背账**，
**不可与墙钟相加**；③`mp_ab` 已证引擎形态无关，`async ON = 34.01 ms/step`。

**真正可优化的两条（有实测支撑）：**

| 手段 | 收益 | 证据 |
|---|---|---|
| **保证所有形状命中 CUDA graph** | **~67 ms/step** | eager `101.02` vs graph `34.01` |
| `async_scheduling=False`（少 1 次 sync） | 待定（OFF 臂卡在 autotune，未取到） | — |

**决定性修正**：真凶是**部分形状未进 graph**。§11 那个「与 ctx 无关的 ~75 ms 截距」
以及 `ctx_scale` 的 125/256/139 ms 冷启动点，**本质都是退回了 eager 路径**
（eager 实测 101 ms/step，同量级）。这解释了为什么它对 ctx 完全不敏感。

> ⚠️ **两个 profiler 在此平台上都不可信**，务必交叉验证：
> * `prof_cpu`（graph 模式）：attention 0.4 ms/step ⇒ 推得 6.8 TB/s > 564 上限 ⇒ **漏计**；
> * `attn_gpu`（eager 模式）：内核和 = wall 的 **302%** ⇒ **重复计数**（profile 窗口含 prefill）。
> **结论：attention 占比必须用 CUDA events 直测 kernel（§14.3）。**

### 14.2 Q2「天数专属的高并行 attention kernel」→ **不必新写，但默认参数是错的**

**① vLLM 硬编码的 `NUM_PAR_SOFTMAX_SEGMENTS=16` 对 16 SM 是错的。**

`vllm/v1/attention/backends/triton_attn.py:55` 写死为 16。实测段数越多越慢：

| ctx | 2D 核 | segm=16（默认） | **segm=1（最优）** | 裸读对照 |
|---|---|---|---|---|
| 1024 | 24.20 | 21.74 | **17.24** | 9.27 |
| 4096 | 94.37 | 71.01 | **64.71** | 32.27 |

⇒ `segm 16→1` 省 **−4.5 ms/step（ctx=1024，−13%）**、**−6.3 ms（ctx=4096）**。
`segm=3/6` 因非 2 的幂在 Triton 编译期失败（`tl.static_assert`）。

**①b 端到端验证 + 数值等价性：均已通过。**

```
① 端到端（同时臂对比，各 3 轮，ctx=1024 conc=64）
segm=16 默认 : 34.01 ms/step  (33.99 / 34.01 / 34.05)
segm=1  补丁 : 30.39 ms/step  (30.35 / 30.39 / 30.40)
差值         : −3.62 ms/step = −10.6%   ← 可复现

② 数值等价性（kernel 级，含 fp32 稠密参考实现）
                    segm=1 vs 16      segm=1 vs fp32ref   segm=16 vs fp32ref
  ctx=1024          max_abs 6.10e-05   max_abs 3.70e-05    max_abs 3.90e-05
  ctx=4096          max_abs 3.05e-05   max_abs 2.00e-05    max_abs 1.93e-05
```

**`6.10e-05` 恰是 bf16 的一个 ULP**（量级 ~0.03 处）⇒ 与默认值**同级精度**；
且 `segm=1` 对 fp32 参考的误差**不劣于**（ctx=1024 时甚至优于）默认值。
`segm=1 vs 2D 路径` 在 ctx=4096 的 `mean_abs` 仅 `3.8e-07` ⇒ 两者实质等价。
**结论：可安全上线。**

**落地方式（不改 vLLM 源码）**：`NUM_PAR_SOFTMAX_SEGMENTS` 在
`TritonAttentionMetadataBuilder.__init__` 里以模块全局读取，因此在**引擎构造前**
执行 `triton_attn.NUM_PAR_SOFTMAX_SEGMENTS = 1` 即可生效。插件侧应在
`IluvatarBackend` 的 `patch_*` 系列（`iluvatar.py` 模块级，随 import 执行）中
加入该赋值，即与现有 `patch_triton_*` 同一位置、同一形式。

**②「高并行」machinery 已存在，无需新写。** `triton_unified_attention` 自带
**2D/3D 双核**，3D 核即并行 softmax + LSE 归约。`num_heads_kv=2` ⇒
`seq_threshold_3D = MIN_LAUNCH_GRID_SIZE_2D(128) // 2 = 64`；`use_3d` 条件为
`num_seqs <= 64`，故 **conc=64 时 3D 路径已激活**。

**③ 真正的天花板在访存模式，不在并行度。**

```
attention 实际      : 130–174 GB/s
同字节裸读（对照）  : 304–349 GB/s     ← 差 ~2x
```

即使 `segm=1` 也只到 163（ctx=1024）/174（ctx=4096）GB/s，离裸读上限仍差 ~1.8x。
根因是 `block_table` 间接寻址 + 每头小粒度 gather（见 kernel `else` 分支的
`physical_block_idx` 逐元素 load）。**这才是需要专属 kernel 的地方**，
而天数官方已有现成原语：

```python
ixformer.batch_paged_attention(
    output, query, kv_cache, lse,
    q_index, kv_index, block_tables,
    cpu_q_index, cpu_kv_index, cpu_block_tables,
    batch_size, max_page_num, page_size, kv_num_heads, scale, causal)
```

`lse` + `q_index`/`kv_index` = 天数官方**两阶段 split-KV + LSE** 原语，
随 `ixformer 0.7.0+corex.4.5.0` 已装。

**建议路径：**

| 步骤 | 成本 | 预期 |
|---|---|---|
| 1. 插件侧 monkeypatch `NUM_PAR_SOFTMAX_SEGMENTS=1`（不改源码） | 极低 | **−4.5 ms/step（−13%）** |
| 2. 用 `attn_kernel.py` 同一基准对比 `ixformer.batch_paged_attention` | 中 | 目标 ~300 GB/s ≈ **−12 ms/step** |
| 3. 大块连续读的专属 kernel | 高 | 追平裸读上限（9.27 ms/step） |

### 14.3 复现入口（本节）

`/root/bench_results/`：
* `attn_kernel.py` / `attn_kernel.txt` —— 三路对比（2D / 3D / 裸读），CUDA events 直测
* `attn_segm_sweep.py` / `attn_segm_sweep.txt` —— 段数 × ctx 扫描
* `attn_segm_best.py` / `attn_segm_best.txt` —— 最优段数定位（segm=1）
* `segm_e2e.py` —— 端到端验证（monkeypatch `NUM_PAR_SOFTMAX_SEGMENTS`）
* `sync_stacks.py` / `sync_stacks.txt` —— sync 频次归因（2.3/step）
* `attn_gpu.py` / `attn_gpu.txt` —— eager 逐核核算（并暴露 eager vs graph 的 3x 差）
* `async_ab.py` / `async_ab.log` —— async scheduling A/B

> **教训（第六条）**：**本平台两个 profiler 都不可用于占比归因** —— graph 模式漏计、
> eager 模式重复计数。任何「某算子占 step 多少」的结论必须用 **CUDA events 直测 kernel**
> 并与**同字节裸读**做对照，否则会像 §8.2.1 / §10.3 那样把方向带偏。
>
> **教训（第七条）**：**eager 与 graph 在此平台差 3 倍**（101 vs 34 ms）。任何"固定开销"
> 若与上下文长度无关，先怀疑**该项退出了 CUDA graph**，而不是内存或调度。

---

## 15. 优化方向路线图（2026-09-27 17:00）

### 15.1 先修正一个分解错误：每步流量包含 4.5 GB 权重，不只 KV

§13 只按 KV 拟合，把截距当成"神秘固定开销"。补上权重后模型完全闭合：

```
权重（bf16，含 tied embedding） = 4.50 GB，与 ctx 无关
KV   = conc(64) × kv_heads(2) × ctx × d(128) × 2 B × 2 (K,V) × 42 层
```

| ctx | KV GB | 权重 GB | 合计 GB | @564 GB/s 下限 | 实测 ms | 实测占天花板 |
|---|---|---|---|---|---|---|
| 512  | 1.41  | 4.50 | 5.91  | 10.47 | 24.80 | 42% |
| 1024 | 2.82  | 4.50 | 7.32  | 12.97 | 33.36 | 39% |
| 2048 | 5.64  | 4.50 | 10.14 | 17.97 | 49.73 | 36% |
| 4096 | 11.27 | 4.50 | 15.77 | 27.97 | 82.90 | 34% |

⇒ **低 ctx 时权重占 61%–76% 的流量**。§13 的"16.5 ms 固定截距"不是开销，
而**主要就是权重流**：`4.50 GB / 16.5 ms = 273 GB/s`。

**两条流的速率不同，必须分开优化：**

```
step(ctx) ≈ W / 273 GB/s  +  KV / 170 GB/s
           └ 16.5 ms ─┘     └ 5.89 ms/GB ┘
```
（ctx=512 代入：16.5 + 8.3 = 24.8 ms ✓ 与实测精确吻合）

| 流 | 现状 | 天花板 | 可省（ctx=1024） | 可省（ctx=4096） |
|---|---|---|---|---|
| 权重 | 273 GB/s（48%） | 564 | **−8.5 ms** | **−8.5 ms** |
| KV | 170 GB/s（30%） | 564 | **−11.6 ms** | **−46 ms** |

**理论最优（ctx=1024）：`4.5/564 + 2.82/564 = 13.0 ms`，相对现在 33.4 ms 有 2.6x。**

### 15.2 优化方向（按「收益 / 可行性」排序）

#### ★T1　权重流 273 → 564 GB/s　—　**−8.5 ms/step，所有 ctx 通用**

权重访问是 M=64 的大 GEMM（`linear_kernel`，192.8 calls/step）。§10.2 曾测得**纯 mm
达 447 GB/s**，但端到端 decode 只有 **273 GB/s** —— 说明差距出在
`addmm`/`bmm` 链路或权重加载路径，不在 GEMM 本身。

* 行动：用 `attn_kernel.py` 同款 CUDA-events 基准**逐层测 decode 的 linear**，
  定位是哪一层/哪种布局把 447 拖到 273。
* 成本：低（纯测量）→ 中（定位后按需修 kernel）。
* **这是当前性价比最高的未开采项**：收益与 ctx 无关，且已验证纯 mm 有 447 GB/s 的先例。

#### ★T2　KV 流 170 → 564 GB/s　—　**ctx=1024 省 11.6 ms，ctx=4096 省 46 ms**

已定位根因：`block_table` 间接寻址 + 每头小粒度 gather（见
`triton_unified_attention.py` 的 `physical_block_idx` 逐元素 load）。

| 手段 | 收益 | 成本 |
|---|---|---|
| **已完成**：`NUM_PAR_SOFTMAX_SEGMENTS 16→1` | −3.62 ms（实测 e2e） | 已提交 |
| 用 `ixformer.batch_paged_attention`（天数官方两阶段 split-KV + LSE） | 目标 300 GB/s ≈ −8 ms | 中 |
| 提高 `TILE_SIZE` / page 内连续读 | 待测 | 中 |
| 大块连续读的专属 kernel | 追平裸读 9.27 ms | 高 |

#### T3　减少字节数：量化　—　**与 T1/T2 叠加，需精度验证**

| 手段 | 字节变化 | 按现速率可省 |
|---|---|---|
| 权重 fp8/int8 | 4.50 → 2.25 GB | **−8.2 ms/step**（所有 ctx） |
| KV fp8 | KV 流量减半 | ctx=1024 −8.3 ms；ctx=4096 −23 ms |

两者合计理论可让 ctx=1024 从 33 ms 降到 ~15 ms。**风险是精度，必须先做等价性校验**
（照 §14.1b 的 `segm_numeric.py` 方法：kernel 级比对 + fp32 参考）。

#### T4　CUDA graph 覆盖率　—　**已测：混合到达下仅 7.2% 掉出，整段 wall 上限 ~5%**

`eager ~101 ms` vs `graph ~30 ms` 的 **67 ms 差只对单步成立**。在近似官方混合到达
（`in=1024 out=64 conc=64 n_prompts=192`，钩 `ModelRunnerFL._determine_batch_execution_and_padding`）下：

```
FULL graph steps : 350  (92.8%)
NOT FULL (NONE)  :  27  ( 7.2%)
upper-bound excess if every NONE paid 70.7 ms: 1.91 s / 34.90 s wall = 5.5%
（NONE 步本身含真 prefill，可归到「纯 eager 浪费」的更少）
```

⇒ **T4 不是一上来就 −67 ms 的大鱼**；对 92.8% 的 FULL 步无效。仍值得做形状扫描防回退，
但是 **~5% 级**，应排在 T1/T2 之后。

* 复现：`/root/bench_results/graph_cover.py`（必须钩 `vllm_fl.worker.model_runner.ModelRunnerFL`，
  不是上游 `GPUModelRunner`）。
* 成本：低。

#### T5　固定开销侧（已基本查清，无大鱼）

| 项 | 状态 |
|---|---|
| 2 次 `cudaEventSynchronize`/step | **不可去**（§14.1）；`async_scheduling=False` 可少 1 次，未取到 |
| `prof_cpu` 报的 23.88 ms | **会计假象**，`unaccounted = −29.82 ms` 证明不可加 |
| `aten::index` 29.4 ms（eager） | 异步调度下已被掩盖（`mp_ab` MP=0/1 无差） |

### 15.3 已达成的结论（可提交）

| 项 | 结果 |
|---|---|
| **`NUM_PAR_SOFTMAX_SEGMENTS` 16→1** | **34.01 → 30.39 ms/step（−3.62 ms，−10.6%）**；数值等价 PASS；已写入插件 `patch_triton_attn_segments_for_iluvatar` |

**最终确认（走真实插件路径，无任何手动 patch）**：`plugin_final.py` 断言
`NUM_PAR_SOFTMAX_SEGMENTS` 在 `LLM()` 构造后为 `1`（证明补丁在
`TritonAttentionMetadataBuilder.__init__` 读取该全局**之前**生效），实测：

```
before LLM()                = 16
after  LLM() construction   = 1        <- 插件补丁生效
PLUGIN-ONLY segm=1  : 30.34 ms/step  (30.30 / 30.34 / 30.36)
baseline   segm=16  : 34.01 ms/step
=> −3.67 ms/step = −10.8%
```

**注意安装位置**：运行时加载的是 `site-packages/vllm_fl`，不是工作区源码。补丁已同时
写入两处：
* 源码 `/workspace/vllm-plugin-FL-comp/vllm_fl/dispatch/backends/vendor/iluvatar/iluvatar.py`
* 运行时 `/usr/local/lib/python3.12/site-packages/vllm_fl/dispatch/backends/vendor/iluvatar/iluvatar.py`

### 15.4 已否决（勿重走）

| 方向 | 否决理由 |
|---|---|
| GEMV M==1 快路径 | CTA 饥饿，0.709x；且 conc=64 时 M≈64 非 1 |
| mm tile 调优 | 无头寸（<0.66%，已达 79–91% 硬件带宽） |
| `fused_add_rms_norm` | <1%（decode split 胜 0.86 ms，prefill fused 胜 2.81 ms） |
| `gate_up_proj + silu` 融合 | 无头寸（Inductor 已融合，0.22%） |
| `flash_attn` D2H 消除 | N/A（天数用 `TRITON_ATTN`，非 `vllm_flash_attn`） |
| `num_splits` 自适应 | N/A（沐曦专属 `flash_attn` 路径）；但**同类问题在天数真实存在**于 T2 |
| 多进程引擎 IPC | 无差异（MP=0 34.03 / MP=1 33.90） |
| 「去掉 cuda 同步」 | 不可行（§14.1，三条硬理由） |

### 15.5 建议执行顺序

1. **T1 定位**（纯测量，半天）→ 可能 −8.5 ms，且全 ctx 通用（对 92.8% FULL 步生效）
2. **T2 的 `ixformer.batch_paged_attention` 基准**（用 `attn_kernel.py` 同款口径对比）
3. **T3 权重 fp8 可行性**（测量 + 精度校验）→ 可能再 −8 ms
4. **T4 形状扫描**（已证整段 wall 上限 ~5%；防回退即可）

> **教训（第八条）**：**做流量分解时必须把权重算进去**。§13 只算 KV，导致把 4.5 GB 的
> 权重流误判为"神秘固定开销"，差点把它当成"CPU/同步开销"去优化。实际它占低 ctx
> 场景 61%–76% 的流量，是**和 KV 同等重要的两个独立优化目标**。

### 15.6 复现入口（本节）

`/root/bench_results/`：`segm_numeric.py`（数值等价）、`plugin_final.py`（插件端到端）、
`attn_kernel.py`（CUDA-events 三路对比）、`attn_segm_best.py`（段数最优）、
`mp_ab.py`（引擎形态 A/B）、`graph_cover.py` / `graph_cover.txt`（T4 混合到达覆盖率）。
权重/KV 分解见本节的 inline 计算。

---

## 16. TTFT / prefill：瓶颈是 prefill attention，且 2D 路径有正确性 bug（2026-09-27 ~ 09-30）

### 16.1 起点：段数补丁后的官方 4k（仍未过门槛）

插件带 §15.3 的 `NUM_PAR_SOFTMAX_SEGMENTS 16→1`，官方口径（`RUNS=4, SKIP_FIRST=1`），
Run 2 后按用户要求停掉：

| 指标 | Run 1（warm-up，不计分） | Run 2（计分） | 门槛 |
|---|---|---|---|
| Total tok/s | 1878.51 | **1877.32** | ≥ 2007.73 ❌（−6.5%） |
| Mean TTFT | 12686 ms | **12597 ms** | ≤ 11689 ❌（+7.8%） |
| Median / P99 TTFT | 3410 / 75388 ms | 3465 / 73825 ms | |
| Mean / Median TPOT | 153.07 / 156.21 ms | 157.00 / 165.79 ms | |
| Median / P99 ITL | 87.89 / 793.47 ms | 87.81 / 756.15 ms | |
| Duration | 697.7 s | 698.2 s | |

对比 §7 同口径的首轮（1690.29 tok/s / TTFT 14341 ms），段数补丁 + 期间修正带来 +11%，
但还差 6.5%。两轮几乎一致 ⇒ warm-up 的编译/autotune 开销可忽略。

**时间结构**（server log 每 10 s 采样）：256 请求按 64 一批、4 批**同步**推进（长度相同，
同时开始同时结束）。每批 ≈ 160 s = **prefill ≈ 80 s**（prompt ≈ 3.3k tok/s，同期 gen 掉到
93–99 tok/s）+ **decode ≈ 80 s**（gen 700–790 tok/s）。⇒ **prefill 占整轮 ~46% wall**，
TTFT 与总吞吐都被它拖住。Median ITL 87.8 ms 才是真实 decode 步长；Mean TPOT 153 ms 是被
prefill 饥饿（P99 ITL ~790 ms）拉高的。

### 16.2 prefill 里 attention 占 77%：用 §11 的 ctx 扫描拟合

§11 表：64 条请求的 prefill 时间 ctx=512/2048/4096 → 2.99 / 21.46 / 69.60 s。ctx 翻倍时间
×3.24（线性应 ×2，纯平方应 ×4）。拟合 `t = a·ctx + b·ctx²`：
`a = 3.96e-3, b = 3.18e-6`，三点均吻合（512 处 2.86 vs 2.99）。ctx=4096 分解：

| 项 | 时间 | 等效算力 |
|---|---|---|
| 线性项（matmul 等） | 16.2 s（23%） | ~64 TFLOPS（1.04 PFLOP） |
| 平方项（attention） | **53.4 s（77%）** | **~3.5 TFLOPS**（184.8 TFLOP） |

⇒ prefill attention 的效率只有 matmul 的 ~1/18。

### 16.3 原因：prefill 用的是 decode 取向的 tiling

`triton_unified_attention.unified_attention`（vLLM 0.24）对 prefill（2D 路径）：
`BLOCK_M = 16`（`num_queries_per_kv=8` ⇒ `BLOCK_Q = 2` 个 query token）、
`TILE_SIZE_PREFILL = 32`、num_warps 默认。上游注释自己承认 "decode-oriented defaults
under-tile it"，只给 B200 + head_size 256 开了大 tile。MiniCPM5-2B（head 128，GQA 8:1）
正好落在小 tile 路径上；BI-V150 warp=64，16×32×128 的 MMA 块远喂不饱。

### 16.4 ❗正确性 bug：2D 路径 softmax 分母多 1

kernel 级比对 fp32 稠密参考（`/root/bench_results/prefill_attn.py` 的 `reference()`）时发现：
**上游 2D 路径每行输出 = 正确值 × S/(S+1)**（S = 真实 softmax 分母）。

| 形状（q_len/kv_len） | out/ref |
|---|---|
| 1 / 1 | **0.500**（输出正好减半） |
| 1 / 16 | 0.921 |
| 1 / 64 | 0.973 |
| 16 / 16，逐 token | 0.500, 0.647, 0.700, 0.767 … |
| 256 / 512 | 0.992 |
| decode 1 / 1024，尖锐注意力（q×4） | 2D 0.9828；**3D（segm 16 或 1）1.0000** |

* 定位：token 0 只能看到 key 0，kernel 输出拟合为 `0.5·v0`（残差 ~1e-7），即 `L=2, acc=v0`。
* 机制：kernel 以 `L = 1.0` 初始化，依赖第一个 tile 的 `alpha = exp(-inf − m) = 0` 把它清掉。
  **天数 2D 路径上这个 1 没被清掉**。单独把 `softmax_step` 放进最小 kernel（含运行时循环）
  结果正确，`exp(-inf)` 本身也为 0 ⇒ 疑似编译器在完整 kernel 上下文中的问题，根因未深挖。
* 影响范围：**所有 prefill、所有混合 prefill+decode 批、>64 并发的 decode（走 2D）**。
  conc≤64 纯 decode 走 3D，正确。prefill 输出进入后续层 ⇒ 误差传播到整段生成。
* 以前的数值校验为何没抓到：随机输入下注意力近乎均匀（S 很大），S/(S+1)≈1。
  **做 attention 数值校验必须包含短序列和尖锐分布**。
* 修法：`L` 初值改 0（有 sinks 时保留单位贡献），2D epilogue 对全遮罩行
  `acc / where(L == 0, 1, L)`。零开销，所有形状/tiling 下对 fp32 误差 ≤ 3.9e-3（bf16 级）。

### 16.5 tiling 扫描（修复后的 kernel，CUDA events，无引擎）

`num_seqs=1, q_len=2048, kv_len=2048`，每层 17.2 GFLOP；"wave s" = 一批 64×4096 prefill
的 attention 外推时间（184.8 TFLOP / 实测算力）：

| 变体 | ms/layer | TFLOPS | wave s |
|---|---|---|---|
| 上游默认（BM16 T32，带 bug） | 5.29 | 3.2 | 56.9（与 §16.2 拟合的 53.4 s 吻合） |
| 修复 + 默认 tiling | 5.30 | 3.2 | 57.0 |
| BM64 T32 w8 | 1.94 | 8.9 | 20.8 |
| BM128 T64 w8 | 2.12 | 8.1 | 22.8 |
| **BM128 T32 w8** | **1.28** | **13.5** | **13.7** |
| BM16 T128 w4（最差） | 25.6 | 0.7 | 275 |

规律：`BLOCK_M` 越大越好，`TILE` 保持 32，8 warps 普遍优于 4。

`ixformer.contrib.vllm_flash_attn.flash_attn_varlen_func(block_table=...)` 以
`CUINFER_STATUS_BAD_PARAM` 报错并直接退出进程（疑似不支持 block_size=16 或要求别的 KV
布局），**未取得数据**。

### 16.6 插件补丁 `patch_triton_unified_attention_prefill_for_iluvatar`

`vllm_fl/dispatch/backends/vendor/iluvatar/iluvatar.py`，不改 vLLM 源码：

* kernel：`kernel._unsafe_update_src` 重写源码（L 初值 0 + 2D 除零保护），清 device cache。
* wrapper：运行时重写 `unified_attention` 源码，仅当 `max_seqlen_q > 1`、非 TD、
  `head_size == 128`、`2 ≤ num_queries_per_kv ≤ 16` 时用 `BLOCK_M=128 / TILE=32 / num_warps=8`；
  decode 保持上游 tiling。替换 `triton_attn.unified_attention` 符号。
* 仅天数（`current_platform.vendor_name == "iluvatar"`），`VLLM_FL_ILUVATAR_PREFILL_TILING=0`
  可关闭做对照，`VLLM_FL_ILUVATAR_PREFILL_BLOCK_M` 可选 64/128/256。
  `patch_triton_attn_segments_for_iluvatar` 同时加了同样的平台守卫（不影响沐曦）。

**走真实插件加载路径验证**（`/root/bench_results/prefill_patch_verify.py`：import vendor
backend 后调用 `triton_attn.unified_attention`，即引擎实际用的符号）：

```
patch applied flag: True ; NUM_PAR_SOFTMAX_SEGMENTS: 1
ns=1  q=1     kv=1     maxerr=0.00e+00                          PASS
ns=1  q=16    kv=16    maxerr=3.74e-03                          PASS
ns=1  q=64    kv=64    maxerr=3.66e-03                          PASS
ns=1  q=2048  kv=2048  maxerr=3.88e-03   1.282 ms  13.41 TFLOPS PASS
ns=1  q=2048  kv=4096  maxerr=1.73e-04   3.486 ms  14.79 TFLOPS PASS
ns=4  q=2048  kv=4096  maxerr=1.67e-04  12.737 ms  16.19 TFLOPS PASS
ns=64 q=1     kv=1024  maxerr=2.02e-04   0.538 ms   (decode 3D) PASS
OVERALL PASS
```

长上下文 chunk 与多序列批次同样有效（14.8–16.2 TFLOPS，原 3.2）。

**预期**：每批 prefill attention ~57 s → ~14 s，一轮省 ~170 s（698 s 中），外推 total
~2300+ tok/s、TTFT 大幅下降 —— 实测见 16.7（省 197 s、2616.65 tok/s）。

### 16.7 官方 4k 实测（prefill 补丁 v1：BM128 T32 w8）—— ✅ 两项门槛大幅通过

官方口径（`launch_official_eval_iluvatar.sh`，带 `--compilation-config`），server log 确认
APIServer 与 EngineCore 两个进程均打出 `patch_triton_unified_attention_prefill_for_iluvatar:
enabled`（当时默认 **TILE=32 / stages=Triton 默认**）。跑完 Run 1（warm-up）与 Run 2
（首个计分轮）后，结论已确定，按用户要求停掉（未跑 Run 3/4 与 16k）。
产物 `/root/bench_results/eval_iluvatar_4k_prefill/`。

| 指标 | 补丁前 Run 2（§16.1） | **补丁后 Run 1**（warm-up） | **补丁后 Run 2**（计分） | 基线 | 门槛 |
|---|---|---|---|---|---|
| Total tok/s | 1877.32 | 2411.96 | **2616.65** | 2028.01 | ≥ 2007.73 ✅ **+29.0% vs 基线** |
| Mean TTFT | 12597 ms | 6511 ms | **6066 ms** | 11573.53 | ≤ 11689.27 ✅ **−47.6% vs 基线** |
| Median / P99 TTFT | 3465 / 73825 | 2093 / 37415 | 2084 / 35198 | | |
| Mean / Median / P99 TPOT | 157.00 / 165.79 / 168.73 | 120.22 / 116.86 / 138.60 | 115.25 / 116.10 / 126.14 | | |
| Median / P99 ITL | 87.81 / 756.15 | 83.44 / 436.45 | 83.55 / 425.17 | | |
| Duration | 698.2 s | 543.4 s | **500.9 s（−197 s）** | 646.31 s | |

**解读**

* **总吞吐 +39.4%、TTFT −51.8%**（Run 2 对 Run 2）。§16.6 外推"省 ~170 s、2300+ tok/s"，
  实测省 197 s、2617 tok/s，方向与量级一致（外推偏保守）。
* **时间结构**：每批 prefill 由 ~80 s 降到 ~45–50 s（prompt 吞吐 3.3k → 5.3–7.4k tok/s），
  每批总时长 ~160 s → ~120 s；decode 段 gen 710–825 tok/s 基本不变（补丁只动 prefill）。
* **TPOT 也降 27%**：纯 decode 步长几乎不变（Median ITL 87.8 → 83.6 ms），下降来自
  prefill 饥饿期缩短（P99 ITL 756 → 425 ms）。
* 补丁后 Run 1 比 Run 2 慢 ~8%：新 tiling 下首轮的 Triton 编译落在 warm-up 里；
  补丁前两轮几乎一致是因为旧 kernel 早已被缓存。

**后续**：默认已升级为 TILE=64 / stages=2（§16.10），端到端见 §16.13。

### 16.8 复现入口

`/root/bench_results/`：`prefill_attn.py`（参考实现 + tiling 扫描 + ixformer 尝试；
`SKIP_IX=1 QUICK=1 CASE_FROM=n` 可裁剪）、`prefill_patch_verify.py`（插件路径数值/速度）、
`prefill_bm_sweep.py` / `prefill_bm_sweep.txt`（BM/TILE/warps/stages 全扫描）、
`t1_weight.py` / `t1_weight.txt`（T1 权重流拆解）、
`eval_iluvatar_4k_prefill/`（§16.7）、`eval_iluvatar_4k_tile64/`（§16.13）、
`/tmp/uam_mod/t3d.py`（2D vs 3D 分母 bug 对照）。

### 16.9 天数侧修改点汇总（截至本节，均在插件内，不改 vLLM 源码、不动沐曦专属代码）

| 修改 | 位置 | 作用 | 实测收益 | 开关 |
|---|---|---|---|---|
| `patch_triton_attn_segments_for_iluvatar` | `vendor/iluvatar/iluvatar.py` | `NUM_PAR_SOFTMAX_SEGMENTS 16→1`（3D decode 路径） | decode 34.01 → 30.34 ms/step（−10.8%，§15.3） | `VLLM_FL_ILUVATAR_NUM_PAR_SOFTMAX_SEGMENTS=1/2/4/8/16`（默认 1） |
| `patch_triton_unified_attention_prefill_for_iluvatar` —— tiling | 同上 | 2D prefill **BM128 / TILE64 / w8 / stages=2**（仅 head 128、GQA 2–16） | prefill attn 3.2 → **21.5 TFLOPS**（§16.10）；e2e 见 §16.13 | `VLLM_FL_ILUVATAR_PREFILL_TILING=0` 关；`BLOCK_M` / `TILE` / `NUM_WARPS` / `NUM_STAGES` |
| 同上 —— softmax 分母修复 | 同上（kernel 源码重写） | `L` 初值 1→0 + 2D 全遮罩行除零保护 | 正确性：修掉 S/(S+1) 缩小（§16.4） | 同上 |
| 两个 attention 补丁的平台守卫 | 同上 | `current_platform.vendor_name == "iluvatar"` 才生效 | 保证沐曦/其他平台行为不变 | — |
| sort-free top-p sampler | `vendor/iluvatar/patches/topk_topp_sampler.py` + `__init__.py` | 天数上的 top-p 采样快路径（§15 前已落地） | — | 见文件内 |

**端到端（官方 4k 计分轮）**：

* v1 TILE=32（§16.7）：1877 → **2616.65 tok/s**，TTFT 12597 → **6066 ms**（相对基线 +29.0% / −47.6%）
* **v2 TILE=64/s2（§16.13）**：→ **2783.73 tok/s**，TTFT → **5180 ms**（相对基线 **+37.3% / −55.2%**）

**部署注意**：运行时加载的是 `site-packages/vllm_fl`，工作区源码改完必须同步到
`/usr/local/lib/python3.12/site-packages/vllm_fl/dispatch/backends/vendor/iluvatar/iluvatar.py`
（本节提交时两处已同步；默认 knobs 与 §16.10/§16.13 一致）。

### 16.10 Prefill tiling 二轮扫描：TILE=64 + stages=2（2026-09-30）

`/root/bench_results/prefill_bm_sweep.py`，在 L=0 修复之上扫
`BM∈{64,128,256} × TILE∈{32,64} × warps∈{4,8} × stages∈{None,2,3}`：

| 配置 | ns=1 q=2048 kv=2048 | TFLOPS | vs 上游默认 |
|---|---|---|---|
| 上游 BM16 T32（有 bug） | 5.29 ms | 3.2 | 1.0× |
| 插件 v1：BM128 T32 w8 | 1.278 ms | 13.4 | 4.1× |
| **插件 v2：BM128 T64 w8 s2** | **0.801 ms** | **~21.5** | **6.6×** |

三形状最优均为 BM128 T64 w8 s2（2048/4096：2.15 ms / 24 TFLOPS；4×2048/4096：
8.24 ms / 25 TFLOPS）。已写入插件默认；`prefill_patch_verify.py` 复验：

```
enabled BLOCK_M=128 / TILE=64 / num_warps=8 / num_stages=2
ns=1 q=2048 kv=2048  maxerr=3.88e-03  0.800 ms  21.49 TFLOPS PASS
ns=1 q=2048 kv=4096  maxerr=1.69e-04  2.149 ms  23.99 TFLOPS PASS
ns=4 q=2048 kv=4096  maxerr=1.78e-04  8.236 ms  25.03 TFLOPS PASS
ns=64 q=1   kv=1024  (decode 3D)      0.537 ms              PASS
OVERALL PASS
```

### 16.11 `ixformer` 原生 attention：当时入口踩坑（已被 §17 取代）

* 当时试过的 `flash_attn_varlen_func(block_table=...)` / `batch_paged_attention` 在
  旧调用约定下失败（`CUINFER_STATUS_BAD_PARAM` / 空 `RuntimeError`），T2 挂起。
* **2026-10-01**：改走 FlagGems-comp fused wrapper + KV-first HND + **decode-only**
  后已在官方 4k 生效，见 **§17**。本小节仅保留历史。

### 16.12 T1 权重流归因更正（2026-09-30）

§15 把 decode 截距 16.5 ms 建模成「权重流 @ 273 GB/s」，并估 −8.5 ms 头寸。
`t1_weight.py`（CUDA events，M=64，无引擎）拆开后：

| 项 | ms（×42 层） |
|---|---|
| 纯 4× `torch.mm` | **10.01**（~396 GB/s） |
| 纯 4× `F.linear` | 10.70 |
| FlagGems mm 在 gate_up 上 | 单次 0.098 ms → **532 GB/s（94% 天花板）** |
| **整层无 attention**（4 linear + 2 rms + silu + residual） | **17.11** |
| §15「权重截距」 | 16.5 |

⇒ **16.5 ms ≈ 无 attention 的整层，不是慢 mm。** 纯 mm 已接近 400 GB/s；再跑满 564
大约只省 ~3 ms，不是 −8.5 ms。截距里另 ~7 ms 是 rms/silu/residual/激活流量。
T1 的「修 mm 布局」优先级下调；若继续抠固定开销，应盯 rms/silu 融合而非 mm tile。

### 16.13 官方 4k 实测（prefill 补丁 v2：BM128 T64 w8 s2）—— ✅ 再上一截

口径：官方 serve（`--compilation-config FULL_DECODE_ONLY`）+ `benchmark_throughput_serve`
仅 4k case，`RUNS=2 / SKIP_FIRST=0`（两轮都报；计分看 Run 2）。server log 确认
`BLOCK_M=128 / TILE=64 / num_warps=8 / num_stages=2`。产物
`/root/bench_results/eval_iluvatar_4k_tile64/`。

| 指标 | 无补丁 §16.1 | v1 TILE=32 §16.7 | **v2 TILE=64 Run 1** | **v2 TILE=64 Run 2（计分）** | 基线 | 门槛 |
|---|---|---|---|---|---|---|
| Total tok/s | 1877.32 | 2616.65 | 2506.86 | **2783.73** | 2028.01 | ≥2007.73 ✅ **+37.3%** |
| Mean TTFT ms | 12597 | 6066 | 6871 | **5180** | 11573.53 | ≤11689 ✅ **−55.2%** |
| Median / P99 TTFT | 3465 / 73825 | 2084 / 35198 | 1737 / 37243 | 1598 / 30411 | | |
| Mean / Median TPOT | 157.0 / 165.8 | 115.3 / 116.1 | 114.7 / 110.2 | 108.6 / 110.5 | | |
| Median / P99 ITL | 87.8 / 756 | 83.6 / 425 | 83.3 / 377 | 83.5 / 341 | | |
| Duration s | 698.2 | 500.9 | 522.9 | **470.9** | 646.31 | |

脚本对两轮平均：Total **2645.3** tok/s，TTFT **6025** ms（仍过门槛）。

**相对 v1（TILE=32 计分轮）**：吞吐 **+6.4%**，TTFT **−14.6%**，耗时 −30 s。
Median ITL 几乎不变（~83.5 ms）⇒ 收益仍在 prefill / TTFT 侧，与 kernel 扫描一致。
Run 1 慢于 Run 2：新 TILE/stages 的 Triton 首次编译落在 warm-up。

> **教训（第九条）**：**先确认 kernel 是对的，再比快慢。** 这次的 bug 是在做性能对比、
> 顺手加 fp32 参考时才暴露的；如果只比时间，会把一个带 bug 的 kernel 当成 baseline。

> **教训（第十条）**：**流量反推带宽 ≠ 该算子的真实带宽。** 把截距整段除以权重字节会
> 把 rms/silu/residual 都算进「权重流」，高估可优化头寸。

---

## 17. Decode-only ixformer attention（2026-10-01）

在 §16 tile64 prefill 之上，把 **纯 decode** 的 `unified_attention` 切到 ixformer
`paged_attention_flashinfer`（HND）；prefill / mixed 继续 Triton tile64。
MetaX 不进此路径（`_is_iluvatar_platform`）。

### 17.1 代码落点

| 层 | 路径 | 作用 |
|---|---|---|
| FlagGems-comp | `_iluvatar/fused/paged_attention_decode.py` | scatter 组页 → ixformer HND decode |
| FlagGems-comp | `_iluvatar/fused/flash_attn_prefill.py` | paged FA 封装（仅纯 prefill 批，见 §17.6） |
| FlagGems-comp | `_iluvatar/fused/ixformer_unified_attention.py` | 形状门控：decode / mixed / short / prefill（§17.6） |
| 插件 | `iluvatar.py` `select_hnd_kv_cache_layout_for_iluvatar` | `VLLM_KV_CACHE_LAYOUT=HND` + **KV-first** stride `(1,0,3,2,4)` |
| 插件 | `iluvatar.py` `patch_triton_unified_attention_ixformer_for_iluvatar` | wrap `unified_attention`；probe / atexit 计数 |

开关（默认开）：

* `VLLM_FL_ILUVATAR_IXFORMER_ATTN=0` → 强制纯 Triton
* `VLLM_FL_ILUVATAR_KV_CACHE_HND=0` → 恢复 vLLM 默认 KV stride

### 17.2 踩坑：为何一度「没生效 / 没收益」

1. **HND 名义 vs 物理连续**  
   仅设 `VLLM_KV_CACHE_LAYOUT=HND` 时 upstream stride 仍是 `(0,1,3,2,4)`，K/V 在
   `dim=1` 交错；`unbind(1)` 后 HND view 的 block stride ×2，**不连续** → fused 门控
   `layout` 拒绝 → **静默回退 Triton**（e2e ≈ tile64）。
2. **KV-first 修好后真走 FA 反而更慢**  
   full ixformer（prefill+decode）在官方 `enable_chunked_prefill` 下：混批 prompt
   ~2–3k、gen ~50–80；同配置 decode-only 混批 prompt ~6–9k、gen ~190–230。纯 decode
   峰值两边都 ~1.5k → 慢在 **chunked/mixed prefill FA**，不是 decode flashinfer。
3. **混合批被 `max_seqlen_q` 绑死**  
   批里只要有 q_len>1，整步（含同批 decode token）曾走 FA，放大 gen 塌陷。故门控改为
   **仅 `max_seqlen_q==1`**。
4. **descale 伪门控**  
   vLLM 对 bf16 仍可能传入 `k_descale`/`v_descale` tensor；早期按「有 descale 即 FP8」
   会误拒。现改为看 KV dtype。
5. **page 组装 D2H**  
   `block_table[mask]` 同步约 0.4 ms/call；已改 scatter 组页。

> **教训（第十一条）**：**微基准赢 ≠ serve 赢。** full-seq FA 可比 tile64 快 1.9–3×，但
> 官方 4k 的短 q / 长 k / 多 seq chunk 上 FA 输给 tile64；必须用 chunked+mixed 口径重测。
>
> **教训（第十二条）**：**静默 fallback 等于没做。** 必须有 hit/layout 计数；否则整轮
> bench 可能测的是旧路径。

### 17.3 官方 4k 计分（decode-only + KV-first + tile64 prefill）

口径：`FULL_DECODE_ONLY`，`CASES=[[4096,1024,64,256]]`，`BENCH_RUNS=2 SKIP_FIRST=0`，
`VLLM_FL_ILUVATAR_IXFORMER_ATTN=1`。产物
`/root/bench_results/eval_iluvatar_4k_decode_only_v2/`。server 确认
`enabled DECODE-ONLY` + `KV-first stride_order=(1, 0, 3, 2, 4)`；probe 捕获期
`hit=True layout=0`。

| 指标 | tile64 §16.13 Run2 | **decode-only Run1** | **decode-only Run2（计分）** | 门槛 |
|---|---|---|---|---|
| Output tok/s | 556.75 | 628.67 | **732.74** | — |
| **Total tok/s** | **2783.73** | 3143.36 | **3663.68** | ≥2007.73 ✅ |
| Mean TTFT ms | 5179.91 | 5983.14 | **5398.24** | ≤11689 ✅ |
| Mean TPOT ms | 108.6 | 85.37 | **77.63** | — |
| Peak output tok/s | — | 1664 | — | — |

两轮 summary：Total **3403.52** tok/s，TTFT **5690.69** ms。  
相对 §16.13 计分轮：Total **+31.6%**（3664 vs 2784）；TTFT 略差（5398 vs 5180，仍过线）。

对照（同机、曾误开 full FA 或未 KV-first 的轮次）e2e ≈ tile64，**不能**当作 ixformer 收益。

### 17.4 正确性（math_500 Level 3 冒烟）

> ✅ **本节结论有效**（曾被 §17.7.3 第一版误判为作废，已按 §17.9 恢复）：
> 引擎在 `FULL_DECODE_ONLY` 下走的是 cudagraph 回放，不触发 §17.7 的 eager
> 错误路径。门控后以同口径重测，结果一致：**29/30，错题同为 index 18**。
> 产物 `/root/bench_results/eval_iluvatar_mathL3_batchgate/partial30_summary.txt`。

完整 105 题 ETA 过长；early-stop 前 **30** 题（evalscope reviews 已打分）：

* **29/30 = 0.9667** ≥ 0.95（错 1 题 index=18）
* 产物：`/root/bench_results/eval_iluvatar_decode_only_mathL3/partial30_summary.txt`
* **非正式完整 Level3**；足以排除系统性数值错乱。完整 105 题待补。

### 17.5 当前推荐配置

* **保留**：tile64 prefill（§16.13）+ **decode-only / split ixformer** + KV-first HND
* **默认开** `VLLM_FL_ILUVATAR_IXFORMER_SPLIT=1`（每步一次元数据；§17.11 的 +52%
  作废，见 §17.13 勘误）
* decode graph 捕获时在 graph 内建 flashinfer 页表（§17.13 修复，必需）
* FlagGems 天数 vendor `CUSTOMIZED_UNUSED_OPS`：`add` + 16 个 eager 小算子
  （§17.13：修复后冷启动 4k×2 汇总 4223 → 5561，+31.7%）
* `MIN_Q=256`、flashinfer ctx-aware 保护（`batch<32 and ctx≥2048` → Triton）
* OpenAI-API 默认 `max_num_batched_tokens` 2048 → 4096（插件 patch
  `EngineArgs.get_batch_defaults`，仅天数；§17.14：4k 汇总 5561 → 5768，+3.7%）；
  `VLLM_FL_ILUVATAR_MAX_NUM_BATCHED_TOKENS=0` 关
* 关拆批：`VLLM_FL_ILUVATAR_IXFORMER_SPLIT=0`

### 17.6 形状门控：mixed → Triton，纯 prefill → FA（2026-10-01 晚）

按 §17.5 最后一条做了形状门控（`ixformer_unified_attention.py`）：

| 批形状 | 判定 | 路径 |
|---|---|---|
| `max_seqlen_q == 1` | decode | ixformer flashinfer |
| 任一 `q_len == 1`（decode 行混入） | `mixed_skip` | Triton tile64 |
| 纯 chunk 但最短 `q_len < MIN_Q` | `short_skip` | Triton tile64 |
| 纯 chunk 且全部 `q_len >= MIN_Q` | prefill | ixformer FA varlen |

`MIN_Q` 由 `VLLM_FL_ILUVATAR_IXFORMER_PREFILL_MIN_Q` 覆盖。

#### 17.6.1 ❗我自己引入的 D2H 同步：第一版门控 TTFT 反而 +2.5 s

第一版用 `q_lens.min().item()` 判 mixed，**每个非 decode 的 attention 调用都同步一次**
（~44k 次/轮）。计数里 FA 只跑了 `prefill=168`（4 步 × 42 层），不可能解释 2 s 的
TTFT 回退 —— 回退来自门控本身。改为先用 host 侧已知量预筛：

* `max_seqlen_q < MIN_Q` → `short`
* `num_q_tokens < batch * MIN_Q` → `mixed`（全 long 必然满足 `tokens >= batch*MIN_Q`）

只有通过预筛的批才做一次 `.item()` 精确判定。serve 典型 mixed（63 decode + 1×2048，
2111 tokens）被第二条零同步拒掉。因此 **`MIN_Q` 不能设 0**：它同时承担免同步识别 mixed。

第二处同步：`flash_attn_prefill._cu_seqlens_from_seqused` 里 `cu[0] = 0`（Python 标量
写 CUDA tensor），`torch.cuda.set_sync_debug_mode("warn")` 抓到；改为 `torch.zeros`
后检测器清零。它只在 FA 路径上触发，量级远小于下表的 kernel 差。

#### 17.6.2 serve 形状下 FA vs Triton（无引擎，CUDA events，单层）

脚本 `/root/bench_results/serve_shape_fa_vs_triton.py`，产物同名 `.txt`：

| 形状 | tokens | FA ms | Triton ms | FA/Triton |
|---|---|---|---|---|
| 63 decode + 1 chunk q=2048 | 2111 | 14.20 | 3.99 | 0.28× |
| 63 decode + 1 chunk q=512 | 575 | 13.56 | 2.60 | 0.19× |
| 32 decode + 4 chunk q=512 | 2080 | 7.59 | 3.94 | 0.52× |
| 纯 chunk 8 × q=256 | 2048 | 0.87 | 3.23 | **3.73×** |
| 纯 chunk 4 × q=512 | 2048 | 0.86 | 3.01 | **3.49×** |
| 纯 prefill 2 × q=2048 | 4096 | 1.43 | 4.21 | **2.93×** |
| 纯 prefill 1 × q=4096 | 4096 | 1.03 | 2.87 | **2.78×** |

* **FA 在 mixed 上的代价随序列数走，不随 token 数走**：≈0.21 ms/seq。q=2048 与 q=512
  两行 token 差 4× 而耗时几乎相同；decode 行各自付一整趟 q-tile 扫描。
* 这才是 §17.2 第 2 条「full FA 慢」的机制：mixed 一步 ~10 ms/层 × 42 层 ≈ 420 ms，
  比任何一次同步大两个数量级。**D2H 不是 FA varlen 慢的原因**。
* 纯 chunk 即便 q=256 FA 也赢 3.7× → `MIN_Q` 默认 512 → **256**。

#### 17.6.3 官方 4k×2 计分

口径同 §17.3；`VLLM_FL_ILUVATAR_IXFORMER_ATTN=1`。

| 轮次 | 产物目录 | Total tok/s（R1 / R2 / 汇总） | Mean TTFT ms（R1 / R2 / 汇总） |
|---|---|---|---|
| decode-only（§17.3） | `eval_iluvatar_4k_decode_only_v2` | 3143.36 / 3663.68 / 3403.52 | 5983.14 / 5398.24 / 5690.69 |
| 门控 v1（`.item()` 同步，MIN_Q=512） | `eval_iluvatar_4k_mixed_fallback` | 2838.51 / 3430.42 / 3134.47 | 8463.18 / 7455.55 / 7959.36 |
| 门控 v2（免同步预筛，MIN_Q=512） | `eval_iluvatar_4k_gate_nosync` | 3267.40 / 3989.17 / 3628.28 | 6177.35 / 5453.70 / 5815.52 |
| 门控 v3（+去 `cu[0]=0`，MIN_Q=256） | `eval_iluvatar_4k_gate256` | TBD | TBD |

server 计数（v2 末尾）：`decode=2940 prefill=168 mixed_skip=43912 short_skip=84`
（decode 计数只含图捕获前的 eager 调用）。**FA 在 serve 中几乎不跑**：官方 64 并发
+ chunked prefill 下，非 decode 步几乎都是 mixed。所以 v2 vs decode-only 的差异
在两轮 bench 的波动内，门控的价值是「不再让 mixed 走 FA」，不是「FA 带来了收益」。

> **教训（第十三条）**：**门控本身也在热路径上。** 判定逻辑每层每步都跑；一个
> `.item()` 乘以 4.4 万次就能吃掉整个优化。先看计数：被优化路径的调用次数
> 解释不了观测到的回退时，回退就在门控里。

### 17.7 ❗flashinfer decode 在「小 batch + 长 ctx」下算错（2026-10-01 晚）

查「拆批」方案时顺带发现的，**影响 §17.3 已落地的 decode-only 路径**。

#### 17.7.1 现象

同一组输入、同一个 fp32 参考（逐序列 gather KV 后全精度 softmax），ixformer
`paged_attention_flashinfer` 与 Triton 的误差比（`flashinfer_safe_map.py`，
产物 `/root/bench_results/flashinfer_safe_map.txt`）：

```
  n\ctx      256     512    1024    2048    4096    8192
     1        1       1       1     473     358     306
     8        1       1       1     368     441     370
    24        1       1       1     412     448     394
    31        1       1       1     420     472     414
    32        1       1       5       1       1       2
    64        1       1       2       5       3       3
   128        1       4       5       4       4       3
```

**`batch < 32` 且 `ctx >= 2048` 时误差是 Triton 的 300–470×**（绝对误差 ~4e-2，
值本身量级 ~0.1）。bf16 累加顺序给不出这个量级；边界在 n=32 上是硬的（n=31 错、
n=32 对）。表里零星的 5–18× 在两次运行间会漂移，是 Triton 误差偶尔很小导致的
比值噪声，不是同一类问题。

已排除测试脚手架：换 seed、换顺序/乱序 block_table，误差稳定在 4e-2
（`flashinfer_smalln_confirm.py`）。ctx 维度上的触发点是 pages/seq ≥ 128
（ctx 512/1024 正常，2048 起全错）。

形态像 split-K：序列数不足以填满 16 个 SM 时，kernel 沿 KV 方向切分以增加并行，
而跨分片的 softmax 合并是错的——和 §16.4 那个「2D 路径分母多 1」同一类。

#### 17.7.2 为什么之前没发现

早期验证 `ix_flashinfer_valid.py` 的取样是 N=1/CTX=64、N=4/CTX=256、
N=64/CTX=1024、N=64/CTX=4096 —— **恰好绕开了「小 N + 大 CTX」这个角**：
小 N 时 ctx 都很短，ctx 大时 N 都是 64。

> **教训（第十四条）**：**数值验证要扫二维，不能只扫对角线。** 按「小规模配小
> 规模、大规模配大规模」取样，看起来覆盖了范围，其实只走了对角线；kernel 的
> 分支恰恰长在反对角线上（小 batch 配长 ctx）。

#### 17.7.3 ❗影响范围（本节已被 §17.9 更正两次，以此为准）

**结论：`FULL_DECODE_ONLY` 下的 serve 从未受影响；此前已报告的 4k 数字和
§17.4 的 Level 3 29/30 都仍然有效。** 受影响的只有 **eager** 调用。

更正过程记录在此，因为它本身是个教训：

* 第一版判断「4k 尾部 token 是错的、§17.4 作废」—— **错误**。它假设微基准的
  eager 行为可以外推到引擎，而没有验证。
* 真实机制见 §17.9：切分与否在 host 侧按 `max_seq_len` 决定，并被烤进 CUDA
  graph；vLLM 捕获时 `seq_lens.fill_(1)` 让引擎始终回放非切分（正确）路径。
* 与之一致的旁证：§17.4 和 §17.7 门控后的重测都是 **29/30 = 0.9667，错题同为
  index 18**。两次独立采样同题出错，说明那是题目难度，不是 kernel。

> **教训（第十五条）**：**「我发现了 bug」不等于「这个 bug 在生产里发生」。**
> 微基准是 eager 的，引擎可能是 graph 的；两者选到的 kernel 路径可以不同。
> 定影响范围必须在真实调用路径上验证，否则会像这里一样，先错误地把一个
> 有效的正确性结论判为作废。

#### 17.7.4 收口

`ixformer_unified_attention._decode_is_unsafe(batch, max_seqlen_k)`：
`batch < 32 and max_seqlen_k >= 2048` → 退回 Triton，计入 `decode_unsafe_skip`。
两个量都是 host 侧已有的（`block_table.shape[0]` 和入参 int），**不引入同步**。

验证（走插件补丁后的 `unified_attention`，对 fp32 参考）：

| n | ctx | 路由 | 误差 |
|---|---|---|---|
| 8 | 4096 | `decode_unsafe_skip` | 1.11e-04（原 4.9e-02） |
| 31 | 4096 | `decode_unsafe_skip` | 1.10e-04 |
| 32 | 4096 | `decode` | 1.27e-04 |
| 64 | 4096 | `decode` | 5.69e-04 |
| 8 | 1024 | `decode` | 2.22e-04 |
| 64 | 8192 | `decode` | 2.55e-04 |

### 17.8 拆批原型：reorder 后 split(flashinfer + FA) 比全 Triton 快 3.1–3.4×

#### 17.8.1 vLLM 已有现成机制，不必改调度器

`vllm/v1/attention/backends/utils.py`：

* `reorder_batch_to_split_decodes_and_prefills()` 把 `q_len <= threshold` 的行搬到
  批首，批重排为 `decode → short_extend → long_extend → prefill`。
* `split_decodes_and_prefills()` 返回 `(num_decodes, num_prefills, num_decode_tokens,
  num_prefill_tokens)`，读的是 **`query_start_loc_cpu`（host 侧）→ 无 D2H 同步**。

开关只是 builder 上的一个属性：`gpu_model_runner._may_reorder_batch` 仅在
`reorder_batch_threshold is not None` 时重排。`TritonAttentionMetadataBuilder`
从未调用 `_init_reorder_batch_threshold()`，继承基类默认 `None`，**所以今天没有重排**
——这正是 decode 行与 chunk 行在批里交错的原因。FlashInfer / 各 MLA 后端都设了
（`flashinfer.py:701`）。

重排同时解决两件事：decode 行变**连续**（切片即可，无需 gather/scatter），
行数从 **CPU 元数据**取（无同步）。

#### 17.8.2 单层实测（`split_batch_proto.py` → `split_batch_proto.txt`）

ctx=4096，KV 按插件的 KV-first 布局分配（否则 decode kernel 拒收非连续 HND）。
计时含切片与 page 元数据构建。

| 形状（decode 行在前） | tokens | Triton | **split** | all-FA | split/Triton |
|---|---|---|---|---|---|
| 63 decode + 1 chunk q=2048 | 2111 | 4.132 | **1.304** | 14.212 | **3.17×** |
| 63 decode + 1 chunk q=512 | 575 | 2.589 | **0.755** | 13.570 | **3.43×** |
| 32 decode + 4 chunks q=512 | 2080 | 3.940 | **1.201** | 7.602 | **3.28×** |
| 48 decode + 2 chunks q=1024 | 2096 | 3.999 | **1.279** | 11.044 | **3.13×** |
| 16 decode + 1 chunk q=2048 | 2064 | 2.679 | **1.109** | 4.165 | 2.42× |

拆解（63 decode + 1 chunk q=2048）：flashinfer 0.515 ms + FA 0.774 ms。
`.item()` 调用 0 次，CUDA 同步检测器无告警。

命中面很大：§17.6.3 的计数里 `mixed_skip=43996`，占全部 attention 调用的 ~93%，
而现在走 FA 的只有 168 次（0.36%）。

> 注：上表最后一行（16 decode）当时报出 `maxdiff dec=4.3e-02`，正是 §17.7 那个
> bug——拆批原型是**顺带**把它暴露出来的。加门控后这一行会退回 Triton。

#### 17.8.3 下一步与风险

* 插件侧在 `TritonAttentionMetadataBuilder` 上设 `reorder_batch_threshold = 1`
  （仅天数，`_is_iluvatar_platform` 门控），再在 fused 门控里按 `num_decode_tokens` 切片。
* 风险点：拆批后 decode 片的 batch 会**小于整批**，更容易掉进 §17.7 的
  `batch < 32` 区 → 必须复用 `_decode_is_unsafe`，不安全时整批退回 Triton。
* cudagraph 不受影响：纯 decode 批 `num_prefills == 0`，路径与现在一致；
  混合批本来就是 eager。
* 重排本身每步有 CPU 侧搬运开销（其他后端也在付），需在 e2e 上确认不抵消收益。

#### 17.8.4 官方 4k×2 实测：拆批没有 e2e 收益

| 配置 | 汇总 Total tok/s | 汇总 TTFT ms |
|---|---|---|
| 门控 MIN_Q=512 | 3628.28 | 5815.52 |
| 门控 MIN_Q=256 | 3630.55 | 5958.41 |
| **+ reorder 拆批** | **3543.45** | 6145.00 |

产物 `/root/bench_results/eval_iluvatar_4k_split/`。两项门槛仍 PASS。

机制确实生效，计数为证：`split=38448 mixed_skip=0 split_unsafe_skip=5292`
——`mixed_skip` 从 43996 归零，38448 次调用真的走了两片派发，**零次校验失败回退**。
所以不是没生效，是生效了但不赚（-2.4%，在轮间波动内）。

未验证的推测（留作下一步）：单层 3.17× 只量了 kernel，而拆批每层多一次 kernel
launch 和一次 flashinfer page 组装；§12 测到 GPU 只占整步 12–20%，host 开销很可能
吃掉了 kernel 的收益。另有 12% 的批因 decode 片 <32 行退回 Triton。

**处置（已被 §17.11 取代）**：当时代码保留、默认 `SPLIT=0`。单层 3× 是真的；
host 开销在 §17.10 被证实后，§17.11 把元数据挪到每步一次，拆批转为默认开。

> **2026-10-02 补**：「-2.4%，在轮间波动内」只对了一半，根因已在 §17.10 定位：
> 拆批的元数据构造每层重做一次，在引擎里被 FlagGems 接管成 ~20 个 Triton
> launch，每层多花 ~6 ms host，含 prefill 步因此变成 host-bound。

### 17.9 ❗更正 §17.7.3：cudagraph 捕获掩蔽了这个 bug，引擎从未受影响

#### 17.9.1 发现过程（先错后对，完整记录）

查 §17.7 的机制时，依次证伪了三个假设：

1. **「是我的 wrapper 建错 page 元数据」** → 否。手搓最简元数据直接调 ixformer，
   误差照样复现，边界仍是 32（`flashinfer_splitk_probe.py` TEST 1）。
2. **「是 `exp(local_max - global_max)` 重加权错」** → 否。q 全零、softmax 完全
   均匀时重加权系数恒为 1，误差仍有 3.45e-2（TEST 2）。
3. **「是 split-K 分片权重错」** → 否。令 V 分段取常数直接读出权重，S=2/4/8/16
   下各段权重精确等于 1/S，和为 1（`flashinfer_shard_weights.py`）。
4. **「是累加精度不够」** → 否。对 fp32 / bf16 / fp16 三种参考的误差完全相同
   （`flashinfer_accum_precision.py`）。

转折点是改看**误差的结构**而不是大小（`flashinfer_error_structure.py`）：
16 个 query head 全都最接近 KV head 1，且不同序列的输出彼此过于相似。于是把
(序列号, head 号) 编码进 V 直接读出来（`flashinfer_index_readout2.py`）：

```
n=8   每行读到的序列号: [1,1,1,1,1,1,1,1]   应为 1..8
n=32  每行读到的序列号: [1,2,3,4,5,6,7,8]   正确
mass=1.0, 平均位置=0.498  → 归一化与覆盖范围都对
```

**真相：不是数值问题，是 n<32 时每一行都去读了序列 0 的 KV**，GQA 的 head 映射
也塌缩到 KV head 0（97%）。误差看起来只有 4e-2，是因为随机 KV 的均值都很小且相近；
把 softmax 调尖（q×16）后误差跳到 **2.12**。

#### 17.9.2 为什么 Level 3 还能拿 29/30：cudagraph 冻结了 launch 配置

上面的结论与「两次 batch=8 的 L3 都是 29/30」直接冲突。不外推、直接测
（`flashinfer_cudagraph_check.py`）：

```
eager,            ctx=4096:  [1,1,1,1,1,1,1,1]   错
graph 捕获@1,     回放@4096:  [1,2,3,4,5,6,7,8]   对
graph 捕获@4096,  回放@4096:  [1,1,1,1,1,1,1,1]   错
```

**切不切分是 host 侧按 `max_seq_len` 决定的，这个决定会被烤进 CUDA graph。**
vLLM 为加速捕获而 `attn_metadata.seq_lens.fill_(1)`（`triton_attn.py`
`build_for_cudagraph_capture`），于是捕获时一律选中非切分路径，回放时即便真实
ctx 是 4k+ 也仍走那条正确路径。

所以：**`FULL_DECODE_ONLY` 下的 serve 从未命中这个 bug**。受影响的只有 eager
调用（关图、未捕获的尺寸、或 cudagraph_mode 改动）。

#### 17.9.3 门控的处置：判据必须带 ctx，不能只看 batch

最终形态 `_decode_is_unsafe(batch, max_seqlen_k)` = `batch < 32 and
max_seqlen_k >= 2048`，即 §17.7.4 的原始判据。中途改成「只看 batch」是
**过度收紧，已回滚**。

推理的关键是：**Python wrapper 只在捕获时或 eager 时被调用，回放不走 Python**，
而 launch 配置正是在这次调用时定下的。所以用「本次调用看到的 `max_seqlen_k`」
判断，在三种情形下都正确：

| 情形 | `max_seqlen_k` | 判定 | 结果 |
|---|---|---|---|
| 捕获（vLLM `fill_(1)`） | 1 | 安全 | 烤进非切分路径，回放任意 ctx 都对 |
| eager 预热 / eager decode | 真实值 | 不安全 | 退 Triton |
| 假如 vLLM 不再 `fill_(1)` | 真实值 | 不安全 | 捕获也会被拦住，仍安全 |

只看 batch 的版本会把低并发（如 batch=8）的 decode 全部赶到 Triton，**白白
丢掉 flashinfer 的 ~3×**，而换不来任何安全性。

当时就有佐证没读懂：ctx 判据那轮 `decode_unsafe_skip=252`（6 次），只看 batch
那轮是 504（12 次）。6 = 小于 32 的捕获尺寸数，12 = 预热 + 捕获。即**预热是
eager 带真实长 ctx（该拦），捕获是 ctx=1（不该拦）**——ctx 判据早就精确区分了。

拆批路径上同样复用它：mixed 批是 eager，没有捕获的庇护，且 decode 片比整批小，
更容易落进该区间（计数 `split_unsafe_skip`）。

> **教训（第十六条）**：**误差的结构比误差的大小信息量大得多。** 四个假设按
> 「数值」方向查全错，改看「哪些元素错」一步定位到索引。先问「错在哪儿」，
> 再问「错多少」。

### 17.10 拆批为何没有 e2e 收益：每层元数据被 FlagGems 接管，含 prefill 步变 host-bound（2026-10-01 深夜）

脚本在 `experiments/src/iluvatar_attn/`，文本结果在
`experiments/bench_results/iluvatar_attn/`（trace 本身 260 MB，不入库，原件在
`/root/bench_results/engine_profile/`）。

#### 17.10.1 官方 4k×2：四种门控配置逐轮对比

| 配置 | 第 1 轮 Total / TTFT | 第 2 轮 Total / TTFT | 汇总 Total | 汇总 TTFT |
|---|---|---|---|---|
| 门控 MIN_Q=512 | 3267.40 / 6177 | 3989.17 / 5454 | 3628.28 | 5815.52 |
| 门控 MIN_Q=256 | 3247.71 / 6283 | 4013.39 / 5634 | 3630.55 | 5958.41 |
| + reorder 拆批（batch-only 保护） | 3280.18 / 6483 | 3806.72 / 5807 | 3543.45 | 6145.00 |
| **最终：不拆批，ctx-aware 保护** | 3244.22 / 6342 | 3883.08 / 5790 | **3563.65** | **6066.06** |

门槛 Total ≥ 2007.73、TTFT ≤ 11689.27，四种全部 PASS。产物
`eval_4k/<配置>/`（`run.sh` / `env_check.txt` / `bench_results.txt`）。

读法：

* **ctx-aware 保护没有代价**：最终配置第 1 轮与 MIN_Q=256 只差 0.1%。
* **轮间波动 ~3%，大于配置差**：最终配置与 MIN_Q=256 在这个负载下 kernel 路径
  几乎相同（FA 只占 ~0.36% 的注意力调用），第 2 轮仍差 130 tok/s。所以
  §17.8.4 的「拆批 -2.4%」单看汇总数**不能**算显著。
* **但拆批确实更慢**：从 server 日志里只挑含 prefill 的 10 s 窗口，平均 prompt
  吞吐 MIN_Q=256 为 5366、最终配置 5507、**拆批 4867（约 -10%）**。

#### 17.10.2 微基准：host 下发开销不是瓶颈（假设被证伪）

`split_host_overhead.py`，单层，`issue` = 不同步时 CPU 入队耗时，`gpu` = CUDA
events；`eng-*` 经过插件 wrapper 与门控（引擎实际付的口径）。单位 ms。

| 形状 | 路径 | issue | gpu | kernels | ×42 issue | ×42 gpu | 瓶颈 |
|---|---|---|---|---|---|---|---|
| 63 decode + 1×q2048 | eng-tri | 0.143 | 4.244 | 1 | 6.0 | 178.3 | gpu |
| | eng-spl | 0.553 | 1.359 | 23 | 23.2 | 57.1 | gpu |
| 32 decode + 4×q512 | eng-tri | 0.142 | 4.161 | 1 | 6.0 | 174.8 | gpu |
| | eng-spl | 0.551 | 1.261 | 23 | 23.2 | 53.0 | gpu |

拆批每层 23 个 kernel、host 0.55 ms，仍远小于其 GPU 1.3 ms。按这张表，含
prefill 步的注意力应从 ~178 ms 降到 ~57 ms——与引擎里的 -10% 完全矛盾，说明
引擎里有微基准没覆盖到的东西。

#### 17.10.3 引擎 profile：拆批的含 prefill 步卡在 host

`run_profile.sh`：官方启动参数（`FULL_DECODE_ONLY`）+ vLLM `--profiler-config`
（torch，无 stack，`max_iterations=120`）。负载 4096/1024、64 并发、128 个请求；
**等第一波 prefill 完、进入纯 decode、再等第二波请求开始补进来**才开抓，即官方
负载的稳态形状「~63 decode + 1 chunk」。`analyze.py` 以 `execute_new_*` 标记
切步、以 `CompiledFxGraph` 调用界定 forward。

> 踩坑：第一版在 bench 开始 15 s 后就抓，那时 decode 行 <32，40 步全被
> `split_unsafe_skip` 退回 Triton，`split=0`——比的根本不是拆批。结果保留为
> `firstwave_analysis.txt`（下表第一行），可作 Triton 含 prefill 步的参照。

| 步类型（每步平均） | 步数 | 墙钟 | host forward | GPU 忙碌 | launch/步 | 同步 |
|---|---|---|---|---|---|---|
| 第一波，Triton 含 prefill（decode ~16–20 行） | 38 | 218.2 | 169.8 | 213.4 | 519 | 无 |
| **稳态，拆批含 prefill（~63 decode + chunk）** | 93 | **429.0** | **405.0** | **187.5** | 1359 | 无 |
| 稳态，不拆批（对照） | _待补_ | | | | | |

拆批步的 GPU 只忙 187.5 ms（GEMM 113、FA 26、flashinfer 21、其他 27），其余
~240 ms GPU 在空等 host。

#### 17.10.4 根因：每层重建元数据，被 FlagGems 接管成 ~20 个 Triton launch

`attn_host_breakdown.py`（`split1_attn_host_breakdown.txt`）：每次
`unified_attention_with_output` host 平均 **6.85 ms**，按自身耗时排前的全是
小张量算子（µs / 次注意力调用）：

| 算子 | self | 次数 |
|---|---|---|
| `aten::where` | 972 | 2 |
| `aten::add` | 694 | 2 |
| `aten::zeros` | 621 | 3 |
| `aten::full_like` | 517 | 2 |
| `sub` / `floor_divide` / `cumsum` / `lt` / `_to_copy` / `remainder` / `eq` / `arange` / `scatter_` | 各 190–370 | 各 1–2 |
| `cuLaunchKernel` 本身 | 136 | 19 |

kernel 名为证：每步 `where_inner_kernel_rank_1`、`full_func_scalar_kernel_rank_1`、
`reduce_then_scan_root_scan_kernel_row`(cumsum)、`floor_div_*`、`rem_ts_*`、
`eq_func_*`、`lt_func_*`、`arange_func`、`_scatter_jit_function` 各 42 或 84 次——
全是 **FlagGems 的 Triton pointwise kernel**，与 `_build_flashinfer_pages`
（indptr / indices / last_page_len）和 FA 的 `cu_seqlens` 计算一一对应。

* 引擎里 `VLLM_PLUGINS=fl` 让 FlagGems 接管 aten，每个小算子都走一遍 Python 层
  Triton launcher，在这台 CPU 上 300–1000 µs/个。
* 微基准没开 FlagGems，同样的算子走原生 aten，10–20 µs/个——所以量出 0.55 ms。
* 每步 42 层 × ~6 ms ≈ 250 ms host，远超注意力在 GPU 上省下的 ~60 ms。

**这些元数据在一步之内对 42 层完全相同**（只依赖 `block_table` / `seqused_k`）。

#### 17.10.5 下一步（§17.11 已完成 1–2）

1. ~~先用开了 FlagGems 的微基准复现~~ → `split_meta_once.py`：rebuild/layer
   issue 0.407 ms，reuse/layer 0.062 ms，步进 host 估计 17.1 → 2.9 ms（−83%）。
2. ~~把 flashinfer page 元数据与 FA `cu_seqlens` 挪到每步一次~~ → §17.11。
3. 纯 decode 走 CUDA graph，没有 host 开销，但同样 ~17 个小 kernel × 42 层每步
   被重放，GPU 上可能有几 ms/步的浪费；同一改法或许也能惠及 decode，待测。
4. 顺带：纯 prefill 判定里 `q_lens.min().item()` 每层一次同步（第一波 trace 里
   纯 prefill 步 43 次 `cudaStreamSynchronize`/步）。拆批 hook 已在 host 上
   知道 `num_decodes`，可以免同步。

> **教训（第十七条）**：**微基准要和引擎跑同一套算子派发。** 插件让 FlagGems
> 接管了 aten，同一行 `torch.where` 在微基准里 15 µs、在引擎里 500 µs。只量
> kernel 的「3× 更快」被 30× 的 host 放大完全吞掉。引擎 profile 才是裁判。

### 17.11 每步一次元数据：拆批 e2e +52%，默认开启（2026-10-02）

> **❗勘误（§17.13）**：本节的 4k 吞吐数字作废。`prebuild_split_meta` 在纯 decode
> 步也发布每步新分配的页表，被 FULL decode graph 捕获后，回放读的是已释放显存，
> decode attention 实际只扫了捕获时（`seq_lens=1`）或被复用的错误页表，decode
> 吞吐虚高约 2.7×。混合步（eager）的 host 优化与 profile 结论仍然成立。

#### 17.11.1 改动

FlagGems-comp：

* `prebuild_split_meta(...)`：在 builder 侧用当前步的 `block_table` /
  `seqused_k` / `cu_seqlens_q` / `page_size` 建好
  `(indptr, indices, last_page_len)`、`prefill_cu_k`、`prefill_cu_q`，与
  `set_batch_split` 一起 stash。
* `paged_attention_decode(..., pages=)`、`flash_attn_prefill(..., cu_seqlens_k=)`
  优先吃 stash；42 层复用，不再每层走 FlagGems pointwise 链。

插件：`patch_triton_attn_reorder_split_for_iluvatar` 的 `build` 钩子改为调
`prebuild_split_meta`（仍仅 Iluvatar；沐曦不受影响）。**默认 `SPLIT=1`**。

#### 17.11.2 引擎 profile（同 §17.10.3 稳态抓法）

| 步类型 | 墙钟 | host forward | GPU 忙碌 | launch/步 | 每次 attn host |
|---|---|---|---|---|---|
| 拆批含 prefill（修前，§17.10.3） | **429.0** | 405.0 | 187.5 | 1359 | **6.85 ms** |
| **拆批含 prefill（修后）** | **199.6** | 166.3 | 177.5 | 580 | **0.58 ms** |

产物 `engine_profile/split1_meta_analysis.txt`。注意力里排前的变成
`cudaLaunchKernel` / `permute` / `slice`，不再有 `where`/`zeros`/`cumsum`。
GPU 忙碌 ≈ 墙钟，host 不再是瓶颈。与 §17.10.5 预测的 429 → ~190 ms 吻合。

#### 17.11.3 官方 4k×2

| 配置 | Run1 Total/TTFT | Run2 Total/TTFT | 汇总 Total | 汇总 TTFT |
|---|---|---|---|---|
| 不拆批（§17.10.1 final） | 3244 / 6342 | 3883 / 5790 | 3564 | 6066 |
| 拆批每层重建（§17.8.4） | 3280 / 6483 | 3807 / 5807 | 3543 | 6145 |
| **拆批 + 每步预建** | **4709 / 5666** | **6146 / 4920** | **5428** | **5293** |

相对不拆批：**Total +52%**（3564 → 5428），TTFT 6066 → 5293。门槛
≥2007.73 / ≤11689.27 PASS。`split=38448 mixed_skip=0`。产物
`eval_4k/split_meta/`。

#### 17.11.4 正确性（math_500 Level 3 冒烟 30 题）

`eval-batch-size=8`、早停 30 题（与 §17.4 / batchgate 同口径）：

* **29/30 = 0.9667**，门槛 ≥0.95 PASS；错题 index=18
* 与 decode_only / batchgate 两次冒烟同为 29/30
* 产物 `mathL3/partial30_summary.txt`；`split=420` 有命中

#### 17.11.5 推荐

天数侧默认：`ATTN=1 SPLIT=1 MIN_Q=256` + ctx-aware flashinfer 保护 + KV-first HND
+ tile64 prefill。关拆批：`VLLM_FL_ILUVATAR_IXFORMER_SPLIT=0`。

> **教训（第十八条）**：**跨层不变的东西不要放进 per-layer 热路径。** 元数据只
> 依赖 `block_table`/`seqused_k`，却被重建了 42 次；在 FlagGems 接管 aten 的
> 环境下，每次重建都是一串 Triton launch。挪到 builder 一次，e2e 直接 +52%。

### 17.12 天数侧不让 FlagGems 接管 `aten::add`：4k +4.6%（2026-10-02）

> **❗勘误（§17.13）**：本节 4k 数字是在 §17.11 的 decode graph bug 上测的，
> 绝对值作废；profile 结论（add 的 host 税）仍然成立，修复后的重测见 §17.13。
> §17.12.5 的 mbt=4096「卡死」与 §17.13 的卡死症状相同，很可能是同一个 bug。

#### 17.12.1 Profile：注意力修好后，host 税转移到 residual add

沿用 §17.11.2 的稳态 trace，按 cpu_op 的 host **自身**耗时排序，对含 prefill
（发 FA kernel）的 85 步取平均（共 238 步，去掉 1 个 1.86 s 的首次 JIT 编译步；
含 prefill 步均 wall 180.1 ms；脚本
`engine_profile/host_self_by_op.py`，输出 `engine_profile/split1_host_self_by_op.txt`）：

| 算子 | 次/步 | host self ms/步 | µs/次 | 备注 |
|---|---|---|---|---|
| `aten::mm` | 168 | 42.0 | 250 | GEMM 本身，不算税 |
| **`aten::add`** | **89** | **30.2** | **339** | GPU 仅 0.059 ms/次（5.1 ms/步） |
| `RmsNorm` | 85 | 18.7 | 220 | |
| `vllm::unified_attention_with_output` | 42 | 16.3 | 387 | §17.11 后已以 kernel 下发为主 |
| `vllm_fl::fused_add_rms_norm` | 84 | 10.4 | 123 | |
| `vllm::unified_kv_cache_update` | 42 | 9.9 | 235 | |

`add` 的 kernel 名是 `add_func_kernel_rank_1`——FlagGems 的 Triton pointwise。
与 §17.10.4 同一机制：每次调用走 Python 层 Triton launcher，host 是 GPU 的 ~6×。
NVIDIA 后端早就把它放进 `CUSTOMIZED_UNUSED_OPS = ("add", "cos")`，天数侧是空的。

#### 17.12.2 改动

FlagGems-comp `_iluvatar/__init__.py`：`CUSTOMIZED_UNUSED_OPS = ("add",)`。
只作用于天数 vendor，沐曦与其他后端不变；`get_unused_ops()` 返回 `['add']`。
开插件后 `x + y`（2048×2048 bf16）host 下发 **0.007 ms/次**（原 ~0.37 ms）。

#### 17.12.3 官方 4k×2（相对 §17.11.3）

| 配置 | Run1 Total / TTFT | Run2 Total / TTFT | 汇总 Total | 汇总 TTFT |
|---|---|---|---|---|
| 拆批 + 每步预建（§17.11） | 4709 / 5666 | 6146 / 4920 | 5427.76 | 5293.03 |
| **+ FlagGems 不接管 add** | **5277 / 5494** | **6083 / 4903** | **5680.24** | **5198.66** |

**Total +4.6%**，TTFT −94 ms，两项门槛 PASS。增益集中在 Run1（+12%，prefill
占比高的冷轮），Run2 基本持平（−1%，在 §17.10.1 测得的 ~3% 轮间波动内）——
与「只省 eager 含 prefill 步的 host」一致：纯 decode 走 CUDA graph，不受影响。
P99 TTFT 不变（27.8 s / 25.0 s），仍是 §7.4.1 的波次排队。产物 `eval_4k/noadd/`。

#### 17.12.4 正确性（math_500 Level 3 冒烟 30 题）

**29/30 = 0.9667**，PASS；错题 index=18，与 §17.11.4 完全相同。
产物 `mathL3_noadd/partial30_summary.txt`。

#### 17.12.5 ❌ 试过但未采用：`max_num_batched_tokens` 2048 → 4096

动机：vLLM 给 <70 GB 设备的 OpenAI-API 默认 `mbt=2048`，4096 token 的 prompt
要切两刀；本版 vLLM 又拒绝 Concurrent Partial Prefill（`arg_utils.py`
`_check_feature_supported`），`max_num_partial_prefills` 锁死为 1。想用 4096
让每个 prompt 一步 prefill 完，改善 TTFT。

实测（插件 patch `EngineArgs.get_batch_defaults`，日志确认
`Chunked prefill is enabled with max_num_batched_tokens=4096`）：Run 1 第一波
prefill 后进入 decode，到 19:11:50 **引擎停止前进**——`Running: 49 reqs`、
生成吞吐 0、`server.log` 8 分钟不再更新，而 GPU 一直 100% 占用、显存满载。
不是 OOM、没有 traceback，只能强杀。根因未查（猜测是某个 4096 量级 eager
形状在 ixformer/Triton 上死循环或死锁）；§7 时期 mbt=8192 也曾跑出 0 tok/s 的轮次。

**处置**：不提交该 patch。若再试，先用小负载单独复现卡死的形状。

> **后续（§17.14）**：卡死是 §17.13 的 decode graph 悬空页表，与 mbt 无关；
> 修复后 mbt=4096 不再卡死，4k 再 +3.7%，已默认开启。

> **教训（第十九条）**：**host 税是一类问题，不是一个点。** §17.10 把注意力
> 里的 FlagGems pointwise 挪走后，同一种税立刻在 residual `add` 上冒头。修完
> 一处就重新 profile，按 host self-time 排序找下一个，而不是只盯 GPU。

### 17.13 ❗勘误：decode graph 读了每步重分配的页表；修复 + eager 小算子回原生（2026-10-02 晚）

#### 17.13.1 现象

扩大 FlagGems 排除列表后的冷启动 4k 跑到第一波请求结束（batch 64 → 48）时
**引擎卡死**：`Running: 48`、生成吞吐 0、EngineCore 主线程停在
`hrtimer_nanosleep`（等 GPU 同步，CPU 0%），GPU 持续 87% 忙——某个 kernel
跑不完。与 §17.12.5 mbt=4096 的卡死完全同一症状。

#### 17.13.2 根因

`patch_triton_attn_reorder_split_for_iluvatar` 的 `build` 钩子在**每次** build
（含 cudagraph 捕获前那次）调 `prebuild_split_meta`；纯 decode 时它把新分配的
`(indptr, indices, last_page_len)` 放进 `_BATCH_SPLIT`，decode 分支
`pages=_cached_decode_pages(...)` 直接拿去用。FULL decode graph 因此把**捕获
那一刻的页表地址**烤进了 graph；之后每步 builder 都分配新张量，旧块回到分配器、
被别的张量复用。回放时 flashinfer 读到的是：

* 捕获时（vLLM 捕获用 `seq_lens.fill_(1)`）每序列 1 页的页表——attention 只扫了
  极短上下文，decode 快得离谱；
* 或被复用的垃圾——`indptr` 变成大数，kernel 扫巨大范围：轻则每步 ~1 s，重则不停。

复现：`tests/test_iluvatar_kv_cache_layout_views.py::
test_decode_graph_replay_ignores_the_per_step_page_stash`（HND 存储、batch 48、
捕获后改 `seq_lens`/`block_table` 并重新 prebuild、搅动分配器再回放）：修前
**98.2% 元素不符**（最大误差 3.92）；修后通过。

#### 17.13.3 修复（FlagGems-comp `ixformer_unified_attention.py`）

* `_cached_decode_pages` 在 `torch.cuda.is_current_stream_capturing()` 时返回 None。
* 捕获时 decode 分支改用 `_graph_decode_pages`：第一层在 graph 内用持久的
  `block_table`/`seq_lens` 建页表（kernel 被捕获，回放时按当前长度重算），其余
  41 层复用同一组张量；`set_batch_split` / `clear_batch_split`（builder 每步、每次
  捕获前都会调）清空它。graph 内每步只多 ~20 个小 kernel，而非每层一次。
* eager decode 与拆批混合步照旧复用 builder 的每步 stash（同一步内生产消费，安全）。

#### 17.13.4 decode 吞吐：+52% 主要是 bug

64 并发纯 decode 时 `Avg generation throughput` 峰值（各次 server.log）：

| 阶段 | 生成 tok/s |
|---|---|
| §17.11 前（kvfirst / split / final 等） | 1540–1590 |
| §17.11–17.12（split_meta / noadd，有 bug） | 3800–4430 |
| **修复后** | **1730–1890** |

修复后比 §17.11 前快 ~15%（graph 内每步建一次页表，而非每层一次），这部分是真的。
压测是随机 prompt + `--ignore-eos`，不校验输出，所以虚高没被吞吐脚本发现；L3 冒烟
29/30 是小 batch 下分配器恰好把新 stash 放回同一地址的运气，不能作为正确性证据。

#### 17.13.5 第一轮慢尾巴：FlagGems `index` 按形状 JIT

修复后（只排除 `add`）仍有尾巴：第一波结束后 batch 43 → 0 时生成吞吐 45–60 tok/s，
拖 ~60 s。worker 的 `TRITON_CACHE_DIR` 在 `VLLM_CACHE_ROOT/torch_compile_cache/<hash>/
inductor_cache/triton`，按时间统计新编译的 kernel：

* 第一轮尾巴（20:53:50–20:55:00）：`_index_linearized_jit_function` **70 个** + `_copy_kernel` 10 个
* 第二轮（暖）尾巴：仍有 9 个 `index`（新出现的 batch 形状）
* 第一轮开跑第一分钟：~95 个小 pointwise 编译（copy/_to_copy/index/sub/eq/floor_div/
  full/rem/where/cumsum…），即冷启动 TTFT 的大头

FlagGems `index` 的代码缓存 key 含**完整输入形状**（`inp_shape_..._m_.._n_..`），每个
新的 `[num_tokens, hidden]` 都生成新模块并 JIT（~1 s）。插件代码一改，compile hash
变、Triton 缓存目录全新，第一轮就要重付这笔账——这也解释了历次 Run1 的大幅波动。

#### 17.13.6 改动：eager 小算子回原生 aten（仅天数 vendor）

FlagGems-comp `_iluvatar/__init__.py`：`CUSTOMIZED_UNUSED_OPS` 从 `("add",)` 扩为
`add, sub, index, copy_, to_copy, eq, eq_scalar, floor_divide, remainder,
where_self, cumsum, full, full_like, fill_scalar_, zeros, zeros_like, zero_`
（按 FlagGems 函数名匹配）。`mm`/`linear`、采样、融合 kernel 仍走 FlagGems。
单次 host：原生 9–32 µs vs FlagGems 130–150 µs（`unused_check.txt`）。沐曦不受影响。

#### 17.13.7 官方 4k×2（均为修复后、全新 `VLLM_CACHE_ROOT` 冷启动）

| 配置 | Run1（冷）Total / TTFT | Run2（暖）Total / TTFT | 汇总 Total | 汇总 TTFT |
|---|---|---|---|---|
| 只排除 `add` | 3749.62 / 8941 | 4697.09 / 4547 | 4223.35 | 6743.88 |
| **+ 16 个 eager 小算子** | **5447.15 / 6651** | **5674.99 / 4320** | **5561.07** | **5485.12** |

整个服务生命周期（启动 + 两轮）新编译的 Triton kernel：444 → **117**，其中
`index` 140 → **0**（`triton_compiles_by_minute.txt`）。

**Total +31.7%，TTFT −18.7%**；暖的 Run2 也 +21%，不只是省了 JIT。尾巴消失
（Run1 第一波结束 10 s 内 64 → 0）。P99 TTFT 43.9 → 33.9 s（Run1）、25.0 → 25.0 s
（Run2）。两门槛 PASS。修复后真实水平相对 §17.11 前（final 汇总 3564）为 +56%。
产物 `eval_4k/cold_fix_base/`、`eval_4k/cold_fix_ext/`。

#### 17.13.8 正确性（math_500 Level 3 冒烟 30 题，修复 + 16 个 eager 小算子）

**29/30 = 0.9667**，门槛 ≥0.95 PASS；错题 index=18，与历次冒烟相同。
`split=294 decode=2688 layout=0`。产物 `mathL3_fix_ext/`。

另：site-packages 里插件天数 vendor 的 `__init__.py`、`patches/__init__.py`、
`patches/topk_topp_sampler.py` 一直停在 09-27 版（缺 `b8dd70a` 的平台门控），其余
文件与仓库一致。差异仅为 Iluvatar 门控，天数上等价；同步后起服务验证
`topp_fast coverage: fast=64 fallback=0 enabled=True`（`topp_check_stats.txt`）。

#### 17.13.9 撤回 / 未采用

* ixformer `residual_rms_norm` 替代 FlagGems rmsnorm（host 170 → 45 µs/次）：属于
  纯算子替换且绕开 `call_op`，与赛题「替换算子而无实际优化」条款风险高，**不提交**。
* mbt=4096：§17.12.5 的卡死确是本 bug，修复后重测见 §17.14。

> **教训（第二十条）**：**凡是喂给 CUDA graph 的张量，地址必须跨步稳定。**
> 「每步一次」的缓存对 eager 是优化，对 graph 是悬空指针。吞吐脚本不校验输出，
> 只看 tok/s 会把错误计算当成加速；decode 吞吐突然翻倍时，先怀疑 graph 读了什么。

### 17.14 `max_num_batched_tokens` 2048 → 4096：修复后不再卡死，4k +3.7%（2026-10-02 晚）

#### 17.14.1 改动（插件 `vendor/iluvatar/iluvatar.py`，仅天数）

`patch_openai_api_mbt_for_iluvatar()`：包一层 `EngineArgs.get_batch_defaults`，
只把 `UsageContext.OPENAI_API_SERVER` 的默认 mbt 抬到 4096（已 ≥4096 不动；
用户显式传 `--max-num-batched-tokens` 时 vLLM 不走默认值，不受影响）。
`_is_iluvatar_platform()` 门控，沐曦不受影响；不改 `vllm serve` 参数。
`VLLM_FL_ILUVATAR_MAX_NUM_BATCHED_TOKENS` 可覆盖目标值，`0` 关。

作用：4096 token 的 prompt 一步 prefill 完（2048 要切两刀，且本版 vLLM 锁
`max_num_partial_prefills=1`）。

#### 17.14.2 官方 4k×2（修复后、全新 `VLLM_CACHE_ROOT` 冷启动，同 §17.13.7 口径）

| 配置 | Run1（冷）Total / TTFT / P99 | Run2（暖）Total / TTFT / P99 | 汇总 Total | 汇总 TTFT |
|---|---|---|---|---|
| mbt=2048（§17.13.7） | 5447.15 / 6651 / 33944 | 5674.99 / 4320 / 24956 | 5561.07 | 5485.12 |
| **mbt=4096** | **5648.52 / 6510 / 30363** | **5888.06 / 4264 / 21927** | **5768.29** | **5387.05** |

**Total +3.7%，平均 TTFT −1.8%，P99 TTFT −10.5% / −12.1%**；duration 240.6 →
232.1 s（Run1）、231.0 → 222.6 s（Run2）。prefill 峰值 11.5k → 13.5k tok/s；
TPOT 51.3 → 49.8 ms。中位 TTFT 反而 1.27 → 1.84 s（每步 prefill 更大，排在后面的
请求多等一步），但均值与 P99 都降。全程无卡死，`index` 新编译仍为 0（总 116）。
两门槛 PASS。产物 `eval_4k/cold_fix_ext_mbt4096/`。

#### 17.14.3 正确性（math_500 Level 3 冒烟 30 题，默认值路径）

不设环境变量、只靠插件默认值起服务（日志确认
`Chunked prefill is enabled with max_num_batched_tokens=4096`）：
**29/30 = 0.9667**，PASS；错题 index=18，与历次相同。产物 `mathL3_mbt4096/`。

注意：第一次跑时 site-packages 的插件仍是默认 `"0"` 的旧版（编辑与拷贝并行，
拷到了旧文件），实际是 mbt=2048，结果作废重跑。起服务前务必 `cmp` 仓库与
site-packages，并在 `server.log` 里确认生效值。

#### 17.14.4 附带观察：小 batch 长 ctx 的 decode 很慢（待查）

Level 3 冒烟（batch=8、max_tokens 32768）中，`Running: 8` 不变而生成吞吐
200 → 146 → 120 → 105 → 94 → 86 → … → 52 tok/s（每分钟一点），近似 ∝ 1/√t，
即单步耗时 ∝ ctx、几乎无固定项。mbt=2048 的上一轮曲线几乎逐点相同，与本改动无关。
粗估 ~9k ctx 时 8 条序列每步 ~150 ms，KV 有效带宽仅 ~20 GB/s。疑为该形状下
decode attention 不沿序列维拆分、并行度不足（`batch<32 and ctx≥2048` 走 Triton
保护分支，见 §17.5）。对官方 4k 影响小（batch 64、ctx ≤5k），16k 需实测。
### 17.15 官方 16k×2 摸底（§17.14 配置，冷 `VLLM_CACHE_ROOT`，2026-10-02 晚）

| | Run1（冷） | Run2（暖） | **汇总** | 门槛 | 基线（§2） |
|---|---|---|---|---|---|
| Total tok/s | 2583.53 | 2609.65 | **2596.59** | ≥906.00 | 915.15 |
| Mean TTFT (ms) | 158454.73 | 155213.16 | **156833.95** | ≤605254.65 | 599262.03 |
| duration (s) | 862.47 | 853.84 | | | 2434.82 |

吞吐 **2.84× 基线**，TTFT 为基线 26%；128/128 成功，两门槛 PASS。Run2 只比 Run1
快 1%，冷 JIT 对 16k 不重要。其他：Median TTFT 170.5 s、P99 TPOT ~228 ms、
Median ITL ~47 ms、P99 ITL ~1.1 s。产物 `eval_16k/cold_mbt4096/`。

16k 的时间去向（`engine_timeline.txt`）：

1. **KV 容量封顶并发 ≈30**：`GPU KV cache size: 521,808 tokens`，每请求 ~17.4k →
   最多 ~30 条同时在跑，另 34 条排队，TTFT 主要是排队。容量由
   `--gpu-memory-utilization` 决定，属 serve 参数，不能动。
2. **长 ctx prefill ~4.9k tok/s**（4k 场景峰值 13.5k）：每步一个 4096 token 的
   prefill 块 ~0.83 s；这类混合步里 ~30 条 decode 合计只出 ~32 tok/s。纯 decode
   段 R=30 时 ~650 tok/s（~46 ms/步）。
3. **小 batch 尾巴 ~210 s/轮（≈25% 时长）**：最后 6–7 条请求纯 decode，合计仅
   ~27 tok/s（~220 ms/步），比 R=30 时每步还慢 ~5×。§17.14.4 Level 3 冒烟的吞吐
   衰减是同一现象。

> **更正（同晚，查 dispatch 计数）**：上面「捕获时保护不会触发、跑的是 flashinfer」
> 是错的。`decode_unsafe_skip=252 = 6 × 42 层`，全部发生在 graph 捕获阶段（probe
> 计数在 call≈2518–2940 间从 0 涨到 252，之后不再变）。35 个捕获尺寸每个调用两次：
> ≥32 的 29 个尺寸两次都走 flashinfer（58 次），<32 的 6 个尺寸（1/2/4/8/16/24）
> 一次 flashinfer、一次 unsafe（6 + 6 次），合计 decode 64 × 42 = 2688、unsafe
> 6 × 42 = 252，与日志一致。即这 6 个尺寸被捕获时传入的 `max_seqlen_k ≥ 2048`，
> **graph 里固化的是 Triton decode**，而插件又把 `NUM_PAR_SOFTMAX_SEGMENTS`
> 16 → 1（不沿 KV 维拆分）。于是 R=30（padding 到 32）→ flashinfer ~46 ms/步，
> R=7（padding 到 8）→ Triton ~220 ms/步，正好对上。§17.9「捕获时 seq_lens=1，
> 所以固化的是短 ctx 的正确路径」对 batch<32 的捕获不成立。

4. **混合步的 decode 段整批留在 Triton**：16k `split=0`、`split_unsafe_skip=44084`
   ——每个混合步的 decode 段都 <32 条且 ctx 长，拆批被保护拒绝，整批走 Triton
   tile64；这是 ② 里 decode 合计只出 ~32 tok/s 的直接原因。4k 也有一部分
   （`split=18864`、`split_unsafe_skip=2688`）。

下一步：让 batch<32、长 ctx 的 decode 不再落到不拆 KV 的 Triton。候选见 §17.16。

### 17.16 小 batch 长 ctx decode：四条现成路线全部走不通（2026-10-02 深夜）

目标是 §17.15 的尾巴：batch<32、ctx≈17k 的 decode 每步 ~220 ms。逐条试：

#### 17.16.1 ❌ 截断 `max_seq_len` 让 flashinfer 走非切分 launch

`flashinfer_clamp_check.txt`（n=8，clamp=2047）：eager 误差 **1.1e3 ~ 2.2e4**，
graph 回放直接 **NaN / 1e37**。kernel 按 `max_seq_len` 分配片上缓冲，真实 ctx
超过它就越界。§17.9.2「捕获@1、回放@4096 正确」是 8 行读出测试的结论，**不能
外推**到 clamp 用法。

#### 17.16.2 ❌ 把 decode 批补齐到 32 行

`flashinfer_pad32_e2e_check.txt`：速度确实到位——n=8/ctx=17408 时 graph 回放
**0.436 ms/层 vs Triton 4.363 ms/层（10.0×）**，×42 层 ≈ 18 ms/步 vs 183 ms/步，
与实测尾巴吻合。但正确性不行：padding 行把 batch 撑到 32 并不能消除 §17.9 的
错读（`flashinfer_repeat_check.txt`：n=4/ctx=2048 padded，**49/50 次** 误差 >0.1，
最大 2.42）。

#### 17.16.3 ❌ ixformer 的其它 paged decode 入口

`ixformer_decode_kernel_probe.txt`：`paged_attention`（algo=-1/0/1）对本卡的
HND 视图 **全部输出 NaN**；`paged_attention_v7` 在 Python 层被重载到
`vllm_paged_attention`，签名不兼容；`vllm_paged_attention` 要求 block_tables
为 int32 contiguous，传 int64 直接 `CHECK_TRUE ERROR`，传 int32 仍不符其内部
检查。都不可用。

#### 17.16.4 ❌ 按 KV head 拆成多次 flashinfer 调用

`flashinfer_gqa_ratio_probe.txt`：group=1（每 KV head 1 个 q head）**0/60 出错**，
group≥2 开始出错——错的是 GQA head 映射，与 §17.9 同源。但 group=1 要调 8 次：
n=64/ctx=4096 合计 **3.368 ms vs Triton ~2.1 ms**，更慢。

#### 17.16.5 ⚠ 勘误：所谓「flashinfer 偶发算错」大半是测量放大

中途一度认为 batch≥32 的线上路径也在偶发算错（`flashinfer_repeat_check.txt`、
`flashinfer_graph_repeat_check.txt`：相同输入重复调用，若干次误差 O(1)）。
`decode_determinism_probe.txt` 推翻了这个归因：**Triton 2D 在同样条件下也不确定**，
而 Triton 2D 没有 split/atomic，不可能真的 race。

`decode_nondeterminism_scope.txt` 定位到探针本身：q 乘了 4 使 softmax 近似 one-hot，
top-1/top-2 接近时 1 ulp 差别就让整行跳变。用正常量级 q：

| 用例 | kernel | self_max | self_mean | ref_max |
|---|---|---|---|---|
| n=32 ctx=2048 | flashinfer | 0.021 | 2e-5 | 0.021 |
| n=32 ctx=2048 | triton_2d | 0.029 | 2e-5 | 0.029 |
| n=64 ctx=4096 | flashinfer | 0.014 | 4e-6 | 0.014 |
| n=64 ctx=4096 | triton_2d | 0.011 | 4e-6 | 0.012 |
| **n=8 ctx=17408** | **flashinfer** | **0.003** | **2.3e-4** | **0.074** |
| **n=8 ctx=17408** | **triton_2d** | **0.000** | **0.0** | **0.000** |

batch≥32 两者无差别，均值都在 bf16 噪声内——**线上 decode 没有偶发错误，已提交的
4k/16k 成绩有效**。唯一真实的异常是 n=8 长 ctx 的 flashinfer：Triton 逐位确定
（self_max 恰为 0），flashinfer 每次都不同且 ref_max 0.074 明显高于噪声，即
§17.9 的 batch<32 bug，门控已经挡住。

> **教训（第二十一条）**：**放大输入来「让 bug 显形」会自己造出 bug。** 近似
> one-hot 的 softmax 把 1 ulp 放大成 O(1)，任何 kernel 都「不确定」。判定非确定性
> 要比较同一 kernel 自身的重复输出（self_max），并同时看均值，不要只看 max 与
> 参考的差；还要有一个已知确定的对照（这里是 Triton 2D）。

#### 17.16.6 结论与下一步

现成路线用尽：flashinfer 在 batch<32 不可用（§17.9），其余 ixformer 入口不可用，
Triton 在天数被 `NUM_PAR_SOFTMAX_SEGMENTS=1` 限制为不拆 KV。剩下的是**在 FlagGems
`_iluvatar` 里自研 split-K decode Triton kernel**（沿 KV 维拆分 + 二段归约），
属于自研实现而非算子替换。微基准表明上限约 10×（§17.16.2 的 0.436 vs 4.363 ms），
对 16k 约 25% 时长、以及 Level 3 这类长生成有效，对官方 4k 影响小 → 见 §17.17。

### 17.17 自研 split-K paged decode kernel：4k +2.6%、TTFT −17.4%（2026-10-02 深夜）

#### 17.17.1 实现

`FlagGems-comp` 新增 `_iluvatar/fused/paged_decode_splitk.py`，两个 Triton kernel：

* `_splitk_partial_kernel`：grid `(num_seqs, num_kv_heads, num_splits)`，每个
  program 处理「一条序列 × 一个 KV head × KV 区间的一段」，在线 softmax 得到局部
  `(acc, m, l)`。MiniCPM5-2B 全 42 层都是 16:2 GQA，所以一个 KV tile 同时喂 8 个
  q head（tile 补到 16 行、多余行 mask），KV 只读一次。
* `_splitk_reduce_kernel`：跨 split 归约，`l == 0` 的空 split 自动退出。

关键约束：**split 数只能由 batch 形状决定，不能看 ctx**。vLLM 捕获 decode graph
时 `seq_lens` 填 1，grid 在捕获时冻结，真实长度回放才写入；kernel 在 device 上读
`seqused_k`，所以任意 split 数都正确，只影响占用率。局部缓冲按形状缓存且**永不
回收**（§17.13 教训二十）。

#### 17.17.2 调参（GPU 空闲时测，`splitk_config_sweep.txt` / `splitk_heuristic_sweep.txt`）

`BLOCK_N=32`、`num_warps=4`、`num_stages=1`。`num_stages=2` 更慢（多一份 KV
缓冲挤占 shared memory），`BLOCK_N=256` 直接 OutOfResources（需 135 KB > 128 KB）。
split 数取 `clamp(pow2(256 / (num_seqs × num_kv_heads)), 1, 32)`，实测各档都在
最优值 5% 以内。

单层调用耗时（ms），对比它替换掉的 Triton：

| batch | ctx=4096 Triton / splitk | ctx=17408 Triton / splitk | 提升 |
|---|---|---|---|
| 1 | 0.987 / 0.037 | 4.214 / 0.103 | 27–41× |
| 4 | 1.011 / 0.078 | 4.312 / 0.284 | 13–15× |
| 8 | 1.190 / 0.143 | 4.398 / 0.547 | 8.0–8.3× |
| 16 | 1.206 / 0.277 | 4.445 / 1.068 | 4.2–4.6× |
| 31 | 1.376 / 0.524 | 5.096 / 2.047 | 2.5–2.7× |

batch≥32 时 splitk 比 flashinfer 慢（n=64/ctx=4096：1.04 vs 0.42 ms），所以**只在
flashinfer 被门控拒绝时启用**，大 batch 路径完全不变。

#### 17.17.3 接入点（两处，均在 `_decode_is_unsafe` 为真时）

* 纯 decode：原 `decode_unsafe_skip` → `decode_splitk`。
* 混合批：原来整批退回 Triton（`split_unsafe_skip`），现在 decode 段走 splitk、
  prefill 段照旧走 FA（`split_splitk`）。

`VLLM_FL_ILUVATAR_SPLITK_DECODE=0` 可关。门控 `_splitk_is_usable` 要求 head_dim
为 2 的幂且连续、GQA 整除，否则回退。

#### 17.17.4 官方 4k×2（冷 `VLLM_CACHE_ROOT`，同 §17.14.2 口径）

| 配置 | Run1 Total / TTFT / P99 | Run2 Total / TTFT / P99 | 汇总 Total | 汇总 TTFT |
|---|---|---|---|---|
| §17.14 mbt=4096 | 5648.52 / 6510 / 30363 | 5888.06 / 4264 / 21927 | 5768.29 | 5387.05 |
| **+ split-K** | **5839.19 / 5135 / 24174** | **6000.94 / 3764 / 19278** | **5920.06** | **4449.53** |

**Total +2.6%，平均 TTFT −17.4%**，P99 TTFT −20% / −12%，duration 232.1→224.5、
222.6→218.4 s。两门槛 PASS。dispatch：`split_unsafe_skip` 2688 → **0**，
`split_splitk` 2688、`decode_splitk` 252（捕获期 6 个小尺寸 × 42 层）。
收益来源是混合步——4k 的纯 decode 几乎都是 batch 64，走 flashinfer 未变。
产物 `eval_4k/splitk/`。

#### 17.17.5 官方 16k×1（冷 `VLLM_CACHE_ROOT`；§17.15 两轮差 <1%，故单轮）

| | §17.15 基准（两轮汇总） | **+ split-K** | 变化 |
|---|---|---|---|
| Total tok/s | 2596.59 | **4730.20** | **+82%** |
| Mean TTFT (ms) | 156833.95 | **104738.59** | **−33%** |
| duration (s) | 862.47 / 853.84 | **471.06** | **−45%** |
| P99 TTFT (ms) | 312860.97 | 214090.45 | −32% |
| Mean TPOT (ms) | 141.83 | 92.61 | −35% |
| Median ITL (ms) | 47.03 | 47.25 | 持平 |

128/128 成功，两门槛 PASS；相对 §2 基线 915.15 从 2.84× 提到 **5.17×**。
Median ITL 不变印证了收益来源：纯 decode（batch 30，走 flashinfer）未动，省下的
全是 §17.15 的两处——

1. **混合步**：`split_unsafe_skip` 44084 → **0**，`split_splitk=21980`；
   prefill 4915 → **8192–9830 tok/s**（混合步里的 decode 段不再整批退回 Triton）。
2. **尾巴**：只剩 7 条请求时 32 → **247 tok/s**（~220 → ~28 ms/步），约 210 s 的
   尾巴基本消失，与微基准 batch 8 的 8× 一致。

产物 `eval_16k/splitk/`。

#### 17.17.6 测试

`tests/test_iluvatar_paged_decode_splitk.py`（16 个，全过）：对 fp32 参考的误差、
split 数无关性（1/2/8/64 同解）、长度参差不跨序列泄漏、极短 ctx（1 token / 1 页 /
短于一个 split）、graph 捕获@1 回放@真实长度、graph 回放经分配器扰动后仍正确、
缓冲区按形状复用、split 数不依赖 ctx。原有 33 个测试仍全过。

#### 17.17.7 正确性（math_500 Level 3 冒烟 30 题，默认开启）

不设环境变量、只靠默认值起服务：**29/30 = 0.9667**，门槛 ≥0.95 PASS；错题
index=18，与历次完全相同。dispatch `decode_splitk=252`、`decode_unsafe_skip=0`、
`split_unsafe_skip=0`。产物 `mathL3_splitk/`。

顺带印证了 §17.14.4 的衰减现象同源：这一轮 **5 分钟**跑完 30 题（mbt=4096 那轮
27 分钟），batch 8 稳定在 **319–323 tok/s**（之前同阶段 52–85 tok/s）。
batch<32 的长生成正是本 kernel 的主场。

### 17.18 裁剪 decode CUDA-graph 捕获尺寸：≤64 + spare 72（2026-10-03）

#### 17.18.1 动机与改动

官方 bench `--max-concurrency 64`；16k 场景 KV 约 30 并发。默认
`cudagraph_capture_sizes` 到 512（51 档，decode 实际捕获到 `max_num_seqs=256`
共 35 档），80–256 的图在计分负载下几乎从不命中，却占图内存估计、拉长捕获。

插件 `patch_cudagraph_capture_sizes_for_iluvatar()`（仅天数）：在
`VllmConfig._set_cudagraph_sizes` 之后保留全部 `≤64`，再加一档 spare（默认
**72**，`VLLM_FL_ILUVATAR_CUDAGRAPH_SPARE`，`0` 关闭）。结果
`[1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72]`。不改 `vllm serve` 参数。

#### 17.18.2 实测

| | 默认（≤512） | **裁剪后（≤64+72）** |
|---|---|---|
| 捕获档数 | 35（到 256） | **12/12** |
| GPU KV cache | **521,712** tokens | **521,712** tokens |

产物 `mem_probe_kv/`、`mem_probe_cg72/`。图内存估计裁掉了，但本机 profiler 路径下
KV 池未放大——扩 16k 并发的目标未达成；裁剪本身仍合理（少捕无用图、启动更快、
burst>64 仍有 72），故保留。

#### 17.18.3 后续

KV 扩并发暂搁置。下一优先：启动期 eager 路径预热，把计分轮内剩余 Triton JIT
挪到 READY 之前（§17.13.7 生命周期约 117 次编译；split-K 后 Run1/Run2 Total
仍 5839 vs 6000）。

### 17.19 剩余头寸 profile + lm_head 两条死路（2026-10-03）

基线：§17.17 split-K + §17.18 cg≤64+72。官方 4k Summary **5920.06**（Run1 5839 /
Run2 **6000.94**）。FULL_DECODE_ONLY 下 decode 图吃掉 rms / kv_update / RoPE host。

#### 17.19.1 当前热路径（`baseline_remain_profile.py`）

Prefill（1×长，`max_tokens=1`）：扣掉 `aten::mm`/`mm_kernel` 双计后 GPU ~111 ms，
**GEMM ~72%**；FA 4.8 ms；silu 6 ms；rms / KV write <2%。CPU 大头是 `aten::mm`
host、`RmsNorm`+`fused_add_rms_norm`、`unbind`/`expand`、`unified_kv_cache_update`。

Decode（64×短，`max_tokens=8`）：图内 FFN GEMM 不再以 `mm_kernel` 出现。图外能
看到的几乎只剩 **lm_head `aten::linear` / `linear_kernel`**：8 step × ~1.13 ms
≈ 9 ms。`lm_head` 在 `compute_logits`，不进 decode CUDA graph。

#### 17.19.2 微基准（M×N×K = M×130560×2048，bf16）

| M | FlagGems `linear_kernel` | FlagGems `mm(x,W.T)` | ixBLAS `F.linear`（未 enable gems） |
|---|---|---|---|
| 8 | **数值错**（vs CPU max~42） | ~0.82 ms，对 | ~0.98 ms 量级 |
| 64 | 1.14 ms，对 | 1.03 ms（1.10×） | 0.98 ms |
| 72 | 1.20 ms | **1.87 ms（0.64×，key 72→96）** | — |

`linear()` 把 `SPLIT_K` 写死为 1；`BLOCK_M` wrap 在 M=8 上错。不能从 FlagGems
aten 实现里调 `torch.mm`（递归），只能调同包 `mm()`。

#### 17.19.3 尝试 A：`linear()` 把无 bias 且 (N<65k 或 M≤64) 转去 `mm()`

改动在 `FlagGems-comp` `_iluvatar/ops/linear.py`，已 **revert**。4k×2 暖缓存
（`eval_iluvatar_4k_lmhead_mm` / `raw_runs_20261003_161930.csv`）：

| | Run1 Total / TTFT | Run2 Total / TTFT | Summary |
|---|---|---|---|
| §17.17 冷 | 5839 / 5135 | **6000.94** / 3764 | 5920 / 4449 |
| mm-route 暖 | 5887 / 5092 | 6017 / 3737 | 5952 / 4414 |

Run2 **+0.3%**，在噪声里。官方 `RUNS=4 SKIP_FIRST=1` 只吃暖轮，这条没有提交价值。

#### 17.19.4 尝试 B：`CUSTOMIZED_UNUSED_OPS` 加 `"linear"`，lm_head 走 ixBLAS

同样只改天数 vendor。4k×2 暖缓存（`eval_iluvatar_4k_nolinear` /
`raw_runs_20261003_163650.csv`）：

| | Run1 | Run2 | Summary |
|---|---|---|---|
| unused linear | 6009 / 3896 | **6019.71** / 3684 | 6015 / 3790 |

Run2 相对 6000.94 仍是 **+0.3%**。Run1 好看是因为没付 FlagGems linear 的首次
autotune，官方不计 Run1。已 **revert**（未提交）。ixBLAS 替换也偏「算子换后端」，
即便 Run2 真涨也不优先。

#### 17.19.5 结论

4k 计分轮已经是 **图内 GEMM + flashinfer（batch 64）+ 混合步 split-K**。lm_head
 theoretically ~2% TPOT，实测省 ~0.15 ms/step，e2e 吃不出来。融合 / tile retune /
warmup / fused add+rms 前几节也已经打过。

**不再为 4k tok/s 做增量融合。** 还可能动的只有 16k TTFT（prefill GEMM / eager
host / 是否 PIECEWISE），且不敢动 `vllm serve` 参数。代码与 site-packages 回到
`126cc1e` + FlagGems `2efaf84ca`。

### 17.20 静态 W8A8 线性层：4k +2.4%、Level 3 103/105（2026-10-05）

权重/激活静态 INT8（SmoothQuant α=0.5，校准集 math_500 Level 3 105 题输入 xmax），
不改 `vllm serve` 参数、不动 KV cache。天数默认开启；`VLLM_FL_ILUVATAR_STATIC_INT8=0`
关闭。沐曦不受影响。

#### 17.20.1 实现

插件 `vllm_fl/.../iluvatar/static_int8.py`：加载后量化 168 个 linear（vLLM 合并后的
qkv / o / gate_up / down，不含 lm_head）。融合 CUDA：bf16 按通道量化到 int8 →
`at::_int_mm` → i32×out_scale 回 bf16。校准文件打在同目录
`minicpm5_2b_l3_static_int8.pt`。

FlagGems Triton `scaled_mm` 在本卡比 bf16 mm 慢 2–3×，未用。Python 量化+反量化会
把 `_int_mm` 的优势吃掉（0.105 vs 0.040 ms），所以 quant/epilogue 必须在 kernel 里。

#### 17.20.2 官方口径 4k×2（`SKIP_FIRST=0`，STATIC_INT8=1，MIX/INT8_KV 关）

| | Total tok/s | Output tok/s | Mean TTFT |
|---|---|---|---|
| Run1 | 6052.66 | 1210.53 | 4301 ms |
| Run2 | 6074.35 | 1214.87 | 4062 ms |
| **avg** | **6063.51** | | **4182 ms** |

相对 §17.17 汇总 **5920.06** / Run2 **6000.94**：Total 约 **+2.4% / +1.2%**，TTFT 更好。
产物 `eval_iluvatar_static_int8_4k2/`。

#### 17.20.3 math_500 Level 3 105 题

官方 evalscope（batch=8，`max_tokens=32768`，t=1.0，top_p=0.95）：
**103/105 = 98.1%**（门槛 ≥0.95；§17.17 基线 101/105=96.2%）。两道错题均为矩阵
答案未抽出 `\boxed{}`，不是乱码。产物 `eval_iluvatar_static_int8_l3/`。

#### 17.20.4 放弃的路径（未提交）

* 动态 packed INT8 KV：decode 每层 unpack 打不过 flashinfer，4k 约 1727 vs 5920。
* MIX_KV（K 紧凑 INT8 + V bf16）：微基准 mix kernel ~0.35 ms，但 cache 写入 /
  CUDA graph / 短 prefill Triton 读 INT8-as-bf16，e2e 4k Run1 4998、Level 3 乱码。
  **MIX_KV 插件补丁与 FlagGems mix dispatch 已删除**，不留开关。

### 17.21 静态 W8A8 在 FULL_DECODE_ONLY 下的 compile/CG 修复（2026-10-07）

§17.20 落地时，AOT 图多数仍走 bf16 `mm`，INT8 未真正进热路径；清 AOT + 强制走
INT8 后 4k 可到 ~7k，但 Level 3 崩成 `'<think>'` 乱码。RCA：

* eager / `cudagraph_mode=NONE` + INT8：**正确**；
* 官方 `FULL_DECODE_ONLY` + INT8 进 decode 图：**错误**（CUDAGraph 只回放
  kernel，不重跑 Python 分支；再叠加 persistent `out` 被下游 alias）。

#### 17.21.1 修复（`static_w8a8_linear_v5`）

仍不改 `vllm serve` 参数。插件内：

1. 预加载 CUDA 扩展，避免 Dynamo 追到 `cpp_extension.load`；
2. opaque `custom_op` + **per-layer** 固定 workspace（cap=72，对齐 capture sizes）；
3. **`M ≤ 72`（及 stream capturing）一律 bf16**，INT8 只给 eager prefill
   （`M > 72`），decode 图里不再录 INT8；
4. 返回 `out[:m].clone()`，切断对 persistent workspace 的 alias。

#### 17.21.2 官方口径结果

| | Total tok/s | 备注 |
|---|---|---|
| 4k Run1 / Run2 / **avg** | 5959.92 / 6135.85 / **6047.89** | `SKIP_FIRST=0`，FULL_DECODE_ONLY |
| Level 3（evalscope） | **101/105 = 96.2%** | 门槛 ≥0.95 → **PASS** |

相对 §17.20 的 6063.51：吞吐约 wash（decode 回到 bf16，INT8 只吃大 prefill）；
相对 §17.17 的 5920 仍约 **+2.2%**。错题 index `[18,71,87,99]`（与历次矩阵/
格式题重叠，非乱码）。产物 `eval_iluvatar_w8a8_v5_4k2/`、
`eval_iluvatar_w8a8_v5_l3/`。

#### 17.21.3 未提交的失败尝试

* drop bf16 权重副本省显存：需清 AOT，且单独不是乱码根因；
* inverse `MIN_M` 门控 / 共享 workspace：4k wash 或 OOM；
* INT8 进 decode CUDAGraph（含 `M∈(16,72]`）：FULL_DECODE 下必崩。

### 17.22 当前栈 remain profile + 后续优先级（2026-10-12）

口径：`FULL_DECODE_ONLY` + 静态 W8A8 v5（§17.21）。脚本
`/root/bench_results/w8a8_v5_remain_profile.py`（LLM.generate 三窗）+
`w8a8_v5_gemm_micro.txt`（单层 GEMM）。**提醒**：本卡 torch profiler 对
CUDAGraph replay 会漏计（§10/§13）；decode 窗里看不见图内 GEMM/FA 是预期现象，
只能看图外 lm_head / sync / eager prefill。

#### 17.22.1 e2e 三窗（相对 §17.19 基线）

| 窗 | wall | 可见 CUDA 大头 | 相对 §17.19 |
|---|---|---|---|
| prefill 1×长 `max_tokens=1` | 262 ms | `_int_mm` 22.4 ms + epilogue/quant/copy；FA 4.8 ms | 旧 bf16 `mm` 双计 ~80 ms → INT8 真热路径 |
| decode 64×短 `max_tokens=8` | 328 ms | **lm_head linear ~9 ms/8 step**；图内 GEMM 不可见；窗内 `_int_mm` 来自短 prompt 的 eager prefill | 与 §17.19 一致：图外只剩 lm_head |
| mixed 8×~1k | 218 ms | INT8 + Triton unified attn（mixed/short 不走 FA） | 混合步仍吃 attention host |

Prefill CPU 仍见 `cudaStreamSynchronize` / `CompiledFxGraph` / rms / kv_update /
`unbind`+`expand`（与 §17.19 同族，非新瓶颈）。

#### 17.22.2 单层微基准（bf16 `F.linear` vs W8A8，H=2048 / I=6144）

| layer | M=512 | M=1024 | M=2048 | M=4096 | 备注 |
|---|---|---|---|---|---|
| gate_up | 1.13× | 1.19× | **1.40×** | **1.44×** | 主收益层 |
| down | ~1.0× | 1.19× | 1.22× | 1.24× | 中等 |
| qkv | 1.11× | 1.25× | 1.15× | 1.09× | 大 M 略赚 |
| o | **0.80×** | 0.96× | 0.99× | 1.02× | **多数更慢** |

`gate_up` M=4096 一次 forward 拆核（20 iter 合计）：`int_mm` 18.3 ms、
**epilogue_i32_bf16 11.0 ms**、quant 1.35 ms、DtoD 1.0 ms。Epilogue ≈ int_mm 的
**60%**，是 INT8 路径上最大的可抠常数。

#### 17.22.3 已证伪 / 低 ROI（不要再抠）

* lm_head → `mm` / ixBLAS：§17.19，暖轮 **+0.3%**；
* decode 图内融合 / tile / fused add+rms：§17.19.5；
* INT8 进 decode CUDAGraph：§17.21，正确性炸；
* 动态/MIX INT8 KV：§17.20.4；
* 改 `vllm serve` / 超参量化：赛规禁止。

#### 17.22.4 后续优先级（按期望 e2e 与风险）

| P | 方向 | 依据 | 风险 |
|---|---|---|---|
| **P0** | **按层选择性 W8A8**：只开 `gate_up`/`down`（+可选大 M `qkv`），**关掉 `o_proj`** | 微基准 o 常 <1×，白付 quant/epilogue | 低；L3 需回归 |
| **P0** | **融合 / 重写 epilogue**（i32→bf16×scale±bias 与 quant 同 kernel 或接 int_mm） | epilogue 占大 GEMM ~37–60% | 中；要自己写 CUDA |
| **P1** | **16k TTFT**：长 prefill 上 INT8+FA 已在，继续抠 host sync / chunk 边界 / Triton mixed | §17.19.5；计分含 16k | 中；别动 serve 参数 |
| **P2** | 降 decode 侧显存：安全 drop bf16 副本（仅 INT8 层、AOT 清干净）扩 KV/并发 | 曾 OOM/空 weight | 高 |
| **P3** | 再探「纯 CUDA、无 Python 分支」的 decode-INT8 CG | 理论 4k 上限接近曾见的 ~7k | **极高**（正确性） |
| — | ixinfer 换后端整层 linear | 偏「换库」赛规灰色带 | 规则风险 |

**结论**：4k 计分轮已是「decode 图内 bf16 GEMM + flashinfer + split-K + prefill
INT8」。下一刀应打 **prefill GEMM 质量**（选层 + epilogue），不是再融 decode
小算子；16k 是第二条的第二战场。

