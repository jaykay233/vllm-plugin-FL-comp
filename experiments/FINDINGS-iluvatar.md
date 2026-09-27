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
