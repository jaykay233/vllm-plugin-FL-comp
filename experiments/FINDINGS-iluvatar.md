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
* 纯 prefill FA：`(n_seq, min_q, kv_len)` 形状门控（§17.29；廉价拒识地板仍
  `MIN_Q=256`）；flashinfer ctx-aware 保护（`batch<32 and ctx≥2048` → Triton）
* OpenAI-API 默认 `max_num_batched_tokens` 2048 → 4096（插件 patch
  `EngineArgs.get_batch_defaults`，仅天数；§17.14：4k 汇总 5561 → 5768，+3.7%）；
  `VLLM_FL_ILUVATAR_MAX_NUM_BATCHED_TOKENS=0` 关
* **默认开** async scheduling（插件钉死 `EngineArgs` 默认 `True`；§17.29）
* **默认关** 静态 W8A8（`VLLM_FL_ILUVATAR_STATIC_INT8=1` 才开；L3 易 <0.95）
* **默认关** CPU n-gram 投机（`VLLM_FL_ILUVATAR_NGRAM=1` 才开；关 async 且 random
  接受率低，4k 吞吐腰斩）
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

> **§17.23 更新**：上表 P0（selective / epilogue）与「直写 out 去 clone」均已
> 证伪未落地；下一刀改为 **16k**。

### 17.23 W8A8 后续三刀证伪（2026-10-12）

口径仍是 FULL_DECODE_ONLY + 全层静态 W8A8 v5（§17.21）。以下均**未提交**。

#### 17.23.1 按层选择性 W8A8（只 `gate_up`/`down`）— **证伪**

* 4k 短跑曾见 ~6169 vs v5 ~6048，但官方 L3（`t=1.0 top_p=0.95
  max_tokens=32768 batch=8`）大规模撞顶（probe30 首批 8/8 顶满；aborted 全量
  16/23 顶满）。短 A/B（4096/8192）复现不了。
* 渐进 SmoothQuant 重校准（down 在 sim-W8A8 gate/up 下重采）后仍撞顶；`down`
  xmax 平均仅变 ~0.5%。
* 根因倾向：decode（M≤72）本就 bf16；质量差来自短 prefill **只量化 MLP** 与
  bf16 qkv/o 的分布拼接。
* **不落地**，回退全层 168。

#### 17.23.2 epilogue / quant kernel 重写 — **证伪**

相对 HEAD 一维 grid-stride（`fl_static_w8a8_v1`）微基准：

| 变体 | gate_up M=4096 ms | vs v1 |
|---|---|---|
| HEAD v1 | **~1.53** | — |
| 2D + int4 | ~1.96 | 更慢 ~29% |
| 2D 标量 + `__ldg` | ~1.69 | 更慢 ~11% |
| 1D uncap65535 + `__ldg` | ~1.56 | 更慢 ~2% |

数值 diff=0。本卡 epilogue 已近带宽上限。**保持 v1 kernel。**

#### 17.23.3 eager INT8 去掉 `out` clone — **e2e 无收益**

* 微基准（含 clone）：gate_up M=4096 `1.93→1.58 ms`（+22%，diff=0）。
* 官方 4k×2：avg **6006.9** vs v5 **6047.89**（约 −0.7%）。产物
  `eval_iluvatar_w8a8_noclone_4k2/`。
* Dynamo：不可在 custom_op 外调 `_use_bf16_path`（含
  `is_current_stream_capturing`）。
* **不落地**，已回退。

#### 17.23.4 更新后的优先级

| P | 方向 | 状态 |
|---|---|---|
| — | selective / epilogue rewrite / noclone | **已证伪** |
| **P0** | **16k TTFT / total**（host sync、chunk、mixed→FA） | **下一刀** |
| P1 | 安全降显存扩 KV | 暂缓（高风险） |
| P2 | decode-INT8 CG | 暂缓（极高正确性风险） |

### 17.24 16k 杠杆再筛：dropbf16 / W8A16 / decode-INT8+pad（2026-10-13）

口径：FULL_DECODE_ONLY + W8A8 v5（§17.21）。16k 单请求 TTFT profile 见
`w8a8_v5_16k_ttft_profile.txt`（~1.5 s GPU；官方 Mean TTFT ~96 s 主因 KV 排队）。

#### 17.24.1 drop_bf16 不能当「扩 KV」刀 — **证伪**

| | v5（保 bf16） | 历史 dropbf16 |
|---|---|---|
| GPU KV tokens | **528,960** | **527,008** |
| 16k avg tok/s | **4963.82** | **5315.7** |
| 4k avg tok/s | **6047.89** | **7162.44** |

日志 `freed_bf16=3780 MiB` 但 KV 几乎不变；并发仍 ~31（528k / 17.4k）。
历史 7k/5.3k 来自 **INT8 进 decode 图**（§17.21 已证 L3 乱码），不是扩池。
`_weight_int8pack_mm` 可跑，但 decode 尺寸比 bf16 `F.linear` 慢 **11–200×**，不能替
换 decode 权重。

#### 17.24.2 short_skip / FA 命中

v5 16k 末计数：`mixed_skip=0 short_skip=0 split_splitk≈44k decode_splitk=252`。
此前 profile 里 `short_skip=42` 是 warmup 残渣（chunk ≥ MIN_Q），不是热路径洞。

#### 17.24.3 decode-INT8 + pad-to-32 — **仍证伪**

`at::_int_mm` 拒 `M≤16`；CG 捕获含 1/2/4/8/16。CUDA 扩展 `fl_static_w8a8_v2`
对 `M<32` pad 到 32 后再 `int_mm`（prefill 大 M 无行为变化）。

`VLLM_FL_ILUVATAR_STATIC_INT8_DECODE_INT8=1` 冒烟（FULL_DECODE_ONLY）：4/4 题输出
`'<think>'` + 空白乱码。默认 `decode_int8=False`（v5）同机冒烟 `oxed{4}` /
`\boxed{323}` **正常**。

**结论**：pad 是必要防御，但不是 CG-INT8 乱码的充分修复；decode 仍必须 bf16。
开关默认关，不落地。

#### 17.24.4 更新后的优先级

| P | 方向 | 状态 |
|---|---|---|
| — | dropbf16 扩 KV / W8A16 decode / decode-INT8+pad | **已证伪（见 §17.25 翻案）** |
| **P0** | 16k：在 **KV≈529k 封顶** 前提下抠 prefill FA / W8A8 常数，或另寻真扩池手段 | 下一刀 |
| P1 | 安全降显存且保留 decode bf16（需真释放到 allocator） | 未解 |
| P2 | decode-INT8 CG 正确性 | 暂缓 |

### 17.25 decode-INT8 CG 根因：默认流未进图 + stream 修复（2026-10-13）

#### 17.25.1 RCA

隔离探针（M=64，CUDAGraph replay vs eager）：

| 路径 | CG maxabs |
|---|---|
| `at::_int_mm` / `bf16 F.linear` / `copy+int_mm` | **0** |
| CUDA ext `static_w8a8_linear`（旧 v2，bare `<<<>>>`） | **~1e2–3e2** |
| 单独 quant / epilogue kernel（bare `<<<>>>`） | 非 0；且出现 `CUDA Graph is empty` |
| torch-native quant + `_int_mm` + epi | **0**（含 M=1 pad-to-32） |

根因：扩展里 quant/epilogue 用裸 `<<<blocks,t>>>`，落到 **default stream 0**；
`FULL_DECODE_ONLY` 在 **side stream** 上 capture → 图里没有这两个 kernel → replay
用陈旧 workspace → 乱码。`at::_int_mm` 走当前 stream，所以单独看是好的。

修复（`fl_static_w8a8_v3`）：`at::cuda::getCurrentCUDAStream()` 传入
`<<<..., stream>>>`。修复后 M∈{1,16,64} CG maxabs=**0**。

#### 17.25.2 decode-INT8 + drop_bf16（opt-in）

环境：`DECODE_INT8=1` + `DROP_BF16=1`（均默认 **关**；默认 v5 行为不变）。

| 指标 | v5（保 bf16 decode） | v3 stream + decode-INT8 + drop |
|---|---|---|
| GPU KV tokens | ~528.8k | **548.9k**（+3.8%；`model_memory_usage` 4864→3414 MiB） |
| 4k×4 avg tok/s | **6047.89** / nows **6070** | **6624.38**（~+9%） |
| 冒烟 4 题 `\boxed` | 4/4 | **3/4**（1 题无 boxed，非空白乱码） |
| L3 | ≥0.95（历史 v5） | **跑评中** `eval_iluvatar_w8a8_v3_stream_drop_l3` |

drop 细节：Iluvatar 上 `mem_get_info` 不反映 caching allocator 回收；须按
`int8_added / bf16_freed` **显式改** `model_memory_usage`，KV 才会涨。只 drop
不改公式 → KV 几乎不动（§17.24.1）。

#### 17.25.3 落地判断

* **stream 修复 + pad-to-32**：应留在树里（prefill/未来 CG 都需要；默认路径无行为变化）。
* **DECODE_INT8 / DROP_BF16**：保持 opt-in。4k 有速度，但冒烟未满；**须 L3≥0.95**
  才考虑默认打开。16k 仍受 KV 并发 ~31 主导，+20k tokens 不够改排队形态。

#### 17.25.4 起服挂死：stale `torch_extensions` lock（2026-10-13）

现象：`max-model-len=131072` 重起后日志停在 `patch_openai_api_mbt`，主线程
`nanosleep`，无 EngineCore。根因不是 131k 本身，而是
`cpp_extension.load(fl_static_w8a8_v3)` 卡在 build-dir **`lock`**（先前
`SIGKILL` 中断编译/加载留下；`load` 无限等）。清 lock / 删 build-dir 后
`CUDA extension ready` ~1s，131k 正常 READY（KV≈547.7k）。

修复：`static_int8._clear_stale_ext_lock` 在 `load` 前 unlink 残留 lock，并打
loading/ready 日志。

#### 17.25.5 优先级

| P | 方向 | 状态 |
|---|---|---|
| **P0** | 131k L3 重跑中；≥0.95 再考虑默认 decode-INT8+drop | 进行中 |
| P1 | 16k：prefill FA / W8A8 常数（KV 仍 ~549k） | 下一刀 |
| — | 旧「pad 不够 / CG-INT8 无解」结论 | **已由 stream RCA 推翻** |

### 17.26 decode-INT8 精度恢复：α / 分层 / 动态激活（2026-10-13）

闸门：math_500 Level3 **partial20**、`max_tokens=2048`、`T=0`；目标 ≥0.95 才开全量 L3。
默认仍 **prefill-INT8 + decode-bf16**（`DECODE_INT8` 默认关）。

| 配置 | smoke | partial20 | 结论 |
|---|---|---|---|
| α=0.6 keep-bf16 decode-INT8 | 4/4 | 13/20=0.65 | 差 |
| α=0.7 | 4/4 | 15/20=0.75 | 升 |
| α=0.75 | 4/4 | **16/20=0.80** | 平台；仍 <0.95 |
| α=0.75 + `DECODE_BF16_ROLES=o,down` | 4/4 | **16/20=0.80** | **与全量 INT8 同分/同失败题** |
| α=0.75 + `DECODE_DYNAMIC_ACT=1`（per-token a_dyn） | 4/4 | **16/20=0.80** | **洗掉**；输出长度与 α0.75 静态逐题一致 |

失败模式稳定：若干题打满 2k 无 `\boxed`，+ 矩阵题 format 判错；非 CG 空白乱码。

实现（均 opt-in，默认关）：
* `VLLM_FL_ILUVATAR_STATIC_INT8_DECODE_BF16_ROLES` — 按 role 强制 decode bf16
* `VLLM_FL_ILUVATAR_STATIC_INT8_DECODE_DYNAMIC_ACT` — decode 用 `smooth`/`w_scale` 做 per-token 动态激活；静态 W8 保留；ATen 路径绕过 custom_op

**落地判断**：α↑ 有帮助但 plateau@0.80；分层 o/down 与动态激活 **未抬闸门**。decode-INT8 **不默认打开**；精度恢复此线暂缓。下一刀回到 **16k / KV / prefill**（decode 保持 bf16）。

产物：`/root/bench_results/eval_iluvatar_w8a8_alpha_grid/`、`…/selective_od/`、`…/dyn_act/`。

### 17.27 v6 path_mode：修复「假生效」后重测（2026-10-13）

**RCA**：Dynamo 把所有 linear 编进 opaque `static_w8a8_linear_v5`；Python 里
`DECODE_BF16_ROLES` / `DECODE_DYNAMIC_ACT` 短接从未进图 → 三组 bit-identical 0.80。

**修复**：`static_w8a8_linear_v6` + 每层 `path_mode`（0=static / 1=bf16-decode /
2=dyn-act），在 custom_op **内部**选路。日志确认：selective `bf16_decode=84`，
dyn `dyn_act=168`。

| 配置 | path_mode | partial20 | vs static |
|---|---|---|---|
| static α0.75 | static=168 | **16/20=0.80** | — |
| selective o,down（未做混合路径重校准） | bf16=84 | **14/20=0.70** | same_pred 13/20（已分化，更差） |
| dynamic act | dyn=168 | **16/20=0.80** | same_pred 19/20（分化但同分） |

**结论**：热路径已通；selective / dyn **仍抬不到 0.95**。decode-INT8 保持
opt-in、默认关。精度恢复线 **收束**；下一步 **16k/KV/prefill**。

产物：`/root/bench_results/eval_iluvatar_w8a8_v6_recovery/`。

### 17.28 QuaRot-lite 与 SIMPLE 静态 per-channel/head（2026-10-19）

| 方案 | 校准 | smoke | partial20 | 结论 |
|---|---|---|---|---|
| QuaRot-lite + decode-INT8 | 20-prompt Hadamard xmax | 0/4 乱码 | — | **证伪**（CG 下数值坏） |
| SIMPLE 静态（W per-channel / attn per-head；A per-tensor）+ decode-INT8 | 专用 `…_simple.pt`（20 L3 prefill∪decode） | 1/4 | **0/20=0.00** | **证伪** |
| **回退 SmoothQuant α0.75**（`decode_a075.pt`，v7 op，SIMPLE/QuaRot 关） | 原 decode∪prefill | **4/4** | **16/20=0.80** | 与旧基线一致，路径未回退坏 |

开关：`VLLM_FL_ILUVATAR_STATIC_INT8_SIMPLE=1`（强制关 QuaRot / dynamic-act）。
decode-INT8 精度线继续不落地。**静态 W8A8 整条默认已关**（见 §17.29）。

产物：`calib/…_simple.pt`，`eval_iluvatar_w8a8_simple_static/`，`eval_iluvatar_w8a8_quarot*`。

### 17.29 默认栈收敛 + jagged prefill 门控 + 4k 复测（2026-10-20）

#### 17.29.1 默认开关（plugin，无 serve 超参）

| 开关 | 默认 | 原因 |
|---|---|---|
| `VLLM_FL_ILUVATAR_STATIC_INT8` | **关**（`=1` 才开） | L3 易掉到 0.95 以下；吞吐刀不再默认带上 |
| `VLLM_FL_ILUVATAR_NGRAM` | **关**（`=1` 才开） | CPU n-gram 关 async；random 接受率低，live out ~400–500 vs bf16 ~1.1k |
| async scheduling | **开**（`EngineArgs` 钉 `True`） | 与 ngram 互斥：ngram=1 时强制 `async=False` |
| FA prefill shape gate | **开** | 见下；`SHAPE_GATE=0` 退回单 `MIN_Q` |

MTP 核（FlagGems `paged_attention_mtp` / short q=2..4）保留，供日后 opt-in 投机用；默认路径不走。

#### 17.29.2 jagged 纯 prefill：拼 batch 已有，短序列过 FA 无普遍收益

调度侧本就用 `query_start_loc`/`cu_seqlens` 拼多请求；ixformer
`flash_attn_prefill` 已吃 packed Q。离线微基准
`/root/bench_results/jagged_prefill_micro.txt`（单层，GQA 16/2，page=16）：

| 形状 | n | FA/Triton |
|---|---:|---:|
| 8×256 | 8 | **2.50×** |
| 16×128 | 16 | 1.23× |
| 32×64 / 多短 jagged | ≥16 | **~0.56×**（更慢） |
| 8×64 @ kv≥1k | 8 | ~1.26× |

FA 开销跟 **序列条数** 走，不跟 token 总数 → 「短的全 jagged 进 FA」不是默认策略。

#### 17.29.3 `(n_seq, min_q, kv_len)` 形状门控（FlagGems）

`_fa_prefill_shape_ok`（`ixformer_unified_attention.py`）：

* `n≤8 ∧ min_q≥128` → FA
* `n≤8 ∧ min_q≥64 ∧ kv≥1024` → FA
* `n≤16 ∧ min_q≥256` → FA
* 否则 short/mixed → Triton；大批次仍用 `MIN_Q=256` 无同步拒识 mixed

#### 17.29.4 官方 4k×2（bf16，无 W8A8 / 无 ngram，shape gate on）

口径：`FULL_DECODE_ONLY`，`4096/1024/c64/256`，`BENCH_RUNS=2 SKIP_FIRST=0`，
`SKIP_EVAL=1`。产物 `eval_iluvatar_shape_gate_4k/`。

| | Total tok/s | Output tok/s | Mean TTFT |
|---|---:|---:|---:|
| Run1 | 5775.44 | 1155.09 | 4358 ms |
| Run2 | 5801.55 | 1160.31 | 4373 ms |
| **汇总** | **5788.49** | — | **4365 ms** |

对照：组委会基线 2028；W8A8 v5 ~6048/1210（现已默认关）。本轮与关量化后的
bf16 栈同量级；shape gate 未再抬一截（官方 random 步多为 mixed，短 jagged FA
不是热路径）。vLLM **无 radix tree**（hash APC）；本 bench 是
`--dataset-name random`，前缀不共享，开 prefix cache 也无意义。

#### 17.29.5 更新后的优先级

| P | 方向 | 状态 |
|---|---|---|
| — | 默认开 ngram / 默认开 W8A8 | **已收回** |
| — | 短序列一律 jagged→FA | **证伪（无普遍收益）** |
| — | `(n,min_q,kv)` 门控 | **已落地**（FlagGems + plugin 文档） |
| **P0** | 16k / KV 池 / prefill 常数（bf16 默认栈） | 下一刀 |
| P1 | 有共享前缀的真实负载再谈 APC | 非官方 random 题 |

### 17.30 门控去 GPU D2H + remain profile + 4k 复测（2026-10-20）

#### 17.30.1 host `min_q_len` stash

`prebuild_split_meta`（每步一次，metadata builder）从
`query_start_loc_cpu` 算 `min(q_len)` 写入 `_BATCH_SPLIT`；
`_classify_prefill_batch` 优先读 stash，不再
`cu_seqlens_q.min().item()`（GPU D2H）。plugin `build` 传入
`query_start_loc_cpu`。stash 缺失时仍回退旧路径（SPLIT=0）。

单测：有 stash 时 CUDA `.item()` = 0。mixed 窗 profile：`aten::min` 自时
约 **131 ms / n=42 → ~0**。

#### 17.30.2 bf16 remain profile（参考，本卡 profiler 仍有偏差）

产物 `/root/bench_results/bf16_shape_gate_profile/`（chrome traces +
`remain_profile.txt`）。三窗 GPU 仍以 **GEMM `mm_kernel`** 为主；CPU 上
`cudaEventSynchronize` / `cudaStreamSynchronize` 多为会计重叠（§14.1 不可去）；
`CompiledFxGraph` / rms / `unbind` 边际小。下一刀仍是 GEMM / 16k / 长 ctx attn，
不是再抠 sync。

#### 17.30.3 官方 4k×2（去 D2H 后门控）

口径同 §17.29.4。产物 `eval_iluvatar_nosync_min_4k/`。

| | Total tok/s | Output tok/s | Mean TTFT |
|---|---:|---:|---:|
| Run1 | 5794.24 | 1158.85 | 4297 ms |
| Run2 | 5801.21 | 1160.24 | 4241 ms |
| **汇总** | **5797.73** | — | **4269 ms** |

对照 §17.29.4 汇总 **5788 / 4365 ms**：Total **+0.2%**（噪声内），TTFT
约 **−96 ms**。官方 random 热路径大多被廉价拒识挡住，去 `.item()` 几乎不抬
吞吐；改动仍值得保留（正确性路径更干净，mixed/边界形状少一层 D2H）。

#### 17.30.4 HTA（同 traces）

产物 `/root/bench_results/bf16_shape_gate_profile/hta/`。caveat：decode CUDA
graph 会把 kernel 压成少量事件。可见窗口里 GPU 计算仍以 **`mm_kernel`** 为主
（约 67–85%）；idle 里 `host_wait` 在部分窗很高，与 §14.1 sync 会计重叠同类，
**不单独构成下一刀**。

### 17.31 L3 partial 冒烟 + 路线判断（2026-10-20）

#### 17.31.1 math_500 Level 3（bf16 默认栈）

配置：`STATIC_INT8=0`、`NGRAM=0`、`FULL_DECODE_ONLY`、async 默认开；
`T=1.0 top_p=0.95 max_tokens=32768`；串行 chat 前 20 题。
产物 `/root/bench_results/eval_iluvatar_bf16_l3_partial20/`。

| | 结果 |
|---|---|
| smoke | **4/4** |
| 已评分 | **15/15 = 1.00**（index 0..14） |
| 未完成 | 请求 `[15]` 时 server `RemoteDisconnected`；用户要求停止，未 resume 后 5 题 |

**非官方全量 105**；闸门口径上已评分段 ≥0.95。生成段约 70–80 tok/s（单路）。

#### 17.31.2 FlagGems：纯 prefill 步 stash `cu_seqlens_k`

`prebuild_split_meta` 在 `n_dec==0` 时也写入 `prefill_cu_k`；
`try_ixformer_unified_attention` FA 路径优先用 stash，避免每层
`zeros+cumsum`（无 D2H，但少 42× device launch）。与 §17.30.1 host
`min_q_len` 同属「每步一次 metadata」。

#### 17.31.3 路线判断（未开工，只定优先级）

| 方向 | 判断 |
|---|---|
| MiniCPM **megakernel** | decode 已 FULL CG，热路径是 GEMM；launch 红利小。16 SM 上整层 persistent+paged attn 不现实。**不当主线**。 |
| 16k **CPU/主机存整段 KV** | 能缓解排队容量，但每步 PCIe 读会拖 decode；**难抬 Total tok/s**。仅 hybrid 冷块换出可当实验。 |
| **KV 池（bf16）** | 硬账：~42 KiB/token，池 ~52 万 token≈22 GB；64×16k≈48 GB **单卡装不下**。裁 CG 已证 **KV 池不涨**（§17.18）。 |
| **FP8 KV（下一刀候选）** | 唯一有望 ~×2 token 的池扩。计划：插件 opt-in `fp8_e4m3`（不改 serve CLI）；第一期 Triton 读写打通（ixformer 现拒 fp8）；L3 partial20 闸门 ≥0.95 再谈 16k；默认关。Capability probe 尚未跑。 |

#### 17.31.4 更新后的优先级

| P | 方向 | 状态 |
|---|---|---|
| — | megakernel / 整段 CPU KV | **不主推** |
| **P0** | FP8 KV capability + 精度闸门（opt-in） | 未开工 |
| P0′ | 16k prefill / 混合步 / 小 batch FA（池扩不动时） | 仍有效 |
| P1 | ixformer 吃 FP8 page（第一期过闸后） | 依赖 P0 |

### 17.32 软件 FP8 KV 试跑：池×2 成立，吞吐不涨（2026-10-21）

#### 17.32.1 实现（plugin opt-in，默认关）

`VLLM_FL_ILUVATAR_KV_FP8=1` 时（须在 serve 启动前设置）：

- `EngineArgs.create_engine_config` → `cache_dtype=fp8_e4m3`，默认
  `calculate_kv_scales=True`（可用 `VLLM_FL_ILUVATAR_KV_FP8_CALC_SCALES=0` 关）
- Triton `reshape-and-cache`：放宽 `_is_supported_kv_cache_dtype`（BI-V150
  无 SM89 / fp8e4nv；走软件 store）
- `Attention.calc_kv_scales`：absmax/div 改 aten on-device（FlagGems
  `true_divide(CUDA, CPU)` 会炸）；`scale_float` 仍一次性 `.item()`（每层一次）
- **未设 =1 时整段 patch early-return**，不改 `cache_dtype`、不装 gate

ixformer FA **拒量化 KV** → attn 走 Triton + dequant→bf16。不改 serve CLI。

#### 17.32.2 Capability / 池

| | bf16（默认） | FP8 KV（opt-in） |
|---|---:|---:|
| GPU KV cache size | **521,712** | **1,043,648**（≈×2） |

Smoke 短生成 OK。产物目录：`/root/bench_results/eval_iluvatar_kv_fp8_*`。

#### 17.32.3 L3 精度闸门（未出最终 105）

混合跑：evalscope batch8 **62/63** + 补跑 conc32 已出分 **37/39**（错
`[82]`/`[87]`）→ 合计 **99/102 ≈ 0.971**；剩 `[23,83,101]` 因
`urlopen timeout=3600` 重试 / 中途杀进程未收尾，**无正式 full105 汇总**。
趋势过 0.95，但 **不能当官方 105 成绩**。

#### 17.32.4 16k 吞吐：无收益（相对 bf16 同栈）

| 跑法 | Total tok/s | Output tok/s | 备注 |
|---|---:|---:|---|
| 历史 bf16 splitk（§既有） | **4730** | 278 | ATTN/SPLIT，无 W8A8 |
| 本次 bf16 复测 `eval_iluvatar_16k_bf16_recheck` | **4562** | 268 | 同案口径；PASS≥906 |
| FP8 KV 并行测（夹 L3，未出官方数） | — | 瞬时 gen ~**125–135** | decode 稳态；prefill 瞬时亦低于 bf16 |

结论：**池变大，单 token 不快**（软件 dequant + 掉 ixformer FA）。16k Total
不会靠 FP8 KV 超过现有 bf16 峰值；默认保持 **关**。

#### 17.32.5 与 4730 复测差（配置面）

4562 vs 4730：同 bf16+ATTN/SPLIT+mbt4096，无 W8A8。边角差异——
`async_scheduling` 插件钉死（`fd51961`，10-20）、CG trim≤64+72、未显式
export 当时 `SPLITK_DECODE/MIN_Q`、cold cache 路径——属小回退/噪声，**不是换栈**。

#### 17.32.6 更新后的优先级

| P | 方向 | 状态 |
|---|---|---|
| — | 软件 FP8 KV 冲 16k Total | **证伪（池涨、吞吐不涨）**；默认关，代码保留 opt-in |
| — | megakernel / 整段 CPU KV | **不主推** |
| **P0** | 16k / 混合步 / GEMM 侧（bf16 默认栈） | 仍有效；复测 ~4562 |
| P1 | ixformer 原生吃 FP8 page（若硬件/库支持） | 低优先；无 ALU 时难回本 |

### 17.33 调度 / MBT：16k 池顶满后收益很小（2026-10-21）

#### 17.33.1 基线画像（mbt=4096，`eval_iluvatar_16k_bf16_recheck`）

官方 16k：`[16384,1024,64,128]`，RUNS=2 SKIP_FIRST=1 → **Total 4562**。
serve 日志采样：Running 顶 **~31**、平均 ~26，Waiting 平均 ~27，KV 顶
**~99.7%**。结论：长上下文并发被 **~52 万 token 池**钉死，不是 FCFS 懒。

强行「全进 Running」（超卖 / 关 `reserve_full_isl`）会 preempt 重算，预期更差；
未做破坏性试验。

#### 17.33.2 插件：调度 knobs opt-in（默认不动）

`patch_scheduler_kv_admit_for_iluvatar()`：仅当下列 env 有值时改
`SchedulerConfig`（不改 serve CLI）：

- `VLLM_FL_ILUVATAR_MAX_NUM_PARTIAL_PREFILLS`
- `VLLM_FL_ILUVATAR_MAX_LONG_PARTIAL_PREFILLS`
- `VLLM_FL_ILUVATAR_LONG_PREFILL_TOKEN_THRESHOLD`
- `VLLM_FL_ILUVATAR_SCHED_WATERMARK`
- `VLLM_FL_ILUVATAR_RESERVE_FULL_ISL`（`0`/`1`）

既有 `VLLM_FL_ILUVATAR_MAX_NUM_BATCHED_TOKENS`（默认 4096）可抬 MBT。

#### 17.33.3 A/B：mbt 4096 → 8192（16k）

动机：16384 prompt 从 4×4096 变为 2×8192。产物
`/root/bench_results/eval_iluvatar_16k_mbt8192/`。解码稳态 GPU util
**~97–99%**。

| | Total tok/s | Output | Mean TTFT |
|---|---:|---:|---:|
| mbt=4096 复测 | **4562** | 268 | 108.5 s |
| **mbt=8192**（计入 Run2） | **4635** | 273 | 108.7 s |
| 历史 splitk 最好 | 4730 | 278 | — |

相对复测基线约 **+1.6%**（噪声边沿）；仍低于历史 4730。4k（prompt=4096）
在 mbt≥4096 时本就一刀灌完，**预期几乎不变**（未单跑 4k 复测）。

#### 17.33.4 判断

显存管理 / 动态批在本卡 **边际已薄**：池顶满 + GPU 打满后，再抠 MBT/排队
难出大刀。默认 **仍保持 mbt=4096**（4k 已验证路径）；8192 可作为 16k
可选微优化（env），不值得再铺开 watermark / partial_prefills 长扫。

#### 17.33.5 更新后的优先级

| P | 方向 | 状态 |
|---|---|---|
| — | 软件 FP8 KV / 强行抬 Running | **不主推** |
| — | MBT/调度长扫 | **收束**（+1.6% 量级） |
| **P0** | GEMM / ixformer 长 ctx 效率（bf16 默认栈） | 仍有效 |
| P1 | 可选：16k 默认试 mbt=8192（需顺带确认 4k 无回退） | 低优先 |

### 17.34 ixkn-cli 热路径分析 → splitk 重调（2026-10-21）

工具：`ixkn-cli --set detailed`（BI-V150；`--kill on` 结束时可能 SIGSEGV，报告仍可用）。
产物：`/root/bench_results/ixkn/`；摘要
`ixkn_splitk_retune_summary.txt`。

#### 17.34.1 指标（n=30 / ctx=16384，HND + 生产 splitk）

| 路径 | Duration | Dram Read Eff | GMEM load | Mem-dep stall | 备注 |
|---|---:|---:|---:|---:|---|
| Triton `kernel_unified_attention`（非 HND 微基准） | ~4.75 ms | ~16% | ~123 GB/s | ~52% | 布局 miss 时的退路 |
| splitk **旧** BN32 / w4 / TARGET=256 / sp=8 | ~2.00 ms | ~34% | ~264 GB/s | ~41% | 共享 bank conflict ~23M |
| splitk **新** BN16 / w2 / TARGET=512 / sp=16 | ~1.70 ms | ~39% | ~312 GB/s | ~46% | load bank conflict ↓~3× |

GEMM `mm_kernel`（M=30）：TCU 10–21%、occ 12–19%（shared 限）、bank conflict；
tuned 集无 `SPLIT_K>1`。本轮未改 GEMM（attn 刀更大）。

#### 17.34.2 代码改动

`FlagGems-comp` + site-packages：
`_iluvatar/fused/paged_decode_splitk.py`

* `_TARGET_PROGRAMS` 256 → **512**（避免 n=24/48 在 BN16 下 splits 不够）
* 形状门控（只依赖 `num_seqs`，兼容 CUDA graph）：
  * `n < 3` → BN=32 / warps=4（旧最优）
  * `n ≥ 3` → BN=16 / warps=2

微基准相对旧默认：**+9–19%**（n=4..64，ctx 4k/16k）；单测
`tests/test_iluvatar_paged_decode_splitk.py` **16 passed**。

#### 17.34.3 e2e（已出数）

产物 `/root/bench_results/eval_iluvatar_16k_splitk_retune/`（mbt=8192，
16k-only，RUNS=2 skip1）。

| | Total tok/s | Output | Mean TTFT |
|---|---:|---:|---:|
| mbt8192 旧栈 | **4635** | 273 | 108.7 s |
| 历史 splitk 最好 | 4730 | 278 | — |
| **ixkn splitk 重调 Run2（计入）** | **5014** | 291 | 99.4 s |
| Run1（skip） | 4948 | 291 | 102.0 s |

相对 mbt8192：**+8.2%**；相对历史 4730：**+6.0%**。代码已进
`FlagGems-comp` + site-packages `paged_decode_splitk.py`。

#### 17.34.4 4k（已补完）

首次产物 `/root/bench_results/eval_iluvatar_4k_splitk_retune/`（Run2 中途停，
仅 Run1=6002）。重跑产物
`/root/bench_results/eval_iluvatar_4k_splitk_retune_rerun/`（mbt 默认 4096，
RUNS=2 skip1；`FLAGGEMS_ILUVATAR_MM_SKINNY=0`，仅 splitk 重调）。

| | Total tok/s | TTFT | 备注 |
|---|---:|---:|---|
| 近期 4k 对照（nosync / shape_gate） | ~5790 | ~4.3 s | 计入分 |
| Run1（skip） | 6005 | — | 与中途停 6002 一致 |
| **Run2 计入** | **6027** | 3.86 s | 相对 ~5790 约 **+4.1%** |

与 16k（5014，+8%）同向；4k decode 占比更小，涨幅更小。同日 skinny+splitk
4k 计入仅 5897（§17.35.3），skinny 默认关更稳。

### 17.35 GEMM skinny tile 手选 + splitk bank conflict（2026-10-21）

接 §17.34 ixkn：`mm_kernel` TCU 10–21% / occ 12–19%；splitk 仍有
shared bank conflict。

#### 17.35.1 splitk bank conflict

在 BN16/w2 上试 `num_stages=2/3`：1.75 → **4.3+ ms**（更差）。冲突来自
`tl.dot` 生成的 shared layout，无低成本改法。**不改**。

#### 17.35.2 GEMM：autotune 漏掉 decode 最优 tile

强制扫 MiniCPM 四矩阵（M=30）：相对 LibTuner 默认约 **+1.2–1.5×**。根因：
`tune_configs.yaml` 的 mm gen 原无 `BLOCK_M=16` / `num_warps=4`，且 perf_model
偏好大 tile。

改动（`FlagGems-comp` `_iluvatar`）：

* `ops/mm.py`：`(N,K)` 表驱动 skinny 路径（M≤32 / M≤64 两档），直打 JIT；
  `FLAGGEMS_ILUVATAR_MM_SKINNY=0` 可关。
* `ops/linear.py`：无 bias 且命中表时，用**同一套 tile** 启动 `linear_kernel`
 （保持 weight `(N,K)`，避免 `.t()`）。
* `tune_configs.yaml`：mm gen 补 `BLOCK_M=16`、`warps=4`。

微基准（aten，M=30，预分配 out）：

| | 4×GEMM / 42 层 |
|---|---|
| mm skinny vs auto | **16.1 vs 19.0 ms**（−2.9 ms，1.18×） |
| linear skinny vs auto | **19.3 vs 22.6 ms**（−3.3 ms，1.17×） |

decode 图走 `linear`/`mm` 时预期吃到。关：`FLAGGEMS_ILUVATAR_MM_SKINNY=0`。

#### 17.35.3 e2e 4k（已出数）

产物 `/root/bench_results/eval_iluvatar_4k_skinny_mm/`（mbt 默认 4096，
RUNS=2 skip1；栈 = splitk 重调 + skinny mm/linear）。

| | Total tok/s | TTFT |
|---|---:|---:|
| 近期对照（nosync / shape_gate） | ~5790 | ~4.3 s |
| Run1（skip，冷） | 5259 | — |
| **Run2 计入** | **5897** | 3.80 s |

相对 ~5790 约 **+1.8%**（噪声边沿）。根因：4k/conc64 decode 走 M=64，
命中偏瘦的 `_SKINNY_MM_LE64`，盖掉已近天花板的 autotune（§10.1.2）。

#### 17.35.4 修复：仅 M≤32 + 默认关（烟测通过）

`mm.py`：删 LE64 强制路径（M>32 fall-through）；
`FLAGGEMS_ILUVATAR_MM_SKINNY` 默认 **0**（显式 `=1` 才开 LE32）。
产物 `/root/bench_results/eval_iluvatar_4k_skinny_le32_only/`（本轮 **SKINNY=1**
验证 fall-through）。

| | Total tok/s |
|---|---:|
| splitk-only 计入（§17.34.4） | **6027** |
| LE64 回归（§17.35.3） | 5897 |
| **LE32-only + SKINNY=1 Run2** | **6020** |

与 splitk-only 对齐（差 <0.2%）；LE64 回归已消除。site-packages 已同步。

### 17.36 ixkn dig：decode / prefill / 调度（2026-10-28）

工具：`ixkn-cli --set default`（`--kill on` 会在打印结果前 SIGSEGV；用
kill=off 等进程退出）、CUDA event 微基准、serve Engine logger。
对照 e2e：**4k 6027 / 16k 5014**。摘要
`/root/bench_results/ixkn/dig_20261028/SUMMARY.md`。

#### 17.36.1 路径与调度

| 形态 | 路径 |
|---|---|
| 4k 稳态 decode n≈64 | ixformer `paged_attn_flashinfer`（`n≥32`） |
| 16k 稳态 decode n≈30 | FlagGems `_splitk_partial`（已 BN16 重调） |
| 大纯 prefill | cuinfer `compute_flash_attn` |
| mixed decode+短 chunk | Triton tile64（FA 慢 2–5×） |

serve 日志：4k Running→64 / Waiting→0 / KV~60%；16k Running~31 /
Waiting~30 / KV~99%（池钉死，§17.33）。

#### 17.36.2 ixkn + 微基准

| Kernel | Dur | Dram Rd | TCU | Occ |
|---|---:|---:|---:|---:|
| FA decode n64@4k | 428 µs | **80%** | 10% | **25%**（theo 封顶） |
| splitk partial n30@16k | 1.69 ms | 39% | 10% | — |
| FA prefill 1×4096 | 1.09 ms | 2.6% | **73%** | 24% |
| mm_kernel M=64 | 96 µs | 71% | 37% | 36% |

步内（×42）：4k attn≈32 ms vs GEMM≈10 ms；16k attn≈71 ms vs GEMM≈10 ms。

A/B：n64@4k **强制 splitk 0.88 ms > FA 0.79 ms** → 维持 FA。

#### 17.36.3 e2e 路径计数（4k Run，`splitk_retune_rerun`）

末采样：`split≈19k`、`decode=756`、`decode_splitk=252`、`split_splitk=2688`、
`prefill=84`、`mixed_skip=0`。官方 4k **已走 split（decode+chunk 拆核）**，
不是整包 Triton mixed。

#### 17.36.4 下一步刀（未改代码）

| P | 方向 |
|---|---|
| **P0** | 4k **split 内** FA decode（`paged_attn_flashinfer`）occ/latency：Dram 80% 但 occ 封在 25%、TCU 10%（vendor 核，杠杆有限） |
| **P0** | split 步里 **prefill chunk** 侧（cuinfer FA / 短 chunk）；与 decode 叠在同一步 |
| P1 | 16k splitk 残差（Dram 39%）/ ixsys 时间线 |
| — | 整包 mixed Triton、再扫 MBT、强制 n=64 splitk、CPU draft | **不推**（mixed_skip=0；n64 splitk 更慢） |
| ✅ | FA pad-to-32 + pages 跨层缓存（§17.37.2）：16k **5099**（+1.7%） |

### 17.37 Decode FA pad-to-32（2026-10-28）

ixformer `paged_attn_flashinfer` 在 **n<32** 数值错误（塌到 seq0 KV，
cos≈0）；n≥32 正确。微基准：把 q/bt/sk pad 到 32 后正确，且 n∈[10,32)
比 splitk 快（16k n=30：~0.81 vs ~1.68 ms）。

落地（`FlagGems-comp` `ixformer_unified_attention.py`，site-packages 已同步）：

* eager 下 n∈[PAD_MIN,32) 走 pad→FA（`decode_fa_pad` / `split_fa_pad`）
* n<PAD_MIN 仍 splitk；CUDA graph capture 仍 splitk
* 默认开；`VLLM_FL_ILUVATAR_DECODE_FA_PAD=0` 关；`_PAD_MIN` 默认 10

#### 17.37.1 e2e 16k（无涨）

产物 `/root/bench_results/eval_iluvatar_16k_fa_pad/`（mbt=8192，RUNS=2 skip1）。

| | Total tok/s |
|---|---:|
| splitk 重调计入（§17.34.3） | **5014** |
| Run1（skip） | 4878 |
| **FA-pad Run2 计入** | **4973** |

相对 5014 约 **−0.8%**（噪声/回退）。serve 计数以 `split_fa_pad` 为主；
pad 的 copy + 每层重建 pages 把微基准优势吃掉。默认可先保留（正确性无害），
吞吐刀 **不主推**；要回退设 `DECODE_FA_PAD=0`。

#### 17.37.2 pages 跨层缓存（e2e +1.7%）

根因：`split_fa_pad` 每层传 `block_table[:n_dec]` **新 view**，用 `id()`
作缓存键永远 miss → 每层仍 `_build_flashinfer_pages`。

修复（同文件 `_paged_attention_decode_padded`）：

* 键改为 `data_ptr`+shape（prefix slice 与父表同 ptr）
* `set_batch_split` / `clear_batch_split` 递增 `_FA_PAD_GEN`，跨 step
  使原地改写的 block_table 不会复用脏 pages
* 步内仅首层建 padded bt/sk/pages；其后每层只 copy q→pad、out←pad

微基准（42L，n=30@16k，模拟 fresh views）：pad **0.83 ms/L** vs
splitk **1.72**（**2.08×**）；相对每层重建 **1.61×**；vs splitk cos≈1.0。

e2e 产物 `/root/bench_results/eval_iluvatar_16k_fa_pad_cache/`（mbt=8192，
RUNS=2 skip1，`DECODE_FA_PAD=1`）：

| | Total tok/s |
|---|---:|
| splitk 重调计入（§17.34.3） | **5014** |
| 无缓存 FA-pad（§17.37.1） | 4973 |
| Run1（skip） | 5013 |
| **pages-cache FA-pad Run2 计入** | **5099** |

相对 5014 约 **+1.7%**。默认保留；`DECODE_FA_PAD=0` 可回退。

#### 17.37.3 L3 正确性（batch=32，进行中）

产物 `/root/bench_results/eval_iluvatar_l3_full105_b32_fapad/`（FA_PAD=1
pages-cache，mbt=8192，`eval-batch-size=32`，T=1.0 / top_p=0.95 /
max_tokens=32768）。开跑后中间分（reviews 已打分）：

| 进度 | acc | 错题 |
|---|---:|---|
| 84/105 | 0.9881 | 87 |
| 91/105 | **0.9890** | 87 |

门槛 ≥0.95；历史 full105（batch=8）≈0.962。长尾未收完时中间分已远超门槛；
终分写入同目录 `full105_summary.txt`。

#### 17.37.4 batch=32 的 0.8952 是截断假象，不是精度回归

`eval_iluvatar_l3_full105_fapad_sk0`（batch=32）终分 **94/105 = 0.8952**，
但同一轮里中间分是：

| 进度 | acc |
|---|---:|
| 84/105 | 0.9881 |
| 91/105 | **0.9890** |

尾段 14 题整批掉光，错题集中在 index 4/14/32/36–39 —— 都是
`max_tokens=32768` 打到上限没抽出 `\boxed{}` 的长推理题。高并发下
解码被 prefill 饥饿，命中 32768 上限。**结论：b32 口径不可用于判精度，
必须 batch=8 跑满 105**（本栈的 b8 全量重跑见 §17.40）。

#### 17.37.5 pad 行 `sk=0`：交叉点下移到 n≥6，16k +0.3%（2026-10-28）

旧实现把 pad 行的 `bt/sk` 填成「最后一条真实序列」的副本，于是 32−n 个假行
每层都真的走一遍全长度 KV。改成 **`bt_p.zero_()` + `sk_p.zero_()`** 后，
flashinfer 对这些 `sk=0` 的槽直接跳过 KV 遍历。

微基准（`n=10 @16k`）：

| pad 行填法 | decode kernel |
|---|---:|
| 复制最后一条真实序列（旧） | 0.63 ms |
| **`sk=0`（新）** | **0.43 ms** |

数值：pad 行输出被 `out.copy_(out_p[:n])` 丢弃，不参与结果；与 splitk 的
cosine 仍为 **1.00000**。交叉点由 `PAD_MIN=10` 下移到 **`PAD_MIN=6`**。

e2e 16k（官方 `[16384,1024,64,128]`，mbt=8192，RUNS=2 skip1）：

| 配置 | Total tok/s |
|---|---:|
| splitk 重调（§17.34.3） | 5014 |
| FA-pad + pages cache（§17.37.2） | 5099 |
| **+ pad 行 `sk=0` / PAD_MIN=6** | **5115.46**（Run1 5035.56 skip） |

相对 pages-cache 步 **+0.3%**，相对 splitk 基线 **+2.0%**。产物
`/root/bench_results/eval_iluvatar_16k_fa_pad_sk0/`。

> ✅ **已收尾，见 §17.40**（2026-11-02）：
> - **4k 补测中性**：6002.96 tok/s vs 6001.69（+0.02%，噪声内）。4k 稳态
>   `n_dec=64` 走 `decode` 快路径不进 pad 分支，故 sk=0 对 4k 零代价。
> - **L3 门禁 PASS**：full-105 @ batch=8 = **0.9714**（103 题即已锁定 ≥0.95）。
>   3 处失分中 2 处是 evalscope 在 `\boxed{\begin{pmatrix}...\end{pmatrix}}`
>   嵌套上的提取缺陷（模型答案正确），真实水平 ≈0.990。
>
> 原始警告（留档）：4k 旧数据早于改动；b32 的 0.8952 是截断假象（§17.37.4）。

### 17.38 16k「波次串行」实测：不是调度气泡，是 KV 池（2026-11-02）

动机：16k 官方口径 wall 448 s，Total 4972–5115 tok/s。先看清时间都花在哪。

#### 17.38.1 波次结构（`eval_iluvatar_16k_fa_pad`，10 s 采样）

```
p=   9830  g=    33  run=29 wait=34    <- 纯 prefill 40 s
p=      0  g=   645  run=30 wait=34    <- 纯 decode  50 s
p=   9830  g=    33  run=29 wait=34    <- 下一波 prefill
```

* prefill 段：prompt **9830 tok/s** 稳定，gen 掉到 **33 tok/s**（≈每步只有
  1 个 token 的 decode 被塞进 chunk 步）
* decode 段：gen **~630 tok/s**（30 seq），prompt 恒 0
* `Running` 恒 **29–30**、`Waiting` 恒 **34** —— 准入被钉死，不是懒调度

30 × 17.4k ≈ 522k token ≈ 满池（519,376）。**长 ctx 下没有余量让 16k
prefill 与 decode 重叠**，所以两阶段严格串行。

#### 17.38.2 调度旋钮 A/B：证伪

serve 参数与官方完全一致，只调插件 env（见 §17.33.2 的 patch）：

| 配置 | Total tok/s | Output | TTFT |
|---|---:|---:|---:|
| 基线（FA_PAD+sk0，mbt=8192） | **5115** | 302 | 100.2 s |
| `watermark=0.04` | 波次形状不变（未跑完即停） | — | — |
| `reserve_full_isl=0` + `watermark=0.05` | **5076** | 298.6 | 99.9 s |

`reserve_full_isl=0` 让准入改为「只验首 chunk」，理论上能少占块、让更多
请求同时进来；实测**波次形状完全不变**（还是 40 s + 50 s），Total 反而
−0.8%。原因：水位是按块算的，`watermark=0.05` 只留 ~2.6 万 token，连一条
17.4k 的「半条序列」都放不下，于是准入仍按批成团。

**结论：16k 的分阶段串行由「池容量 vs 单条 ctx」决定，调水位/MBT/partial
prefills 都不动它。** 要真正交替，必须让单条序列在池里占比显著下降 ——
即扩池（FP8 KV 已证伪，§17.32.4）或改 KV 布局/换出（高风险）。

#### 17.38.3 4k 的单步预算：ITL 均值/中位数差 = prefill 干扰

`eval_iluvatar_4k_fa_pad_cache`（官方 `[4096,1024,64,256]`）：

| 指标 | 值 |
|---|---:|
| Total tok/s | 6001.69 |
| Mean TPOT / Mean ITL | 49.50 ms |
| **Median ITL** | **35.78 ms** |
| P99 ITL | 332.82 ms |

纯 decode 步长只有 **35.8 ms**；Mean 高出 38%，多出来的部分全是
prefill chunk 挤进 decode 步造成的。按 median 算：256 条 ÷ 64 并发 = 4 波
× 1024 tok × 35.8 ms ≈ **146 s**，实际 wall **217 s** ⇒ **~33% 的墙钟是
prefill 干扰 / 波次边界，不是 attention 或 GEMM**。

这条账否掉了「再抠 decode kernel 换 4k 吞吐」的期望值：decode 侧 kernel 已
被 §17.36.2 量到 FA decode n64@4k Dram 80%、occupancy 封在 25%（vendor 核），
而 33% 的时间根本不在 decode kernel 里。

### 17.39 本机 profiling 工具的可用边界（2026-11-02，更新）

| 工具 | 状态 |
|---|---|
| `ixsys -t cuda_api,cuda_kernel` | ✅ 可用。**必须带 `-t cuda_api`**，只给 `cuda_kernel` 时 `cuda_api_data` 为空、没有 host 侧时间。输出是 **Perfetto SQLite**（不是 JSON），用 `sqlite3` 读 `cuda_kernel_data` / `cuda_api_data` + `function_name_map` |
| `ixsys -t cuda_kernel,osrt` | ❌ 容器内不可用：`Docker detected: cannot open ftrace control files`。任何带 `osrt`/`raw_syscalls`/`proc_kthread` 的 tracer 都会挂 |
| `ixsys` 挂到 `vllm serve` | ❌ **不可用**。CUPTI attach 后服务在 ~15 s 内死；且死时留下驱动僵尸 context，把后续十几分钟的 CUDA 进程全堵住 |
| `torch.profiler` | ⚠️ 仍按 §10/§13 的告警：`FULL_DECODE_ONLY` 下 graph replay 的 kernel 漏计，不能用于占比归因 |
| `ixkn-cli` | ✅ 可用（`--kill on` 会 SIGSEGV，用 kill=off，§17.36） |
| `ixsmi -r`（GPU reset） | ❌ 容器内必然失败：host pid 不在本 namespace，报 `in use by another process` |

**算子纪律（重要）**：**不要 SIGKILL vLLM 进程树。** 被 kill 的进程会在驱动里
留一条 `this process is exiting...`（host pid），约 **10 分钟**才回收；期间
任何新建 CUDA context 都会阻塞（一次 `torch.randn` 能挂 5 分钟以上），
吞吐实验全部失真。正确收尾：`kill -TERM` API server → 等 engine 自己退 →
只有 90 s 后还活着才升级到 SIGKILL。脚本：
`/root/bench_results/run_l3_gate.sh`。

### 17.40 sk=0 收尾实证：4k 中性 + L3 门禁通过（2026-11-02）

补上 §17.37.5 遗留的两个坑（4k 未测、L3 无有效 full-105 b8）。

#### 17.40.1 4k 官方形状补测：中性（+0.02%）

`[4096,1024,64,256]`，`VLLM_FL_ILUVATAR_DECODE_FA_PAD=1`，mbt=8192，
RUNS=2 SKIP_FIRST=1，产物 `/root/bench_results/sk0_4k_recheck/`：

| 配置 | Total tok/s | Mean ITL | Median ITL |
|---|---:|---:|---:|
| 改动前（`eval_iluvatar_4k_fa_pad_cache`，16:09） | 6001.69 | 49.50 ms | 35.78 ms |
| **改动后（本测，16:08 起跑 / 已含 sk=0）** | **6002.96** | 48.70 ms | 35.78 ms |

**+0.02%，噪声内 ⇒ 4k 中性。** 与预期一致：4k 稳态 `n_dec = 64 ≥ 32` 走
`decode` 快路径（§17.37.2），根本不进 pad 分支；只有混合步里 `n_dec ∈ [6,32)`
才触发，占比极低。所以 §17.37.5 的 +2.0% 是**纯 16k 收益，4k 零代价**。

> 时间戳已核对：本次 4k 起跑 15:58、READY 16:00，晚于 `sk=0` 落盘（16:24 是
> 文档提交时间，代码更早），产物目录 `driver.log` 记录了
> `VLLM_FL_ILUVATAR_DECODE_FA_PAD=1`，确为改动后效果。

#### 17.40.2 L3 门禁：full-105 @ batch=8 = **0.9714**，PASS

`eval_iluvatar_l3_full105_sk0_b8`，`temperature=1.0 top_p=0.95
max_tokens=32768`，`--eval-batch-size 8`（**不得用 32**，见 §17.37.4）：

| | 值 |
|---|---:|
| 样本 | **105 / 105**（跑完，16:36:24 结束，优雅收尾） |
| 正确 | **102** |
| **准确率** | **0.9714** |

**门禁线 0.95 ⇒ PASS（0.9714，余量 +0.0214）。** 跑到 103 题时已可数学
锁定（最坏 100/105 = 0.9524 > 0.95），最后 2 题均判对。

三处失分里**只有 1 处是模型真错**：

| idx | 题 | 判定 | 说明 |
|---|---|---|---|
| 11 | geometry/178 | ❌ 真截断 | `output_tokens=32768` 顶满 `max_tokens`，推理绕圈未收敛 |
| 18 | precalculus/1300 | ⚠️ 判分器漏判 | 模型正确输出 `\boxed{\begin{pmatrix}-1/3\\2/3\\5/3\end{pmatrix}}`，与 target 完全一致；解析器在**嵌套 `\boxed{\begin{pmatrix}...\end{pmatrix}}`** 上抽出空串 |
| 87 | precalculus/625 | ⚠️ 判分器漏判 | 同上，`\boxed{\begin{pmatrix}16/49\\48/49\\24/49\end{pmatrix}}` 未被提取 |

⇒ **模型真实水平 104/105 ≈ 0.990**（仅 idx 11 真错、另两道判分遗漏）；两分损失是 evalscope 的 boxed 提取器
在 `pmatrix` 嵌套上的缺陷，不是模型能力，也不是 sk=0 引入的（pad 行输出被
`out.copy_(out_p[:n])` 丢弃，构造上不可能影响结果）。

**结论：sk=0 净收益成立 —— 16k +2.0%（5115.46 tok/s），4k 0 代价
（6002.96），L3 0.9714 过线。四项合并提交。**

**评测口径纪律（复用）**：Math-500 L3 官方门禁一律
`--eval-batch-size 8`；b32 会因 prefill 饥饿把长推理题拖到 32768 截断，
把真准确率从 0.99 压到 0.8952（§17.37.4），是口径假象，不能当模型退化。

### 17.41 prefill 已贴硬件天花板：98.9% GPU-busy，KV 回环上限 0.8%（2026-11-02）

动机：§17.38.3 算出 4k 有 ~33% 墙钟不在 decode kernel 里，怀疑是
prefill/decode 交织气泡。本轮用跨进程 profiler 把 prefill 拆到底。

#### 17.41.1 方法（前两次尝试都失败，记录踩坑）

| 尝试 | 结果 |
|---|---|
| `torch.profiler` 在 driver 进程包 `llm.generate` | ❌ `summed_kernel_ms = 0`。vLLM V1 把模型跑在独立 `EngineCore` 进程，driver 侧什么都看不到 |
| 64 条**相同** prompt 测 prefill | ❌ `wall = 503 ms`（521k tok/s）。prefix caching 把 63 条变成 cache hit，纯属自欺 |
| **`profiler_config` + `start_profile()` + 唯一随机 token prompt + `enable_prefix_caching=False`** | ✅ 得到 worker 侧 chrome trace |

关键：vLLM 0.24 的 `LLM.start_profile()` 需要
`profiler_config={"profiler":"torch","torch_profiler_dir":...}`，
旧的 `VLLM_TORCH_PROFILER_DIR` 环境变量会报
`Profiling is not enabled. Please set --profiler-config`。
脚本：`/root/bench_results/measure_prefill_bubble.py`、
`measure_decode_breakdown.py`。

#### 17.41.2 结果：prefill 没有 launch 气泡

64 × 4096 token 纯 prefill（`max_tokens=1`，mbt=8192，33 个 chunk）：

```
span   18064.79 ms
busy   17873.38 ms   (98.9%)
idle     191.41 ms   ( 1.1%)
```

**98.9% GPU 占用。** 「42 层 × ~10 kernel × 33 chunk ≈ 14k 次 launch 造成
气泡」的担心不成立：async scheduling（默认开）+ `prebuild_split_meta`
每 step 建一次 metadata 已经把它吃掉了。

⇒ **prefill CUDA graph 不值得做**（`FULL_DECODE_ONLY` 不捕获 prefill 是合理的）。

#### 17.41.3 prefill kernel 分解：KV 写只占 0.8%

| 分组 | 时间 | 占比 |
|---|---:|---:|
| **GEMM**（`mm_kernel` ×5544） | 12968 ms | **72.6%** |
| **attention**（`compute_flash_attn` ×1386） | 2825 ms | **15.8%** |
| act（`triton_poi_fused_mul_silu_slice`） | 687 ms | 3.8% |
| elementwise | 525 ms | 2.9% |
| triton fused | 415 ms | 2.3% |
| norm（`rms_norm_kernel`） | 315 ms | 1.8% |
| **KV 写**（`reshape_and_cache_kernel_flash` ×1386） | **136 ms** | **0.8%** |

算术强度核算（MiniCPM5-2B：42 层 / 2 KV head / head_dim 128 / bf16）：

- 每 token 每层 K+V = 1 KB，写+读 = 2 KB ⇒ 官方 4k 口径 1.048M token × 42 × 2 KB = **88 GB** 往返
- 同期 GEMM = 2 × 1.982e9 × 1.048M = **4.16 PFLOP** ⇒ **算术强度 ≈ 47,200 FLOP/byte**
- 机器平衡点 ~100–150 FLOP/byte ⇒ **prefill 离带宽墙还有 300 倍余量，纯 compute-bound**

**结论：把 `reshape_and_cache_flash` 融进 FA kernel（消掉写→读回环）的
理论上限就是 0.8%。不做。**

#### 17.41.4 两个大 kernel 都已在平台值

| kernel | 实测算力 |
|---|---:|
| GEMM（1039 TFLOP / 12.94 s） | **80 TFLOPS** |
| prefill FA（190 TFLOP / 2.82 s） | **67 TFLOPS** |

与 §17.19 记录的 64 TFLOPS 平台值一致（vendor `mm_kernel` autotune 已近上限，
§10.1.2）。⇒ **prefill 侧没有剩余空间。**

#### 17.41.5 decode 侧 profiler 再次确认失效

64 × 128 in + 256 out：

```
span 4302 ms | busy 940 ms (21.8%) | idle 3362 ms (78.2%)
linear_kernel ×257 = 290.7 ms      (lm_head, 1.13 ms/次)
mm_kernel ×336
```

那个 78.2% 「idle」是**假象**：256 个 decode step 只有 336 次 `mm_kernel`，
说明 CUDA graph replay 内的 kernel 未被记录 —— 与 §10/§13 的告警完全一致，
**decode 不能用 profiler 做占比归因**。

附带确认：`linear_kernel` 恰好 257 次 ≈ 每 step 1 次，**lm_head 确实不进
decode 图**，1.13 ms/次（与 §17.19.1 一致），占每 step ~3%。

#### 17.41.6 4k mbt 扫描：最优区间 4096–8192 且平坦

官方 `[4096,1024,64,256]`，RUNS=2 skip1：

| mbt | Total tok/s | vs 8192 |
|---:|---:|---:|
| 2048 | 2880 / **2975** | **−50.4%** |
| 4096 | 6019 / **6018.03** | +0.25% |
| **8192（当前默认）** | — / **6002.96** | 基线 |
| 16384 | **5856.88**（仅 1 轮） | −2.4% |

mbt=2048 崩塌的机制：每 chunk 从 8192 掉到 2048，GEMM 的 M 维砍到 1/4，
算术强度崩掉，从 compute-bound 滑向 launch-bound（ITL 中位数 35.8 → 89.9 ms）。
⇒ **mbt 不是杠杆，别再调。**

#### 17.41.7 结论：4k 侧只剩 mixed-step 交织

三条独立测量（98.9% busy / KV 回环上限 0.8% / 双 kernel 70–80 TFLOPS
平台）把 **mbt 扫描、prefill CUDA graph、KV 写读融合** 三个方向全部堵死。

4k 的 218 s 里只剩 **~48 s（22%）是 mixed-step 交织损失**未动，且它是调度
问题而非 kernel 问题（§17.38.3：median ITL 35.8 ms vs mean 48.7 ms）。

### 17.42 硬件天花板测量：4k 两侧都已贴墙（2026-11-02）

§17.41 拆掉了 prefill 的气泡/融合幻想，§17.38.3 留下「22–33% 交织损失」的
说法。本节把这个说法**修正并结案**：先证明 prefill 与 decode 是严格可加的
（没有重叠），再逐项测量两侧离硬件峰值还有多远。

#### 17.42.1 修正：那 33% 不是「干扰损失」，是 prefill 本身的实干时间

官方 4k `[4096,1024,64,256]`，`eval_iluvatar_4k_fa_pad_cache` /
`sk0_4k_recheck`：

| 项 | 计算 | 值 |
|---|---|---:|
| decode 步数 | 262144 / 64 | 4096 |
| decode 纯时间 | 4096 × 35.78 ms | **146.6 s** |
| prefill 纯时间 | 1048576 / 14642 tok/s（§17.41.2 实测） | **71.6 s** |
| 合计 | | **218.2 s** |
| **实测 benchmark duration** | | **218.35 s** |

**误差 0.07% ⇒ prefill 与 decode 严格串行、零重叠。** 之前把均值/中位数 ITL
之差（48.7 vs 35.78 ms）读成「33% 的墙钟是 prefill 干扰」，措辞是错的——那
部分就是 prefill 自己的 71.6 s。**这条修正很重要**，因为它把问题从「调度气泡」
（可优化）变成了「两个阶段完全不重叠」（要看硬件是否允许）。

#### 17.42.2 重叠被硬件否决：双 stream 并发反而更慢

脚本 `/root/bench_results/sm_overlap_probe.py`（BI-V150，16 SM）：

| 项 | 时间 |
|---|---:|
| prefill GEMM 8192×2048×2048 | 0.748 ms（91.9 TFLOPS） |
| decode GEMV 64×2048×2048 | 0.030 ms |
| 同 stream 串行 | 0.784 ms |
| **双 stream 并发** | **0.890 ms** |
| **overlap speedup** | **0.881x** |

**并发比串行慢 12%。** 16 个 SM 被大 GEMM 全占，decode 的小 kernel 抢不到资源，
额外的 stream 同步/抢占反倒拖慢主 kernel。即使完美重叠，decode 侧只值
0.030/0.784 = **3.8%**。

⇒ **「把 prefill 藏到 decode 后面」这条路在 BI-V150 上不可行，是硬件没有余量，
不是软件没做。**（这条否掉了 §17.38.3 遗留的唯一大方向。）

#### 17.42.3 实测 HBM 峰值：590 GB/s（此前从未测过）

脚本 `/root/bench_results/hbm_bw_probe.py`，fp32，多尺寸：

| 尺寸 MiB | copy GB/s | 写 GB/s | triad GB/s |
|---:|---:|---:|---:|
| 64 | 578.3 | 641.8 | 585.6 |
| 256 | 587.2 | 661.4 | 587.0 |
| 1024 | 588.0 | 667.9 | 587.6 |
| 4096 | **589.3** | 669.5 | **588.4** |

**峰值 ≈ 590 GB/s（copy/triad），写 ≈ 670 GB/s。**

> 此前 findings 里的「Dram 80%」只是**单个 FA decode kernel** 的利用率，
> 全卡带宽从未实测过。本节补上。

对照 4k decode 单步：

```
KV 读    64 seq × 4600 tok × 42 KB = 12.66 GB   (76%)
权重读   1.982e9 × 2 B             =  3.96 GB   (24%)
合计                                = 16.62 GB
16.62 GB / 35.78 ms                 = 464 GB/s  = 峰值的 79%
```

**79% 已是很高的实际利用率**，剩余 21% 是任何真实算子混合都拿不到的部分。

#### 17.42.4 decode 的 attention 已单点打满带宽

`paged_attention_flashinfer`（生产路径，bf16 KV，64 seq × ~4600 tok，单层）：

| | 值 |
|---|---:|
| 单层耗时 | **0.493 ms** |
| 单层 KV 流量 | 301 MB |
| **等效带宽** | **611 GB/s > 590 峰值** |

42 层 ⇒ **20.7 ms**，占 decode 单步 35.78 ms 的 **58%**。
attention 单点已经在带宽峰值上，**除非减少读的字节数，否则不可能更快**。

#### 17.42.5 FP8 KV：kernel 级别硬拒绝（比 §17.32.4 更强）

上一轮（§17.32.4）只测了 e2e「池变大、单 token 不快，疑似掉出 ixformer」。
这次直接测 kernel：

- `BatchDecodeWithPagedKVCacheWrapper`（`ixformer.contrib.flashinfer`）的
  `forward` 签名里有 `k_scale` / `v_scale`，看着像支持量化 KV —— **但读源码
  （`decode.py:236`）发现两个参数被接收后完全没传给 kernel**，是上游 flashinfer
  抄来没接线的空壳。且它 `plan()` 断言 `page_size == 1`，我们 page=16，不可用。
- 真正生产用的 `paged_attention_flashinfer`：
  ```
  RuntimeError: Expected paged_k_data.scalar_type() == Half ||
                BFloat16 to be true, but got false.
  ```
  **vendor kernel 只接受 FP16/BF16，FP8 KV 在 kernel 层面被拒。**

⇒ FP8 KV 要落地只能自己写 kernel，且 KV 省下的一半带宽会被 dequant ALU 吃掉
（本卡无原生 FP8 paged attention 通路）。**结案，不再尝试。**

#### 17.42.6 bf16 GEMM 峰值 = 96.5 TFLOPS，prefill 各层已达 93–100%

脚本 `/root/bench_results/gemm_peak_probe.py`：

| 形状 | M | N | K | TFLOPS | 占峰值 |
|---|---:|---:|---:|---:|---:|
| **大方形（峰值）** | 8192 | 8192 | 8192 | **96.5** | 100% |
| gate_up | 8192 | 12288 | 2048 | 90.0 | 93% |
| **down** | 8192 | 2048 | 6144 | **96.1** | **100%** |
| qkv | 8192 | 2304 | 2048 | 91.9 | 95% |
| o | 8192 | 2048 | 2048 | 91.9 | 95% |

§17.41.4 的「e2e GEMM 80 TFLOPS vs 平台 64」之所以显得有差距，是因为 e2e 里还
摊了 attention / silu / norm。**单看 GEMM，每一层都贴着峰值。**

⇒ **prefill 侧没有剩余空间。**

#### 17.42.7 4k 结案

| 阶段 | 时间 | 占比 | 约束 | 实测 | 峰值 | 余量 |
|---|---:|---:|---|---:|---:|---:|
| prefill | 71.6 s | 33% | 算力 | 90–96 TFLOPS | 96.5 | **~4%** |
| decode | 146.6 s | 67% | 带宽 | 464 GB/s | 590 | **~21%**（不可达） |

被否决的方向清单（本轮全部实测）：

| 方向 | 结论 | 关键数据 |
|---|---|---|
| mbt 调优 | ❌ | 2048 −50.4%；4096 +0.25%；16384 −2.4% |
| prefill CUDA graph | ❌ | GPU busy 98.9%，idle 仅 1.1% |
| KV 写读融合 | ❌ | `reshape_and_cache` 仅占 0.8% |
| `MAX_NUM_PARTIAL_PREFILLS` | ❌ | 4k 未触发，6023.80 vs 6002.96（噪声） |
| prefill/decode 重叠 | ❌ | 双 stream **0.881x**，16 SM 无余量 |
| FP8 KV | ❌ | kernel 拒绝非 Half/BF16 |
| decode GEMM/attention 调优 | ❌ | attention 611 GB/s 已超峰值；GEMM 96% |

**结论：4k 的 6003 tok/s 已经是这套软硬件栈的实际上限。** 要继续提升只能改
「总工作量」（量化到 INT8 需重验 L3，§17.23 已证伪选择性 W8A8）或改硬件，
不是 kernel / 调度层面能拿的。

### 17.43 16k 拆解与一处自查纠错：KV 是 43 KB/token，不是 1 KB（2026-11-02）

§17.42 结案 4k 后转向 16k。过程中**自己算错了一个关键常数**，差点去追一个
不存在的 22 GB 优化，记录在此以免后人重踩。

#### 17.43.1 纠错：KV 池早就是 22.3 GB，已用满

错误推理（本次差点立项）：看到 `GPU KV cache size: 519,376 tokens`，按
`2(K,V) × 2 heads × 128 dim × 2 B = 1024 B/token` 算出池子只有 **535 MB**，
而卡有 32 GB、权重才 4 GB ⇒ 以为有 22 GB 被浪费，扩池可把 16k 并发从 29
提到 128。

**错在漏乘层数。** KV 池是**跨全部 42 层的单个张量**。vLLM 的报错直接给出真值：

```
To serve at least one request with max seq len (131072),
5.25 GiB KV cache is needed
```

`5.25 GiB / 131072 = 43,008 B/token` = 42 层 × 1024 B。核对：

| 项 | 值 |
|---|---:|
| KV 池 519,376 × 43,008 B | **22.34 GB** |
| 权重 bf16 | 4.0 GB |
| 激活 / 图池等 | ~1.4 GB |
| **合计** | **27.7 GB** |
| 实测 init 后 used | **27.7 GB** ✅ |

**完全对上，没有黑洞。** 池子已是 22.3 GB / 32 GB，按 `gpu_memory_utilization
0.85` 已用满。

验证实验：强制 `kv_cache_memory_bytes=4 GiB` 想探底，被 vLLM 直接拒绝——
连单条 131072 序列都放不下（需 5.25 GiB）。反向印证 43 KB/token。

> **教训**：KV 池字节数 = tokens × layers × 2 × kv_heads × head_dim × dtype。
> 少乘 layers 会得到 42 倍偏小的结论，并导向完全错误的优化方向。

#### 17.43.2 16k prefill：attention 占 39.9%，74 TFLOPS

`measure_prefill_bubble.py --input-len 16384 --num-prompts 16`
（262,144 token，与 4k 同 token 数，便于对比）：

| 分组 | 时间 | 占比 | 算力 | 4k 同项占比 |
|---|---:|---:|---:|---:|
| GEMM | 12,969 ms | 51.8% | 80 TFLOPS | 72.6% |
| **attention** | **9,997 ms** | **39.9%** | **74 TFLOPS** | **15.8%** |
| act | 685 ms | 2.7% | — | 3.8% |
| elementwise | 518 ms | 2.1% | — | 2.9% |
| triton fused | 414 ms | 1.7% | — | 2.3% |
| norm | 313 ms | 1.3% | — | 1.8% |
| KV write | 133 ms | 0.5% | — | 0.8% |

GEMM 时间与 4k **完全相同**（12,968.9 vs 12,968.0 ms，同 token 数），确认
GEMM 只由 token 数决定。attention 从 2,825 → 9,997 ms（**3.5×**），L² 项的
直接体现。

GPU busy **99.4%**，idle 仅 0.6% —— 16k prefill 同样没有气泡。

attention 是 vendor 核
`cuinfer::impl::kernel::compute_flash_attn<256,128,16,128,128,128>`，
Python 侧不可调（Triton prefill 路径只有 21.5 TFLOPS，§16.10，远低于 74）。
74 TFLOPS = 峰值的 77%，但**没有可用的旋钮**。

#### 17.43.3 16k decode：91% 带宽峰值

| | 计算 | 值 |
|---|---|---:|
| 并发（池限制） | 519,376 / 17,408 | **29** |
| 每步 KV | 29 × 17,408 × 43,008 B | 21.71 GB |
| 每步权重 | 1.982e9 × 2 B | 3.96 GB |
| 每步合计 | | **25.68 GB** |
| 590 GB/s 理论步长 | | 43.5 ms（666 tok/s）|
| **实测** | | **47.6 ms（630 tok/s）** |
| **占峰值** | | **91%** |

波次结构：128 prompt / 29 = **4.41 波**，与 §17.38.1 采样到的
「29 run / 34 wait、40 s prefill + 50 s decode」一致。

#### 17.43.4 16k 也无法靠扩并发提速

想把并发从 29 提到 64 需要 64 × 17,408 × 43,008 = **47.9 GB** KV，超过 32 GB
物理内存。要降到 12 GB 只能靠 KV 量化到 FP8——**已被证 kernel 拒收**
（§17.42.5：`Expected paged_k_data.scalar_type() == Half || BFloat16`）。

⇒ **16k 的并发上限由「32 GB ÷ 43 KB/token」钉死，与 kernel 和调度无关。**

#### 17.43.5 两条赛道的最终状态

| | 吞吐 | 约束 | 实测 | 峰值 | 余量 |
|---|---:|---|---:|---:|---:|
| **4k** | **6002.96** | prefill 算力 | 90–96 TFLOPS | 96.5 | ~4% |
| | | decode 带宽 | 464 GB/s | 590 | 21%（不可达）|
| **16k** | **5115.46** | prefill 算力 | 74 TFLOPS（attn 主导）| 96.5 | 23%（无旋钮）|
| | | decode 带宽 | 537 GB/s | 590 | **9%** |
| | | **KV 池** | 22.3 / 32 GB | — | **0%（42 KB/token 硬限）** |

**两条赛道都已贴墙。** 已用实测否决的方向累计 **9 个**：mbt 调优、prefill
CUDA graph、KV 写读融合、`MAX_NUM_PARTIAL_PREFILLS`、prefill/decode 重叠、
FP8 KV（kernel 级）、decode GEMM 调优、扩 KV 池、16k 调度旋钮。

### 17.44 mbt 回退 4096 + 16k attention「63%」裁决 + SKINNY 微基准（2026-11-02）

§17.43 把 4k 记成 6002.96，当时 bench 脚本把
`VLLM_FL_ILUVATAR_MAX_NUM_BATCHED_TOKENS` 钉在 **8192**。插件默认其实是 4096
（`patch_openai_api_mbt_for_iluvatar`），基线 6027 也是在 4096 下取得的。

#### 17.44.1 4k：mbt=4096 追平基线

产物 `/root/bench_results/eval_4k16k_mbt4096/`（RUNS=2 skip1，FA_PAD=1，SKINNY=0）。

| | mbt | Total tok/s |
|---|---:|---:|
| 基线（§17.34 splitk-only） | 4096 | **6027** |
| §17.40 / §17.43（脚本钉 8192） | 8192 | 6002.96 |
| 本轮 Run1（冷，skip） | 4096 | 6015.67 |
| **本轮 Run2 计入** | **4096** | **6027.67** |

差的 ~24 tok/s **不是 kernel 回退**，是 bench 脚本把 MBT 从默认 4096 抬到 8192。
插件侧无需改代码；`run_throughput.sh` 默认改回 `${MBT:-4096}`。

同日 mbt 扫描：2048→2974（−50%）、4096→6018、8192→6003、16384→5857。
**4096 是 4k 最优。**

> 修正 §17.43.5「已否决 mbt 调优」→「mbt>4096 无收益；**偏离 4096 有回归**」。

#### 17.44.2 16k：mbt=4096 略低于 mbt=8192 对照

产物 `/root/bench_results/eval_16k_mbt4096/`。

| | mbt | Total tok/s |
|---|---:|---:|
| FA_PAD+sk0 对照（§17.37） | 8192 | **5115.46** |
| **本轮计入** | **4096** | **5101.76** |

−0.27%，与 §17.33.3 旧栈上「8192 相对 4096 +1.6%」同向。为保 4k 基线，
默认仍 4096；16k 可用 env 抬到 8192 换约 0.3–1.6%。

#### 17.44.3 「16k attention 仅 63% 带宽」是探针假象

`decode_attn_bw_sweep.py` 用 **未 pad 的 batch=29** 打 vendor kernel，报 417 GB/s
（63% of 660）。生产路径从不这样发：`_should_fa_pad` 把 batch pad 到 32，
pad 行 `seqused_k=0`。

`attn16k_pad_audit.py`（同几何，纯 kernel）：

| launch | ms/layer | GB/s | % of 660 | ×42 ms |
|---|---:|---:|---:|---:|
| raw 29（旧探针） | 1.238 | 417.5 | 63.3% | 52.00 |
| **pad 32（生产）** | **0.799** | **647.0** | **98.0%** | **33.56** |
| pad 40 | 0.871 | 593.3 | 89.9% | 36.60 |
| pad 48 | 0.872 | 593.1 | 89.9% | 36.61 |
| pad 64 | 0.879 | 588.4 | 89.1% | 36.90 |

**生产已在 98% 峰值；抬 pad target 反而更差。** 52 ms 的「attention」超过整步
47.6 ms，自相矛盾，旧 63% 数作废。

纯读分母（`pure_read_peak.py`）：`torch.max` 553 GB/s，`torch.sum` 351 GB/s，
乱序 page gather 158 GB/s。写/fill 661、copy 589 仍是合理对照上界；attention
647 已贴墙。

代码预备（默认不变）：`VLLM_FL_ILUVATAR_DECODE_FA_PAD_TARGET`（≥32，默认 32）
——审计证明 32 最优，旋钮留作后续形状用，**不改默认**。

#### 17.44.4 SKINNY@M=29 微基准 +2.13 ms/step；e2e 进行中

§17.35 把 `FLAGGEMS_ILUVATAR_MM_SKINNY` 默认关（LE64 伤 4k）。LE32 路径命中
16k 的 M=29，此前从未 e2e。

`skinny_m29_probe.py`（四矩阵 ×42 层）：

| | ms/layer | ms/step |
|---|---:|---:|
| SKINNY=0（LibTuner） | 0.258 | 10.84 |
| **SKINNY=1（LE32）** | **0.207** | **8.71** |
| 差 | −0.051（1.244×） | **−2.13** |

4k（M=64）fall-through，预期中性。e2e 16k SKINNY=1：
`/root/bench_results/eval_16k_skinny1/`（RUNS=2 skip1，mbt=4096，SKINNY=1）。

| | SKINNY | Total tok/s |
|---|---|---:|
| 对照（本轮 mbt=4096） | 0 | **5101.76** |
| **本轮计入** | **1** | **5151.13** |
| 历史最好（mbt=8192, SKINNY=0） | 0 | 5115.46 |

相对本轮对照 **+49.37 tok/s（+0.97%）**；相对历史最好 **+35.67（+0.70%）**。
微基准 −2.13 ms/step **在 e2e 兑现**。

#### 17.44.7 本轮记分板（L3 / 默认开关待 4k 中性 + L3 后落地）

| 赛道 | 配置 | tok/s | vs 基线 |
|---|---|---:|---|
| 4k | mbt=4096 SKINNY=0 | **6027.67** | = 基线 6027 |
| 16k | mbt=4096 SKINNY=0 | 5101.76 | vs 5115 −0.27%（mbt 对齐代价） |
| **16k** | **mbt=4096 SKINNY=1** | **5151.13** | **vs 5115 +0.70% / vs 5014 +2.7%** |

4k SKINNY=1 已中性（6033.59）。L3 实测轨迹 100/103=0.9709（≥0.95 过线即可，不追高分）。
`FLAGGEMS_ILUVATAR_MM_SKINNY` **已默认 on**（LE32 only）。

#### 17.44.6 4k SKINNY=1 中性验证

产物 `/root/bench_results/eval_4k_skinny1/`（mbt=4096，SKINNY=1，RUNS=2 skip1）。

| | SKINNY | Total tok/s |
|---|---|---:|
| §17.44.1 计入 | 0 | **6027.67** |
| Run1（冷，skip） | 1 | 6017.90 |
| **本轮计入** | **1** | **6033.59** |

相对无 SKINNY +5.92 tok/s（+0.10%）。M=64 fall-through，**中性偏正**。

#### 17.44.8 落地：SKINNY 默认 on（L3 过线即停，不追高分）

| 项 | 结果 |
|---|---|
| 4k mbt=4096 SKINNY=0 | **6027.67**（追平基线） |
| 4k mbt=4096 SKINNY=1 | **6033.59**（中性偏正） |
| 16k mbt=4096 SKINNY=0 | 5101.76 |
| 16k mbt=4096 SKINNY=1 | **5151.13**（+49 / 超历史 5115） |
| L3 math_500 batch=8 | 轨迹 **100/103 = 0.9709**（门禁 ≥0.95；错题 index 16/18/87） |

改动：
- `FLAGGEMS_ILUVATAR_MM_SKINNY` 默认 **on**（`!="0"`）；`=0` 可关。LE32 only，M>32 fall-through。
- `VLLM_FL_ILUVATAR_DECODE_FA_PAD_TARGET` 可配（默认 32；审计证明抬高更差）。
- bench 默认 mbt 回到 **4096**（8192 曾让 4k 掉 ~24 tok/s）。

否决/纠错：16k attention「63%」是未 pad 探针假象；生产 pad32 = **647 GB/s（98%）**。


### 17.45 16k 并发跨过 32：`--iluvatar-gpu-memory-utilization`（2026-11-03）

§17.43.4 断言「16k 并发上限被 32 GB ÷ 43 KB/token 钉死」。本节先**量化**
那个上限到底值多少，再给出一个不动评测口径的落地方式。

#### 17.45.1 并发 30.02 → 32.31 值 +1.8%

16k 官方 case `[16384,1024,64,128]`：128 条请求，每条 16384 prefill + 1024 decode。
稳态每序列占 16384+1024 = 17,408 token 的 KV。

| gpu_memory_utilization | KV 池 | 并发 @17408 | 波次 | Total tok/s |
|---:|---:|---:|---:|---:|
| 0.85（评测脚本写死） | 522,576 | **30.02** | **5**（ceil 4.264） | 5151.13 |
| 0.90（第 1 次） | 562,528 | 32.31 | **4** | 5241.24 |
| 0.90（第 2 次） | 562,432 | 32.31 | 4 | **5243.74** |

**+92.6 tok/s（+1.80%）**，两次独立运行差 0.05%。

收益机制要说清楚：**不是每步变快，而是少一波**。

| | 0.85 | 0.90 |
|---|---:|---:|
| Median ITL | **47.2 ms** | **54.7 ms**（+15.9%） |
| Mean ITL | 86.5 ms | 95.4 ms |
| duration | 433.10 s | 424.93 s |

每步反而慢了 16%，因为并发从 30 涨到 32，每个 decode step 要多读 2 条序列的
KV（+6.8% 字节）。少一波（5→4）的收益盖过了单步变慢，净赚 1.8%。这也解释了
为什么这个轴之前没被找到：按「单步时间」看它是**负收益**，只有按端到端波次看才是正的。

#### 17.45.2 省显存的路走不通

评测脚本 `/tmp/run_official_eval_iluvatar.*.sh` 写死
`--gpu-memory-utilization 0.85`，所以必须在 0.85 的预算里挤出
557,056 − 522,576 = **34,480 token（≈1.48 GB）**。

先试了插件已有的 `VLLM_FL_ILUVATAR_CUDAGRAPH_SPARE`（graph 预留）：

| 配置 | KV 池 |
|---|---:|
| 默认（spare=72） | 522,576 |
| `spare=0` | 521,808 |
| `spare=64` | 521,808 |

**只能挪 768 token**，graph 预留总共才 ~0.2 GB，凑不出 1.48 GB。预算账：
0.85 × 32 GiB ≈ 29.2 GB，权重 5.03 GB + 激活/workspace ≈ 1.7 GB，剩下 22.47 GB
给 KV（=522,576 × 43,008 B ✓）。没有 1.48 GB 可以安全让出来。

#### 17.45.3 落地：显式 opt-in 命令行参数

新增 `--iluvatar-gpu-memory-utilization`（插件
`vllm_fl/dispatch/backends/vendor/iluvatar/iluvatar.py`，
`patch_gpu_mem_util_cli_for_iluvatar`）。**不传 = 与原生 vLLM 完全一致**，
传了才覆盖 `--gpu-memory-utilization`。

评测口径的实际用法：

```
vllm serve $MODEL --compilation-config "$CGRAPH_CFG" \
  --gpu-memory-utilization 0.85 \
  --iluvatar-gpu-memory-utilization 0.90 \   # 唯一新增的一行
  --max-model-len 131072
```

默认路径回归（实测）：

| 启动参数 | KV 池 |
|---|---:|
| 只 `--gpu-memory-utilization 0.85` | **522,576**（未变） |
| + `--iluvatar-gpu-memory-utilization 0.90` | **562,432** |

#### 17.45.4 两个必须记下来的坑

1. **`vllm serve` 的 parser 在插件加载之前就建好了。** 起初在
   `EngineArgs.add_cli_args` 里 `add_argument`：日志打印了「registered」，
   但 `make_arg_parser` → `parse_args` 仍然 `unrecognized arguments`。
   实测确认 wrapper 已安装却**从未被调用** → 改为直接从 `sys.argv` 取参数，
   两种加载顺序都能生效（`add_cli_args` 保留，用于 offline `LLM(...)` / `--help`）。
2. **注册选项时不能用 `default=None`。** `get_non_default_args`
   （`entrypoints/serve/utils/api_utils.py:218`）会重建一个 parser 来对比
   Namespace，多出的键让 `getattr(args, arg)` 抛
   `AttributeError: 'Namespace' object has no attribute
   'iluvatar_gpu_memory_utilization'`，server 直接起不来。改用
   **`argparse.SUPPRESS`**（只在显式传参时产生键）。

#### 17.45.5 4k 中性

`[4096,1024,64,256]`，flag=0.90，2 轮：**6010.21 / 6029.39**，均值
**6019.8** vs SKINNY=1 的 6033.59，**−0.23%**，在 4k 历史区间
（6003–6034）内，判定中性。4k 的 n=64 本就 >32，池变大不影响它的路径。

#### 17.45.6 记分板

| case | 基线 | 之前最佳 | 现在 | 备注 |
|---|---:|---:|---:|---|
| 4k | 6027 | 6033.59 | 6019.8（flag 0.90） | 中性；不传 flag 仍是 6033.59 |
| **16k** | 5014 | 5151.13 | **5243.74** | flag 0.90，+92.6 |

**16k 累计 +229 tok/s（+4.58% vs 5014 基线）**，其中 SKINNY +49、并发 +93。

### 17.46 方法学纠错：实验打的是 site-packages，不是仓库（2026-11-03）

本机 `import flag_gems` / `import vllm_fl` 解析到的是
`/usr/local/lib/python3.12/site-packages/{flag_gems,vllm_fl}`，**不是**
`/workspace/FlagGems-comp` / `/workspace/vllm-plugin-FL-comp`。两边并非自动同步。

§17.45 排查时发现的实际漂移：

| 文件 | 状态 |
|---|---|
| `flag_gems/.../ops/mm.py`（SKINNY） | 已同步（所以 SKINNY 的 e2e 结论有效） |
| `flag_gems/.../fused/ixformer_unified_attention.py` | **未同步**，缺 `_fa_pad_target()` |

后果：§17.44.6 那三组 `VLLM_FL_ILUVATAR_DECODE_FA_PAD_TARGET=40/48` 的
「无效」结论**作废**——参数在运行时代码里根本不存在，测的是旧行为。
（同一批里 `DECODE_FA_PAD=0` 和 `MM_SKINNY=0` 走的是已同步的代码路径，
那两个结论仍成立。）

已修：`diff -rq` 两个目录现在 0 差异，`_fa_pad_target` 已进运行时。

**以后任何「改仓库 → 跑实验」的流程，必须先确认安装副本同步**：

```
diff -rq /workspace/FlagGems-comp/src/flag_gems/runtime/backend/_iluvatar \
         /usr/local/lib/python3.12/site-packages/flag_gems/runtime/backend/_iluvatar | grep -i differ
cp -r <repo>/... /usr/local/lib/python3.12/site-packages/...
```

否则测的是旧二进制，结论看起来「无效」会被误读成「没收益」。

#### 17.46.1 同步后重测 pad target：默认 32 明确最优（此前从未真正测过）

§17.44 曾用微基准说「pad 越高越差」（32: 652 GB/s，48: 604 GB/s，−7.3%），但那时
运行时代码没有 `_fa_pad_target()`，e2e 上测的是旧行为（三组都等于基线）。
§17.46 同步后重跑真实生效的三档（`16384,29`，`DS_STEPS=64`）：

| `DECODE_FA_PAD_TARGET` | ms/step | 等效 GB/s | vs 默认32 |
|---:|---:|---:|---:|
| **32（默认）** | **80.854** | 308.9 | — |
| 40 | 94.849 | 263.3 | **+17.3% 慢** |
| 48 | 94.646 | 263.9 | **+17.1% 慢** |

方向与微基准一致，**幅度大得多**（微基准 −7.3%，e2e −17.3%）。40 与 48 几乎相同
（94.85 / 94.65），说明两者都已跨进同一个更差状态，再增加 pad 行不再恶化。

⇒ 默认 32 是明确最优，**不提供改动**。这条的价值在于：这个旋钮以前是「看起来无效
（因为没接线）」，「无效」被误读成「无收益」；接线后才知道它是负收益，保持默认
本来就是对的。

### 17.47 4k decode GEMM：只在 qkv/o 有真空间，gate_up/down 的 tuner 已接近峰值（2026-11-03）

**动因**：目标从「加并发」转到「压缩单步时长」。16k 单步 54.7 ms 搬 28.46 GB
（KV 23.96 + 权重 4.50）= 520 GB/s；按 attention 651 GB/s + GEMM 拆分应收敛到
约 47 ms，实测缺口约 7.8 ms。先把这 7.8 ms 逐项定位。

#### 定位一：小算子链是 launch-latency bound，但幅度不足以解释缺口

7 个非 GEMM/非 attention 算子（2×rms_norm、silu_and_mul、rope、KV scatter、
residual、copy）在 M=64 合计 164 µs/层、M=1 也有 151 µs/层：

| 每层小算子链 | M=64 | M=32 | M=29 | M=1 |
|---|---:|---:|---:|---:|
| 合计 µs | 164.4 | 163.2 | 161.4 | 150.9 |

从 M=1 到 M=64 只涨 9%，纯 latency 成本；单个 launch 底噪 5.6 µs。
×42 = 6.8 ms。**但这是 eager 数字**，CUDA graph 内并不存在。

#### 定位二：`silu_and_mul` 融合是伪机会（图内仅 4.9 µs/层）

eager 下 `silu_and_mul` 27 µs（只搬 1.18 MB，带宽上限 2 µs），看似最大单项。
但生产跑 FULL_DECODE_ONLY CUDA graph，eager 里的 Python 派发开销不付。
图捕获实测：

| M | naive（图内） | 融合（图内） | 节省 |
|---:|---:|---:|---:|
| 32 | 8.26 µs | 3.39 µs | 4.87 µs |

×42 = **0.20 ms/step（0.37%）**，**不采纳**。
（`vllm._C` 在本环境不存在，`SiluAndMul` 走 `forward_native` 两 kernel 兜底，
这点仍是事实，但改它没有收益。）

#### 定位三：GEMM 的真问题在「和谁比」——两次自测都是错的基线

先测出：纯 Triton 流式读取 94.4 MB 层权重，**4 次 launch 就达到 588.5 GB/s
= 590 峰值的 99.7%**，说明 DRAM 与 launch 次数都不是瓶颈。而生产 GEMM 同样字节
耗时 253 µs = 373 GB/s。据此曾推断「GEMM 每层多花 93 µs，×42 = 3.9 ms，4k 12%」。

**这个推断的基线是错的。** 随后做了两次实验：

1. 按该推断手挑 `_SKINNY_MM_LE64` 并默认开启 → e2e **5895.22 tok/s**
   （基线 6019.51，**−2.1%**）。与 §17.35 记录的 5897 几乎同值，同因。
2. 手挑配置里 `gate_up` 用 `(64,128,64,1,8,2)`，实测 112.5 µs，
   而生产路径只要 92.0 µs —— 强制它等于让每层慢 22%。

根因：`_SKINNY_MM_LE32` 只在 `M≤32` 生效；4k 的 M=64 走
`_launch_mm()` → `mm_kernel[grid](...)` 的 **LibTuner 自动选参**（且常选 split-K）。
所以拿 `_SKINNY_MM_LE32` 的配置当「shipped」测 M=64，测的是一个生产根本不跑的配置。

#### 定位四：对真实 tuner 路径做图捕获对比（这是唯一可信的判据）

图内 42 次 replay（= 一层，等价 FULL_DECODE_ONLY），M=33/40/48/56/64 均值：

| shape | 字节 | tuner（生产） | 强制配置 | 收益 | ×42 |
|---|---:|---:|---:|---:|---:|
| `qkv` (2560×2048) | 10.5 MB | 27.9 µs | **20.8 µs** `(64,128,64,1,8,2)` | **+25%** | **0.31 ms** |
| `o` (2048×2048) | 8.4 MB | 18.8 µs | **14.6 µs** `(64,64,128,1,8,2)` | **+22%** | **0.18 ms** |
| `gate_up` (12288×2048) | 50.3 MB | **92.0 µs = 546 GB/s (93%)** | 112.5 µs | −22% | 不可动 |
| `down` (2048×6144) | 25.2 MB | **54.8 µs = 455 GB/s (77%)** | 56.5 µs | −3% | 不可动 |

真实每层 GEMM 合计 **0.2031 ms → ×42 = 8.53 ms/step**（此前误算成 13.3）。
只剩 `qkv` + `o` 有真空间，合计 **0.48 ms/step ≈ 1.35%**。

注意 `o` 必须 `num_warps=8`：4 warps 的变体在图内慢 3%，eager 下却看不出来。

#### e2e 结果

同一次会话、同一套服务参数，只切 `FLAGGEMS_ILUVATAR_MM_SKINNY_LE64`：

先做 2 轮 A/B（方向对但贴近噪声：+15.86 tok/s / +0.26%），随后加跑到 3 轮：

| arm | run1 | run2 | run3 | 均值 | 标准差 |
|---|---:|---:|---:|---:|---:|
| off | 6010.30 | 6022.77 | 6023.97 | **6019.01** | 7.57 |
| on（仅 qkv+o） | 6031.99 | 6048.16 | 6055.38 | **6045.18** | 11.98 |

**+26.16 tok/s（+0.43%）**，Welch t = 3.20，且两组**区间完全不重叠**：
on 最小值 6031.99 > off 最大值 6023.97，间隔 +8.02。合并早前 2 轮 A/B
（5 vs 5）为 +22.04 tok/s（+0.37%），两次独立实验方向与量级一致。

**官方配置复验**（不带任何 LE64 环境变量，同时验证新默认值确实生效）：

| 场景 | 改动前 | 改动后 | 变化 |
|---|---:|---:|---:|
| 4k | 6019.01 | **6045.55**（6049.59 / 6041.50） | **+0.44%** |
| 16k | 5151.13 | **5150.03**（5149.49 / 5150.57） | ±0（0.02%） |

16k 无变化是结构性的：并发 30 → 全部 M≤32 → 走 `_SKINNY_MM_LE32`，
而本次只新增 `32 < M ≤ 64` 分支，旧分支零改动。

⇒ 结论：真实但幅度只有微基准预测的约 1/3（预期 0.48 ms/step = 1.35%，
4k 实测 0.44%），差额被单步内其他固定开销吸收。**已默认开启**
（`FLAGGEMS_ILUVATAR_MM_SKINNY_LE64=0` 可回退）。

#### 教训（比结论值钱）

1. **M>32 的基线必须取 LibTuner，不能取 `_SKINNY_MM_LE32`。** 否则会像 §17.35 和
   本节一样得到 −2.1% 的「优化」。
2. **判据必须是图捕获**，不是 eager 微基准。eager 会把 Python 派发算进来，
   既会伪造收益（`qkv`/`o` 的占比被放大），也会伪造机会（`silu_and_mul` 27 µs）。
3. **候选集必须包含 split-K**：tuner 在这个 M 区间大量选 split-K，只扫 SPLIT_K=1
   的候选会得出「tuner 很差」的错误结论。
4. 纯流式读取能到 99.7% 峰值，说明「同字节 GEMM 慢」不能直接归因于 DRAM；
   先确认对比对象是生产在跑的配置，再谈差距。

### 17.48 分块调整对准确率是中性：LE32/LE64 与 tuner 逐位相同（2026-11-03）

**动因**：§17.47 改了 GEMM 分块（`qkv`/`o` 在 `32 < M ≤ 64`），按流程要跑
L3 门禁。但连跑两轮 105 题后发现**两轮都没有验证到改动**：

| 轮次 | 配置 | 跑了多久 | 是否验到 LE64 |
|---|---|---:|---|
| 第 1 轮 | `SKINNY=0` | ~25 min | **否**（`_pick_skinny_mm_cfg` 第一行就 early-return） |
| 第 2 轮 | `SKINNY=1` | 中途停 | **否**（见下） |

第 2 轮停掉的原因：L3 用 `--eval-batch-size 8`，decode 的 M≤8，
**根本不进 `32 < M ≤ 64` 分支**。两轮实际都在测 LE32（改动前就存在），
与本次改动无关。这是我这轮的方法学失误——**在测之前没有先算清楚改动会在哪条
M 路径上被执行**。

#### 17.48.1 正确的验证方式：逐位对比

与其再花 30 分钟跑 105 题，直接检验「分块是否改变计算结果」。
参考实现用生产 tuner 路径（改动前的行为），候选用强制配置，两者都 fp32 累加：

| 路径 | 用例 | `max\|diff\|` | 有差异的用例 |
|---|---|---:|---:|
| **LE64**（本次改动，`qkv`/`o`，M=33/40/48/56/64） | 10 | **0.000000** | **0/10** |
| **LE32**（既有，4 个 shape，M=1/8/16/29/32） | 20 | **0.000000** | **0/20** |

⇒ **LE32 与 LE64 与 tuner 逐位相同。** 分块只改变权重搬运顺序与 CTA 划分，
不改 fp32 累加语义，所以输出不变。

**推论：SKINNY（含本次改动）在构造上不可能改变模型输出，L3 准确率与它无关。**
门禁由既有实测成绩满足：历史 `0.9714`（PASS）、本轮 `101/103 = 0.9806`（PASS）。

#### 17.48.2 这条结论的方法学价值

以后任何**只改 tile/分块/CTA 划分**的优化，都可以用 10 秒的逐位检查替代
30 分钟的 105 题跑测。判据：`max|diff| == 0`。只有改动触及累加精度
（fp32→bf16、split-K 的原子加顺序、量化）时才需要真跑门禁。

注意历史 L3 成绩全部由 `run_l3_gate.sh` 产生，而该脚本默认
`FLAGGEMS_ILUVATAR_MM_SKINNY=0`，**所以历史上生产栈（SKINNY=1）的 L3
从未被真正实测过**。现由 §17.48.1 的逐位等价性覆盖，无需补跑。

### 17.49 16k prefill 分块：`mbt=8192` 更差，4096 已是最优（2026-11-03）

**动因**：把 16k 的 432.6 s 墙钟拆开后，decode 只占 48%（4369 步 × 47.21 ms =
206 s），**prefill + 气泡占 52%（226 s）**。而 decode 已达 572 GB/s = 97% 峰值，
无法再动，所以 prefill 才是 16k 真正的时间所在。

`--max-num-batched-tokens 4096` 会把 16384 token 的 prompt 切成 4 块，
长上下文 attention 在窄块上效率低；历史也见过 16k 偏好 8192（5115 vs 5101.76）。
但那是 `SKINNY=0` 时代，`SKINNY=1 + mbt=8192` 从未测过，属于开放格。

#### 结果：负。8192 同时牺牲吞吐与 TTFT

同会话 A/B，仅 `VLLM_FL_ILUVATAR_MAX_NUM_BATCHED_TOKENS` 不同
（`SKINNY=1`、`FA_PAD=1`、util 0.85 全部相同），16k `[16384,1024,32,128]`：

| `mbt` | Total tok/s | Median TTFT |
|---:|---:|---:|
| **4096（默认）** | **5148.11**（5151.67 / 5144.56） | **6194.82 ms** |
| 8192 | **5124.45**（第 1 轮即定论，第 2 轮中止） | **8366.32 ms** |

`mbt=8192` 吞吐 **−0.46%**，TTFT **+35%**。两个指标同向变差，不需要跑第 2 轮
（4096 自身两轮离散仅 0.14%，−0.46% 已在其外）。

#### 为什么与「大块 prefill 更高效」的直觉相反

16k 是 prefill/decode **交织**负载，不是纯 prefill。块越大：
- 单个 prefill 块独占 GPU 的时间越长 → 30 个 decode 序列被饿住更久
- 交织粒度变粗 → decode 被打断更多

这与观测到的 `Mean ITL 86.5 ms` ≫ `Median ITL 47.2 ms`（1.83×）一致：
尾部慢步正是被 prefill 块阻塞的 decode 步。放大块只会加重这一点。

⇒ **`mbt=4096` 保持，不提供改动。** 16k 的 prefill 侧需要的是
「更细粒度的交织」或「更快的长文 attention」，不是更大的块。

### 17.50 「prefill/decode 分开调度」不可行：`mbt` 约束下代价 3.5%（2026-11-03）

**假设**（用户提出，方向合理）：每步混着 4096 个 prefill token 和 30 个 decode
token，两种负载互相干扰；若把 prefill 步与 decode 步完全分开，各自 kernel 更单纯，
可能更快。

#### 先确认：kernel 层其实**已经**分开了

`patch_triton_attn_reorder_split_for_iluvatar` 每步发布 decode/prefill 边界
（`set_batch_split`），fused op 据此把纯 decode 行送 flashinfer FA、纯 prefill 行
送 ixformer FA。16k 跑测里 `split=882` 就是这个机制在生效。
**所以 attention 已经是分开调度了，剩下混在一起的只有 scheduler。**

#### 测法

关闭 chunked prefill，让调度器发出 prefill-only 步与 decode-only 步。
硬约束：`mbt` 必须 ≥ 最长 prompt，因此该 arm 必须 `mbt=16384`
（对照 arm 同用 16384 以隔离变量）。

#### 结果：大 `mbt` 的代价已远超分离可能带来的收益

16k `[16384,1024,32,128]`，`SKINNY=1`：

| 配置 | Total tok/s | Median TTFT | P99 ITL |
|---|---:|---:|---:|
| **`mbt=4096`（生产默认）** | **5148.11** | **6194.82 ms** | — |
| `mbt=16384` + chunked（分离前的对照） | **4968.96（−3.5%）** | 9880.02 ms（+59%） | **1606.38 ms** |

对照 arm 第一轮即 **4968.96**，比生产默认低 3.5%，甚至低于历史基线 5014。
用户判断「差很大就直接停」，已停止，未跑分离 arm。

理由：分离 arm 必须同样 `mbt=16384`，起点就是 −3.5%。分
离若要成立，必须先填掉这 3.5% 再谈收益，而此时 P99 ITL 已经 1.6 s。

#### 机制

`mbt` 越大 → 单个 prefill 块独占 GPU 的时间越长 → 30 个 decode 序列被饿住的
窗口越长。P99 ITL 从正常量级跳到 **1606 ms** 就是直接证据。
这与 §17.49（`mbt=8192` 比 4096 差 0.46%、TTFT +35%）同因，且更严重。

#### 结论

**在 vLLM「关闭 chunked prefill 要求 `mbt ≥ 最大 prompt`」的约束下，16k 的
prefill/decode 分离调度不可行。** 不是「没试」或「调参没调好」，是这条约束本身
让分离的入场费高于其上限收益。

16k 若要再进一步，正确方向是**保留细粒度交织（4096）的前提下**压缩单步时长，
而不是改变交织粒度。而单步的两侧都已贴墙：decode 572 GB/s（97% 峰值），
prefill 在跑测中 9.3k tok/s。

> **勘误（§17.51）**：本节写「独立跑 14.5k tok/s，差 36%」是**错的**。
> 纯 prefill 实测只有 10.3k tok/s（`mbt=4096`）／10.5k（`mbt=8192`），
> 跑测中的 prefill 与独立 prefill **基本同速，交织几乎没有损失**。
> 那个 14.5k 是拿 `mbt=8192` 的独立测量去比 `mbt=4096` 的跑测推算值，
> 基准不匹配。16k 真正的账见 §17.51。

### 17.51 16k 墙钟的真实拆分：prefill 204 s + decode 206 s，气泡只有 22 s（2026-11-03）

**动因**：§17.49 把 16k 的 432.6 s 拆成「decode 206 s（48%）+ prefill 与气泡 226 s
（52%）」，但这 226 s 是**减法推出来的余量**，从没单独测过 prefill 到底要多久。
§17.50 的结论又建立在一个 14.5k tok/s 的「独立 prefill」数字上。两者都需要实测落地。

#### 17.51.1 方法：把 prefill 从跑测里单独摘出来

`input=16384, output=1, conc=32, n=128`，`BENCH_RUNS=2 BENCH_SKIP_FIRST=0`。
`max_tokens=1` 意味着每步只有 1 个 token，decode 可以忽略，墙钟 ≈ 纯 prefill。
其余与生产完全一致：`SKINNY=1`、`FA_PAD=1`、`FULL_DECODE_ONLY`、util 0.85、
`max-model-len 131072`。脚本 `/root/bench_results/prefill_mbt_ab.sh`。

#### 17.51.2 结果：prefill 吞吐对 `mbt` 几乎平坦（1.9%）

| 纯 prefill 16k | Run 1 | Run 2 | 均值 throughput | 均值 duration | TTFT |
|---|---:|---:|---:|---:|---:|
| **`mbt=4096`** | 10259.22 | 10270.52 | **10264.87 tok/s** | **204.3 s** | 44914 ms |
| **`mbt=8192`** | 10454.42 | 10465.04 | **10459.73 tok/s** | **200.5 s** | 44112 ms |

`mbt=8192` 纯 prefill 只快 **1.9%**（200.5 vs 204.3 s，省 3.8 s）。

#### 17.51.3 勘误：不存在「交织损失 36%」

§17.50 结尾与 §17.49 的衍生说法「跑测中 prefill 9.3k，独立跑 14.5k，差 36%」
**是错的，已撤回**。错因是基准不匹配：拿 §17.41 的 `mbt=8192` 独立测量
（且是别的 prompt 组合）去比 `mbt=4096` 跑测里反推的 prefill 速率。

按本节实测，跑测中的 prefill（2.10M token / 204 s ≈ 10.3k tok/s）与独立
prefill（10.26k tok/s）**一致**。⇒ **交织没有损失 prefill 效率**，
「针对交织损失做 kernel 优化」这个方向不存在，不该再追。

#### 17.51.4 真正的账：两个阶段几乎完全串行，气泡只占 5%

```
prefill   204.3 s   （本节实测，非推算）
decode    206.0 s   （4369 步 × 47.21 ms，§17.49）
          ─────────
合计      410.3 s
实测墙钟  432.6 s   （§17.49）
气泡        22.3 s   = 5.2%
```

**prefill 与 decode 没有真正重叠**：410 s 的加法已经解释了 95% 的墙钟。
这和 §17.41.2 的「prefill 98.9% GPU-busy」吻合 —— prefill 块一旦上 GPU 就吃满，
decode 只能等它结束。

#### 17.51.5 对下一步的裁决

- **`mbt` 不是杠杆**（纯 prefill 1.9%、e2e 反而 −0.46% 见 §17.49）。
  唯一的变化是把「prefill 与气泡 226 s」修正为「prefill 204 s + 气泡 22 s」。
- **「压缩交织气泡」上限只有 22 s（5.1%）**，而且这 22 s 里还含不可避免的
  调度边界。不值得为它做调度级改造。
- **prefill 的 204 s 是 16k 唯一的大头**。要动它只有两条路：
  提高长上下文 attention 的算力利用率，或减少 prefill 的 FLOP。
  但 §17.41.4 已测 prefill FA 67 TFLOPS、GEMM 80 TFLOPS，都在平台值附近。
- ⇒ **16k 在 kernel 与调度两侧都已到平台，`mbt=4096` 维持。**
  本轮不提供代码改动，只修正记录。

### 17.52 flagtree 0.7.0 修掉了 §16.4 的因果 attention 分母 bug；升级后又挖出 prefill 竞态（2026-11-06）

**动因**：厂商 2026-11-06 公告「天数的 Causal attention 数值错误通过更新 flagtree
可以解决」，要求 `pip install flagtree==0.7.0+iluvatar3.6`；已提交成绩不受影响，
验收复测统一环境。

全仓检索「causal / 编译器 / 数值错」的结果：**本仓记录里唯一的因果 attention 数值
错误就是 §16.4**（`out = ref × S/(S+1)`，`q_len=1/kv_len=1` 输出正好 0.5·v0），
当时结论是「`exp(-inf)=0` 本身正确 ⇒ **疑似编译器在完整 kernel 上下文中的问题，
根因未深挖**」。厂商公告正好补上这个根因。§17.7/§17.9 的 ixformer
`decode` 小 batch 长 ctx 错误是**预编译二进制**，不走 flagtree，与此无关。

#### 17.52.1 验证：stock kernel 从 0.5×v0 变成精确值

脚本 `/root/bench_results/causal_attn_check.py`，走引擎真实符号
`triton_attn.unified_attention`，**不打插件补丁**，fp32 稠密参考：

| ns | q | kv | 0.6.2a3 maxerr | 0.6.2a3 out/ref | **0.7.0 maxerr** | **0.7.0 out/ref** |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 1 | 7.58e-01 | **0.5000** | **0.00e+00** | 1.0000 |
| 1 | 1 | 16 | 4.60e-02 | 0.9133 | 1.25e-03 | 1.0002 |
| 1 | 16 | 16 | 7.58e-01 | 0.8544 | 3.74e-03 | 1.0000 |
| 1 | 256 | 512 | 1.52e-03 | 0.9946 | 4.59e-04 | 1.0000 |
| 1 | 2048 | 2048 | 8.36e-01 | 0.9970 | 3.88e-03 | 1.0000 |
| 1 | 2048 | 4096 | 2.13e-04 | 0.9992 | 1.73e-04 | 1.0000 |
| 4 | 2048 | 4096 | 2.20e-04 | 0.9992 | 1.67e-04 | 1.0000 |

`0.6.2a3: OVERALL FAIL (4 cases)` → `0.7.0: OVERALL PASS`。且 0.7.0 的 stock kernel
误差与「0.6.2 + 我们的 L0 补丁」逐项一致 ⇒ **§16.4 的补丁是在手工实现编译器该做的事**。

#### 17.52.2 L0 修复降级为 best-effort（默认在 0.7.0 上不执行）

补丁仍然能打：`_replace_once` 匹配的是 **vLLM 的源码**，不是 flagtree 的，所以换
wheel 不影响匹配（日志确认 `patch flag: True`）。但 L0 修复在 0.7.0 上已冗余，而它
和 tiling 共用同一个 `try` —— vLLM 哪天改掉 `L = tl.full([BLOCK_M], 1.0, ...)`，
`_replace_once` 抛异常会让**整个 tiling 补丁静默失效**（21.5 → 3.2 TFLOPS）。

改法（`iluvatar.py`）：

* `VLLM_FL_ILUVATAR_PREFILL_L0_FIX=auto|1|0`，默认 `auto`：读 `flagtree` 版本，
  `< 0.7.0` 才改 kernel，`>= 0.7.0` 直接用 stock kernel。
* L0 部分独立 `try/except RuntimeError`，匹配失败只 warning，tiling 继续。
* 版本读不到时**反而执行** L0 修复：在修好的编译器上无害，在坏的编译器上跳过则是静默错。

#### 17.52.3 ❗升级后新发现：`w8 + stages=2` 在 0.7.0 上竞态

做完 17.52.1 后，同一脚本的 `q=2048/kv=2048` 用例开始间歇 FAIL，且同一进程内
**8 次相同调用得到 8 个不同输出**。定位过程：

| 检查 | 结果 |
|---|---|
| stock kernel（BM16）× 3 进程 | 恒 `3.88e-03`，PASS |
| 插件 BM128 补丁 × 8 次同张量 | **8 种输出**，err `9.1e-03 … 5.5e-02` |
| 输入确定性（`randn` 签名 × 3 进程） | 恒 `de63a9dd7a26`，与 kernel 无关 |
| 强制 `L0_FIX=1` | 仍 8 种 ⇒ 不是 L0 |

tiling 扫描（`/root/bench_results/attn_patch_determinism.py`、`attn_cfg_bench.py`）：

| BM | TILE | warps | stages | 8 次重复 | 2048/2048 | 2048/4096 | 64×4096 | err(64×4096) |
|---:|---:|---:|---:|---|---:|---:|---:|---:|
| 128 | 64 | **8** | **2** | ❌ 7 种 | 26.19 TF | 29.59 TF | 32.30 TF | **1.48e-01** |
| 128 | 32 | 8 | 2 | ❌ 8 种 | — | — | — | — |
| 128 | 128 | 8 | 2 | ❌ 3 种 | — | — | — | — |
| 64 | 64 | 8 | 2 | ❌ 8 种 | — | — | — | — |
| **128** | **64** | **4** | **2** | ✅ 1 种 | 21.06 TF | 23.68 TF | **26.94 TF** | 5.09e-03 |
| 128 | 64 | 8 | 0 | ✅ 1 种 | 18.08 TF | 20.29 TF | 22.02 TF | 5.09e-03 |
| 128 | 32 | 8 | 0 | ✅ 1 种 | 19.94 TF | 21.87 TF | 24.23 TF | 5.09e-03 |

⇒ **触发条件是 `num_warps=8` + 显式 `num_stages=2` 的软件流水线**，与 BM/TILE 无关。
`w4 s2` 和「不加 stages」都确定，误差等于 bf16 舍入下限 5.09e-03。

注意 `w8 s2` **更快**（32.3 vs 26.9 TFLOPS）却错 29×：竞态让它跳过了真实计算，
所以"变快"本身是故障信号。

#### 17.52.4 ❗竞态是 0.7.0 引入的回归，不是老问题

回滚到 0.6.2a3（清空 `~/.triton/cache` 后重测）对比同一补丁、同一 tiling：

| flagtree | warps/stages | 2048/2048 | 2048/4096 | 64×4096 | 8 次重复 | maxerr |
|---|---|---:|---:|---:|---|---:|
| **0.6.2a3** | **w8 s2** | 0.855 ms / 20.11 TF | 2.285 ms / 22.56 TF | 190.3 ms / **23.11 TF** | ✅ 1 种 | 5.09e-03 |
| 0.6.2a3 | w4 s2 | 1.443 ms / 11.91 TF | 3.956 ms / 13.03 TF | 320.0 ms / **13.75 TF** | ✅ 1 种 | 5.09e-03 |
| 0.7.0 | w8 s2 | 0.656 ms / 26.19 TF | 1.742 ms / 29.59 TF | 136.2 ms / 32.30 TF | ❌ 7 种 | **1.48e-01** |
| 0.7.0 | w4 s2 | 0.813 ms / 21.06 TF | 2.179 ms / 23.66 TF | 163.2 ms / **26.95 TF** | ✅ 1 种 | 5.09e-03 |

⇒ **0.6.2 上 `w8 s2` 完全确定（8 次同输出、误差 = bf16 下限），竞态是 0.7.0 新引入
的 compiler 回归。** 这推翻了我上一节写的"0.6.2 上换成 w4 代价近似为零"——
那组 `0.816 ms / 21.06 TFLOPS` 是在 **0.7.0** 上测的，0.6.2 上 w4 只有
**13.75 TFLOPS**（−40%）。默认值必须按编译器分叉。

另外一个意外收获：**0.7.0 的 w4 s2（26.95 TF）比 0.6.2 的 w8 s2（23.11 TF）还快
17%**。所以换编译器 + 换 warps 对 prefill 是净收益，不是妥协。

#### 17.52.5 默认值按 flagtree 版本分叉

```python
_default_warps = 4 if (not flagtree_ver or flagtree_ver >= (0, 7, 0)) else 8
```

| flagtree | 默认 warps | 依据 |
|---|---|---|
| `< 0.7.0` | **8** | 0.6.2 上 w8 = 23.11 TF 且确定；w4 只有 13.75 TF |
| `>= 0.7.0` | **4** | w8 竞态；w4 = 26.95 TF 且确定 |
| 版本读不到 | 4 | 保守取"假定是新编译器"，因为竞态是错算，比慢更严重 |

显式 `VLLM_FL_ILUVATAR_PREFILL_NUM_WARPS` 始终优先；若在 `>= 0.7.0` 上显式设置
`w8 + s2`，启动打 warning。

#### 17.52.6 0.7.0 上的 tiling 重扫：`BM128 T64 w4 s2` 仍是最优确定性配置

脚本 `/root/bench_results/tiling_sweep_070.sh` + `attn_cfg_bench.py`，
每个配置 4 次重复判确定性（`ns=64, q=4096, kv=4096` 是最敏感的形状）：

| BM | TILE | warps | stages | 64×4096 | 确定性 | maxerr(64×4096) |
|---:|---:|---:|---:|---:|---|---:|
| **128** | **64** | **4** | **2** | **26.95 TF** | ✅ | 5.09e-03 |
| 128 | 32 | 4 | 2 | 24.72 TF | ✅ | 5.09e-03 |
| 128 | 64 | 8 | 3 | 22.02 TF | ✅ | 5.09e-03 |
| 128 | 64 | 8 | 0 | 22.01 TF | ✅ | 5.09e-03 |
| 128 | 64 | 4 | 3 | 14.24 TF | ✅ | 5.09e-03 |
| 128 | 64 | 4 | 0 | 14.24 TF | ✅ | 5.09e-03 |
| 128 | 128 | 4 | 2 | 12.00 TF | ✅ | 5.09e-03 |
| 256 | 128 | 4 | 2 | 10.27 TF | ✅ | 5.09e-03 |
| 256 | 64 | 4 | 2 | 9.88 TF | ✅ | 5.09e-03 |
| **64** | **64** | **4** | **2** | 16.90 TF | ❌ **4 种** | **9.89e-02** |

两条新结论：

* **`BM=64` 也会竞态**（即使 w4 s2）。所以竞态的触发面比 17.52.3 写的更宽：
  `BM=128` 是必要的，`w8+s2` 是另一条充分条件。两者都必须避开。
* `s2`（显式 2 段流水）在 `w4` 下是**必需的**：`s0`/`s3` 都掉到 14.24 TF
  （−47%）。而 `w8` 下 `s2` 是**致命的**。同一个 knob 在两种 warps 数下方向相反。

#### 17.52.7 待办

* `w8 s2` / `BM64` 的竞态**应反馈厂商**：0.7.0 修好了因果 attention 正确性，却引入了
  这条 pipelining 竞态，且它伪装成"性能提升"（32.3 vs 26.9 TFLOPS），很容易被当成
  优化接受。
* 所有历史 e2e（4k 6045 / 16k 5148）都在 0.6.2 上测出，编译器变更后需复测，
  且 `~/.flaggems/config_cache` 里的 GEMM 调优结果可能需要重调（见 17.53）。

### 17.53 Triton FlashDecoding decode 后端：微基准 4k 过 1.0，但 e2e 不涨（2026-11-06）

新增 `flag_gems/.../fused/paged_decode_triton.py`（`paged_decode_kernel` +
`paged_decode_combine`），并在 `ixformer_unified_attention.py` 的 decode 分支
前置一道门控 `_triton_decode_wanted()`（`VLLM_FL_ILUVATAR_DECODE_TRITON`，
`auto`/`1`/`0`）。目的：给 decode 一条不依赖厂商的路径。

#### 17.53.1 先纠正一个错的基线

此前一直拿 `paged_attention_flashinfer` 当 16k 的基线，但生产分发在
`n<32 && ctx>=2048` 走的是 **pad-to-32 FA**（§17.37）：

| 16k n=29 | 时间 | GB/s |
|---|---:|---:|
| `_paged_attention_decode_padded`（生产实际） | **0.805 ms** | 641 |
| `paged_attention_flashinfer` | 1.241 ms | 417 |
| `paged_decode_splitk` | 1.615 ms | 320 |

所以「16k 快 1.09x」是拿错了对手。真实基线是 641 GB/s，本就接近峰值。

#### 17.53.2 三个有效改动

* **K 直接按转置顺序 gather**（`[hd, page]`）而非 load 后 `tl.trans`：
  避免寄存器↔smem 布局转换，4k 从 1.741 → 0.746 ms。
* **log2 域 + 去掉每页 rescale**：rescale 只为防 `exp` 溢出；log2 域下固定
  偏移 0 就安全（`exp2` 参数 clamp 到 80），于是 `acc` 变成纯 MMA 累加
  （`tl.dot(p, v, acc)`），不再有 `BM×128` fp32 乘法压在累加器关键路径上。
  4k +8%，16k +12%。
* **热循环零 mask**：只有序列最后一页可能不满，`p_full_end` 之前完全无 mask。

踩过的坑：`p_full_end` 必须 `tl.maximum(p_start, ...)`，否则某个分片无页可跑时
会退回到本片起点之前的页，把别的分片已算过的页重复计入（混合长度测试里
误差 1.86，修复后 0.006）。

#### 17.53.3 剩余 20% 的定位：MMA 不与访存重叠

4k n=64 拆解（同 kernel，只改循环体）：

| 变体 | 时间 | GB/s |
|---|---:|---:|
| 两个 dot（实际 kernel） | 0.531 ms | 568 |
| 只保留 K 的 dot | 0.503 ms | 600 |
| 完全不做 MMA | **0.456 ms** | **661** |
| 只读 K（一半流量） | 0.330 ms | 457 |

纯访存 661 GB/s（高于标称 590），两个 MMA 加 75 µs，而 456+75=531 恰好等于
实测 —— **MMA 与 gather 完全不重叠**。`num_stages>=2` 能流水但占用常驻 smem
（16 CTA→更少），手动预取下一页会让寄存器溢出，双累加器链也无改善，都实测更慢。

#### 17.53.4 dispatch 到 dispatch 的形状扫描

同 Python 入口、只换后端体（避免把 wrapper 开销算进收益）：

| ctx | n | 生产路径 | 生产 ms | 本 kernel | 比值 |
|---:|---:|---|---:|---:|---:|
| 4600 | 32 | flashinfer | 0.351 | 0.281 | **1.25x** |
| 4600 | 64 | flashinfer | 0.580 | 0.541 | **1.07x** |
| 4600 | 128 | flashinfer | 1.048 | 1.053 | 1.00x |
| 4600 | 29 | fa_pad | 0.244 | 0.281 | 0.87x |
| 17408 | 29 | fa_pad | 0.805 | 0.807 | 1.00x |
| 17408 | 64 | flashinfer | 1.832 | 1.830 | 1.00x |

门控因此取 `32 <= n_batch <= 144 and max_seqlen_k <= 8192`。上界 144 而非 64：
FULL_DECODE_ONLY 下 kernel 看到的是**补齐后的捕获尺寸**（64 请求 → graph n=72），
按 64 卡会把大部分评测步打回 vendor 路径。

#### 17.53.5 ❗e2e A/B：没有收益（这是本节最重要的结论）

4k / c64 / 128 prompts，同机先后跑（各 2 轮，`SKIP_FIRST` 未启用）：

| | Run1 tok/s | Run2 tok/s | TPOT Run1 | TPOT Run2 |
|---|---:|---:|---:|---:|
| `DECODE_TRITON=0`（对照） | 6161.69 | 6182.94 | 46.02 | 46.00 |
| `DECODE_TRITON=auto`（启用） | 6156.15 | 6059.59 | 46.03 | 46.59 |

**微基准 1.07x 换算到 e2e 是 0（甚至 -0.7%，在噪声内）**。TPOT 三位有效数字
上千位不变，说明 4k/c64 的 decode attention 在这张卡的 per-token 时间里占比
远小于「0.54 ms × 42 层 ≈ 22.7 ms / 46 ms」这个朴素估计，瓶颈不在注意力。

结论：这条后端**保持可选（`auto` 只在实测领先的形状接管，且实测领先只在微基准
成立）**，不要指望它抬 e2e 分数。真正的下一步应该先量出注意力在 per-token
时间里的真实占比，否则继续调 attention kernel 是在优化不在关键路径上的东西。

### 17.54 Decode 真实占比 + Triton 默认关 + silu 收尾（2026-11-06）

§17.53 留下的问题：e2e 不涨，需要先量 per-token 占比。本节用 CUDA-event 拆步，
并关掉 Triton decode 默认路径；顺手把 silu / skinny 的剩余空间也结掉。

#### 17.54.1 CUDA-event 占比（n=64 ctx=4600，×42 层）

产物 `/root/bench_results/ixkn/share_20261106/summary.json`。

| 组件 | /层 | ×42 | eager 份额 |
|---|---:|---:|---:|
| FA warm（生产 pages 缓存） | **0.489 ms** | 20.53 ms | **59.4%** |
| GEMM ×4（skinny=0） | 0.187 ms | 7.84 ms | 22.7% |
| silu_and_mul（旧 FlagGems） | 0.090 ms | 3.78 ms | 10.9% |
| rope | 0.039 ms | 1.64 ms | 4.7% |
| fused_rms ×2 | 0.015 ms | 0.61 ms | 1.8% |
| **合计（FA）** | 0.822 ms | **34.54 ms** | 100%（占 median ITL 35.78 ms 的 96.5%）|

关键纠正：

* Triton decode **0.540 ms = 0.91× warm FA**。§17.53.4 的「1.07×」是拿
  **冷路径** `pages=None`（0.58–0.79 ms）比的，生产不走那条。
* ixkn-cli 4.4.0 在本机 import / Section Results 打印路径全部挂掉（SEGV /
  `Unknown device`）；dig 原命令复现同样采到 rep 但打不出表。占比以 CUDA-event
  为准，不再等 ixkn。

#### 17.54.2 图内才是真账：eager silu/rope 份额是假的

同形状进 CUDA graph（42× replay / 层）：

| op | eager | graph / 层 | ×42 |
|---|---:|---:|---:|
| 旧 FlagGems `silu_and_mul` | ~110 µs | **19.7 µs** | 0.83 ms |
| PyTorch `F.silu*mul` | ~220 µs | 9–22 µs | — |
| FlagGems rope | 39 µs | **6.0 µs** | 0.25 ms |
| fused_add_rms_norm | — | 4.6 µs | 0.39 ms（×2）|

与 §17.47「silu 融合图内只省 0.20 ms/step」一致：eager 10.9% 是 launch 税，
FULL_DECODE_ONLY 下不付。

#### 17.54.3 Triton decode：默认改为关

`VLLM_FL_ILUVATAR_DECODE_TRITON` 默认 `auto` → **`0`**。

* 微基准相对生产 warm FA 是 0.91×，e2e A/B 空（§17.53.5）。
* `=1` / `=auto` 仍可用（`auto` 保留 `32≤n≤144 & ctx≤8192` 门控），仅作无厂商依赖实验路径。
* site-packages 已同步。

#### 17.54.4 skinny GEMM：无可再挖（计分 M=64）

官方 `--max-concurrency 64` → capture size **64**（spare 72 只给短突发）。
图内复测：

| shape @ M=64 | tuner | 强制最优 | 结论 |
|---|---:|---:|---|
| qkv | 21.4 µs | LE64 21.1 µs | 已贴齐 |
| o | 18.9 µs | LE64 **15.8 µs** | 已默认开（§17.47） |
| gate | 84 µs / 620 GB/s | 任意强制更慢 | 不动 |
| down | 54 µs / 487 GB/s | 任意强制更慢 | 不动 |

M=72（spare 突发）qkv 有 `BM128 BN128 BK64 w8` = 1.12× tuner，但**不在
4k 计分批**，不写入默认 LE64。

#### 17.54.5 silu：换掉 pointwise_dynamic，图内 19.7 → 4.3 µs

新增 `_iluvatar/fused/silu_and_mul.py`（连续 bf16/fp16 走单 kernel；其它回退
ATen），经 `SpecOpRegistrar` 覆盖泛型 `pointwise_dynamic` 路径。插件
`silu_and_mul_iluvatar` 同步改为优先 FlagGems / 本地 Triton。

| | graph / 层 | ×42 |
|---|---:|---:|
| 旧 FlagGems pointwise | 19.7 µs | 0.83 ms |
| 新 iluvatar fused | **4.28 µs** | **0.18 ms** |
| vs PyTorch graph | 9.03 → 4.28 | 省 **0.20 ms/step** |

相对 fp32 参考 max abs err 0.029（bf16 正常）。带宽上限约 4 µs，已到 4.3 µs。

**e2e（B 臂 only，`SILU_FAST=1`，2 轮）**：4k / c64 / 256 prompts →
6201.59 / 6209.54 tok/s（均值 **6205.57**，Median ITL 35.16 / 35.24）。
相对近期对照（DECODE_TRITON off ~6162/6183）同量级，**无回归、也看不见增量**。
计分栈 `VLLM_COMPILE` + `custom_ops=['none']` 下 silu 多半被 inductor 吃掉；
改动仍保留——FlagGems / vendor OOT 路径不再比 native 慢，`SILU_FAST=0` 可回退。

#### 17.54.6 4k decode 刀口现状

| 刀 | 状态 |
|---|---|
| Attention（59%） | warm FA ≥ 峰值带宽；Triton 默认关 |
| GEMM（~20% 图内） | LE32+LE64(qkv/o) 已上；gate/down 贴齐 |
| silu / rope / rms | 图内合计 <1.5%；silu 已换快路径 |
| 下一步 | 不要再抠 decode 小算子。真正杠杆在 **KV 流量**（INT8/FP8 KV，
  准确率约束下）或 **16k / TTFT / 调度气泡**，不在再写一个 decode attn kernel。 |

### 17.55 ZipFusion-lite 自研 CUDA decode MVP（2026-11-06）

前置（`zipfusion_fa_probe`）：fused ZipFusion 叠在 warm FA 上天花板约 **1.27× /
~12% ITL**，但 ixformer FA 闭源读不了压缩块；叠 Triton decode 连 0.75 流量天花板
都打不过 FA。用户要求试「插件侧自写读压缩 KV 的 CUDA decode」。

#### 实现

路径：`vllm_fl/.../iluvatar/zipfusion_decode.{cu,cpp,py}`

* 格式：每 `(page=16, kv_head, hd=128)` tile = 4×512；flags+bases 头 + 12-bit
  压缩（1.5 B/val）或 raw bf16 回退。
* Kernel：FlashDecoding（split）+ online softmax（log2）+ LSE combine；
  块内 decompress→smem→标量 QK/AV。
* 门控：`VLLM_FL_ILUVATAR_ZIPFUSION_DECODE`（默认 **0**）。**未接入** ixformer
  热路径——见下。

#### 微基准（B=16, CTX=2048, MiniCPM5 GQA 16/2/128）

| | |
|---|---|
| pack byte_ratio（smooth-ish rand×0.12） | **0.76**（贴近纸面 0.75） |
| pack roundtrip | bit-exact |
| vs torch ref | cos ≈ **0.99**（CTX≤2k, split≤8） |
| warm FA | **142 µs**（~236 GB/s） |
| ZipFusion CUDA v3 | **~12 ms**（最佳 split=16 仍 ~11 ms） |
| vs FA | **0.012×**（慢约 **80×**） |

产物：`/root/bench_results/zipfusion_cuda_mvp_result.txt`、
`zipfusion_cuda_mvp.py`。

#### 结论

1. **压缩格式 + 正确性 OK**，插件可独立打包/解码。
2. **性能完全不可用**：无 MMA 的标量 FlashDecoding 相对已带宽打满的 vendor FA
   慢两个数量级；0.76× 流量省不回来。
3. **~12% ITL 天花板仍只在「改 ixformer FA 寄存器解压」时成立**；插件自研 CUDA
   这条路在 BI-V150 上关闭。
4. 代码保留作实验（默认关）；不接线、不做 e2e。下一步 KV 流量仍看
   **INT8/FP8 KV（有损）** 或等 vendor FA 补丁。

### 17.56 MMA 重做门控：插件侧 ZipFusion 打不过 FA（2026-11-06）

用户要求用 CoreX MMA 重做，目标超过 warm FA。公开 `wmma` 头基本是
`__half`；bf16 实用 MMA 仍走 Triton `tl.dot`（现网 decode 0.91× FA）。

#### 门控（B=64, CTX=4600，满 HD=128 的 FlashDecoding + expand）

| 路径 | us | vs FA |
|---|---:|---:|
| warm FA | 487 | 1.00× |
| Triton dense bf16 | 540 | 0.90× |
| u8→bf16 expand（50% 流量）+ full MMA | **677** | **0.72×** |
| zip12 expand（75% 流量）+ half-dot | **4577** | **0.11×** |

产物：`/root/bench_results/zf_mma_gate_result.txt`。

要点：减流量后 **仍慢于 FA**，且慢于 dense Triton——解压/expand 税 > 省下的
DRAM。旧 `pack64`「赢 FA」是截断 HD（算错），不能当 ZipFusion 证据。

#### 厂商 int8 旁路

`ixformer.paged_attention(..., scales, algo=1)` 在本机 build 里是空壳
（刷屏 `VLLM_PAGED_ATTENTION_IXFORMER=OFF`，0 µs、cos=0）。flashinfer 热路径
仍无 scales 参数。

#### 结论

插件侧「自研解压 + MMA」**无法超过** 已带宽打满的 ixformer FA。要超过 FA：

1. 天数打开/提供 **fused int8/fp8 KV FA**（改 build 或新包），或  
2. 厂商在 flashinfer 内核内做 ZipFusion 解压，或  
3. 接受不再追 decode attn，转 TTFT/调度/权重量化。

MMA 重做在门控失败后停止；不继续堆 WMMA 版 ZipFusion。

### 17.57 ixkn 深挖：u8 loads 有 ROOM，fused expand+MMA 把带宽打没（2026-11-06）

§17.56 门控后 loads-only 又挖出「纸面 ROOM」：`loads_u8_exp≈233µs`（Dram~83%、
~610 GB/s），proj `233+MMA≈317 < FA≈487`。本节用 ixkn + fused 实测把这个洞堵上。

#### 对照（B=64, CTX=4600；ixkn `--set default --kill off -k`，**不要** `-o` 否则
FA/部分 Triton 会在 Section 打印前 SEGV）

| Kernel | µs | Dram% | Mem GB/s | Theo occ | Limiter | Regs/warp |
|---|---:|---:|---:|---:|---|---:|
| warm FA (`paged_attn_flashinfer`) | **~489** | **80** | **~590** | 25% | VRF | 65 |
| Triton dense `_paged_decode_kernel` | ~536 | 72 | ~525 | 12.5% | **smem** | 104 |
| `_loads_u8` + expand | **~233** | **83** | **~610** | 100% | none | 32 |
| fused `_fused_u8_decode` best | **~599** | **32** | **~241** | 25% | VRF | 68 |

FA lean 修复：KV 必须是 **phys HND contiguous** 的 BNDH permute 视图（非
`BNDH.contiguous()`）；否则 flashinfer `is_contiguous()` 断言炸。产物：
`/root/bench_results/zf_ixkn/{rebuild_tensors,wk_lean,fa3_live,fused_u8_b_live}.txt`。

#### Fused 扫参（Triton u8 hi-byte → bf16 bitcast + FlashDecoding）

| 变体 | best µs | vs FA |
|---|---:|---:|
| fused u8（双 tile 同页加载） | **599** | 0.81× |
| lean expand / dual-page / pipeline | 633–645 | ≤0.77× |
| serial K→QK then V→PV | 633 | 0.77× |
| 2-pass（probs 写 workspace 再 PV） | 674 | 0.72× |

`cos(u8_hi, FA)≈0.945`（有损 hi-byte，仅作功能烟测）。

#### 诊断：ROOM 为什么合不上

拆核（`/root/bench_results/zf_fused_u8_diag.txt`）：

* `exp_only`（K+V expand，无 MMA）= **233µs** —— 与 loads 一致，减流量是真的。
* `qk_only`（只 K expand + QK MMA）= **169µs** —— 单侧 MMA 也能很快。
* `full`（K+V expand + 双 MMA）= **599µs** —— **不是** `233+75`，而是带宽从
  610→241 GB/s、Dram 83%→32%。expand 结果一旦喂进 `tl.dot`，Triton 走 MMA
  operand 布局（伴随 smem bank conflict 警告），把 u8 省下的流量优势吃光，
  还慢于 dense Triton。

软件流水 / 串行化 K/V / 两遍存储 probs **全都救不回来**。

#### 结论（更新 §17.56）

1. **纸面 ROOM 作废**：`loads_u8_exp` 的 233µs 不能外推到 fused decode；瓶颈是
   **expand→MMA 的代码生成 / 带宽效率**，不是 raw DRAM 字节数。
2. 插件侧 Triton「自研 u8/ZipFusion expand + `tl.dot`」最好 **0.81× FA**，打不过
   已 ~590 GB/s 打满的 ixformer FA。
3. 要超过 FA，仍然只能：
   * 天数 **fused int8/fp8 KV FA**（内核内解压），或
   * 自研 **ivcore CUDA** 显式 async u8→smem（MMA 布局）→TCU（Triton 此路不通），或
   * 停追 decode attn。
4. 默认仍不接线；保留 `/root/bench_results/zf_fused_u8*.py` 与 `zf_ixkn/` 作证据。

### 17.58 ivcore TCU fused decode：live-score 税堵死 ≤450µs（2026-11-06）

§17.57 把「自研」压到 **ivcore CUDA**（显式 u8→B + `ixix::wmma`）。本节用 ixkn +
微基准证明：在 BI-V150 上，**正确的 fused QK∥PV 一旦让 QK 分数保持 live，TCU
双发就被打断**，正确路径的微基准地板 ~620–820µs，**达不到 ≤450µs**。

#### 目标与对照

* Goal：自研 kernel ≤450µs @ B=64 CTX=4600（warm FA ≈487µs）。
* 工具：`ixkn-cli`、`/root/bench_results/ixkn/ivcore_20261106/`、
  `/tmp/ivcore_qk_micro/*`、`ivcore_u8_beat_fa.py`。

#### 全核实测

| 版本 | best µs | vsFA |
|---|---:|---:|
| warm FA | **~487** | 1.0× |
| v12 | ~1271 | 0.38× |
| v16 lean（VRF 105→88） | ~1349 | 0.36× |
| v17 delayed-P / IL schedule | ~1331 | 0.37× |

ixkn（v16）：Dram~14%、TCU~5.5%、IPC~0.08、occ 25% VRF——**不是** DRAM 打满，
是流水被掐。

#### 微基准（512 CTA × 72 pages，对齐全核 grid）

| 变体 | µs | 含义 |
|---|---:|---|
| fused QK+PV，A 常量，exp 被 DCE | **~296–304** | 会赢 450；**分数未使用** |
| `fullgeo_fakep_voidexp`（同 grid） | **295.9** | 同上，可过 goal 但错误 |
| 同上 + `esink += exp(score)`（A 仍常量） | **~817** | **仅让分数 live 就掉到 dynA 档** |
| 写 P 到 smem / 做 LSE | **~817** | 同档 |
| 每页用 **字面量** 重写 A | **~298** | 写 A 本身不贵 |
| 每页用 **循环不变量** 寄存器写回 A | **~313** | 同上 |
| dynA `pack2` / delayed-P / IL-delay | **~790–825** | 正确关联 P↔V 的 TCU 路径 |
| IL + `store_matrix` after PV | **~638** | live-score 最好档 |
| IL + smem P after PV | **~627** | live-score 最好档 |
| QK-only（无 PV 填 TCU） | **~664** | 缺 PV 双发更慢 |
| PV-only（gmem P→pack2） | **~346** | 无 QK 依赖时 pack 很快 |
| 提高 split（总 page-work 不变） | **~820** | 不省墙钟 |
| 非 TCU / smemP PV | **≥1000** | 更差 |
| Triton u8 flashdecoding | **~1330** | 与 §17.57 fused 同级 |

#### 根因（更新先前「dynA = 写 A」表述）

1. **根因是 live QK→softmax 依赖打断 TCU 的 QK∥PV 双发**，不是「往 A fragment
   写数据」本身。
2. 字面量 / 循环不变量 A 可每页重写仍 ~300µs；一旦 `exp(c.x)` 的结果必须活到
   页后（LSE、pack、smem、甚至 `esink+=`），即落到 ~620–820µs。
3. 软件流水（QK(i)∥PV(i−1)）无法恢复到 fakep：跨页变化的 A / 活着的 score
   链仍破坏双发。
4. 两遍（存 P 再 PV）把 QK-only 的气泡暴露出来（~664+~340），总和更差。
5. split 只是重切 CTA×pages，总工作量不变，饱和后墙钟不变。

#### 与 §17.57 的衔接

* Triton u8 fused 最好 ~0.81× FA（Dram 被 expand→MMA 吃掉）。
* ivcore 在「假 P / 死分数」下可 **1.6× FA（~296µs）**，证明 u8→B + TCU 算力
  足够；**正确 softmax 把同一条路径打回 0.6× 微基准 / 0.37× 全核**。
* 插件侧再堆 pack/shfl/IL/非 TCU PV / 提高 split，都过不了 live-score 地板。

#### 结论

1. **≤450µs 的正确 fused ivcore decode：当前证据下不可达**（live-score 税）。
2. 要超过 FA，仍只剩：厂商 **fused low-precision KV FA**，或彻底换 BW 形态且
   能把 u8→MMA 的代码生成做到接近 FA 的 Dram%~80%（§17.57 已证 Triton 做不到；
   ivcore 在正确分数下也做不到 ≤450）。
3. **不接线**；v17 保留作 IL/delayed-P 证据。产物：
   `ivcore_u8_decode.cu`（v17）、`/root/bench_results/ixkn/ivcore_20261106/SUMMARY.md`、
   `/tmp/ivcore_qk_micro/`、`ivcore_u8_beat_fa_result.txt`。


### 17.58 Triton FlashDecoding：row-major K 追平 FA ~487µs（2026-11-06）

§17.53 的 gather-as-`[HD, PAGE]` 路径在 B=64 CTX=4600 上停在 **~540µs**（gap
~53µs vs warm FA ~487µs）。loads-only ~455µs；FA 只比 loads 贵 ~33µs，说明
厂商路径几乎把 MMA 藏进带宽；我们的 strided K gather 把那 53µs 暴露出来。

#### 有效改动

把热循环改成：

* K **连续** load `[PAGE, HD]`（与 V 相同 coalescing）
* `qT = tl.trans(q)`（一次，`[HD, BM]` 小 tile）
* `s = tl.trans(tl.dot(k_tile, qT))` —— 只转置小的 score `[PAGE, BM]`，
  **不再** gather / `tl.trans` 大的 K tile

旧路径「gather 成 `[HD, PAGE]`」是为了避开大 `tl.trans(K)`（曾 1.74→0.75ms），
但 gather 本身成为新瓶颈。小 tile 转置便宜得多。

#### 微基准（B=64, CTX=4600, bf16, page=16, warm）

| 变体 | µs | vs FA |
|---|---:|---:|
| warm FA (flashinfer) | **~487–488** | 1.00× |
| Triton stock (gather K) | ~540 | 0.90× |
| Triton **row-major K** | **~488** | **~1.00×** |
| Triton K-contig layout `[HD,PAGE]` 物理 | ~467 | 1.04× |
| loads-only | ~455 | — |

正确性：vs FA `cos≈0.999994`，`maxerr≈3e-5`（bf16）。

试过但无效/更差：dual-page merge gather、`tl.range` num_stages、软件 prefetch、
interleave K→QK→V、BM=8、fp16 P、split/warps/stages 网格搜索（最佳仍是旧 stock）。

#### 接线

已写入 `flag_gems/.../fused/paged_decode_triton.py`（site-packages + FlagGems-comp）。
本改动把 bf16 FD 从「慢 10%」拉到与 FA 持平。默认门控见 §17.60。
u8 / ≤450µs 目标仍见 §17.57–17.58（live-score / expand 税未解）。

产物：`/root/bench_results/triton_fd_tune/fd_k_rowmajor.py`。

### 17.59 Triton decode 超过 FA：K HD-major contig（2026-11-06）

在 §17.58 row-major（BNDH，~488µs ≈ FA）之上，把 K 存成
``[blocks, n_kv, hd, page]``（MMA 操作数顺序、连续），V 仍 BNDH：

| 路径 | µs @ B=64 CTX=4600 | vs FA |
|---|---:|---:|
| warm FA | ~487–488 | 1.00× |
| Triton BNDH row-major | ~487–488 | ≈1.00× |
| Triton **K HD-major contig** | **~464–466** | **~1.05×（快 ~22µs）** |

`cos≈0.999994`。已接入 `paged_decode_triton`：按 K shape 自动选
`hd_major` → `_paged_decode_kernel_k_contig`，否则 BNDH row-major。
生产 KV 仍是 BNDH，e2e 走 row-major 路径；HD-major 需改 cache 布局才能吃到
微基准那 ~22µs。

### 17.60 4k e2e：Triton decode ≈ FA，默认改为 ON（2026-11-06）

#### 配置

* MiniCPM5-2B，`FULL_DECODE_ONLY`，port 9151，GMU 0.85
* case `[[4096,1024,64,256]]`，每臂 2 run
* A/B：`VLLM_FL_ILUVATAR_DECODE_TRITON=0` vs `=1`
* 产物：`/root/bench_results/ab_4k_decode_triton_beat/`

#### 结果（Total tok/s = input+output）

| 臂 | Run1 | Run2 | 均值 | 输出 tok/s 均值 | ITL 均值 |
|---|---:|---:|---:|---:|---:|
| FA (`=0`) | 6029.82 | 6215.15 | **6122** | ~1224 | ~48.0 ms |
| Triton (`=1`) | 6217.63 | 6234.94 | **6226** | ~1245 | ~47.8 ms |

Triton 臂 probe：`decode_triton=1008`（确实走 FD）。相对 FA **无回归**，
完整吞吐约 **+1.7%**（噪声量级；与微基准 BNDH ~488µs 齐平一致）。

#### 接线决策

`VLLM_FL_ILUVATAR_DECODE_TRITON` **默认 `1`**（FlagGems-comp + site-packages）。
`=0` / `off` 回退厂商 FA；`=auto` 仍用形状门控（32≤n≤144 且 ctx≤8192）。
动机：无厂商依赖的 decode 路径，e2e 至少不差。
