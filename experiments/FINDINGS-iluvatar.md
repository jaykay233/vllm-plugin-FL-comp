# MiniCPM5-2B on 天数 BI-V150：环境与已验证状态

本文件是 [`FINDINGS-mx.md`](./FINDINGS-mx.md)（沐曦 MetaX C500）的**天数对应篇**。
区别很重要：两张卡的基线相差 **2.5×（4k）～7.7×（16k）**，沐曦篇里的收益数字、
根因结论、门槛都不能直接搬到这张卡上。

> **当前状态：尚无已量化的优化收益。** 本文件目前只记录**环境**与**已验证的
> 事实**，以及哪些沐曦结论**尚未在本卡复测**。等本卡跑出 A/B 结果后，按沐曦篇的
> 体例往下续写「收益清单 / 负结果 / 排查经验」。

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
| FlagOS 融合算子白名单（最大单笔收益） | 未测 |
| `flash_attn` 阻塞 D2H 移除（`2fc71f9`） | 未测 |
| decode attention `num_splits` 自适应（`4899f91`） | 未测 |
| `fused_add_rms_norm` 去掉 clone（`dc27f60`） | 未测 |
| M==1 `mm` 路由到 GEMV | 未测 |
| 4k 双峰现象与 20 s 停顿（§4） | 未观测，本卡是否出现未知 |
| 采样器 sort-free top-p v2（§10） | 未测 |

另外本卡的 **flagtune/cost-model 路径根本不可用**（§3.3），所以沐曦篇里围绕
autotune 停顿（§4.8、§4.10）的结论在 5.3.5 上也不一定同构。

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
