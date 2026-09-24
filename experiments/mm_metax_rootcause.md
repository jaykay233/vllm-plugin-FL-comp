# 为什么 FlagGems 的 MetaX `mm` 比原生慢 —— 实测与根因

> ⚠️ **§1–§5 的结论已被 §6 推翻（保留原文以记录过程）。**
> `pipeline: "null"` 这个缺陷是真的（改掉后警告消失），但**它不影响性能**。
> 真正的瓶颈是：**每次调用 ~0.1 ms 的主机侧开销** + **prefill 的 kernel 质量** +
> **`linear` 在 M>1 时回落到通用 kernel**。请直接看 §6、§7。

> 目的：回答「修 autotune 还是修 kernel」。结论是**修 kernel/配置**，autotune 不是吞吐瓶颈。
> 脚本：`/root/src/mm_metax_probe.py`、`mm_metax_probe2.py`、`splitk_route.py`
> 环境：MetaX C500，bf16，MiniCPM5-2B 真实形状；`flag_gems` 取
> `/opt/conda/envs/mx/lib/python3.12/site-packages/flag_gems`（server 实际加载的那份）。

## 1. 方法上的两个坑（先记下来，否则会测错）

1. **布局**。`_m1_gemv`（M==1 快速路径）要求 `b.stride() == (1, K)`，即 b 是
   `W.t()` 的**转置视图**（W 为 (N,K) 连续）—— 这正是模型 `x @ W.t()` 的形态。
   若用 `b = randn(K, N)` 连续（stride `(N,1)`），GEMV 路径被跳过，M==1 的数字无意义
   （v1 测出 6–7x，v2 修正后是 1.1–3.5x）。
2. **调用路径**。直接 `flag_gems.mm()` 与服务器走的 aten 派发不是一回事。
   v2 改为两臂都调 `torch.mm`，只用 `only_enable(include=["mm"])` 切换后端；
   native 臂在 `import flag_gems` **之前**测，避免被全局 patch 污染。

顺带确认：`flag_gems.mm` 确实走 MetaX 实现 —— autotune DB 里
`mm_kernel_general` 表数为 **0**，全是 `mm_kernel_nt/nn/splitk`。

## 2. 稳态结果（autotune 已热；单位 ms）

| shape | impl | M1 | M2 | M8 | M32 | M512 | M2048 |
|---|---|---|---|---|---|---|---|
| qkv (2560,2048) | native | 0.0120 | 0.0228 | 0.0166 | 0.0167 | 0.0385 | 0.1319 |
| | flaggems | 0.0380 | 0.1102 | 0.1114 | 0.1102 | 0.1169 | 0.2466 |
| gate_up (12288,2048) | native | 0.0360 | 0.0371 | 0.0387 | 0.0397 | 0.1347 | 0.5220 |
| | flaggems | 0.0394 | 0.1107 | 0.1118 | 0.1130 | 0.3085 | 0.8935 |
| o_proj (2048,2048) | native | 0.0111 | 0.0193 | 0.0157 | 0.0158 | 0.0378 | 0.1009 |
| | flaggems | 0.0383 | 0.1199 | 0.1219 | 0.1096 | 0.1152 | 0.1870 |
| down (2048,6144) | native | 0.0204 | 0.0360 | 0.0361 | 0.0392 | 0.0977 | 0.2773 |
| | flaggems | 0.0378 | 0.1183 | 0.1182 | 0.1113 | 0.1790 | 0.5759 |

**倍率 flaggems/native（<1 为 FlagGems 更快）**

| shape | M1 | M2 | M8 | M32 | M512 | M2048 |
|---|---|---|---|---|---|---|
| qkv | 3.17 | 4.84 | **6.71** | **6.60** | 3.04 | 1.87 |
| gate_up | 1.09 | 2.98 | 2.89 | 2.84 | 2.29 | 1.71 |
| o_proj | 3.46 | 6.22 | **7.77** | **6.92** | 3.05 | 1.85 |
| down | 1.85 | 3.29 | 3.27 | 2.84 | 1.83 | 2.08 |

要点：
- **prefill 尺度（M=2048）仍慢 1.7–2.1x**；
- **M=8–32 最差，2.8–7.8x** —— 这正是 decode 里「多序列并发」的区间；
- M=1 只有窄 N 的投影落后（有效带宽：qkv 276 vs 原生 875 GB/s、o_proj 219 vs 758），
  而最宽的 `gate_up` 已基本持平（1276 vs 1397 GB/s）→ 窄 N 的 grid 太小、占不满带宽。

## 3. autotune **不是**吞吐瓶颈

在「新形状首次调用」上测冷/热：`M=1009 cold=0.4ms, M=1777 cold=0.5ms`，
`cold - warm ≈ 0.3–0.4 ms`。**没有 20s 级别的 autotune 开销**，
所以「把 autotuner 修好」对吞吐没有帮助。

> 需要修正 `FINDINGS.md` §4.10 的归因：白名单排除 `mm` 后停顿确实消失（A/B 已证），
> 但「停顿 = mm 的 autotune」这条**可能不成立于全部路径** —— 至少在我探针命中的
> 路径上 autotune 只有亚毫秒级。停顿更可能只在特定 kernel（如 splitk 的 REPLAY 基准）
> 上出现，需要单独复测才能定性。

## 4. 已定位的具体缺陷

**缺陷 A：`mm_splitk` 的 tuned config 带 `pipeline: "null"`。**

`tune_configs.yaml`（`_metax/`）里：

| 算子 | pipeline 取值 |
|---|---|
| `mm` | 8×basic, 1×null |
| `mm_nt` | 7×basic, 1×cpasync |
| `mm_nn` | 6×basic, 2×cpasync |
| **`mm_splitk`** | **8×null（全部）** |
| **`mm_splitk_two_step`** | 2×basic, 1×cpasync, **8×null** |

而 Triton 的 MetaX 后端只支持 `basic` / `cpasync` / `cpasync-mixed` / `mixed`：

```startLine:275:304:/opt/conda/envs/mx/lib/python3.12/site-packages/triton/backends/metax/compiler.py
            if use_opt_maca_mma:
                if opt.pipeline == "basic":
                    ...
                elif (opt.pipeline == "cpasync" or opt.pipeline == "cpasync-mixed") and not mla:
                    ...
                elif mla and opt.num_stages == 2 and opt.pipeline == "cpasync":
                    ...
                elif mla and opt.num_stages == 2 and opt.pipeline == "mixed":
                    ...
                else:
                    print("no avalilable pipeline for maca")
```

`null` 落进 `else` → **所有软件流水线 pass 被跳过**，kernel 无 prefetch/async。
实测确实打印了该警告（v1 8 次、v2 1 次）。

**它会触发**：`splitk_mm_scenario` 在 M 小（通用 kernel tile 数不足）时命中。
实测（SM=104, L2=8MB, 预算 78）：

```
o_proj  M2:SPLITK(prog=64)  M8:SPLITK(prog=64)  M32:general(prog=128) ...
down    M2:SPLITK(prog=64)  M8:SPLITK(prog=64)  M32:general(prog=128) ...
qkv/gate_up  全程 general
```

**缺陷 B：小 M（M=8–32）的 `nt` 路径仍然慢 2.8–6.7x。**
这些 shape 走 `mm_kernel_nt`（pipeline `basic`，**有**流水线），却仍慢得多。
`_prune_mm_dense_configs` 在 `M < 1024` 分支里 `continue` 掉 `block_m == 128 or warps == 8`
的配置；若可用 config 的 BLOCK_M 普遍偏大，M=8 时一个 tile 只用了 8/16 行 → 严重浪费。
（此项待进一步确认：需要打印实际选中的 config。）

## 5. 结论与建议

**结论：修 kernel/配置，别修 autotune。**

按收益/工量排序：

1. **缺陷 A —— 给 `mm_splitk` / `mm_splitk_two_step` 的 MetaX config 填上真实
   `pipeline`（`basic` 或 `cpasync`），或让编译器把 `null` 落到一个安全默认值。**
   改动小、可验证（看警告是否消失 + M=2–8 的 o_proj/down 倍数），属于**真正的算子编译优化**。
2. **缺陷 B —— 补小 `BLOCK_M` 的 `nt` config 或放宽剪枝**，收 M=8–32 那 2.8–6.7x。
3. 长期：窄 N 的 M=1 GEMV grid 太小 → 考虑按 K 切分提高占用。

**对合规的意义**：把「把 `mm` 移出白名单」换成「修好 FlagGems 的 splitk 流水线与小 M 配置」，
收益来源从**配置切换**变成**算子优化** —— 既拿得到 30% 创新分，也不再有「绕过 FlagGems」的观感问题。

## 6. 修正：把「设备时间」和「主机时间」分开之后

§2 的数字用的是 CUDA event 包住一个循环 —— 它测的是**GPU 时间线**，
主机 launch 不上来时会把它算成气泡。所以 §2 的「kernel 慢」结论是错的。

`mm_host_vs_device.py` 用 `torch.profiler` 单独取设备时间，并单独测纯 launch 开销：

**`mm`（torch.mm 派发）**

| case | 平台 | 设备倍数 gems/native | gems 主机 ms | native 主机 ms |
|---|---|---|---|---|
| qkv M1 | 1.29 | 0.0383 | 0.0081 |
| qkv M8 | **0.94（更快）** | 0.1115 | 0.0110 |
| o_proj M2 | **0.85（更快）** | 0.1185 | 0.0139 |
| down M8 | **0.90（更快）** | 0.1185 | 0.0136 |
| qkv M2048 | 1.82 | 0.1101 | 0.0104 |
| down M2048 | 2.06 | 0.1118 | 0.0095 |
| gate_up M2048 | 1.70 | 0.1121 | 0.0090 |

结论：
- **小 M（decode）：kernel 不慢，甚至更快。** 恒定 ~0.11 ms 的**主机开销**才是瓶颈
  （原生只要 ~0.01 ms，即 10x）。每步 42 层 × 4 个 mm ≈ 168 次调用 → 约 17 ms/step。
- **大 M（prefill）：kernel 本身慢 1.7–2.1x。**

## 7. `linear` 才是模型真正走的算子 —— 并且 M>1 回落到通用 kernel

模型的投影是 `F.linear(x, W)`（aten::linear），MetaX 有独立的 `linear.py`。
它**只为 M==1 做了 GEMV 快速路径**，M>1 一律
`return _generic_linear(...)`（通用 `linear_kernel`）—— 而同一个包里
**已经存在**调好的 MetaX `mm`（nt/nn/splitk）。

同一个 (2048, 6144) 形状、M=2048，设备时间：

| 路径 | 设备 ms |
|---|---|
| 原生 | 0.2758 |
| `linear` → 通用 `linear_kernel` | **1.1373**（4.12x） |
| `mm` → `mm_kernel_splitk`/`nt` | **0.5683**（2.06x） |

**同一个包内两条路差 2x。**

### 7.1 修复：M>1 也走 `mm`

`_metax/ops/linear.py` 里，在回落到 `_generic_linear` 之前，把无 bias 的
2-D case 交给 `mm(input, weight.t())`。实测设备时间：

| case | 改前 | 改后 | 倍数变化 | 对原生 |
|---|---|---|---|---|
| qkv M8 | 0.0291 | **0.0150** | 1.94x ↓ | 0.89x（更快） |
| o_proj M8 | 0.0286 | **0.0149** | 1.92x ↓ | 0.93x |
| down M8 | 0.0745 | **0.0316** | 2.36x ↓ | 0.92x |
| qkv M2048 | 0.3108 | **0.2384** | 1.30x ↓ | 1.82x |
| o_proj M2048 | 0.2968 | **0.1792** | 1.66x ↓ | 1.80x |
| down M2048 | 1.1373 | **0.5683** | 2.00x ↓ | 2.06x |
| gate_up M2048 | 1.2698 | **0.8853** | 1.43x ↓ | 1.70x |
| lm_head M8 | 0.4468 | **0.3533** | 1.26x ↓ | 0.55x（更快） |

数值：max|diff| ≤ 2.0（bf16 噪声级，与原生一致）。

**副作用**：每次调用的主机开销从 ~0.05 ms 升到 ~0.15 ms（多一层 Python 派发，
且 `mm` 入口自身有 ~0.1 ms 开销）。**是否可接受取决于真实服务路径有没有被
CUDA graph 捕获** —— 若 decode/prefill 都被 graph 掉，主机开销只在 capture 时付一次，
那么设备时间的收益是净赚；否则 168 次/step × 0.15 ms 会反噬。**必须做端到端 A/B 才能定。**

## 8. 当前状态与下一步

已做：
- `_metax/tune_configs.yaml`：17 处 `pipeline: "null"` → `"basic"`。
  **警告消失，性能无变化（负结果）**，但 `null` 本就不是合法 pipeline 值，保留为清理项。
- `_metax/ops/linear.py`：M>1 走 `mm`。**设备时间大幅改善，待端到端验证。**

待办（按价值）：
1. **端到端 A/B**：linear→mm 补丁在真实服务下是否净赚（主机开销 vs 设备时间）。
2. **压掉每次调用 ~0.1 ms 的主机开销**（`mm` 入口的 168 次/step 派发）。
3. **prefill M=2048 的 kernel 质量**（仍比原生慢 1.7–2.1x）—— 这是剩下的硬骨头。
4. autotune：**确认不是吞吐瓶颈**（冷启动 ~0.4 ms），不要在这上面花时间。

## 9. 端到端判决：`linear -> mm` 是**灾难性负优化**，并锁定 20s 停顿的真因

§6 结论「autotune 不是瓶颈」**也是错的**。我把 `linear` 加进白名单跑了真实服务，
结果是 100x 级的崩塌。

### 9.1 实测（4k，concurrency 64，256 prompts，官方口径单轮）

| | ARM A 基线（linear 走原生） | ARM B（linear -> mm） |
|---|---|---|
| prompt 吞吐 | 6553 – 19248 tok/s | **409 – 819** |
| generation 吞吐 | 819 – 2399 tok/s | **7.3 – 22.5** |
| Running / Waiting | 64 / **0** | 48 / **16 – 26**（持续积压） |
| 单轮 Total tok/s | **8876.55** | 永远跑不完（≈8 s/token） |

ARM B 在预热阶段就跑了 5:40 仍未结束（ARM A 全程 3:30），已手动终止。

### 9.2 直接证据：autotune DB 在 ARM B 期间增长

| 算子族 | ARM B 之前 | ARM B 之后 | 变化 |
|---|---|---|---|
| `mm_kernel_nt` | 6 表 / 4686 行 | 6 表 / **5154 行** | **+468 行** |
| `mm_kernel_splitk` | 4 表 / 252 行 | **5 表** / 288 行 | **+1 表 / +36 行** |
| `linear_kernel` / `mm_kernel_nn` | — | 不变 | 0 |

DB 文件在整个 ARM B 期间被持续写入（mtime 跟随 wall clock）。

### 9.3 机制：autotune key 含 M

| 算子 | autotune key |
|---|---|
| `mm_kernel_nt`（MetaX） | `["M", "N", "K", "stride_am", "stride_bk"]` |
| `linear_kernel`（通用） | `["M", "N", "K"]` |
| `rms_norm`（白名单内） | `["N"]` ← **不含 M** |

`M` 是**运行期 batch 维度**，在 serving 里每步都在变（请求异步结束，实测出现
M=2,3,4,…,25,256 共 26 个不同值）。每个新 M 都是新的 autotune key，于是
**每一步都在推理热路径上跑一遍完整 autotune**（`BenchmarkMode.REPLAY`，
即 `do_bench_cudagraph`）。这解释了：

- **为什么之前默认配置（`prefer: flagos`，全放 FlagGems）会出 ~20 s 停顿**；
- **为什么白名单（排除 `mm`/`linear`）能完全消除停顿** —— 它留下的 3 个算子
  key 不含 M，只会 autotune 一次；
- **为什么我 §6 的单次冷调用探针（0.4 ms）严重低估** —— 它只测了**一个新 M**
  的代价，而真实服务里新 M 每步都出现。

### 9.4 结论与正确修法

- `linear -> mm` 补丁**不能以当前形式合入**。它单测下来的设备收益是真的
  （§7.1），但被 autotune 风暴彻底淹没。
- 正确修法**不在白名单，而在 autotune 策略**：
  **不要把 `mm` / `linear` autotune 的 key 绑在运行期 M 上。**
  M 是 batch 维度而非编译期 tile 参数，应当按 M **分桶**（或直接阈值启发式选
  config），使 autotune 次数有界。这样：
  1. 20 s 停顿从根上消失；
  2. `mm` / `linear` 可以**正常使用**（进白名单或直接默认），无需靠排除来绕过；
  3. §7.1 的 1.3–2.4x 设备收益才可能真正兑现。

  这属于「算子编译优化」，正是赛事 30% 创新分想要的维度，且完全合规 ——
  因为它是**修 FlagGems 的缺陷**，不是「在没有算子优化的情况下切换算子」。

### 9.5 环境状态
- site-packages 的 `linear.py` 已**恢复为原始版本**（ARM B 留下的补丁态已清除）。
- `flagos-2026-s2/metax-linear-mm-route` 分支保留补丁与全部证据，但**不应合入**。

## 10. 根因确认：`align32` 分桶策略在 MetaX `mm` 上没生效

§9 把矛头指向「autotune key 含运行期 M」。这里把它落到具体一行。

### 10.1 ARM C 判决：`linear` 是安全的

只把 `linear` 加进白名单、`linear.py` 保持原始版（M>1 走通用 `linear_kernel`）：

| | ARM A 基线 | ARM B（→mm 补丁） | ARM C（→通用 kernel） |
|---|---|---|---|
| Total tok/s | 8876.55 | 跑不完 | **8825.67**（-0.57%，在 ±1% 噪声内） |
| generation 吞吐 | 819–2399 | **7–22** | 563–2393 |
| Waiting | 0 | 16–26 | 0 |
| autotune DB | 无变化 | **+468 行 / +1 表** | **零变化** |

ARM C 的 trace 有 65 个去重形状（全是 lm_head，N=130560，M=1..256），
即 `linear` 确实被调用了几十次不同的 M，**却完全没有风暴**。
所以风暴不是「`linear` 进白名单」造成的，而是 `mm` 路径特有的。

### 10.2 唯一的差异是 `strategy`

| kernel | 解析出的 strategy |
|---|---|
| `linear_kernel`（通用 `ops/linear.py`） | **`align32_strategy` ×3** |
| `mm_kernel` / `mm_kernel_nt` / `mm_kernel_nn` / `mm_kernel_splitk`（MetaX） | `default_strategy` ×N |

`align32_strategy` 把 key 按 32 对齐，所以 M=1..32 归一化到同一个 key、
33..64 归一化到下一个 —— **autotune 次数被分桶限住**。
`default_strategy` 是恒等函数，M 原样进 key，于是每个新 M 都是一次冷启动。

框架其实**知道** `mm` 该分桶，`runtime/common.py`:

```python
DEFAULT_STRATEGIES = {
    ...
    "mm": ["align32", "align32", "align32", "align32", "align32"],
    "mm_nt": ["align32", "align32", "align32"],
    "mm_splitk": ["align32", "align32", "align32", "align32", "align32"],
    ...
}
```

但这张表**只在 `TuningMode.EXPANDED` 下被消费**（`configs_loader.py:493-494`
构造 expand config 时才读 `OP_KEY_ORDERS` / `DEFAULT_STRATEGIES`）。
MetaX 的 mm decorator 既没有显式传 `strategy=`，运行时又是
`_flagtune_mode = TuningMode.DEFAULT`（实测），于是回落到
`_flagtune_default_strategy = "default"`（恒等）。
通用 `linear_kernel` 之所以免疫，只是因为它**显式写了**
`strategy=["align32", "align32", "align32"]`。

### 10.3 修法（一行/decorator）

给 MetaX 的 `mm` / `mm_nt` / `mm_nn` / `mm_splitk` decorator 补上

```python
strategy=["align32", ...]   # 长度与 key 相同
```

与 `DEFAULT_STRATEGIES` 已声明的值、以及 `linear_kernel` 的既有写法一致。
效果：
1. autotune 次数有界，热路径风暴与 ~20 s 停顿从根上消失；
2. `mm` / `linear` 可以正常使用，不必靠白名单排除；
3. §7.1 那 1.3–2.4x 的设备收益才有机会兑现。

这是**修 FlagGems 自身的缺陷**（策略声明的值与实际生效值不一致），
属于「算子编译优化」，对应赛事 30% 创新分，且不存在「绕过 FlagGems」的合规问题。
