# 为什么 FlagGems 的 MetaX `mm` 比原生慢 —— 实测与根因

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
