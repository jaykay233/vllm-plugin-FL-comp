# Decode 阶段 `mm` 与 `zero_` 开销归因（MetaX / FlagGems v5.3.5）

环境：`conda activate mx`，MiniCPM5-2B (LlamaForCausalLM, 42 层, hidden=2048)，
`VLLM_COMPILE` + CUDA Graph，batch=1 decode，TPOT ≈ 6766 µs/token。

## 1. 单步 kernel 预算（ir_split，即当前默认）

| kernel | µs/step | 次数/step | 占比 |
|---|---|---|---|
| `mm_kernel_nt` | 2257.3 | 84.66 | 33.0% |
| `mm_kernel_splitk` | 1692.5 | 84.66 | 24.7% |
| `flash_fwd_splitkv_combine` | 402.5 | 42.33 | 5.9% |
| `rms_norm_kernel` | 375.4 | 85.67 | 5.5% |
| `flash_fwd_splitkv_kernel` | 337.3 | 42.33 | 4.9% |
| `_gemv_kernel` (lm_head) | 332.7 | 1.01 | 4.9% |
| `add_func_kernel_rank_1` (残差加) | 231.7 | 86.68 | 3.4% |
| `zero_persistent_kernel` | 223.2 | 84.66 | 3.3% |
| `triton_poi_fused_cat_1/2` (RoPE) | 515.2 | 128.0 | 7.5% |
| `reshape_and_cache_flash` | 182.3 | 42.33 | 2.7% |

GEMM 合计 3949 µs = **58%**，是 decode 的绝对大头。

## 2. `zero_persistent_kernel` 的真正来源（此前误判为 inductor 胶水）

用 `TORCH_LOGS=graph_code` dump dynamo 图 → **图里没有任何 zero 节点**。
用 `TorchDispatchMode` / chrome trace 重建 Python 调用链，得到：

```
[inductor 生成代码] torch.mm
  → flag_gems/runtime/backend/_metax/ops/mm.py:1197 mm_out
    → Tensor.zero_()                    ← 每层 2 次，共 84 次/step
```

`mm_out` / `mm` 的 split-k 分支：

```python
    if splitk_mm_scenario(M, N, K):
        c.zero_()
        return splitk_mm(a, b, c, M, N, K)
```

`c.zero_()` 是**语义必需**的：`mm_kernel_splitk` 主循环用 `tl.atomic_add` 累加
（`mm.py:719`），所以累加器必须先清零。

- 触发条件（`splitk_mm_scenario`，`mm.py:1065`）是**并行度启发式**：
  `max_general_programs <= sm_count*3//4 (104*3//4=78)` 且 `M*N*4 <= L2/32`。
- 本模型 M=1：`qkv(N=2560)` / `gate_up(N=12288)` → 不走 split-k（`mm_kernel_nt`）；
  `o_proj(N=2048)` / `down(N=6144)` → 走 split-k。
- 代价：84 次/step × 2.6 µs = **223 µs/step（3.3%）**，且 FlagGems 的 MetaX `zero`
  是 persistent kernel（`_metax/ops/zero.py`，256 blocks × BLOCK 16384），对 4 KB 张量
  纯属 launch 开销。

**否定的方案**：直接把 `splitk_mm_scenario` 关掉更慢（见第 4 节）。

## 3. 每层 4 个 M=1 matmul 的形状与带宽

`aten::mm`（`record_shapes=True`）：每层 4 次 + lm_head 1 次：

| 形状 | M | N | K | 权重 MB | 当前路径 | 有效带宽 |
|---|---|---|---|---|---|---|
| qkv | 1 | 2560 | 2048 | 10.5 | `mm_kernel_nt` | **699 GB/s** |
| gate_up | 1 | 12288 | 2048 | 50.3 | `mm_kernel_nt` | 1275 GB/s |
| o_proj | 1 | 2048 | 2048 | 8.4 | `mm_kernel_splitk` | **625 GB/s** |
| down | 1 | 2048 | 6144 | 25.2 | `mm_kernel_splitk` | **862 GB/s** |
| lm_head | 1 | 130560 | 2048 | 534.8 | `_gemv_kernel` | **1607 GB/s** |

单步权重流量 ≈ 4.5 GB / 147.8 tok/s ≈ 665 GB/s 平均 —— 纯 memory-bound，
但四条投影里有三条只跑到 625–862 GB/s。

## 4. A/B：禁用 split-k（❌ 负结果）

`splitk_mm_scenario → False`，其余不变：

| 形状 | as-is | no-splitk |
|---|---|---|
| qkv | 15.19 µs | 15.17 µs |
| gate_up | 39.52 | 39.56 |
| o_proj | 13.26 | 13.34 |
| down | 29.54 | **33.33（更慢）** |
| 每层合计 | **97.5 µs** | **101.4 µs (+3.9)** |

结论：split-k 在这个启发式下是合理的，**不要动它**。

## 5. A/B：把 M=1 的 `mm` 路由到已有的 MetaX GEMV kernel（✅ 正结果）

`_metax/ops/linear.py` 里已有 `_gemv_kernel`（lm_head 在用，1607 GB/s）。

| 形状 | `mm` | GEMV | 加速 | max rel err |
|---|---|---|---|---|
| qkv | 14.99 µs | 13.24 µs | 1.13x | 4.2e-04 |
| gate_up | 39.48 | 39.04 | 1.01x | 1.4e-03 |
| o_proj | 13.43 | 11.36 | 1.18x | 5.4e-03 |
| down | 29.20 | 22.98 | **1.27x** | 7.5e-03 |
| lm_head | 332.83 | 332.09 | 1.00x | 2.3e-03 |
| **每层合计** | **97.10 µs** | **86.62 µs** | **1.12x** | — |

预期收益：10.48 µs/层 × 42.33 层 = **≈444 µs/step ≈ 6.5% TPOT**。

实现位置：`flag_gems/runtime/backend/_metax/ops/mm.py` 的 `mm_out` / `mm`，
在 `M == 1` 且 `b` 是 `[N,K]` 连续权重的转置视图（`b.stride() == (1, K)`）时，
走 GEMV kernel 并写入已有 `out`。

## 6. 实现：M=1 mm → GEMV 路由（已落地，e2e 待验证）

改动文件：`/workspace/FlagGems/src/flag_gems/runtime/backend/_metax/ops/mm.py`
（+81 行，并已同步到 `/opt/conda/envs/mx/lib/python3.12/site-packages/...`）

新增内容：

- `_ENABLE_MM_GEMV`：env `FLAG_GEMS_METAX_MM_GEMV`，**默认 1（开启）**，`=0` 可 kill switch。
- `_m1_gemv(a, b, c, M, N, K)`：仅在 `M==1`、`a.stride()==(K,1)`、
  `b.stride()==(1,K)`（即 `weight.t()` 的透视图，vLLM Linear 的传法）、dtype 均为
  bf16/fp16/fp32、`c` 连续时启用；否则返回 `None` 回退原路径。
  内核直接复用 `linear.py` 的 `_gemv_kernel`（`importlib` 加载模块，因为包
  `_metax.ops.__init__` 会把 `linear` 这一名字重绑定为函数）。
- 在 `mm()` 与 `mm_out()` 中，`N == 1` 判断**之前**调用，命中即返回。

微基准 A/B（`src/bench_mm_gemv_route.py`，profiler `device_time`）：

| 形状 | off | on | 加速 | max rel err |
|---|---|---|---|---|
| qkv (N=2560) | 15.15 µs | 13.40 | 1.13x | 0 |
| gate_up (N=12288) | 39.47 | 39.09 | 1.01x | 3.0e-03 |
| o_proj (N=2048) | 13.37 | 11.42 | 1.17x | 5.5e-03 |
| down (N=2048,K=6144) | 29.44 | 23.26 | 1.27x | 6.2e-03 |
| lm_head | 333.21 | 330.69 | 1.01x | 2.5e-03 |
| **每层合计** | **97.10** | **86.62** | **1.12x** | — |

分 arm 时 `on` 臂的 kernel 完全变成 `_gemv_kernel`，`mm_kernel_splitk` 与
`vectorized_elementwise`（即 `zero_`）一并消失。

**待办**：`src/ab_mm_gemv_e2e.py` 的端到端 A/B 尚未跑完（off 臂第 1 次尝试中途失败，
第 2 次被手动停止）。预期 TPOT 改善 ≈ 10.3 µs/层 × 42.33 ≈ **440 µs/step ≈ 6.5%**。
该脚本同时会报告 `zero_persistent_kernel` 是否随路由消失，以及 token-hash 交叉校验。

备份：`/root/bench_results/mm_gemv_route/mm.gemv_route.py`
还原 stock：`cd /workspace/FlagGems && git checkout src/flag_gems/runtime/backend/_metax/ops/mm.py`
再同步到 site-packages。

## 复现脚本

- `src/probe_cfg_budget.py` — 三配置（ir_split / ir_fused / ir_off）全 kernel 预算
- `src/probe_zero_eager.py` — 关闭 CUDA Graph，统计 `zero_` 与父节点
- `src/probe_zero_trace.py` + `src/parse_zero_parents.py` — chrome trace 重建 zero_ 调用链
- `src/probe_mm_aten_shapes.py` — 每层 mm 形状
- `src/bench_mm_splitk_ab.py` — split-k A/B（负结果）
- `src/bench_gemv_vs_mm.py` — GEMV vs mm（正结果）
