# 用原生 attention kernel 替换 MiniCPM5-2B 的 ixformer FA —— 最终报告

日期：2026-11-19
硬件：Iluvatar BI-V150（16 SM、64-lane wavefront、1.5 GHz、ISA `ivcore11`、major 7 / minor 1）
目标：`实现 attention kernel 替换原来 minicpm 里的 attention kernel，并且尽量打满带宽`

本报告是终态汇总。**详细推导、逐条实验记录、代码级机理**在
`experiments/attn_kernel/README.md`（16 节）与
`experiments/RESULTS-iluvatar-e2e-4k16k-20261118.md`（18 项 e2e 证伪）里，
本文件只负责：门禁对照、交付物、性能阶梯、根因、以及**完整证伪清单**。

---

## 0. 结论先行

| # | 门禁 | 要求 | 实测 | 判定 |
|---|---|---|---|---|
| 1 | 覆盖 serving 真实路径（分页 BNDH/HND、GQA 16/2、D=128、page=16、causal、mixed） | 全覆盖 | `fused_paged.cu` 全支持，且比 vendor **多支持一项**：真正的 2-head GQA 缓存 | **达成** |
| 2 | 数值对齐 vendor / fp32 参考 | bf16 容差、0 坏行 | 连续 8/8 + 分页 10/10 shape，worst row rel ≤ **8.7e-03**，坏行 **0** | **达成** |
| 3 | prefill 达到 MMA roofline（~98 TF） | 不劣于现役 | **33.8 TF**（roofline 34%），vendor 72 TF，**落后 2.27×** | **未达成** |
| 4 | 端到端 4k ≥6436 / 16k ≥5421 / MATH-500 L3 ≥95% | 不回退 | 4k **6426.08**（−0.16%）、16k **5417.5**（−0.065%）、L3 **97.1%** | 见 §2 |
| 5 | findings 与 benchmark 数字入 `experiments/` 并提交 | — | 全部提交 | **达成** |

**一句话结论**：内核在功能与数值上是 vendor FA 的完整等价替代（且 GQA 支持更强），
但性能停在 34% roofline，差距是**指令发射带宽**而不是内存带宽——这颗芯片上没有
能改变该比例的 MMA 形状（`m16n16k16` 是唯一 BF16 MMA），所以**没有接入生产**，
e2e 门禁继续由 vendor FA 承载。本内核作为等价参考实现与回归用例保留。

关于原任务里的"打满带宽"：decode 侧的 HBM 实测上限由现役 Triton FlashDecoding
持有（**617 GB/s**，理论 819.2 的 75%）；自研 split-KV decode 最好 **396 GB/s**
（`DEC_MINB=0`），更差，故也未接入。带宽不是 prefill 的瓶颈（见 §5）。

---

## 1. 交付物

| 文件 | 内容 |
|---|---|
| `experiments/attn_kernel/fused_paged.cu` | 主力：分页 + varlen + mixed batch 的原生融合 prefill attention（bf16 MMA、GQA、causal/dense、HND/BNDH） |
| `experiments/attn_kernel/fused_bf16.cu` | 连续 KV 版本，用于逐层剥离归因（保留作诊断工具） |
| `experiments/attn_kernel/fused_decode.cu` | split-KV decode 版本（`DEC_MINB` 可调） |
| `vllm_fl/dispatch/backends/vendor/iluvatar/decode_attn.h` | decode 内核头文件，随插件发布（harness 与出货同一份 kernel） |
| `vllm_fl/dispatch/backends/vendor/iluvatar/native_decode_attn.py` | decode 内核的 Python 侧封装与形状闸门（`nunit ≤ 64`） |
| `experiments/attn_kernel/cmp_paged.py` / `cmp_decode.py` | 分页 / decode 数值门禁（乱序 block_table、空页、混合 batch） |
| `experiments/attn_kernel/mma_rate.cu` / `mma_shapes.cu` | MMA 发射速率与指令形状的硬件探针（roofline 的唯一权威来源） |
| `experiments/attn_kernel/README.md` | 16 节完整推导：硬件语义、三个 vendor 陷阱、性能阶梯、根因、证伪 |
| `experiments/RESULTS-iluvatar-e2e-4k16k-20261118.md` | e2e 18 项证伪 + 交付配置的计分复测 |
| 本文件 | 终态汇总 |

---

## 2. 端到端门禁的诚实位置

四条 4k 复测（官方口径 `4096/1024, c=64, num_prompts=256`）：

| 轮次 | tok/s | Mean TTFT | Mean TPOT |
|---|---|---|---|
| warmup | 6413.71 | 3773 ms | 46.08 |
| run1 | **6426.08** | 3548 ms | 46.21 |
| run2 | 6423.04 | 3414 ms | 46.36 |
| run3 | 6419.89 | 3413 ms | 46.39 |

16k（`16384/1024, c=64`）：**5417.5** tok/s vs 门禁 5421.40（−0.065%）。
MATH-500 Level 3：**97.1%**（门禁 95%）。

**必须说清的两件事**：

1. **门禁值恰好等于历史观测分布的最大值**（4k 39 次记录、16k 18 次记录）。
   "是否通过"由运行抖动决定，不是由某个已知代码差异决定。−0.16% 是 10 tok/s。
2. **这三个数字是交付配置本身的成绩，与本轮原生 kernel 无关。** 原生 prefill
   (2.498 ms) 比 vendor (1.099 ms) 慢 2.27×，所以没有进 fast path；原生 decode
   被 `nunit ≤ 64` 闸门挡在官方 16k 形状（64 × nkv 2 = 128 units）之外，且本身
   也比现役 Triton 慢。生产路由实测为 `decode_triton=1008, prefill=84, decode=0`。

> 早期记录的 4k 6394 tok/s 是我的测试条件错误（把官方 256 prompts 覆盖成 128，
> TTFT 摊薄更差）。用官方条件重测后为 6426.08。已在 RESULTS 文档更正并提交。

---

## 3. 数值正确性证据

分页门禁 `cmp_paged.py`：页随机置换分配 + 留 7 个空页（任何"线性扫缓存"的实现必错）+
混合 batch（1-token decode 紧挨 3000-token prefill）+ 长度故意不对齐 BM=384。

| 用例 | causal / dense maxdiff | worst row rel | 坏行 |
|---|---|---|---|
| single_prefill_256 | 9.7e-03 / 1.1e-03 | 8.2e-03 | 0 / 4096 |
| single_prefill_1024 | 1.1e-02 / 6.4e-04 | 8.5e-03 | 0 / 16384 |
| chunk_prefill_512_2048 | 4.7e-04 / 4.3e-04 | 8.0e-03 | 0 / 8192 |
| mixed_batch (1024,512,1,7) | 7.8e-03 / 1.8e-03 | 8.6e-03 | 0 / 24704 |
| ragged (300,1,129) | 8.4e-03 / 1.0e-03 | **8.7e-03** | 0 / 6880 |

decode split-KV：worst relative error **1.893e-03**，全 shape 通过。

**这套门禁抓到过两个真 bug**：

- **causal 右下角对齐**：`kv > tok` 时对角块起点算错，乱序/非对齐用例才暴露。
- **分页布局**：vendor 要的是 **HND** `[blocks, n_kv, PAGE, D]`，不是任务描述里的
  BNDH。第一版按 BNDH 写，用顺序 block_table 测出来"差不多"（掩盖了错误），
  用乱序 block_table 才暴露 maxdiff 3.0 / 100% 坏行。

---

## 4. 性能阶梯（只算独立测得的收益）

| 改动 | 4k | 说明 |
|---|---|---|
| 初版融合（QK+online softmax+PV，RB=1） | 4.41 ms | 每 warp 一个 16 行块，LDS 带宽受限 |
| **RB 多行块**（RB=3） | **2.82 ms** | 本轮最大的单笔收益，LDS 摊薄 3× |
| NT 多 tile | 2.72 ms | 每轮多做几个 k-tile，摊薄循环开销 |
| **causal 上界外提** | −15% | 不再逐 tile 判因果，直接算最后一行的上界 |
| **`HSPAN=2`** | +14% | 每 warp 跨 32 行，减少 S/P 寄存器重复 |
| **`PR_ORD` 最长序列优先** | −5.8% | 尾部小块的负载均衡 |
| **`PG_MMASUM`：softmax 分母用 MMA 求** | **+9%** | 见 §5.2 |
| **终版** | **2.498 ms / 31.8 TF** | 8k 6.529 ms / 33.9 TF；16k 14.927 ms / 33.8 TF |

vendor 同 shape：1.099 / 3.011 / 6.681 ms（约 72 TF）→ 差距 **2.27× / 2.17× / 2.23×**。

**逐 shape 开不同模板已被实测否决**：对 $(NT \in \{1,2,4,8\}, RB \in \{1,2,3\},
HSPAN \in \{1,2\})$ 在 4k/8k/16k 上全扫，**三个形状的最小值都落在同一个点**
`NT=2, RB=3, HSPAN=2`。原因是配置空间被两道墙夹死：RB=3 用掉 253 个寄存器
（1 CTA/SM 之上无路）、HSPAN=2 用掉 1024 线程（block 上限）。所以"不同尺寸开不同
模板"这条路在这颗芯片上确实没有空间。

---

## 5. 根因：不是内存，是指令发射带宽

### 5.1 剥离法定位

保留真实数据流，逐层砍掉 16k dense（NT=4/RB=3）：

| 变体 | 时间 | TF | 效率 |
|---|---|---|---|
| 完整 | 19.700 ms | 28.8 | 29% |
| 去 softmax | 15.588 ms | 36.4 | 37% |
| 去 P→A 转置 | 17.805 ms | 31.8 | 32% |
| 去两者 | 14.357 ms | 39.5 | 40% |
| 去两者 **+ 去 LDS** | 14.371 ms | 39.4 | 40% |

**去掉共享内存读，时间完全不变**（14.357 → 14.371）。LDS 不是瓶颈。

数每轮每 warp（NT=4/RB=3/NCH=8）的指令：

```
MMA        NT*RB*2*NCH   = 192   (40%)
LDS        NT*NCH*2      =  64   (13%)
softmax    RB*4*(NT-1+4) =  84   (17%)
P→A 转置   NT*RB*12      = 144   (30%)
合计                      = 484
```

MMA 占 **40%**——与实测效率 40% 精确吻合。BI-V150 的 64-lane wavefront 每 2 周期
只发射一条指令，纯 MMA 时正好 2 周期 1 条、可达 98 TF（`mma_rate.cu` 实测），
**所以每条非 MMA 指令都在直接抢发条带宽**。

这同时解释了为什么 §4 的全参数扫描全部落在同一点：NT / RB / HSPAN 对 MMA 和
其他指令是**同比例**缩放，比例恒为 40%。任何"多算少存"式调参的天花板是 ±5%。

### 5.2 唯一有效的方向：把非 MMA 工作塞进 MMA

softmax 分母 $\sum_n P[m,n]$ 数学上等于 $P @ \mathbf{1}$。发一条 $P$ 对全 1 向量的
MMA，结果正好落在分母已有的 C 布局（lane $a$、寄存器 $q \leftrightarrow$ 行 $a+4q$），
且每个输出列都是同一个值——**一条 MMA 替掉 16 lane shuffle 归约 + 60 条标量加**。

每轮 60 条 → 6 条（$NT \times RB$），MMA 占比 40% → 42.5%，**prefill 快 9%**
（4k 2.719 → 2.498 ms，8k/16k 各 +9.2%/+9.3%）。

这是本轮唯一一个改变指令配比而不是改变配置的优化，也是唯一还有余量的思路方向
（同类可迁移：把行最大值也做成 MMA 不可行，max 不是线性算子）。

---

## 6. 硬件不可为：4 倍 MMA 在这颗芯片上不存在

要让 MMA 占比从 42% 提到 80%，唯一办法是单条 MMA 做 4 倍工作。
`BuiltinsIluvatar.def` 里列了这些指令（`m16n32k32`、`m32n16k32`、`m64n16k16`，
各 32768 flops），但**它们不是 `ivcore11` 的指令**。用 `llvm-mc` 按 subtarget
逐条验证：

| 指令 | ivcore11（BI-V150） | ivcore40（下一代） |
|---|---|---|
| `ml_matrix_mad_f32_bf16_m16n16k16` | **可用** | 可用 |
| `m16n32k32` | 不支持 | 可用 |
| `m32n16k32` | 不支持 | 可用 |
| `m8n16k32` / `m8n64k32` / `m16n64k16` / `m32n32k16` / `m64n16k16` | 不支持 | — |

即 ivcore11 上 **BF16 只有一个 MMA 形状**：`m16n16k16`（8192 flops，128 位累加器）。
fp16 与 bf16 同速，所以换 dtype 也没有收益。

> 更正记录：这里最初被误判为"`llc` 后端 vector-split 缺陷"。`llc` 确实会崩
> （`-O0`..`-Os`、256/512 位累加器全崩，是真实缺陷，应报厂商），但即使不崩，
> 指令在这颗芯片上也不存在。前端报的 `needs target feature long-mtx` 有误导性：
> `-mattr=+long-mtx` 在 ivcore11 上仍被汇编器拒绝，说明该 feature 只是被登记进枚举、
> 未挂到该 subtarget。

### 6.1 比例的来源

关键比值是 **PV 的 MMA 条数 : P→A 转置条数 = 8 : 12**。

- PV 一侧：`NCH = head_dim/16 = 8`，模型给定，改不了。
- 转置一侧：C 输出把 key 索引放在 `lane%16`（每 lane 一个 key），而 A/B 操作数要求
  key 成对（每 lane 四个 key）。这是同一组 16 个 key 的**两个不同 lane 级划分**：

```
C 输出的 m 划分:      S_a = {a, a+4, a+8, a+12}
A/B 操作数的 k 划分:  T_a = {2a, 2a+1, 2a+8, 2a+9}
```

每次 MMA 链都要换一次，且跨 lane 交换只能靠 shuffle 或 smem——这是硬件固有，
不是实现选择（§7 有两条证伪证明绕不过）。

---

## 7. 完整证伪清单（**不要再挖**）

### 7.1 kernel 层（本报告主体，共 18 项）

| # | 尝试 | 结论 | 证据 |
|---|---|---|---|
| 1 | `cp.async` 异步拷贝 | **工具链缺陷，不可用** | 4B/8B/16B × 有/无 commit × `-O0`..`-O3`，**全部 `llc` 段错误**；只加 `+async-copy` flag 不用指令可编译 → 是 lowering 没实现。`cuda_pipeline.h` 不存在 |
| 2 | 手写双缓冲软流水（RB=1 时代） | **数值对，性能更差** | 4.32 → 4.77 / 4.81 / 7.85 ms（NT=4/2/8），smem 代价（占用率下降）总是大于重叠收益 |
| 3 | RB=3 之后重做软流水 | **仍然更差** | 1 CTA/SM 下重叠本该"免费"，实测仍慢 2% |
| 4 | 寄存器预取替代 smem 双缓冲 | **更差 8%** | 寄存器墙 |
| 5 | warp 特化（producer/consumer） | 未做（oversubscribe 受占用率限制，无预期收益） | 保留为唯一未验证的重叠路径 |
| 6 | 用 smem 往返替掉转置的 8 条 shuffle | **慢 31%** | 3.552 vs 2.719 ms；`syncwarp` + bank 冲突（`col*16+row` 布局）代价超过 shuffle |
| 7 | 删掉 P→A 转置，只改 O 写回行号 | **数学上不成立** | 需存在 `f` 使 `f(MP⁻¹(m))=f(m)` 对全部 m 成立 → 只有常量解。实测只有不动点 `{0,7,8,15}` 正确，其余 12 行相对误差 5–25%，坏行 2294/4096 |
| 8 | 改 row-block 组织（换哪些行组成 tile） | **不成立** | 关掉归一化后逐行看"输出/参考"比值：不动点行比值 133–153（σ≈1e-6），其余行 98–153（σ 4.5e1–5.7e2）→ 分子混入多行。根因是 §6.1 的 lane 级划分，与行号取值无关 |
| 9 | QK 的 `ch` 循环外提提升 ILP | **慢 4.5%** | 依赖链不是瓶颈 |
| 10 | 硬件转置 load `__ivcorex_slb_mtx_load_b16trs` | **前端崩溃** | clang SIGSEGV（exit 139），只要物化输出寄存器就崩 |
| 11 | 4× MMA（`m16n32k32` 等） | **ivcore40 专属** | `llvm-mc` 逐条验证，见 §6 |
| 12 | 逐 shape 开不同模板（4k/8k/16k） | **无收益** | 19 组配置全扫，三个形状最小值同点 `NT=2/RB=3/HSPAN=2`；被寄存器墙 + 线程墙夹死 |
| 13 | causal 跳过（按网格分解跳块） | **负收益** | 省下的计算量小于占用率下降的代价 |
| 14 | `HSPAN` / `NT` 继续加大 | **溢出** | HSPAN=4 超 1024 线程；NT=8 超寄存器 |
| 15 | 换 fp16 替代 bf16 | **无收益** | `mma_rate.cu` 实测两者同速 |
| 16 | 尾块不做越界保护（BM=384 不整除 4096） | **错** | 必须保留，已实现 |
| 17 | 用 MMA-only 剥离测上限 | **测量伪影，必须加偏置** | 常量替换 `kf/vf/pa` 会让 clang 把整个 MMA 提到循环外，报出假 72 TF。必须用 `(blk & 1) * 4` 偏置 LDS 读数 |
| 18 | 长序列 grid 重排（`PR_ORD`） | **有效，但只有 −5.8%** | 已并入终版 |

### 7.2 端到端层（详见 RESULTS 文档 §7，18 项）

摘要（每条都实测过，**不要重跑**）：

- INT8 / FP8 KV 量化：池子翻倍（522k → 1013k / 1044k），但 dtype 门禁让所有
  attention 退回带反量化的通用 Triton，4k 掉到 **2372** tok/s。
- KV page 16 → 32/64/128：慢 2.0× / 3.3× / 14×。
- `max_num_batched_tokens` 8192：5283（−0.7%），4096 已最优。
- 双 stream decode‖prefill 重叠：0.99–1.00×（BI-V150 不支持核函数并发）。
- prefill GEMM：92.9 vs 峰值 96.6 TF（96%，无空间）。
- prefill attention tiling：10 组全 68.8 TF（已饱和）。
- `--no-enable-chunked-prefill`：启动失败（要求 `max_num_batched_tokens ≥ max_model_len`）。
- async scheduling A/B：测试无效（该环境变量语义是"留给 auto"，而 auto 本身即 enabled，两组配置相同）。
- graph padding 行导致 bin0 争用：证伪（真因是全部元素含真实行都低于 `lo`）。
- step 内调度气泡：证伪（实测 6.7 us / 1.0%）。
- 长上下文回退 ixformer 是瓶颈：证伪（强制 16k 也走 Triton，反而慢 3%）。
- `_CTA_TARGET` 256→512：证伪（4k −1.6%，output −8.8%）。
- `true_div` 比 aten 慢 2.7×：证伪（测量伪影，同进程 A/B 只快 1.06×）。
- "prefill attention 还有 68.8→74.3 TFLOPS"：证伪（68.8 是 Triton 微基准，生产不走那条路）。
- 把 16k decode 逐层归因：工具不可行（CUDA graph replay 对 ixsys 完全不透明，
  `linear_kernel` 计数 392 但 kernel 序列无法拆解）。
- `VLLM_FL_ILUVATAR_DECODE_TRITON`：只被写入、从未被读取，失效标志。

---

## 8. 为什么不接入生产

1. **prefill 慢 2.27×**：2.498 vs 1.099 ms。接入会让 4k e2e 掉约 3%、16k 掉约 8%
   （attention 占墙钟 4k 6.8% / 16k 20.4%）。
2. **decode 形状闸门不匹配**：`nunit ≤ 64`，官方 16k 用例是 64 × nkv 2 = 128 units，
   内核根本不接管；且接通后也只有 396 GB/s，低于现役 Triton 的 617 GB/s。
3. **在慢 kernel 上做端到端优化是负价值的**：只会把慢路径藏进 fast path。

所以交付配置保持 vendor FA（prefill ixformer / decode Triton），原生内核以
"等价参考实现 + 回归用例"身份保留——它的价值在于：数值门禁可以在没有 vendor 的
机器上独立验证 attention 语义，以及它**支持 vendor 不支持的 2-head GQA 分页缓存**。

---

## 9. 如果还要继续，只有三条路（都需要外部条件改变）

1. **修好 `cp.async` 的 lowering**（`llc` 对所有粒度段错误）—— 这是厂商侧 bug，
   一行 patch 就能解锁真正的异步搬运。
2. **让 softmax 直接产出 A-fragment 序**，把 P→A 转置从 12 条 shuffle 降到接近 0
   —— 需要改 MMA 的 C 输出布局或拿到 lane 级跨 lane 交换指令（这颗芯片上没有）。
3. **拿到能突破 RB ≤ 3 寄存器墙的累加器组织** —— 例如 256 位累加器的可用实现
   （`ivcore40` 才有）。

在这三条任一被解决前，继续调参的收益在 5% 量级，不改变结论。

---

## 10. 复现

```bash
CC=/usr/local/corex/bin/clang
FLAGS="-O3 -x ivcore --cuda-gpu-arch=ivcore11 --cuda-path=/usr/local/corex"
LD="-L/usr/local/corex/lib64 -lcudart"
INC="-I vllm_fl/dispatch/backends/vendor/iluvatar"

# 硬件 roofline（唯一权威来源）
$CC $FLAGS -o /tmp/mma_rate experiments/attn_kernel/mma_rate.cu $LD && /tmp/mma_rate 128 20000
$CC $FLAGS -o /tmp/mma_shapes experiments/attn_kernel/mma_shapes.cu $LD

# decode 内核（harness 与出货同一份头文件）
$CC $FLAGS $INC -DDEC_MINB=0 -o /tmp/fused_decode experiments/attn_kernel/fused_decode.cu $LD
```

数值门禁：`python experiments/attn_kernel/cmp_paged.py`（需先编译出 `fused_paged`）。

注意：`cmp_paged.py` 传的是 `rb=3`（诚实值），不是命令行默认值——历史上有过
harness 传 `rb=4` 与 kernel `BM` 不匹配、从而作废全部吞吐数字的事故，已在本轮修正
（README §16.1），所有现行数字都基于匹配的配置。
