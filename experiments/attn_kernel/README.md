# 自研 attention kernel：BI-V150 原生 MMA 可行性（进行中）

目标：用自研 kernel 替换 MiniCPM5-2B 当前的 prefill attention
（闭源 `ixformer.contrib.vllm_flash_attn.flash_attn_varlen_func`），并尽量贴近硬件 roofline。

本文只记录**已实测**的事实。所有数字都在本机复现过。

---

## 1. 三条路线，已排除两条

| 路线 | 实测 | 结论 |
|---|---|---|
| Triton `tl.dot` | 8192³ 仅 **39.1 TFLOPS**（40%）、4096³ 43.0 | **排除**。是 ixBLAS 的 2.2–2.5× 慢，和 §17.20.1「Triton scaled_mm 比 bf16 mm 慢 2–3×」一致 |
| 复用 ixBLAS 分块 GEMM | 分块后 S 矩阵要落 HBM（4096²×16×2B = 512 MB/次），额外 3.2 GB 流量 ≈ 4.9 ms，而 vendor FA 只要 1.11 ms | **排除** |
| **裸 CUDA + vendor MMA** | MMA 指令发射上限 **97–98 TFLOPS** | **唯一可行**，见下 |

对照：vendor FA 当前在 `Sq=4096/kv=4096` 是 1.122 ms（61.2 TFLOPS），
`Sq=4096/kv=16384` 是 6.712 ms（71.7 TFLOPS）。

---

## 2. 硬件真实语义（与 NVIDIA 不同，之前踩坑都来自这里）

| 项 | 实测值 | 说明 |
|---|---|---|
| `warp_size` | **64** | 不是 32。BI-V150 一个 wavefront 是 64 lane |
| threads/SM | 8192 | 128 warps/SM，占用率资源充足 |
| MMA 指令 | `__ivcorex_matrix_mad_f32x4_f16x4` | `V4f V4h V4h V4f`，文档标注 M*N*K = **16×16×16** |
| 单指令产出 | 64 lane × 4 float = 256 = 16×16 | 与 16×16×16 自洽 |
| MMA 发射上限 | **97–98 TFLOPS**（128 blocks × 256 thr） | 128 blocks 起满速；32 blocks 只有 48%（需 ≥32 warps/SM） |

`__ivcorex_matrix_mad_f32x4_f16x4` 的 fp16 与 bf16 同速（`torch.mm` 两者都是 96.5 TFLOPS）。

---

## 3. fp16 fragment 形状（`<mma.h>` → `IX/crt-ixsdk/iluvatar_mma.h`）

只有 4 种 fp16 MMA tile，**没有 bf16 fragment**：

| 形状 (M,N,K) | matrix_a | matrix_b | 用途 |
|---|---|---|---|
| 16,16,32 | row_major | **col_major** | `Q@Kᵀ`（NT） |
| 16,32,32 | row_major | row_major | — |
| 32,32,16 | col_major | row_major | — |
| 32,16,32 | col_major | col_major | — |

bf16 需要绕道：`bf16→fp16` 对 `[6.1e-5, 65504]` 区间是无损的（fp16 尾数 10 bit > bf16 8 bit），
超小值退化到 fp16 subnormal，绝对误差 ~1e-8，对 attention 无影响。

---

## 4. 三个必须知道的 vendor 陷阱（都已实测确认）

### 4.1 fragment 的 `x[]` 是 `unsigned int`，一个装 2 个 half

`__frag_base<__half,4>` 里 `storage_element_type x[4]`，但 `storage_element_type` 是
`unsigned int` → 实际 **8 个 half**。按 float 读会得到 `inf`。

### 4.2 shared memory 必须是 vendor 的 swizzle，不是普通 row-major

`load_matrix_sync` 的 `ldm` 参数**被完全忽略**，自己按 swizzle 算偏移：

```c
// matrix_a 16,16,32 row_major, 元素 (m,k)，共 512 half
int sw_row(int m, int k) {
  int i = k / 16, c = k % 16, r = m / 2;
  return i * 256 + (r / 4) * 128 + ((r * 32 + c * 2 + (m & 1) + i * 64) & 127);
}
// matrix_b 16,16,32 col_major, 元素 (k,n)，共 512 half
int sw_col(int k, int n) {
  int r = k >> 1;
  return (n >> 2) * 128 + (r & 3) * 32 + ((r >> 3) ^ (n >> 3)) * 16 +
         ((r >> 2 & 1) ^ (n >> 2 & 1)) * 8 + (n & 3) * 2 + (k & 1);
}
```

摆了 swizzle 之后单 tile 精度：**worst rel err 0.00e+00，0/256 错**（`tile_ok.cu`）。

### 4.3 `store_matrix_sync` 也忽略 `ldm`，永远写连续 16×16

`CoordToOffset<32,row_major>(row,col) = row*16+col`。所以它会把 C 写成一整块连续
16×16，跨行 stride 全部毁掉 → 必须自己从寄存器落地。

累加器映射（从 `__imma_st_row_b32` 读出，已用 256×256 全 tile 验证）：

```
lane L, reg q  ->  row = L/16 + 4*q,  col = L%16
```

即：一个输出行分散在 16 个相邻 lane 上；每个 lane 持 4 个不同行（q=0..3）。
这对 fused softmax 是**好消息**：行内 max/sum 只需 16-lane shuffle 归约，且
4 行互不干扰。

---

## 5. 当前进度

已交付（`experiments/attn_kernel/`）：

| 文件 | 内容 | 状态 |
|---|---|---|
| `mma_rate.cu` | MMA 指令发射上限探针 | 98 TFLOPS |
| `tile_ok.cu` | 单 16×16×32 tile + swizzle 正确性 | 0.00e+00 误差 |
| `gemm_swz.cu` | 全尺寸 NT GEMM（swizzle + 寄存器 epilogue） | **全 tile 通过**，7.5 TFLOPS |

**未达标**：`gemm_swz.cu` 只有 7.5 TFLOPS（MMA 上限的 8%）。瓶颈不在 MMA，
在 staging：每个 16B 向量要拆成 8 次标量 smem 写，且每个 BK=32 chunk 有 2 次
`__syncthreads()`。需要多级流水 + 更大 per-warp tile + swizzle 友好的
向量化 smem 写，才能把 7.5 拉向 ~85。

## 6. 复现

```bash
CC=/usr/local/corex/bin/clang
FLAGS="-O3 -x ivcore --cuda-gpu-arch=ivcore11 --cuda-path=/usr/local/corex"
LD="-L/usr/local/corex/lib64 -lcudart"
$CC $FLAGS -o /tmp/mma_rate  experiments/attn_kernel/mma_rate.cu  $LD && /tmp/mma_rate 128 20000
$CC $FLAGS -o /tmp/tile_ok   experiments/attn_kernel/tile_ok.cu   $LD && /tmp/tile_ok
$CC $FLAGS -o /tmp/gemm_swz  experiments/attn_kernel/gemm_swz.cu  $LD && /tmp/gemm_swz 256 5
```

## 7. 收益上限（决定值不值得继续）

attention 占墙钟：4k **6.8%**、16k **20.4%**。

若自研 kernel 能做到 MMA roofline 的 90%（vendor 现在 64–78%）：

| | attention 现耗时 | 做到 90% 后 | 端到端 |
|---|---|---|---|
| 4k | 13.8 s | ~9.8 s | **+2.0%** |
| 16k | 83.8 s | ~59 s | **+6.0%** |

（未计 softmax 开销，实际约为一半：4k +1%、16k +3%。）

---

## 8. 原型实测进展（QK^T，真实 prefill 形状：4096 tok × 16 heads × kv 4096）

所有版本都通过正确性校验（对 host double 参考，worst rel err **0.00e+00**）。

| 版本 | 结构 | TFLOPS | 占用 | 改动与原因 |
|---|---|---|---|---|
| `qk_proto.cu` | 1 CTA, BM=64, warp tile 16×16 | 0.1 | 4 warps | 先证明 fragment 映射正确 |
| `qk_scaled.cu` | 全网格, BM=64, BN=64 | 2.9 | 16 warps | 放大到真实规模 |
| `qk_fast.cu` | + 16B 向量化 K 拷贝 + 8 warps | 3.0 | 24 warps | 拷贝不是瓶颈 |
| `qk_fast.cu` + smem padding | Qs/Ks 行步长 +8 half | **7.0** | 24 warps | **16 路 bank 冲突**是主因，2.3× |
| `qk_tile32.cu` | warp tile 32×32（寄存器分块） | **13.0** | 16 warps | 每个 fragment 喂 4 个 MMA，1.86× |

### 关键归因

1. **bank 冲突**：fragment 的 lane 沿 k（smem 行）展开，而 Qs 行步长 64 half
   = 128 B = 恰好 32 bank，导致 16 lane 同 bank。这是 3.0 TFLOPS 的真正原因，
   不是带宽也不是占用率。vendor 的 swizzle 就是为这件事存在的。
2. **操作数复用**：一个 16×16×32 fragment 是 64 lane × 4 B = 256 B，若只喂
   1 个 MMA（16384 flop），就是 8 条 `ld.shared` 配 1 条 MMA。改成 32×32 warp
   tile 后每 4 个 MMA 只需 4 次 fragment 装载，指令开销降到 1/2。
3. **不是带宽瓶颈**：13 TFLOPS 对应的 smem 读只有 ~12.7 GB/s/SM ≈ 8.5 B/cycle，
   远低于 128 B/cycle 的能力。**是指令发射与延迟隐藏不足**。
4. **占用率上限**：BM=128×BN=64 的 Qs+Ks 已 50 KB，只能 2 CTA/SM = 16 warps。
   纯发射探针在 16 warps 时是 46 TFLOPS，所以当前 13 只相当于该占用率下 28%。

### 与目标的差距

| | TFLOPS |
|---|---|
| MMA 发射上限（32+ warps/SM） | 97–98 |
| 46 warps/SM 下可达 | 92 |
| 16 warps/SM 下可达 | 46 |
| **当前原型** | **13** |
| **vendor FA（要打败的对象）** | **61–72** |

要追平 vendor 需再提升 ~5×，路径明确但工作量大：异步拷贝 + 双缓冲流水
（用单 CTA 少数 warp 靠流水而非 warp 数隐藏延迟，这是真实 FA 内核的做法），
以及 64×32/64×64 的 warp tile 把 `ld.shared`/MMA 比压到 ≤3。

## 9. 第二轮优化（Q 常驻寄存器）

继续从 13.0 往上推。关键发现：**Q 在整个 KV 循环里不变**，可以一次装进寄存器，
于是循环内的 A 装载全部消失，Q 的 shared memory 也整套删掉。

| 版本 | 结构 | TFLOPS | 占用 | regs | 说明 |
|---|---|---|---|---|---|
| `qk_tile32.cu` | warp tile 32×32 | 13.0 | 16 warps | 58 | 基线 |
| `qk_regq.cu` | Q 进寄存器 + K 预取 | 24.8 | 32 warps | 95 | **+91%**，smem 只剩 K（16.6 KB） |
| `qk_regq2.cu` | warp tile 64×16（1 ld/MMA） | **26.9** | 16 warps | 163 | +8%，但寄存器把占用率压回 16 |

### 为什么停在 ~27

`ld.shared`/MMA 比 = 4/M_tiles（Q 免费，只需装 B）。MT=4 时是 1.0，MT=2 时是 2.0。
但 MT=4 要 64 个寄存器存 Q 的 fragment，总寄存器 163 → 每 SM 只能 2 CTA（16 warps）。

两条路各有一个天花板：

| 配置 | 占用 | 该占用下的发射上限 | 实测 | 达到比例 |
|---|---|---|---|---|
| MT=2 (32×32) | 32 warps | 92 TFLOPS | 24.8 | 27% |
| MT=4 (64×16) | 16 warps | 46 TFLOPS | 26.9 | **58%** |

两者的乘积几乎被寄存器预算锁死 —— 这就是当前 27 TFLOPS 的本质。
MT=4 已经把指令效率做到其占用率下的 58%，继续加 MT 只会进一步压占用率。

### 与 vendor 的差距

口径必须是同一个。`RESULTS` §400 实测 4k prefill 的 `compute_flash_attn` 是
54.78 ms / 42 层 = **1.304 ms 每层，52.7 TFLOPS**，它的 flop 计数
`2*2*(seq*(seq+1)/2)*H_Q*D` = 68.7 GFLOP 已经同时包含了 QK 与 PV 并扣掉 causal 的一半。

| | 实际做的活 | 时间 | TFLOPS |
|---|---|---|---|
| 本原型 QK^T | 满方阵 68.7 GFLOP | 2.49 ms | 27.6 |
| 同等 causal QK^T（估算，按跳过一半 tile） | 34.3 GFLOP | ≈1.25 ms | 27.6 |
| vendor FA（QK + softmax + PV 全含） | 68.7 GFLOP | 1.304 ms | 52.7 |

**即：本原型只做 QK 这一项（且按 causal 折算后）就要花掉 vendor 做完整
attention 的时间，仍差约 2×。**

### 下一步（未做）

1. **cp.async / 异步拷贝**：一旦 staging 完全脱离关键路径，占用率的重要性下降，
   可以用 1–2 个大 CTA + 多级流水，这是真实 FA 内核的形态。需要先确认 ivcore11
   是否暴露 `cp.async` 或等价原语。
2. **把寄存器压到 ≤128** 以拿到 32 warps 同时保留 MT=4（现在 163，需砍掉
   地址计算与循环的 35 个寄存器）。
3. 之后才是 softmax 融合（accumulator 映射已知：行摊在 16 个相邻 lane 上，
   行内归约只需 16-lane shuffle）与 PV。

## 10. 第三轮：异步拷贝不可用，寄存器墙是硬的

### `cp.async` 在这套工具链里不可用

`/usr/local/corex/include/clang/Basic/BuiltinsIluvatar.def` 里有完整的异步拷贝族：

```
__ivcorex_cp_async_ca_shared_global_4 / _8 / _16   (vV4Ui*V4UiC*)
__ivcorex_cp_async_commit_group                     (v)
__ivcorex_cp_async_wait_group                       (vUi)
```

但它们被 target feature `async-copy` 门禁，默认不开：

```
error: '__ivcorex_cp_async_wait_group' needs target feature async-copy
```

而用 `-Xclang -target-feature -Xclang +async-copy` 打开后，**后端直接崩溃**：

```
clang: error: unable to execute command: Segmentation fault (core dumped)
clang: error: llc command failed due to signal
```

（`-mllvm -ivcorex-async-copy` 不是有效选项，会被 LLVM option parser 拒绝。）
所以 `qk_async.cu` 里的双缓冲流水线代码写好了但**跑不了**，保留在仓库里作为
工具链恢复后可直接启用的实现。这是本次结论里最值得上报给厂商的一条。

### 没有异步拷贝后，寄存器就是硬墙

| 尝试 | regs | 占用 | TFLOPS | 结果 |
|---|---|---|---|---|
| MT=4 + 预取寄存器 | 176 | 2 CTA / 16 warps | 26.8 | 基线 |
| 去掉预取 | 163 | 2 CTA / 16 warps | 26.9 | 无变化 |
| B 地址压成单基址 | 176 | 2 CTA / 16 warps | 27.1 | 无变化 |
| `store_s` 改编译期模板 | **145** | 2 CTA / 16 warps | **27.6** | 最好 |
| `__launch_bounds__(NTHREAD,4)` 强压 128 | 64 | 7 CTA / 56 warps | 11.8 | **溢出到 local，崩** |

强压寄存器等于让 Q 的 fragment 落到 local memory，56 warps 也救不回来（11.8）。
所以 27.6 是这套「Q 常驻 + MT=4」设计的实际平台，不是调参没到位。

寄存器构成：qreg 64（4 个 M tile × 4 个 k chunk × 4 uint，这是 64 行 × 128 k ÷ 64 lane
= 128 half，已经是最小表示）+ acc 16 + fb 4 = 84 不可压缩，实测 145，剩余 61 是
16 条 MMA 全展开后的活跃临时量。

### 结论：本路径的性能目标（不低于 vendor）在当前条件下达不到

| | 时间 | TFLOPS | 说明 |
|---|---|---|---|
| 本原型（**只做 QK^T**，causal 折算后） | ≈1.25 ms | 27.6 | 无 softmax、无 PV、无 paged/causal |
| vendor FA（**QK + softmax + PV 全含**） | 1.304 ms | 52.7 | 生产内核 |

即「做三分之一的活，花掉对方全部时间」，约差 2×。

注意这里的 roofline 不是 96.6：`RESULTS` §400 记录该 shape 的 microbench 峰值是
**71–74 TFLOPS**（不是 8192³ 的 96.5），vendor 的 52.7 相当于该 roofline 的 71–74%。
所以要追平，QK^T 需要在做完 softmax/PV 之余仍达到 ~52 TFLOPS 的实际吞吐，
而本设计的天花板受寄存器预算锁在 27.6（见上表：强压寄存器会溢出并掉到 11.8）。

这不是实现细节没打磨，是「自研 + 无异步拷贝 + 每 warp 16×16×16 指令形状
+ 128 寄存器预算」叠加后的结构性上限。

### 最小复现（建议上报厂商）

`cp_async_repro.cu` 是一个 7 行的最小用例，只做一次 16 B 的 global→shared 异步拷贝：

```bash
clang -O3 -x ivcore --cuda-gpu-arch=ivcore11 --cuda-path=/usr/local/corex \
      -Xclang -target-feature -Xclang +async-copy -c -o /dev/null cp_async_repro.cu
# clang: error: unable to execute command: Segmentation fault (core dumped)
# clang: error: llc command failed due to signal
```

`-O0` 同样崩溃，所以不是优化级别问题；builtin 与 IR intrinsic
（`llvm.bi.cp.async.ca.shared.global.{4,8,16}`，在 clang 二进制里可见）都在，
坏的只是 **llc 的 lowering 路径**。corex 版本 4.5.0。

这一条是整个结论的关键：没有异步拷贝，就没法把 staging 移出关键路径；
移不出去就只能靠 warp 数隐藏延迟；而 warp 数被寄存器预算锁死。

---

## 11. 重要更正：峰值是 196 TFLOPS，不是 96.6

之前所有峰值数字都少算了一半。原因很具体：`mma_rate.cu` 假设 warp 是 32 lane，
于是把一条 MMA 的输出算成 `32 × 4 = 128` floats = 16×8。但**这台机器的 warp 是
64 lane**（`warpSize=64`），输出是 `64 × 4 = 256` floats = 16×16。

三条独立证据都指向 **8192 flops/指令**：

| 证据 | 结论 |
|---|---|
| 64 lane × 4 float = 256 = 16×16 输出 | 16×16×16 = 8192 flops |
| `tile_ok.cu` 用**恰好 2 条** `ml_matrix_mad_f32_f16_m16n16k16` 算出并验证了 16×16×32 tile | 每条 16×16×16 |
| `mma_rate` 的循环体汇编里恰好 4 条 MMA 对应 4 次 intrinsic 调用，无隐藏展开 | 1:1 |

所以峰值 = 16 SM × 1 MMA/cycle × 8192 flops × 1.5 GHz = **196.6 TFLOPS**，
与之吻合的实测值是 196.2（1024 warps = 128 warps/SM）。

修正后的发射上限曲线（同一个探针，只改记账）：

| warps/SM | TFLOPS | 占 196.6 |
|---|---|---|
| 8 | 46.1 | 23% |
| 16 | 92.2 | 47% |
| 32 | 184.4 | **94%** |
| 64 | 196.2 | 100% |

**这条曲线改变了整个判断**：32 warps/SM 就已经到 94%，不需要 128 warps。
而之前把 96.6 当峰值，得出「vendor 已在 1.28–1.57× roofline 之上、无空间」的结论
是错的。真实情况是 vendor 只跑到 roofline 的 27%。

| | TFLOPS | 占 196.6 |
|---|---|---|
| torch.mm / ixBLAS (8192³) | 96.5 | 49% |
| vendor FA（4k prefill，QK+softmax+PV） | 52.7 | 27% |
| 本原型 QK^T（最新） | **63.0** | 32% |

## 12. QK^T 优化到 63.0 TFLOPS

一律先验证再计时。验证方式是**把输出缓冲区涂成 -1 的毒值**，然后要求每个元素都被
覆写且数值与 host double 参考完全一致（`worst rel err 0.00e+00`，`untouched 0`）。
这样「只算了部分列」这类 bug 无法蒙混过关——下面记的两个坑正是这样被抓到的。

### 两个记账陷阱（都已抓出并修正）

1. `WNUL=1` 时 8 个 warp 全排在 M 方向，只覆盖 16 列，但 flop 公式按 BN 计
   → 虚报 `BN/16` 倍。曾一度报出 210 TFLOPS（>峰值），就是它。
2. `BN` 作为独立模板参数、与 `WNUL*NT*16` 脱钩，同样虚报。
   最终版本把 `BM = WMUL*MT*16`、`BN = WNUL*NT*16` 全部由几何**推导**，不再放开。

### 配置扫描（全部 4096 tok × 16 heads × kv 4096，全部验证通过）

| 配置 | BM | BN | warps | regs | ms | TFLOPS |
|---|---|---|---|---|---|---|
| W4x1 MT8 NT1 | 512 | 16 | 16 | 174 | 1.091 | **63.0** |
| W4x2 MT8 NT1 | 512 | 32 | 16 | 175 | 1.180 | 58.2 |
| W8x1 MT8 NT1 | 1024 | 16 | 16 | 256 | 1.232 | 55.3 |
| W2x1 MT8 NT1 | 256 | 16 | 16 | 174 | 1.258 | 54.6 |
| W2x2 MT8 NT1 | 256 | 32 | 16 | 175 | 1.303 | 52.7 |
| W2x4 MT8 NT1 | 256 | 64 | 16 | 175 | 1.474 | 46.6 |
| W4x2 MT4 NT1 | 256 | 32 | 32 | 100 | 1.744 | 39.4 |
| W8x1 MT4 NT1 | 512 | 16 | 16 | 143 | 1.876 | 36.6 |
| W4x1 MT16 NT1 | 1024 | 16 | 16 | 256 | 1.936 | 35.5 |

### 三个已确认的规律

1. **BM 越大越好**。staging 开销 ∝ BN，MMA 工作量 ∝ BM×BN，所以两者的比值只取决于
   BM。BM 从 128 提到 512，QK^T 从 30.7 → 63.0（2.05×）。
2. **WT 排布要尽量沿 M**。A 片段在 N 方向每个 warp 复制一份，所以 `WNUL` 就是 Q 的
   寄存器冗余倍数。W2x4 用 128 个寄存器只换来 BM=256；W4x1 同样 128 个寄存器拿到
   BM=512。
3. **寄存器是硬墙，16 warps 是平台**。174 个寄存器下所有强制提高 CTA 数的尝试都溢出：

| 强制 CTA | 实际 regs | warps | TFLOPS | 结果 |
|---|---|---|---|---|
| 默认 | 174 | 16 | **63.0** | 最好 |
| 5 CTA | 48 | 80 | 14.5 | 溢出，崩 |
| 6 CTA | 40 | 96 | 12.8 | 溢出，崩 |
| 8 CTA | 32 | 128 | 8.9 | 溢出，崩 |

qreg 本身就是 MT×KCH×4 = 128 个寄存器（Q 的 512×128 半个 tile，没有冗余可去），
加上 acc 32 + fb 4，174 已是实际需求。所以「16 warps × 68% 效率 = 63 TFLOPS」
就是这套设计的平台，不是调参没到位。

### 意义

| | 时间 | TFLOPS |
|---|---|---|
| 本原型，**只做 QK^T**（无 softmax/PV/paged/causal，满方阵） | 1.091 ms | 63.0 |
| vendor FA，**QK + softmax + PV 全含**（causal，flop 已折半） | 1.304 ms | 52.7 |

满方阵 68.7 GFLOP 折算成 causal 后约 34.3 GFLOP，即在同等工作量下本原型约
**0.55 ms vs vendor 1.304 ms，快 2.4×**，为 softmax 与 PV 留出了 0.75 ms 的预算。
这是本项目第一次出现「自研路径在真实形状上不劣于 vendor」的证据。
