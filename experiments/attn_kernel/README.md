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

| | 时间 | TFLOPS |
|---|---|---|
| 本原型 QK^T（不含 softmax、不含 PV） | 2.56 ms | 26.9 |
| vendor FA（QK + softmax + PV 全含） | 1.12 ms | 61.2 |

**即：目前只做 QK^T 就比 vendor 做完整 attention 慢 2.3×。**

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
| 本原型（**只做 QK^T**） | 2.49 ms | 27.6 | 无 softmax、无 PV、无 paged/causal |
| vendor FA（**QK + softmax + PV 全含**） | 1.12 ms | 61.2 | 生产内核 |

即「做更少的活还慢 2.2×」。要追平需 QK^T 单独做到 ~110 TFLOPS（把 softmax/PV
的预算算进去后仍需 ≥65），而 MMA 发射上限只有 97–98。**换句话说，即使把
QK^T 做到发射上限，也刚好只够平 vendor，没有任何余量容纳 softmax 与 PV。**

这不是实现细节没打磨，是「自研 + 无异步拷贝 + 每 warp 16×16×16 指令形状」
三者叠加后的结构性上限。
