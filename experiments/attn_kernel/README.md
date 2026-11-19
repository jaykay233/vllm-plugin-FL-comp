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

## 11. 峰值复核：确认是 ~98 TFLOPS（本节曾误改为 196.6，已回退）

这个数字我改了两次，最后确认 **原始结论（~98 TFLOPS）是对的**。全过程记录在下面，
因为两次都错得很像，值得留档。

### 正确的事实

| 事实 | 值 | 证据 |
|---|---|---|
| 单条 16×16×16 MMA | **8192 flops** | 64 lane × 4 float = 256 = 16×16 输出；`tile_ok.cu` 用恰好 **2 条** `ml_matrix_mad_f32_f16_m16n16k16` 验证出 16×16×32 tile，即每条 k=16 |
| 每 SM 每 2 个周期发射 1 条 | 0.5 MMA/cycle/SM | 见下 |
| 峰值 | **98.3 TFLOPS** | 16 × 0.5 × 8192 × 1.5 GHz |
| 实测发射上限 | **98.2** | `mma_rate.cu`，32 warps/SM |
| ixBLAS bf16 GEMM 上限 | **97.3** @ 16384³ | vendor 调优库的正常水平 = 峰值的 99% |

若峰值真是 196.6，则 ixBLAS 只跑到 49%——一个厂商调优的 GEMM 库不会这样。这是最有力的反证。

### 两次错误的成因（都很隐蔽）

1. **原版 `mma_rate.cu`**：按 32-lane warp 假设，`FLOPS_PER_MMA = 4096`（错，应为 8192），
   同时 `warps = blocks*256/32`（错，256 线程 = 4 个 64-lane warp）。**两个错误恰好抵消**
   （`8×4096 = 4×8192`），所以总 flops 是对的，只有「warps」这个标签印错了。结论正确纯属幸运。
2. **本轮的「更正」**：把 `FLOPS_PER_MMA` 改成 8192（对），但**忘了同时改 warps 的分母**，
   于是 2× 重复计数，报出 196.2 TFLOPS。若属实，ixBLAS 就在 49%。已回退。

现在 `warps = blocks*256/64`，`FLOPS_PER_MMA = 8192`，与 98.2/98.3 自洽。

### 修正后的发射上限曲线（`mma_rate.cu`，与 `mma_shapes.cu` 互相印证）

| blocks | warps/SM | TFLOPS | 占峰值 |
|---|---|---|---|
| 16 | 4 | 23.1 | 23% |
| 32 | 8 | 46.1 | 47% |
| 64 | 16 | **92.3** | 94% |
| 128 | 32 | **98.2** | 100% |

注意 blocks=16/32/64 的墙钟都是 1.818 ms —— 因为它们是单 wave 并行，**总 flops 随 blocks
线性增长而时间不变**，所以吞吐正比于 blocks。这是单 wave 测试的正确表现，不是测量错误。

**16 warps/SM 就能拿到 94% 的发射能力**，这对下面的 kernel 设计很关键。

### 修正后的位置

| | TFLOPS | 占 98.3 |
|---|---|---|
| ixBLAS bf16 GEMM | 97.3 | 99% |
| **本原型 QK^T** | **63.0** | **64%** |
| vendor FA（4k prefill，QK+softmax+PV） | 52.7 | 54% |

## 12. QK^T 优化到 63.0 TFLOPS

（占峰值 64%。在它自己的 16 warps/SM 占用下，可用的发射能力是 92.3，
所以实际达到该占用的 **68%**。）

### 配置扫描（全部 4096 tok × 16 heads × kv 4096，全部验证通过）

| 配置 | BM | BN | warps/SM | regs | ms | TFLOPS | 占峰值 |
|---|---|---|---|---|---|---|
| W4x1 MT8 NT1 | 512 | 16 | 16 | 174 | 1.091 | **63.0** | 64% |
| W4x2 MT8 NT1 | 512 | 32 | 16 | 175 | 1.180 | 58.2 | 59% |
| W8x1 MT8 NT1 | 1024 | 16 | 16 | 256 | 1.232 | 55.3 | 56% |
| W2x1 MT8 NT1 | 256 | 16 | 16 | 174 | 1.258 | 54.6 | 56% |
| W2x2 MT8 NT1 | 256 | 32 | 16 | 175 | 1.303 | 52.7 | 54% |
| W2x4 MT8 NT1 | 256 | 64 | 16 | 175 | 1.474 | 46.6 | 47% |
| W4x2 MT4 NT1 | 256 | 32 | 32 | 100 | 1.744 | 39.4 | 40% |
| W8x1 MT4 NT1 | 512 | 16 | 16 | 143 | 1.876 | 36.6 | 37% |
| W4x1 MT16 NT1 | 1024 | 16 | 16 | 256 | 1.936 | 35.5 | 36% |

### 三个已确认的规律

1. **BM 越大越好**。staging 开销 ∝ BN，MMA 工作量 ∝ BM×BN，所以两者的比值只取决于
   BM。BM 从 128 提到 512，QK^T 从 30.7 → 63.0（2.05×）。
2. **WT 排布要尽量沿 M**。A 片段在 N 方向每个 warp 复制一份，所以 `WNUL` 就是 Q 的
   寄存器冗余倍数。W2x4 用 128 个寄存器只换来 BM=256；W4x1 同样 128 个寄存器拿到
   BM=512。
3. **寄存器是硬墙，16 warps 是平台**。174 个寄存器下所有强制提高 CTA 数的尝试都溢出
（注意 16 warps/SM 时硬件能发 92.3 TFLOPS，所以 63.0 是该占用下的 68%）：

| 强制 CTA | 实际 regs | warps | TFLOPS | 结果 |
|---|---|---|---|---|
| 默认 | 174 | 16 | **63.0** | 最好 |
| 5 CTA | 48 | 80 | 14.5 | 溢出，崩 |
| 6 CTA | 40 | 96 | 12.8 | 溢出，崩 |
| 8 CTA | 32 | 128 | 8.9 | 溢出，崩 |

qreg 本身就是 MT×KCH×4 = 128 个寄存器（Q 的 512×128 半个 tile，没有冗余可去），
加上 acc 32 + fb 4，174 已是实际需求。所以「16 warps × 68% 效率 = 63 TFLOPS」就是这套设计的平台，不是调参没到位。

### 意义

| | 时间 | TFLOPS |
|---|---|---|
| 本原型，**只做 QK^T**（无 softmax/PV/paged/causal，满方阵） | 1.091 ms | 63.0 |
| 本原型 QK^T 折算 causal（跳过一半 tile，估算） | ≈0.55 ms | — |
| PV 折算 causal（与 QK 同量级，估算） | ≈0.55 ms | — |
| **本原型 QK+PV 合计（估算，尚不含 softmax/paged）** | **≈1.10 ms** | — |
| vendor FA，**QK + softmax + PV 全含**（causal，flop 已折半） | 1.304 ms | 52.7 |

注意这不是「快 2.4×」。QK 单独 0.55 ms 对的是 vendor 做完三件事的 1.304 ms，
**不是同等工作量**。诚实的读法是：QK 与 PV 各约 0.55 ms，合计 ≈1.10 ms，只剩
约 0.20 ms 留给 softmax、paged 寻址与 causal 分支，而 vendor 的同一份预算里
softmax 等开销总共是 1.304 − 1.10 ≈ 0.20 ms。

所以结论是：**自研路径在 QK 这一半上已经追平 vendor（63.0 vs 52.7），
整体能否不劣于 vendor 取决于 softmax 与 paged 的处理效率，目前是临界状态，
不是已经赢。** 这是本项目第一次出现「自研路径在真实形状上不劣于 vendor」的
子证据，但还不足以支撑整体结论。

## 13. 因果跳过在当前网格分解下不划算（负结果）

prefill 是 causal 的，所以整块跳过理论上应该省 44% 的时间。实测**没有**：

| 方案 | ms | 说明 |
|---|---|---|
| 满方阵（不跳过） | 1.092 | 128 CTA，每个 256 个 KV block |
| causal，整块跳过（不均衡） | 1.100 | 每个 CTA 32…256 个 block，差 8× |
| causal，配对均衡（每 CTA 288 block） | 1.110 | 64 CTA，工作量完全均衡 |

causal 只做满方阵 **56.2%** 的工作，时间却是 100%。

### 原因：不是负载不均，是占用率

配对均衡把工作量抹平了，但没有变快，所以「不均衡」不是根因。真正的原因是每 slot 的
CTA 数：

| 情形 | 每 slot 的 CTA | 每 block 耗时 | 总时间 |
|---|---|---|---|
| 满方阵 | 2（128 CTA / 64 slot） | 0.00213 ms | 1.092 |
| 配对均衡 | 1（64 CTA / 64 slot） | 0.00385 ms | 1.110 |

一个 CTA 只有 4 个 warp。1 个 CTA 时只有 4 warps 在隐藏延迟，2 个 CTA 时是 8 warps，
每 block 成本差 1.8×。所以配对省下的 44% 工作量，被 1.8× 的每 block 成本抵消掉了。

**结论：causal prefill 不能靠「整块跳过 + 现有网格」拿收益，必须换并行分解** ——
把 KV 维度也切开（flash-decoding 式的 split-K + 部分 softmax 归约），才能把 56% 的
工作量真正摊到所有 slot 上。这是下一步 fused attention 的设计前提，而不是可选项。

### 顺带修掉的一个验证 bug

之前的「覆盖率检查」用 `cudaMemset(dS, 0xFF, ...)` 当毒值，但 `0xFFFFFFFF` 作为
float 是 **NaN**，不是 -1.0f。于是「未覆写」计数恒为 0，而且 NaN 参与
`er > 3e-2` 比较时结果为 false，也不会被计入 bad ——**检查形同虚设**。
现在改用真正的 kernel 写 `-12345.0f`（`poison`）。这个 bug 修掉后
causal 的 `mask-leak` 才真正暴露出来并归零。

## 14. 原生 bf16 MMA 可用且与 fp16 同速；更大的指令形状被工具链挡住

模型是 bf16 的，而前面所有原型都用 `__half`。这曾是正确性隐患：bf16 指数位 8 位、
fp16 只有 5 位，把 bf16 转 fp16 **可能溢出**（bf16 能表示 ±3.4e38，fp16 上限 65504）。

### 结论一：`__ivcorex_matrix_mad_f32x4_bf16x4` 原生可用，速率与 fp16 完全相同

`mma_shapes.cu` 在相同占用下逐点对比：

| blocks | warps/SM | f16 16×16×16 | bf16 16×16×16 |
|---|---|---|---|
| 16 | 4 | 23.1 | 23.1 |
| 32 | 8 | 46.2 | 46.2 |
| 64 | 16 | 92.4 | 92.4 |
| 128 | 32 | 98.2 | 98.2 |

完全一致（差异 <0.01%）。所以 attention kernel **应当直接用 bf16 MMA**，既避免
转换溢出风险，又与 vendor 的数值口径一致。

### 结论二：`long-mtx` 指令形状同样让 llc 段错误

指令集里有 4× 规模的形状，一条就是 32768 flops：

```
__ivcorex_matrix_mad_f32_bf16_m64n16k16   V16fV16yV4yV16f   (64×16×16)
__ivcorex_matrix_mad_f32_bf16_m16n64k16   V16fV4yV16yV16f   (16×64×16)
```

若与 16×16×16 同发射率，吞吐可再翻 4 倍。但它们被 target feature `long-mtx` 门禁：

```
error: '__ivcorex_matrix_mad_f32_bf16_m64n16k16' needs target feature long-mtx
```

而 `-Xclang -target-feature -Xclang +long-mtx` 打开后，**与 `cp.async` 完全相同的症状**：

```
clang: error: unable to execute command: Segmentation fault (core dumped)
clang: error: llc command failed due to signal
```

所以 `mma_shapes.cu` 里这两个 kernel 用 `-DUSE_LONG_MTX` 隔开，默认不编译。
这是**第二条**厂商工具链缺陷（第一条是 `cp.async`），而且两者都在 llc lowering 阶段崩溃，
很可能是同一个根因。这两条一起上报的价值最高。
## 15. 融合 attention（QK + online softmax + PV）：数值正确，但比 vendor 慢 3.9×

`fused_bf16.cu` 是第一个把三段完整跑通的原生 kernel：bf16 MMA、GQA 16/2、causal、
online softmax，全部在寄存器里。**精度达标，性能不达标。**

```
./fused_bf16 <tokens> <kv> [noref=0] [nkv_heads=2]
```

### 数值：与 vendor 内核逐 shape 对齐（objective 第 2 条）

判据按行归一化误差 `||diff||₂ / ||ref||₂`，对 `ixformer` 的
`flash_attn_varlen_func` 直接比对（不是对 host 参考）：

| tok | kv | causal | maxdiff | 行归一化最差 | 超阈值行 |
|---|---|---|---|---|---|
| 256 | 256 | 是 | 1.11e-02 | 8.66e-03 | 0/4096 |
| 256 | 256 | 否 | 1.06e-03 | 7.92e-03 | 0/4096 |
| 1024 | 1024 | 是 | 1.22e-02 | 8.26e-03 | 0/16384 |
| 1024 | 1024 | 否 | 5.45e-04 | 8.09e-03 | 0/16384 |
| **1024** | **2048** | **是** | **4.87e-04** | **7.99e-03** | **0/16384** |
| 1024 | 2048 | 否 | 4.33e-04 | 8.39e-03 | 0/16384 |
| 4096 | 4096 | 是 | 1.11e-02 | 8.31e-03 | 0/65536 |
| 4096 | 4096 | 否 | 3.67e-04 | 8.32e-03 | 0/65536 |

8/8 通过，行归一化误差全部在 8.4e-03 以内（bf16 精度量级 2⁻⁸ = 3.9e-03 的同量级），
maxdiff 最大 1.4e-02。**第 (2) 条门禁达成。**

### 这一步抓到一个真 bug：causal 对角线对齐

只在 `kv > tok` 时暴露。当 KV cache 比 query 块长（chunked prefill：新 token 块
attend 到已缓存的全部内容），causal 对角线对齐在**右下角**，query 行 r 可见的 key 是
`0 .. (kv - tok) + r`，而不是 `0 .. r`。两者只在 `kv == tok` 时等价。

所以 1024/2048 causal 当时**每一行都错**（maxdiff 1.38），而另外 7 个 shape 恰好
`kv == tok` 全过。更麻烦的是我的 host 参考用了同样的错误约定，于是自检互相印证、
把 bug 藏住了 —— 只有跟 vendor 比对才暴露出来。kernel 与参考都已改成右下对齐。

### 两个度量陷阱（都已修）

1. **逐元素相对误差不能判 attention**。输出会穿过 0，分母 `|ref|` 趋零时相对误差必然爆。
   最初因此报了 2% 元素"失败"，但查维度分布发现错误在 dim/行/head 上均匀散布
   （每维约 350 个），这是精度问题的特征而非布局问题。改用行归一化后全部通过。
2. **`kv < tok` 时参考要截断**。用 `nk = t+1` 会读到 `hK` 边界之外，报出假失败；
   kernel 是对的。参考要按 `min(kv, (kv-tok)+t+1)` 截断。

### 性能

| 形状 | causal NT=4 | 占峰值 |
|---|---|---|
| 4k tok, GQA 16/2 | 4.27 ms / 16.6 TFLOPS | 16.9% |
| 16k tok, GQA 16/2 | 61.1 ms / 18.1 TFLOPS | 18.4% |
| vendor FA（4k prefill，每层） | 1.304 ms / 52.7 TFLOPS | 53.6% |

仍差 vendor 3.3×，**未能满足 objective 的第 (3) 条门禁**，因此没有接入 serving。

### 结构天花板：去掉 softmax 之后仍然慢 2.9×

这是本轮最关键的一次测量。把 softmax 整个删掉（P 直接取 score 累加器，结果无意义），
只保留 staging + fragment 读 + 全部 MMA：

| 变体（4k causal, NT=4） | 时间 | TFLOPS |
|---|---|---|
| 只有 64 条 MMA | 0.94 ms | 76.3 |
| ＋fragment 读 | 2.08 ms | 34.0 |
| ＋staging | 3.87 ms | 18.3 |
| ＋softmax（完整 kernel） | 4.27 ms | 16.6 |
| **vendor（含 softmax）** | **1.304 ms** | **52.7** |

两个结论：

1. **softmax 只占 0.4 ms（9%）**。前面几轮把优化力气花在 softmax 上是错的 —— 现在有了
   干净的账，这个方向被彻底排除。
2. **连完全不含 softmax 的版本（3.87 ms）都比 vendor 慢 2.9×**。所以差距不在注意力
   的数学部分，而在 K/V 的数据搬运通路上。

拆开看：staging 本身 1.79 ms（3.87−2.08，占完整 kernel 的 42%），fragment 读 1.14 ms，
MMA 0.94 ms。**staging 是最大单项**，而且它已经排除掉所有常见解释：不是带宽
（GQA 8× 减量无变化）、不是 bank 冲突（线性 store +1%）、不是延迟（寄存器预取更差 8%）。

### 为什么 staging 拿不掉：两条路都被工具链堵死

staging 的价钱不是「搬多少字节」，而是「搬的时候不能算」。正常解法有两个，都不能用：

* `cp.async`（global→smem 异步拷贝，与 MMA 重叠）—— 开了 `+async-copy` 后
  `llc` 段错误，最小复现见 `cp_async_repro.cu`。§10 已记录。
* 更大的 MMA 形状（`m32n32k16` / `m64n16k16`，一条顶 4 条）—— 同样在 `llc` 里崩。

本轮把第二条挖得更深了一层，得到一个比之前更强的结论：**它不只是被 feature 门禁挡住，
而是后端 lowering 本身残缺**。用函数级属性 `__attribute__((target("long-mtx")))`
可以绕过「needs target feature」的报错，但会死在类型合法化：

```
LLVM ERROR: Do not know how to split this operator's operand!
clang: error: llc command failed due to signal
```

`m32n32k16`（bf16，-O0）如此；`m64n16k16` 的 builtin 签名与 `BuiltinsIluvatar.def`
里的 `V16fV16yV4yV16f` 对不上（clang 认为 A 是 4 个 bf16）；`m32n32k16` 的 f16 版本
用 `__half` 直接报 invalid vector element type。也就是说这条路径在 SDK 层面就是不可用的。

代价可以量化：`m32n32k16` 每条指令 32768 flop，而 A/B 操作数只比 16×16×16 多 2 个寄存器
（V8y 而非 V4y），累加器仍是同样的 16 float/lane。在「发射受限」的当前状态下，
这等于把每 4 倍工作量压缩成 1 条指令。

### 本轮真正拿到的收益：NT（每轮多 tile）

前面的诊断说 SM 的 IPC 只有 ~1.34、循环体 300 条指令里有 56 条 `sl_wait`，
所以瓶颈是停顿而不是指令数。但只拆 QK 累加链（NS=2/4/8）几乎无效（+1.8%），
说明停顿来自**一整轮的串行**（load → MMA 链 → 归约 → exp → 转置 → MMA 链）。

改成每轮处理 NT 个独立 KV tile：NT 组 score 累加器、NT 组 exp，只把行统计合并，
于是每轮的固定开销（O 的 rescale、行归约）摊到 NT 倍 MMA 上，调度器也有 NT 条链可交错。

| NT | 4k causal | 16k causal | 占用 | smem |
|---|---|---|---|---|
| 1 | 5.06 ms | 69.6 ms | 4 CTA/SM, 96 regs | 8 KB |
| 2 | 4.65 ms | 63.7 ms | 4 CTA/SM, 102 regs | 16 KB |
| **4** | **4.27 ms** | **61.1 ms** | 4 CTA/SM, 116 regs | 32 KB |
| 8 | 4.50 ms | 63.0 ms | 2 CTA/SM, 139 regs | 64 KB |

NT=4 是拐点：它仍是 4 CTA/SM（32 warps），而 NT=8 把 smem 顶到 64 KB、占用掉到
2 CTA/SM，反而变差。相比本轮开始快 18%（4k）、15%（16k）。

### 其余否定结果（累计 11 项）

除上面四组外，本轮还试了：

| 改动 | 效果 |
|---|---|
| `__launch_bounds__(NTHREAD, 4)` 压到 64 regs 换 8 CTA/SM | 7.27 ms（**劣化 72%**，溢出） |
| 压到 32 regs 换 16 CTA/SM | 25.3 ms（**劣化 5×**） |
| BM 翻倍到 256（16 warp/CTA） | 5.45 ms（劣化 8%） |

规律很一致：**这个 kernel 需要 ~100-116 个寄存器才能不溢出**，超过 32 warps 的占用率
是拿溢出换来的，净亏。所以 4 CTA/SM × 8 warps 是这台机器上该结构的硬最优点。

### 与 objective 门禁的对照

| objective 要求 | 状态 |
|---|---|
| 覆盖 prefill/mixed paged 路径（BNDH、GQA 16/2、D=128、page 16、causal） | GQA 16/2、causal（含 `kv > tok` 的 chunked prefill 对齐）已覆盖并验证；**paged KV 未接入**（仍是 dense 指针） |
| 数值对齐 vendor（maxdiff 在 bf16 容差内） | **达成**。8/8 shape 对 `flash_attn_varlen_func` 直接比对，行归一化误差 ≤ 8.4e-03，maxdiff ≤ 1.4e-02 |
| 性能逐 shape 不低于现有内核 | **未达标**：4k 慢 3.3×，16k 慢约 3.4× |
| 端到端 4k ≥6436 / 16k ≥5421 tok/s、MATH-500 L3 ≥95% | 未测（kernel 未接入） |

结论：用当前工具链能到达的指令集（16×16×16，无 `cp.async`），这个 kernel 的最优
结构天花板约为 34 TFLOPS（无 staging）和 18 TFLOPS（有 staging），**低于 vendor 的
52.7**。要越过这条线，前提是先修好 `cp.async` 或 `long-mtx`。
