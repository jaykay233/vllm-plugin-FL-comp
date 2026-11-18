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
