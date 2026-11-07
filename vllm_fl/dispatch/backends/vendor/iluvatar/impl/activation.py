# Copyright (c) 2026 BAAI. All rights reserved.

"""Iluvatar SiLU-and-mul: prefer FlagGems fused, else a local Triton kernel."""

from __future__ import annotations

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _silu_mul_single_kernel(x_ptr, o_ptr, N, I, BLOCK: tl.constexpr):
    """``x`` is contiguous ``[..., 2*I]``; write ``silu(x[..., :I]) * x[..., I:]``."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    row = offs // I
    col = offs % I
    d = tl.load(x_ptr + row * (2 * I) + col, mask=mask).to(tl.float32)
    g = tl.load(x_ptr + row * (2 * I) + col + I, mask=mask).to(tl.float32)
    s = d * tl.sigmoid(d)
    tl.store(o_ptr + offs, (s * g).to(o_ptr.dtype.element_ty), mask=mask)


def _triton_silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    out = torch.empty(*x.shape[:-1], d, device=x.device, dtype=x.dtype)
    flat_n = out.numel()
    x_c = x.contiguous()
    block = 512 if flat_n >= 512 else max(64, triton.next_power_of_2(flat_n))
    _silu_mul_single_kernel[(triton.cdiv(flat_n, block),)](
        x_c.view(-1, 2 * d),
        out.view(-1),
        flat_n,
        d,
        BLOCK=block,
        num_warps=4,
    )
    return out


def silu_and_mul_iluvatar(obj, x: torch.Tensor) -> torch.Tensor:
    """
    SiLU activation followed by element-wise multiplication.

    Prefers FlagGems' vendor fused kernel when available; otherwise a local
    Triton fused kernel for contiguous CUDA bf16/fp16; ATen last.
    """
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    try:
        import flag_gems

        if hasattr(flag_gems, "silu_and_mul"):
            return flag_gems.silu_and_mul(x1, x2)
    except Exception:
        pass
    if (
        x.is_cuda
        and x.dtype in (torch.bfloat16, torch.float16)
        and x.is_contiguous()
    ):
        return _triton_silu_and_mul(x)
    return F.silu(x1) * x2
