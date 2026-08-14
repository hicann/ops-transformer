# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
from typing import Optional, Tuple
import torch
from torch.library import impl
from cann_ops_transformer.op_builder.builder import OpBuilder
from cann_ops_transformer.op_builder.builder import AS_LIBRARY


class CompressorOpBuilder(OpBuilder):
    def __init__(self):
        super(CompressorOpBuilder, self).__init__("compressor")

    def sources(self):
        """Path to C++ source code."""
        return ["ops/csrc/compressor.cpp"]

    def schema(self):
        """PyTorch operator signature."""
        pass

    def register_meta(self):
        """
        Registers the Meta implementation (Shape/Dtype inference).
        Essential for Autograd and FakeTensor support.
        """
        pass


compressor_op_builder = CompressorOpBuilder()
op_module = compressor_op_builder.load()


# ===========================================================================
# Register compressor forward
# ===========================================================================
@torch.library.custom_op(
    "cann_ops_transformer::_compressor_forward", mutates_args=(), device_types="npu"
)
def _compressor_forward(
    x: torch.Tensor,
    wkv: torch.Tensor,
    wgate: torch.Tensor,
    state_cache: torch.Tensor,
    ape: torch.Tensor,
    state_block_table: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    seqused: Optional[torch.Tensor] = None,
    start_pos: Optional[torch.Tensor] = None,
    cmp_ratio: int = 4,
    coff: Optional[int] = 1,
    cache_mode: Optional[int] = 1,
    grad_enabled: Optional[bool] = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    cmp_kv, softmax_score, kv = op_module.compressor(
        x,
        wkv,
        wgate,
        state_cache,
        ape,
        cmp_ratio,
        state_block_table,
        cu_seqlens,
        seqused,
        start_pos,
        coff,
        cache_mode,
        grad_enabled,
    )
    updated_state_cache = state_cache.detach().clone()
    return cmp_kv, softmax_score, kv, updated_state_cache


@torch.library.register_fake("cann_ops_transformer::_compressor_forward")
def _compressor_forward_fake(
    x: torch.Tensor,
    wkv: torch.Tensor,
    wgate: torch.Tensor,
    state_cache: torch.Tensor,
    ape: torch.Tensor,
    state_block_table: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    seqused: Optional[torch.Tensor] = None,
    start_pos: Optional[torch.Tensor] = None,
    cmp_ratio: int = 4,
    coff: Optional[int] = 1,
    cache_mode: Optional[int] = 1,
    grad_enabled: Optional[bool] = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    d = wkv.size(0) // coff
    coff_cmp = coff * cmp_ratio
    if x.dim() == 3:
        b = x.size(0)
        s = x.size(1)
        cmp_size = (s + cmp_ratio - 1) // cmp_ratio
        cmp_kv_size = (b, cmp_size, d)
        softmax_score_size = (b, cmp_size, coff_cmp, d)
        kv_size = (b, cmp_size, coff_cmp, d)
    else:
        b_size = cu_seqlens.size(0) - 1
        t = x.size(0)
        cmp_size = min(t, t // cmp_ratio + b_size)
        cmp_kv_size = (cmp_size, d)
        softmax_score_size = (cmp_size, coff_cmp, d)
        kv_size = (cmp_size, coff_cmp, d)

    cmp_kv_out = torch.empty(cmp_kv_size, dtype=x.dtype, device=x.device)
    softmax_score_out = torch.empty(
        softmax_score_size, dtype=torch.float32, device=x.device
    )
    kv_out = torch.empty(kv_size, dtype=torch.float32, device=x.device)
    state_cache_out = torch.empty_like(state_cache)
    return (cmp_kv_out, softmax_score_out, kv_out, state_cache_out)


# ===========================================================================
# Register compressor backward
# ===========================================================================
@torch.library.custom_op(
    "cann_ops_transformer::_compressor_backward", mutates_args=(), device_types="npu"
)
def _compressor_backward(
    d_cmp_kv: torch.Tensor,
    x: torch.Tensor,
    wkv: torch.Tensor,
    wgate: torch.Tensor,
    softmax_score: torch.Tensor,
    kv: torch.Tensor,
    cu_seqlens: Optional[torch.Tensor] = None,
    seqused: Optional[torch.Tensor] = None,
    start_pos: Optional[torch.Tensor] = None,
    cmp_ratio: int = 4,
    coff: Optional[int] = 1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    d_x = torch.empty_like(x)
    d_wkv = torch.empty_like(wkv)
    d_wgate = torch.empty_like(wgate)
    d_ape_size = (cmp_ratio, wkv.size(0))
    d_ape = torch.empty(d_ape_size, dtype=torch.float32, device=x.device)
    return op_module.compressor_backward(
        d_cmp_kv,
        x,
        wkv,
        wgate,
        softmax_score,
        kv,
        cu_seqlens,
        seqused,
        start_pos,
        cmp_ratio,
        coff,
    )


@torch.library.register_fake("cann_ops_transformer::_compressor_backward")
def _compressor_backward_fake(
    d_cmp_kv: torch.Tensor,
    x: torch.Tensor,
    wkv: torch.Tensor,
    wgate: torch.Tensor,
    softmax_score: torch.Tensor,
    kv: torch.Tensor,
    cu_seqlens: Optional[torch.Tensor] = None,
    seqused: Optional[torch.Tensor] = None,
    start_pos: Optional[torch.Tensor] = None,
    cmp_ratio: int = 4,
    coff: Optional[int] = 1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    d_x = torch.empty_like(x)
    d_wkv = torch.empty_like(wkv)
    d_wgate = torch.empty_like(wgate)
    d_ape_size = (cmp_ratio, wkv.size(0))
    d_ape = torch.empty(d_ape_size, dtype=torch.float32, device=x.device)
    return (d_x, d_wkv, d_wgate, d_ape)


# ===========================================================================
# Register AutoGrad
# ===========================================================================
def setup_context(ctx, inputs, output):
    x, wkv, wgate = inputs[:3]
    cu_seqlens, seqused, start_pos, cmp_ratio, coff = inputs[6:11]

    cmp_kv, softmax_score, kv, _ = output
    ctx.save_for_backward(
        x, wkv, wgate, cu_seqlens, seqused, start_pos, softmax_score, kv
    )
    ctx.cmp_ratio = cmp_ratio
    ctx.coff = coff


def backward(ctx, dout, *grads):
    """
    Args:
        dout: d_cmp_kv
    """
    x, wkv, wgate, cu_seqlens, seqused, start_pos, softmax_score, kv = ctx.saved_tensors
    cmp_ratio = ctx.cmp_ratio
    coff = ctx.coff

    d_x, d_wkv, d_wgate, d_ape = _compressor_backward(
        dout,
        x,
        wkv,
        wgate,
        softmax_score,
        kv,
        cu_seqlens,
        seqused,
        start_pos,
        cmp_ratio,
        coff,
    )
    return d_x, d_wkv, d_wgate, None, d_ape, *((None,) * 8)


_compressor_forward.register_autograd(backward, setup_context=setup_context)


def compressor(
    x: torch.Tensor,
    wkv: torch.Tensor,
    wgate: torch.Tensor,
    state_cache: torch.Tensor,
    ape: torch.Tensor,
    cmp_ratio: int = 4,
    *,
    state_block_table: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    seqused: Optional[torch.Tensor] = None,
    start_pos: Optional[torch.Tensor] = None,
    coff: Optional[int] = 1,
    cache_mode: Optional[int] = 1,
) -> Tuple[torch.Tensor]:
    """
    dispatcher implementation for NPU.
    'PrivateUse1' is the combine key for custom NPU backends.
    """
    grad_enabled = x.requires_grad
    cmp_kv, softmax_score, kv, new_state_cache = _compressor_forward(
        x,
        wkv,
        wgate,
        state_cache,
        ape,
        state_block_table,
        cu_seqlens,
        seqused,
        start_pos,
        cmp_ratio,
        coff,
        cache_mode,
        grad_enabled,
    )
    if state_cache is not None:
        state_cache.copy_(new_state_cache)
    return cmp_kv
