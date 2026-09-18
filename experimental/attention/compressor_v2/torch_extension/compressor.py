# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
from typing import Optional

import torch
from torch.library import impl
from cann_ops_transformer.op_builder import OpBuilder


class CompressorOpBuilder(OpBuilder):
    def __init__(self):
        super(CompressorOpBuilder, self).__init__("compressor_v2", category="attention")

    def sources(self):
        """Path to C++ source code."""
        return ["csrc/attention/compressor_v2.cpp"]

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
compressor_op_builder._ensure_initialized()


# ===========================================================================
# Register compressor forward
# ===========================================================================
@torch.library.custom_op(
    "cann_ops_transformer::ds41.compressor", mutates_args=(), device_types="npu"
)
def _compressor_forward(
    x: torch.Tensor,
    wkv: torch.Tensor,
    wgate: torch.Tensor,
    state_cache: torch.Tensor,
    state_block_table: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    seqused: Optional[torch.Tensor] = None,
    start_pos: Optional[torch.Tensor] = None,
    cmp_ratio: int = 4,
) -> torch.Tensor:
    op_module = compressor_op_builder.load()
    return op_module.compressor(
        x,
        wkv,
        wgate,
        state_cache,
        cmp_ratio,
        state_block_table,
        cu_seqlens,
        seqused,
        start_pos,
    )


@torch.library.register_fake("cann_ops_transformer::ds41.compressor")
def _compressor_forward_fake(
    x: torch.Tensor,
    wkv: torch.Tensor,
    wgate: torch.Tensor,
    state_cache: torch.Tensor,
    state_block_table: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    seqused: Optional[torch.Tensor] = None,
    start_pos: Optional[torch.Tensor] = None,
    cmp_ratio: int = 4,
) -> torch.Tensor:
    d = wkv.size(0)
    if x.dim() == 3:
        b = x.size(0)
        s = x.size(1)
        cmp_size = (s + cmp_ratio - 1) // cmp_ratio
        cmp_kv_size = (b, cmp_size, d)
    else:
        b_size = cu_seqlens.size(0) - 1
        t = x.size(0)
        cmp_size = min(t, t // cmp_ratio + b_size)
        cmp_kv_size = (cmp_size, d)

    cmp_kv_out = torch.empty(cmp_kv_size, dtype=x.dtype, device=x.device)
    return cmp_kv_out


def compressor(
    x: torch.Tensor,
    wkv: torch.Tensor,
    wgate: torch.Tensor,
    state_cache: torch.Tensor,
    cmp_ratio: int = 4,
    *,
    state_block_table: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    seqused: Optional[torch.Tensor] = None,
    start_pos: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    dispatcher implementation for NPU.
    """
    return _compressor_forward(
        x,
        wkv,
        wgate,
        state_cache,
        state_block_table,
        cu_seqlens,
        seqused,
        start_pos,
        cmp_ratio,
    )
