# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
from enum import IntEnum
from typing import Optional
import torch
from torch.library import impl
from cann_ops_transformer.op_builder import OpBuilder, get_as_library


class QuantMode(IntEnum):
    A8W8_A_HIFP8_PER_TENSOR_W_HIFP8_PER_CHANNEL = 1


class CacheMode(IntEnum):
    LINEAR_BUFFER = 1
    RING_BUFFER = 2


class QuantCompressorOpBuilder(OpBuilder):
    def __init__(self):
        super(QuantCompressorOpBuilder, self).__init__(
            "quant_compressor", category="attention"
        )

    def sources(self):
        """Path to C++ source code."""
        return ["csrc/attention/quant_compressor.cpp"]

    def schema(self):
        """PyTorch operator signature."""
        return (
            "quant_compressor(Tensor x, Tensor wkv, Tensor wgate, Tensor(a!) state_cache, "
            "Tensor ape, "
            "int quant_mode, int cmp_ratio, *, "
            "Tensor? x_descale=None, Tensor? wkv_descale=None, Tensor? wgate_descale=None, "
            "Tensor? state_block_table=None, Tensor? cu_seqlens=None, "
            "Tensor? seqused=None, Tensor? start_pos=None, "
            "int coff=1, int cache_mode=1) -> Tensor"
        )

    def register_meta(self):
        """
        Registers the Meta implementation (Shape/Dtype inference).
        Essential for Autograd and FakeTensor support.
        """

        @impl(get_as_library(), self.name, "Meta")
        def quant_compressor_meta(
            x,
            wkv,
            wgate,
            state_cache,
            ape,
            quant_mode,
            cmp_ratio,
            *,
            x_descale=None,
            wkv_descale=None,
            wgate_descale=None,
            state_block_table=None,
            cu_seqlens=None,
            seqused=None,
            start_pos=None,
            coff=1,
            cache_mode=CacheMode.LINEAR_BUFFER,
        ):
            d = wkv.size(0) // coff
            if x.dim() == 3:
                b = x.size(0)
                s = x.size(1)
                sr = (s + cmp_ratio - 1) // cmp_ratio
                cmp_kv_size = (b, sr, d)
            else:
                t = x.size(0)
                b_size = cu_seqlens.size(0) - 1
                sr = min(t, t // cmp_ratio + b_size)
                cmp_kv_size = (sr, d)

            return torch.empty(cmp_kv_size, dtype=torch.bfloat16, device="meta")


quant_compressor_op_builder = QuantCompressorOpBuilder()
quant_compressor_op_builder._ensure_initialized()


@impl(get_as_library(), quant_compressor_op_builder.name, "PrivateUse1")
def quant_compressor(
    x: torch.Tensor,
    wkv: torch.Tensor,
    wgate: torch.Tensor,
    state_cache: torch.Tensor,
    ape: torch.Tensor,
    quant_mode: QuantMode,
    cmp_ratio: int,
    *,
    x_descale: Optional[torch.Tensor] = None,
    wkv_descale: Optional[torch.Tensor] = None,
    wgate_descale: Optional[torch.Tensor] = None,
    state_block_table: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    seqused: Optional[torch.Tensor] = None,
    start_pos: Optional[torch.Tensor] = None,
    coff: Optional[int] = 1,
    cache_mode: CacheMode = CacheMode.LINEAR_BUFFER,
) -> torch.tensor:
    """
    dispatcher implementation for NPU.
    'PrivateUse1' is the combine key for custom NPU backends.
    """
    op_module = quant_compressor_op_builder.load()
    return op_module.quant_compressor(
        x,
        wkv,
        wgate,
        state_cache,
        ape,
        quant_mode,
        cmp_ratio,
        x_descale,
        wkv_descale,
        wgate_descale,
        state_block_table,
        cu_seqlens,
        seqused,
        start_pos,
        coff,
        cache_mode,
    )


quant_compressor.QuantMode = QuantMode
quant_compressor.CacheMode = CacheMode
