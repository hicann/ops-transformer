#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


from typing import Optional, Tuple

import torch


_qc_op_module = None


def _get_quant_compressor_op():
    global _qc_op_module
    if _qc_op_module is not None:
        return _qc_op_module
    from cann_ops_transformer.op_builder.builder import OpBuilder

    class _QuantCompressorBuilder(OpBuilder):
        def __init__(self):
            super().__init__("quant_compressor")

        def sources(self):
            return ["csrc/attention/quant_compressor.cpp"]

        def schema(self):
            pass

        def register_meta(self):
            pass

    _qc_op_module = _QuantCompressorBuilder().load()
    return _qc_op_module


_GRAPH_OP_REGISTERED = False


def _ensure_graph_op_registered():
    global _GRAPH_OP_REGISTERED
    if _GRAPH_OP_REGISTERED:
        return
    _GRAPH_OP_REGISTERED = True

    @torch.library.custom_op(
        "cann_ops_transformer::_quant_compressor_forward_graph",
        mutates_args=("state_cache",),
        device_types="npu",
    )
    def _quant_compressor_forward_graph(
        x: torch.Tensor,
        wkv: torch.Tensor,
        wgate: torch.Tensor,
        state_cache: torch.Tensor,
        ape: torch.Tensor,
        quant_mode: int = 1,
        cmp_ratio: int = 4,
        x_descale: Optional[torch.Tensor] = None,
        wkv_descale: Optional[torch.Tensor] = None,
        wgate_descale: Optional[torch.Tensor] = None,
        state_block_table: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        seqused: Optional[torch.Tensor] = None,
        start_pos: Optional[torch.Tensor] = None,
        coff: Optional[int] = 1,
        cache_mode: Optional[int] = 1,
    ) -> torch.Tensor:
        op = _get_quant_compressor_op()
        return op.quant_compressor(
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

    @torch.library.register_fake(
        "cann_ops_transformer::_quant_compressor_forward_graph"
    )
    def _quant_compressor_forward_graph_fake(
        x: torch.Tensor,
        wkv: torch.Tensor,
        wgate: torch.Tensor,
        state_cache: torch.Tensor,
        ape: torch.Tensor,
        quant_mode: int = 1,
        cmp_ratio: int = 4,
        x_descale: Optional[torch.Tensor] = None,
        wkv_descale: Optional[torch.Tensor] = None,
        wgate_descale: Optional[torch.Tensor] = None,
        state_block_table: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        seqused: Optional[torch.Tensor] = None,
        start_pos: Optional[torch.Tensor] = None,
        coff: Optional[int] = 1,
        cache_mode: Optional[int] = 1,
    ) -> torch.Tensor:
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
        return torch.empty(cmp_kv_size, dtype=torch.bfloat16, device=x.device)


class QuantCompressorGraphNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        _ensure_graph_op_registered()

    def forward(
        self,
        x,
        wkv,
        wgate,
        state_cache,
        ape,
        x_descale=None,
        wkv_descale=None,
        wgate_descale=None,
        state_block_table=None,
        cu_seqlens=None,
        seqused=None,
        start_pos=None,
        quant_mode=1,
        cmp_ratio=4,
        coff=1,
        cache_mode=1,
    ):
        cmp_kv = torch.ops.cann_ops_transformer._quant_compressor_forward_graph(
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
        return cmp_kv, state_cache
