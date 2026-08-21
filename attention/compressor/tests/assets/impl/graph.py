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


_compressor_op_module = None


def _get_compressor_op():
    global _compressor_op_module
    if _compressor_op_module is not None:
        return _compressor_op_module
    from cann_ops_transformer.op_builder.builder import OpBuilder

    class _CompressorBuilder(OpBuilder):
        def __init__(self):
            super().__init__("compressor")

        def sources(self):
            return ["ops/csrc/compressor.cpp"]

        def schema(self):
            pass

        def register_meta(self):
            pass

    _compressor_op_module = _CompressorBuilder().load()
    return _compressor_op_module


_GRAPH_OP_REGISTERED = False


def _ensure_graph_op_registered():
    global _GRAPH_OP_REGISTERED
    if _GRAPH_OP_REGISTERED:
        return
    _GRAPH_OP_REGISTERED = True

    @torch.library.custom_op(
        "cann_ops_transformer::_compressor_forward_graph",
        mutates_args=("state_cache",),
        device_types="npu",
    )
    def _compressor_forward_graph(
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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        op = _get_compressor_op()
        return op.compressor(
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

    @torch.library.register_fake("cann_ops_transformer::_compressor_forward_graph")
    def _compressor_forward_graph_fake(
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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        return (cmp_kv_out, softmax_score_out, kv_out)


class CompressorGraphNetwork(torch.nn.Module):
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
        state_block_table=None,
        cu_seqlens=None,
        seqused=None,
        start_pos=None,
        cmp_ratio=4,
        coff=1,
        cache_mode=1,
    ):
        cmp_kv, softmax_score, kv = (
            torch.ops.cann_ops_transformer._compressor_forward_graph(
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
                False,
            )
        )
        return cmp_kv, state_cache
