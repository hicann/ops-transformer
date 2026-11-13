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

import torch


_GRAPH_OP = None


def _graph_op():
    global _GRAPH_OP
    if _GRAPH_OP is None:
        from cann_ops_transformer.ops.attention import (
            compressor as _real,  # noqa: F401
        )

        _GRAPH_OP = torch.ops.cann_ops_transformer._compressor_forward_graph
    return _GRAPH_OP


class CompressorGraphNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()

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
        cmp_kv, softmax_score, kv = _graph_op()(
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
            False,
        )
        return cmp_kv, state_cache
