#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""TTK dtype adapter for KvQuantSparseFlashAttentionV2."""

from typing import Optional

import torch
import torch_npu
from cann_ops_transformer.ops import kv_quant_sparse_flash_attention


def _get_kv_torch_dtype(kv_dtype):
    if kv_dtype is None:
        return None
    name = str(kv_dtype).strip().lower().removeprefix("torch.")
    if name in ("hifloat8", "hifp8"):
        return torch_npu.hifloat8
    return None


def kv_quant_sparse_flash_attention_v2_ttk(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    sparse_indices: torch.Tensor,
    scale_value: float,
    key_quant_mode: int,
    value_quant_mode: int,
    *,
    key_dequant_scale: Optional[torch.Tensor] = None,
    value_dequant_scale: Optional[torch.Tensor] = None,
    block_table: Optional[torch.Tensor] = None,
    actual_seq_lengths_query: Optional[torch.Tensor] = None,
    actual_seq_lengths_kv: Optional[torch.Tensor] = None,
    sinks: Optional[torch.Tensor] = None,
    sparse_block_size: int = 1,
    layout_query: str = "BSND",
    layout_kv: str = "BSND",
    sparse_mode: int = 3,
    pre_tokens: int = (1 << 63) - 1,
    next_tokens: int = (1 << 63) - 1,
    attention_mode: int = 0,
    quant_scale_repo_mode: int = 1,
    tile_size: int = 128,
    rope_head_dim: int = 64,
    key_dtype: Optional[int] = None,
    value_dtype: Optional[int] = None,
    return_softmax_lse: bool = False,
):
    """Call the installed API with explicit key/value dtype semantics."""
    return kv_quant_sparse_flash_attention(
        query,
        key,
        value,
        sparse_indices,
        scale_value,
        key_quant_mode,
        value_quant_mode,
        key_dequant_scale=key_dequant_scale,
        value_dequant_scale=value_dequant_scale,
        block_table=block_table,
        actual_seq_lengths_query=actual_seq_lengths_query,
        actual_seq_lengths_kv=actual_seq_lengths_kv,
        sparse_block_size=sparse_block_size,
        layout_query=layout_query,
        layout_kv=layout_kv,
        sparse_mode=sparse_mode,
        pre_tokens=pre_tokens,
        next_tokens=next_tokens,
        attention_mode=attention_mode,
        quant_scale_repo_mode=quant_scale_repo_mode,
        tile_size=tile_size,
        rope_head_dim=rope_head_dim,
        key_dtype=_get_kv_torch_dtype(key_dtype),
        value_dtype=_get_kv_torch_dtype(value_dtype),
        sinks=sinks,
        return_softmax_lse=return_softmax_lse,
    )
