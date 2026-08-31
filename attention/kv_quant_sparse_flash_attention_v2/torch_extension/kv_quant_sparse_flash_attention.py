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
import torch_npu  # noqa: F401
from torch.library import impl
from cann_ops_transformer.op_builder import OpBuilder, get_as_library

INT64_MAX = 9223372036854775807


class KvQuantSparseFlashAttentionOpBuilder(OpBuilder):
    def __init__(self):
        super().__init__("kv_quant_sparse_flash_attention", category="attention")

    def sources(self):
        return ["csrc/attention/kv_quant_sparse_flash_attention.cpp"]

    def schema(self) -> str:
        return (
            "kv_quant_sparse_flash_attention("
            "Tensor query, Tensor key, Tensor value, Tensor sparse_indices, "
            "float scale_value, int key_quant_mode, int value_quant_mode, *, "
            "Tensor? key_dequant_scale=None, Tensor? value_dequant_scale=None, "
            "Tensor? block_table=None, Tensor? actual_seq_lengths_query=None, "
            "Tensor? actual_seq_lengths_kv=None, "
            "int sparse_block_size=1, "
            'str layout_query="BSND", str layout_kv="BSND", '
            "int sparse_mode=3, int pre_tokens=9223372036854775807, "
            "int next_tokens=9223372036854775807, int attention_mode=0, "
            "int quant_scale_repo_mode=1, int tile_size=128, int rope_head_dim=64, "
            "int? key_dtype=None, int? value_dtype=None, "
            "Tensor? sinks=None, bool return_softmax_lse=False) -> (Tensor, Tensor, Tensor)"
        )

    def register_meta(self):
        @impl(get_as_library(), self.name, "Meta")
        def _meta(
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
            sparse_block_size: int = 1,
            layout_query: str = "BSND",
            layout_kv: str = "BSND",
            sparse_mode: int = 3,
            pre_tokens: int = INT64_MAX,
            next_tokens: int = INT64_MAX,
            attention_mode: int = 0,
            quant_scale_repo_mode: int = 1,
            tile_size: int = 128,
            rope_head_dim: int = 64,
            key_dtype: Optional[int] = None,
            value_dtype: Optional[int] = None,
            sinks: Optional[torch.Tensor] = None,
            return_softmax_lse: bool = False,
        ):
            if query.numel() == 0:
                raise ValueError("The shape size of query should not be 0")
            out_shape = list(query.shape)
            out_shape[-1] -= rope_head_dim
            attn_out = torch.empty(out_shape, dtype=query.dtype, device="meta")
            empty_lse = torch.empty(0, dtype=torch.float32, device="meta")
            if not return_softmax_lse:
                return attn_out, empty_lse, empty_lse
            kv_head_num = (
                key.shape[2]
                if layout_kv == "BSND" or layout_kv == "PA_BSND"
                else key.shape[1]
            )
            g = (
                query.shape[-2] // kv_head_num
                if layout_query == "BSND"
                else query.shape[-2] // kv_head_num
            )
            if layout_query == "BSND":
                lse_shape = [query.shape[0], kv_head_num, query.shape[1], g]
            else:
                lse_shape = [kv_head_num, query.shape[0], g]
            softmax_max = torch.empty(lse_shape, dtype=torch.float32, device="meta")
            softmax_sum = torch.empty(lse_shape, dtype=torch.float32, device="meta")
            return attn_out, softmax_max, softmax_sum


_op_builder = KvQuantSparseFlashAttentionOpBuilder()
_op_builder._ensure_initialized()


@impl(get_as_library(), _op_builder.name, "PrivateUse1")
def kv_quant_sparse_flash_attention(
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
    sparse_block_size: int = 1,
    layout_query: str = "BSND",
    layout_kv: str = "BSND",
    sparse_mode: int = 3,
    pre_tokens: int = INT64_MAX,
    next_tokens: int = INT64_MAX,
    attention_mode: int = 0,
    quant_scale_repo_mode: int = 1,
    tile_size: int = 128,
    rope_head_dim: int = 64,
    key_dtype: Optional[int] = None,
    value_dtype: Optional[int] = None,
    sinks: Optional[torch.Tensor] = None,
    return_softmax_lse: bool = False,
):
    """
    Dispatcher implementation for NPU.
    'PrivateUse1' is the dispatch key for custom NPU backends.
    """
    op_module = _op_builder.load()
    return op_module.npu_kv_quant_sparse_flash_attention(
        query,
        key,
        value,
        sparse_indices,
        key_dequant_scale,
        value_dequant_scale,
        block_table,
        actual_seq_lengths_query,
        actual_seq_lengths_kv,
        scale_value,
        key_quant_mode,
        value_quant_mode,
        sparse_block_size,
        layout_query,
        layout_kv,
        sparse_mode,
        pre_tokens,
        next_tokens,
        attention_mode,
        quant_scale_repo_mode,
        tile_size,
        rope_head_dim,
        key_dtype,
        value_dtype,
        sinks,
        return_softmax_lse,
    )
