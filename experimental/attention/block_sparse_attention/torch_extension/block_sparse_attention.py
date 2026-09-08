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
from typing import List, Optional
import torch
import torch_npu  # noqa: F401
from torch.library import impl
from cann_ops_transformer.op_builder import OpBuilder, get_as_library


class QuantMode(IntEnum):
    NO_QUANT = 0
    FP8_QUANT = 1
    MXFP4_OCP_QUANT = 2
    MXFP4_CX_QUANT = 3


class BlockSparseAttentionOpBuilder(OpBuilder):
    def __init__(self):
        super().__init__("block_sparse_attention")

    def sources(self):
        return ["csrc/attention/block_sparse_attention.cpp"]

    def schema(self) -> str:
        return (
            "block_sparse_attention("
            "Tensor query, Tensor key, Tensor value, "
            "Tensor block_sparse_mask, "
            "int[] block_shape, *, "
            'str q_input_layout="TND", '
            'str kv_input_layout="TND", '
            "int num_key_value_heads=1, "
            "float scale_value=1.0, "
            "int inner_precise=1, "
            "int[]? actual_seq_lengths=None, "
            "int[]? actual_seq_lengths_kv=None, "
            "bool return_softmax_lse=False, "
            "int mask_type=0, "
            "int quant_mode=0, "
            "int block_size=0, "
            "int pre_tokens=2147483647, "
            "int next_tokens=2147483647, "
            "float dst_type_max=0.0, "
            "Tensor? atten_mask=None, "
            "Tensor? block_table=None, "
            "Tensor? q_dequant_scale=None, "
            "Tensor? k_dequant_scale=None, "
            "Tensor? v_dequant_scale=None, "
            "Tensor? p_quant_scale=None, "
            "ScalarType? attention_out_dtype=None"
            ") -> (Tensor, Tensor)"
        )

    def register_meta(self):
        @impl(get_as_library(), self.name, "Meta")
        def _meta(
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            block_sparse_mask: torch.Tensor,
            block_shape: List[int],
            *,
            q_input_layout: str = "TND",
            kv_input_layout: str = "TND",
            num_key_value_heads: int = 1,
            scale_value: float = 1.0,
            inner_precise: int = 1,
            actual_seq_lengths: Optional[List[int]] = None,
            actual_seq_lengths_kv: Optional[List[int]] = None,
            return_softmax_lse: bool = False,
            mask_type: int = 0,
            quant_mode: int = 0,
            block_size: int = 0,
            pre_tokens: int = 2147483647,
            next_tokens: int = 2147483647,
            dst_type_max: float = 0.0,
            atten_mask: Optional[torch.Tensor] = None,
            block_table: Optional[torch.Tensor] = None,
            q_dequant_scale: Optional[torch.Tensor] = None,
            k_dequant_scale: Optional[torch.Tensor] = None,
            v_dequant_scale: Optional[torch.Tensor] = None,
            p_quant_scale: Optional[torch.Tensor] = None,
            attention_out_dtype: Optional[torch.dtype] = None,
        ):
            # 与 C++ 实现保持一致：quant_mode != 0 时 attention_out_dtype 必填，
            # 否则 meta 推断的 dtype 会与实际输出不一致（错误迟到运行时才抛出）
            if attention_out_dtype is not None:
                out_dtype = attention_out_dtype
            elif quant_mode == QuantMode.NO_QUANT:
                out_dtype = query.dtype
            else:
                raise ValueError(
                    "attention_out_dtype must be specified when quant_mode != 0"
                )
            # FP4: two values pack into one uint8 byte, so last dim is 2*D
            if query.dtype == torch.uint8 and quant_mode in (
                QuantMode.MXFP4_OCP_QUANT,
                QuantMode.MXFP4_CX_QUANT,
            ):
                attn_shape = list(query.shape)
                attn_shape[-1] *= 2
                attn = query.new_empty(tuple(attn_shape), dtype=out_dtype)
            else:
                attn = query.new_empty(query.shape, dtype=out_dtype)
            if return_softmax_lse:
                if q_input_layout == "TND":
                    lse_shape = (query.size(0), query.size(1), 1)
                else:
                    lse_shape = (query.size(0), query.size(1), query.size(2), 1)
                lse = query.new_empty(lse_shape, dtype=torch.float32)
            else:
                lse = query.new_empty((0,), dtype=torch.float32)
            return attn, lse


_op_builder = BlockSparseAttentionOpBuilder()
_op_module = _op_builder.load()


@impl(get_as_library(), _op_builder.name, "PrivateUse1")
def block_sparse_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    block_sparse_mask: torch.Tensor,
    block_shape: List[int],
    *,
    q_input_layout: str = "TND",
    kv_input_layout: str = "TND",
    num_key_value_heads: int = 1,
    scale_value: float = 1.0,
    inner_precise: int = 1,
    actual_seq_lengths: Optional[List[int]] = None,
    actual_seq_lengths_kv: Optional[List[int]] = None,
    return_softmax_lse: bool = False,
    mask_type: int = 0,
    quant_mode: int = 0,
    block_size: int = 0,
    pre_tokens: int = 2147483647,
    next_tokens: int = 2147483647,
    dst_type_max: float = 0.0,
    atten_mask: Optional[torch.Tensor] = None,
    block_table: Optional[torch.Tensor] = None,
    q_dequant_scale: Optional[torch.Tensor] = None,
    k_dequant_scale: Optional[torch.Tensor] = None,
    v_dequant_scale: Optional[torch.Tensor] = None,
    p_quant_scale: Optional[torch.Tensor] = None,
    attention_out_dtype: Optional[torch.dtype] = None,
):
    return _op_module.npu_block_sparse_attention(
        query,
        key,
        value,
        block_sparse_mask,
        block_shape,
        q_input_layout,
        kv_input_layout,
        num_key_value_heads,
        scale_value,
        inner_precise,
        actual_seq_lengths,
        actual_seq_lengths_kv,
        return_softmax_lse,
        mask_type,
        quant_mode,
        block_size,
        pre_tokens,
        next_tokens,
        dst_type_max,
        atten_mask,
        block_table,
        q_dequant_scale,
        k_dequant_scale,
        v_dequant_scale,
        p_quant_scale,
        attention_out_dtype,
    )
