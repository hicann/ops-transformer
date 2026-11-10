# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
from typing import List, Optional

import torch
import torch_npu  # noqa: F401
from torch.library import impl

from cann_ops_transformer.op_builder import OpBuilder, get_as_library


def _check_quant_mode(query: torch.Tensor, quant_mode: Optional[int]) -> None:
    if quant_mode is None:
        return  # Compatibility: infer the mode from dtype.
    if quant_mode not in (0, 5):
        raise ValueError(
            "quant_mode only supports 0 (FP16/BF16) or 5 (HiF8 per-tensor)"
        )
    valid_dtype = (
        query.dtype == torch.uint8
        if quant_mode == 5
        else query.dtype in (torch.float16, torch.bfloat16)
    )
    if not valid_dtype:
        raise ValueError(
            "quant_mode must match query dtype: 0=FP16/BF16, 5=HiF8-encoded uint8"
        )


class BlockSparseAttentionGradOpBuilder(OpBuilder):
    def __init__(self):
        super().__init__("block_sparse_attention_grad")

    def sources(self):
        # setup.py 会把 csrc 收敛到包内 csrc/attention/ 下（wheel 内相对包根的路径）
        return ["csrc/attention/block_sparse_attention_grad.cpp"]

    def schema(self) -> str:
        return (
            "block_sparse_attention_grad("
            "Tensor d_out, Tensor query, Tensor key, Tensor value, "
            "Tensor attention_out, Tensor softmax_lse, Tensor block_sparse_mask, "
            "int[]? block_shape=None, *, "
            'str q_input_layout="BNSD", '
            'str kv_input_layout="BNSD", '
            "int num_key_value_heads=1, "
            "float scale_value=1.0, "
            "int[]? actual_seq_lengths=None, "
            "int[]? actual_seq_lengths_kv=None, "
            "Tensor? atten_mask=None, "
            "int mask_type=0, "
            "Tensor? q_dequant_scale=None, "
            "Tensor? k_dequant_scale=None, "
            "Tensor? v_dequant_scale=None, "
            "Tensor? do_dequant_scale=None, "
            "Tensor? p_quant_scale=None, "
            "Tensor? ds_quant_scale=None, "
            "int? quant_mode=None"
            ") -> (Tensor, Tensor, Tensor)"
        )

    def register_meta(self):
        @impl(get_as_library(), self.name, "Meta")
        def _meta(
            d_out: torch.Tensor,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            attention_out: torch.Tensor,
            softmax_lse: torch.Tensor,
            block_sparse_mask: torch.Tensor,
            block_shape: Optional[List[int]] = None,
            *,
            q_input_layout: str = "BNSD",
            kv_input_layout: str = "BNSD",
            num_key_value_heads: int = 1,
            scale_value: float = 1.0,
            actual_seq_lengths: Optional[List[int]] = None,
            actual_seq_lengths_kv: Optional[List[int]] = None,
            atten_mask: Optional[torch.Tensor] = None,
            mask_type: int = 0,
            q_dequant_scale: Optional[torch.Tensor] = None,
            k_dequant_scale: Optional[torch.Tensor] = None,
            v_dequant_scale: Optional[torch.Tensor] = None,
            do_dequant_scale: Optional[torch.Tensor] = None,
            p_quant_scale: Optional[torch.Tensor] = None,
            ds_quant_scale: Optional[torch.Tensor] = None,
            quant_mode: Optional[int] = None,
        ):
            _check_quant_mode(query, quant_mode)
            grad_dtype = torch.bfloat16 if query.dtype == torch.uint8 else query.dtype
            dq = query.new_empty(query.shape, dtype=grad_dtype)
            dk = key.new_empty(key.shape, dtype=grad_dtype)
            dv = value.new_empty(value.shape, dtype=grad_dtype)
            return dq, dk, dv


_op_builder = BlockSparseAttentionGradOpBuilder()
_op_module = _op_builder.load()


@impl(get_as_library(), _op_builder.name, "PrivateUse1")
def block_sparse_attention_grad(
    d_out: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_out: torch.Tensor,
    softmax_lse: torch.Tensor,
    block_sparse_mask: torch.Tensor,
    block_shape: Optional[List[int]] = None,
    *,
    q_input_layout: str = "BNSD",
    kv_input_layout: str = "BNSD",
    num_key_value_heads: int = 1,
    scale_value: float = 1.0,
    actual_seq_lengths: Optional[List[int]] = None,
    actual_seq_lengths_kv: Optional[List[int]] = None,
    atten_mask: Optional[torch.Tensor] = None,
    mask_type: int = 0,
    q_dequant_scale: Optional[torch.Tensor] = None,
    k_dequant_scale: Optional[torch.Tensor] = None,
    v_dequant_scale: Optional[torch.Tensor] = None,
    do_dequant_scale: Optional[torch.Tensor] = None,
    p_quant_scale: Optional[torch.Tensor] = None,
    ds_quant_scale: Optional[torch.Tensor] = None,
    quant_mode: Optional[int] = None,
):
    """block sparse attention backward（aclnnBlockSparseAttentionGrad 的 torch 扩展封装）。

    mask_type=0 → atten_mask 必须为 None（无 counts）；
    mask_type=1 → atten_mask 必传（INT32 counts：4D [B,N,>=maxBlocks,2]）。
    quant_mode=0 为非量化，5 为 HiF8 per-tensor；None 保留按 dtype 推断。
    返回 (dq, dk, dv)。
    """
    _check_quant_mode(query, quant_mode)
    return _op_module.npu_block_sparse_attention_backward(
        d_out,
        query,
        key,
        value,
        attention_out,
        softmax_lse,
        block_sparse_mask,
        block_shape,
        q_input_layout,
        kv_input_layout,
        num_key_value_heads,
        scale_value,
        actual_seq_lengths,
        actual_seq_lengths_kv,
        atten_mask,
        mask_type,
        q_dequant_scale,
        k_dequant_scale,
        v_dequant_scale,
        do_dequant_scale,
        p_quant_scale,
        ds_quant_scale,
        quant_mode,
    )
