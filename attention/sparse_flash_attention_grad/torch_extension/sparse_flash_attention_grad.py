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
from cann_ops_transformer.op_builder import OpBuilder, get_as_library


class SparseFlashAttentionGradOpBuilder(OpBuilder):
    def __init__(self):
        super(SparseFlashAttentionGradOpBuilder, self).__init__(
            "sparse_flash_attention_grad", category="attention"
        )

    def sources(self):
        """Path to C++ source code."""
        return ["csrc/attention/sparse_flash_attention_grad.cpp"]

    def schema(self) -> str:
        """PyTorch operator signature."""
        return (
            "sparse_flash_attention_grad(Tensor query, Tensor key, Tensor? value, Tensor sparse_indices, "
            "Tensor d_out, Tensor out, Tensor softmax_max, Tensor softmax_sum, Tensor? sinks, "
            "float scale_value, int sparse_block_size, "
            "*, Tensor? query_rope=None, Tensor? key_rope=None, Tensor? actual_seq_qlen=None, "
            'Tensor? actual_seq_kvlen=None, str layout="BSND", int sparse_mode=3, '
            "int win_left=9223372036854775807, int win_right=9223372036854775807, int attention_mode=0) "
            "-> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)"
        )

    def register_meta(self):
        """
        Registers the Meta implementation (Shape/Dtype inference).
        Essential for Autograd and FakeTensor support.
        """

        @impl(get_as_library(), self.name, "Meta")
        def sparse_flash_attention_grad_meta(
            query,
            key,
            value,
            sparse_indices,
            d_out,
            out,
            softmax_max,
            softmax_sum,
            sinks=None,
            scale_value=1.0,
            sparse_block_size=1,
            query_rope=None,
            key_rope=None,
            actual_seq_qlen=None,
            actual_seq_kvlen=None,
            layout="BSND",
            sparse_mode=3,
            win_left=9223372036854775807,
            win_right=9223372036854775807,
            attention_mode=0,
        ):
            dq = query.new_empty(query.shape, dtype=query.dtype, device="meta")
            dk = key.new_empty(key.shape, dtype=key.dtype, device="meta")
            # KV merge 场景（value 不传）下 dValue 返回空 meta tensor（shape [0]），
            # 与 C++ 侧空输出张量保持一致；普通场景下 shape/dtype 跟随 value。
            dv = (
                value.new_empty(value.shape, dtype=value.dtype, device="meta")
                if value is not None
                else query.new_empty([0], dtype=query.dtype, device="meta")
            )
            dq_rope = (
                query_rope.new_empty(
                    query_rope.shape, dtype=query_rope.dtype, device="meta"
                )
                if query_rope is not None
                else None
            )
            dk_rope = (
                key_rope.new_empty(key_rope.shape, dtype=key_rope.dtype, device="meta")
                if key_rope is not None
                else None
            )
            d_sinks = (
                sinks.new_empty(sinks.shape, dtype=sinks.dtype, device="meta")
                if sinks is not None
                else None
            )
            return (dq, dk, dv, dq_rope, dk_rope, d_sinks)


# Instantiate the builder (registers schema + meta at import time)
sfag_op_builder = SparseFlashAttentionGradOpBuilder()
sfag_op_builder._ensure_initialized()


@impl(get_as_library(), sfag_op_builder.name, "PrivateUse1")
def sparse_flash_attention_grad(
    query,
    key,
    value,
    sparse_indices,
    d_out,
    out,
    softmax_max,
    softmax_sum,
    sinks=None,
    scale_value=1.0,
    sparse_block_size=1,
    query_rope=None,
    key_rope=None,
    actual_seq_qlen=None,
    actual_seq_kvlen=None,
    layout="BSND",
    sparse_mode=3,
    win_left=9223372036854775807,
    win_right=9223372036854775807,
    attention_mode=0,
):
    """SparseFlashAttentionGrad（SFAG，MLA OSS Sink）反向计算，封装 aclnnSparseFlashAttentionGradV2。

    Args:
        query (Tensor): query 张量，dtype 支持 bfloat16/float16，layout 决定 shape（TND: [total_q, N1, D+Dr]）。
        key (Tensor): key 张量，dtype 与 query 一致（TND: [total_kv, N2, D+Dr]）。
        value (Tensor, optional): value 张量，dtype 与 query 一致（TND: [total_kv, N2, D]）。
            可选项。传入 None 时启用 KV merge，内部按 value=key 处理（arch22/A2 与 Ascend950 均支持；
            KV merge 场景要求 d_value 输出为空 tensor）。
        sparse_indices (Tensor): topk 索引，int32，shape [T1, N2, K]（K=select_block_count）。
        d_out (Tensor): 反向梯度 dy，dtype 与 query 一致，shape 与 out 相同。
        out (Tensor): 前向输出，dtype 与 query 一致，shape 与 query 相同。
        softmax_max (Tensor): 前向 softmax max，float32。
        softmax_sum (Tensor): 前向 softmax sum，float32。
        sinks (Tensor, optional): oss-sink 输入，float32，shape [N1]。
        scale_value (float): query@key 缩放系数，默认 1.0（通常为 1/sqrt(head_dim)）。
        sparse_block_size (int): block 大小，支持 {1,8,16,32,64}，默认 1。
        query_rope (Tensor, optional): query 的 rope 部分，dtype 与 query 一致，仅 layout=TND 时使用。
        key_rope (Tensor, optional): key 的 rope 部分，dtype 与 query 一致，仅 layout=TND 时使用。
        actual_seq_qlen (Tensor, optional): query 实际序列长度，int32，shape [B]。
        actual_seq_kvlen (Tensor, optional): key/value 实际序列长度，int32，shape [B]。
        layout (str): 输入排布，支持 "BSND"/"TND"，默认 "BSND"。
        sparse_mode (int): 稀疏模式，默认 3。
        win_left (int): 左侧窗口 token 数，默认 9223372036854775807（int64 max）。
        win_right (int): 右侧窗口 token 数，默认 9223372036854775807（int64 max）。
        attention_mode (int): 注意力模式占位参数（暂不参与计算），默认 0。

    Returns:
        Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
            dq、dk、dv、dq_rope、dk_rope、d_sinks。
            dq/dk/dv/dq_rope/dk_rope dtype 与对应输入一致；d_sinks 为 float32 shape [N1]。
            可选输入（query_rope/key_rope/sinks）未提供时，对应输出为 shape [0] 的空 tensor；
            KV merge（value=None）时 dv 为 shape [0] 的空 tensor。
    """
    op_module = sfag_op_builder.load()
    return op_module.sparse_flash_attention_grad(
        query,
        key,
        value,
        sparse_indices,
        d_out,
        out,
        softmax_max,
        softmax_sum,
        sinks,
        scale_value,
        sparse_block_size,
        query_rope,
        key_rope,
        actual_seq_qlen,
        actual_seq_kvlen,
        layout,
        sparse_mode,
        win_left,
        win_right,
        attention_mode,
    )


sparse_flash_attention_grad = torch.ops.cann_ops_transformer.sparse_flash_attention_grad
