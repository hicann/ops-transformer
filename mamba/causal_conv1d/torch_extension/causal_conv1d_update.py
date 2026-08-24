# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ---------------------------------------------------------------------------
from typing import Optional

import torch
from torch.library import impl
from cann_ops_transformer.op_builder import OpBuilder, get_as_library


class CausalConv1dUpdateOpBuilder(OpBuilder):
    def __init__(self):
        super(CausalConv1dUpdateOpBuilder, self).__init__(
            "causal_conv1d_update", category="mamba"
        )

    def sources(self):
        """Path to C++ source code."""
        return ["csrc/mamba/causal_conv1d_update.cpp"]

    def schema(self) -> str:
        """PyTorch operator schema."""
        return (
            "causal_conv1d_update("
            "Tensor x, Tensor(a!) conv_state, Tensor weight, "
            "*, "
            "Tensor? bias=None, "
            'str activation="silu", '
            "Tensor? conv_state_indices=None, "
            "Tensor? num_accepted_tokens=None, "
            "Tensor? query_start_loc=None, "
            "int max_query_len=-1, "
            "int null_block_id=0, "
            "Tensor? block_idx_last_scheduled_token=None, "
            "Tensor? initial_state_idx=None"
            ") -> Tensor"
        )

    def register_meta(self):
        """Register Meta implementation for shape/dtype inference."""

        @impl(get_as_library(), self.name, "Meta")
        def causal_conv1d_update_meta(
            x: torch.Tensor,
            conv_state: torch.Tensor,
            weight: torch.Tensor,
            *,
            bias: Optional[torch.Tensor] = None,
            activation: str = "silu",
            conv_state_indices: Optional[torch.Tensor] = None,
            num_accepted_tokens: Optional[torch.Tensor] = None,
            query_start_loc: Optional[torch.Tensor] = None,
            max_query_len: int = -1,
            null_block_id: int = 0,
            block_idx_last_scheduled_token: Optional[torch.Tensor] = None,
            initial_state_idx: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            y = torch.empty_like(x)
            return y


_causal_conv1d_update_op_builder = CausalConv1dUpdateOpBuilder()
_causal_conv1d_update_op_builder._ensure_initialized()


@impl(get_as_library(), _causal_conv1d_update_op_builder.name, "PrivateUse1")
def _causal_conv1d_update(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    *,
    bias: Optional[torch.Tensor] = None,
    activation: str = "silu",
    conv_state_indices: Optional[torch.Tensor] = None,
    num_accepted_tokens: Optional[torch.Tensor] = None,
    query_start_loc: Optional[torch.Tensor] = None,
    max_query_len: int = -1,
    null_block_id: int = 0,
    block_idx_last_scheduled_token: Optional[torch.Tensor] = None,
    initial_state_idx: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    _op_module = _causal_conv1d_update_op_builder.load()
    return _op_module.causal_conv1d_update(
        x,
        conv_state,
        weight,
        bias,
        activation,
        conv_state_indices,
        num_accepted_tokens,
        query_start_loc,
        max_query_len,
        null_block_id,
        block_idx_last_scheduled_token,
        initial_state_idx,
    )


def causal_conv1d_update(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    *,
    bias: Optional[torch.Tensor] = None,
    activation: str = "silu",
    conv_state_indices: Optional[torch.Tensor] = None,
    num_accepted_tokens: Optional[torch.Tensor] = None,
    query_start_loc: Optional[torch.Tensor] = None,
    max_query_len: int = -1,
    null_block_id: int = 0,
    block_idx_last_scheduled_token: Optional[torch.Tensor] = None,
    initial_state_idx: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """因果一维卷积状态更新（decode / update），封装 aclnnCausalConv1dUpdate。

    沿序列维度使用缓存数据对各序列头部进行 padding，确保输出依赖当前及历史输入；
    卷积完成后可选施加 SiLU 激活；计算完成后将当前序列部分数据更新到缓存。

    Args:
        x (Tensor): 输入序列，shape 为 [batch, 1, dim]（固定 batch）或 [cu_seq_len, dim]（变长），
            dtype 支持 float16/bfloat16。
        conv_state (Tensor): 卷积状态缓存，shape 为 [num_cache_lines, state_len, dim]，
            dtype 与 x 一致。计算后原地更新。
        weight (Tensor): 卷积权重，shape 为 [kW, dim]，其中 kW∈{2, 3, 4}，dtype 与 x 一致。
        bias (Tensor, optional): 偏置，shape 为 [dim]，dtype 与 x 一致，默认 None 表示不使用。
        activation (str): 激活函数类型，"silu" 或 "none"，默认 "silu"。
        conv_state_indices (Tensor, optional): 缓存索引，shape 为 [batch]，int32，
            默认 None 使用恒等映射。
        num_accepted_tokens (Tensor, optional): 投机解码中每个 batch 已接受的 token 数，
            shape 为 [batch]，int32，仅 kW=4 支持，默认 None。
        query_start_loc (Tensor, optional): 变长序列起始位置索引，shape 为 [batch+1]，int32，默认 None。
            提供时，batch 大小由 query_start_loc.size(0)-1 决定，而非由 x 的 shape 推导。
        max_query_len (int): 最大查询长度，-1 表示不限制，默认 -1。
        null_block_id (int): 无效缓存槽位标记 ID，conv_state_indices[i]==null_block_id 时跳过该序列，默认 0。
        block_idx_last_scheduled_token (Tensor, optional): 最后调度 token 块索引，shape 为 [batch]，int32，默认 None。
        initial_state_idx (Tensor, optional): 初始状态索引，shape 为 [batch]，int32，默认 None。

    Returns:
        Tensor: 卷积输出 y，shape 与 x 一致，dtype 与 x 一致。
    """
    return _causal_conv1d_update(
        x,
        conv_state,
        weight,
        bias=bias,
        activation=activation,
        conv_state_indices=conv_state_indices,
        num_accepted_tokens=num_accepted_tokens,
        query_start_loc=query_start_loc,
        max_query_len=max_query_len,
        null_block_id=null_block_id,
        block_idx_last_scheduled_token=block_idx_last_scheduled_token,
        initial_state_idx=initial_state_idx,
    )
