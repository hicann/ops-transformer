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


class CausalConv1dFnOpBuilder(OpBuilder):
    def __init__(self):
        super(CausalConv1dFnOpBuilder, self).__init__(
            "causal_conv1d_fn", category="mamba"
        )

    def sources(self):
        """Path to C++ source code."""
        return ["csrc/mamba/causal_conv1d_fn.cpp"]

    def schema(self) -> str:
        """PyTorch operator schema."""
        return (
            "causal_conv1d_fn("
            "Tensor x, Tensor weight, Tensor? bias, "
            "Tensor(a!) conv_states, "
            "Tensor query_start_loc, "
            "*, "
            "Tensor? cache_indices=None, "
            "Tensor? has_initial_state=None, "
            'str activation="silu", '
            "int pad_slot_id=-1, "
            "int null_block_id=0, "
            "Tensor? block_idx_first_scheduled_token=None, "
            "Tensor? block_idx_last_scheduled_token=None, "
            "Tensor? initial_state_idx=None, "
            "Tensor? num_computed_tokens=None, "
            "int block_size_to_align=0"
            ") -> Tensor"
        )

    def register_meta(self):
        """Register Meta implementation for shape/dtype inference."""

        @impl(get_as_library(), self.name, "Meta")
        def causal_conv1d_fn_meta(
            x: torch.Tensor,
            weight: torch.Tensor,
            bias: Optional[torch.Tensor],
            conv_states: torch.Tensor,
            query_start_loc: torch.Tensor,
            *,
            cache_indices: Optional[torch.Tensor] = None,
            has_initial_state: Optional[torch.Tensor] = None,
            activation: str = "silu",
            pad_slot_id: int = -1,
            null_block_id: int = 0,
            block_idx_first_scheduled_token: Optional[torch.Tensor] = None,
            block_idx_last_scheduled_token: Optional[torch.Tensor] = None,
            initial_state_idx: Optional[torch.Tensor] = None,
            num_computed_tokens: Optional[torch.Tensor] = None,
            block_size_to_align: int = 0,
        ) -> torch.Tensor:
            y = torch.empty_like(x)
            return y


_causal_conv1d_fn_op_builder = CausalConv1dFnOpBuilder()
_causal_conv1d_fn_op_builder._ensure_initialized()


@impl(get_as_library(), _causal_conv1d_fn_op_builder.name, "PrivateUse1")
def _causal_conv1d_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    conv_states: torch.Tensor,
    query_start_loc: torch.Tensor,
    *,
    cache_indices: Optional[torch.Tensor] = None,
    has_initial_state: Optional[torch.Tensor] = None,
    activation: str = "silu",
    pad_slot_id: int = -1,
    null_block_id: int = 0,
    block_idx_first_scheduled_token: Optional[torch.Tensor] = None,
    block_idx_last_scheduled_token: Optional[torch.Tensor] = None,
    initial_state_idx: Optional[torch.Tensor] = None,
    num_computed_tokens: Optional[torch.Tensor] = None,
    block_size_to_align: int = 0,
) -> torch.Tensor:
    _op_module = _causal_conv1d_fn_op_builder.load()
    return _op_module.causal_conv1d_fn(
        x,
        weight,
        bias,
        conv_states,
        query_start_loc,
        cache_indices,
        has_initial_state,
        activation,
        pad_slot_id,
        null_block_id,
        block_idx_first_scheduled_token,
        block_idx_last_scheduled_token,
        initial_state_idx,
        num_computed_tokens,
        block_size_to_align,
    )


def causal_conv1d_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    conv_states: torch.Tensor,
    query_start_loc: torch.Tensor,
    *,
    cache_indices: Optional[torch.Tensor] = None,
    has_initial_state: Optional[torch.Tensor] = None,
    activation: str = "silu",
    pad_slot_id: int = -1,
    null_block_id: int = 0,
    block_idx_first_scheduled_token: Optional[torch.Tensor] = None,
    block_idx_last_scheduled_token: Optional[torch.Tensor] = None,
    initial_state_idx: Optional[torch.Tensor] = None,
    num_computed_tokens: Optional[torch.Tensor] = None,
    block_size_to_align: int = 0,
) -> torch.Tensor:
    """因果一维卷积前向计算（prefill / chunk-prefill），封装 aclnnCausalConv1dFn。

    沿序列维度使用缓存数据（长度为卷积核宽减1）对各序列头部进行 padding，
    确保输出依赖当前及历史输入；卷积完成后可选施加 SiLU 激活；
    计算完成后将当前序列部分数据更新到缓存。

    Args:
        x (Tensor): 输入序列，shape 为 [batch, seq_len, dim]（固定 batch）或 [cu_seq_len, dim]（变长），
            dtype 支持 float16/bfloat16。
        weight (Tensor): 卷积权重，shape 为 [kW, dim]，其中 kW∈{2, 3, 4}，dtype 与 x 一致。
        bias (Tensor, optional): 偏置，shape 为 [dim]，dtype 与 x 一致，None 表示不使用。
        conv_states (Tensor): 卷积状态缓存，shape 为 [num_cache_lines, state_len, dim]，
            dtype 与 x 一致。计算后原地更新。
        query_start_loc (Tensor): 变长序列起始位置索引，shape 为 [batch+1]，int32。
            变长场景必须提供有效值，固定 batch 场景可传空 Tensor。
        cache_indices (Tensor, optional): 缓存索引，shape 为 [batch]，int32，
            默认 None 使用恒等映射。
        has_initial_state (Tensor, optional): 初始状态标志，shape 为 [batch]，int32。
            1=使用缓存历史，0=零初始化（冷启动），默认 None 全部零初始化。
        activation (str): 激活函数类型，"silu" 或 "none"，默认 "silu"。
        pad_slot_id (int): padding slot id，默认 -1。
        null_block_id (int): 无效缓存槽位标记 ID，cache_indices[i]==null_block_id 时跳过该序列，默认 0。
        block_idx_first_scheduled_token (Tensor, optional): 首个调度 token 块索引，shape 为 [batch]，int32，默认 None。
        block_idx_last_scheduled_token (Tensor, optional): 最后调度 token 块索引，shape 为 [batch]，int32，默认 None。
        initial_state_idx (Tensor, optional): 初始状态索引，shape 为 [batch]，int32，默认 None。
        num_computed_tokens (Tensor, optional): 各序列已完成的 token 数，shape 为 [batch]，int32，默认 None。
        block_size_to_align (int): 缓存状态对齐的块大小，默认 0。

    Returns:
        Tensor: 卷积输出 y，shape 与 x 一致，dtype 与 x 一致。
    """
    return _causal_conv1d_fn(
        x,
        weight,
        bias,
        conv_states,
        query_start_loc,
        cache_indices=cache_indices,
        has_initial_state=has_initial_state,
        activation=activation,
        pad_slot_id=pad_slot_id,
        null_block_id=null_block_id,
        block_idx_first_scheduled_token=block_idx_first_scheduled_token,
        block_idx_last_scheduled_token=block_idx_last_scheduled_token,
        initial_state_idx=initial_state_idx,
        num_computed_tokens=num_computed_tokens,
        block_size_to_align=block_size_to_align,
    )
