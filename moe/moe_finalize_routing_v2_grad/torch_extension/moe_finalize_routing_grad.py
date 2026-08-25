# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
from typing import Optional, Tuple

import torch
from torch.library import impl
from cann_ops_transformer.op_builder import OpBuilder, get_as_library


def _has_value(t):
    if t is None:
        return False
    if isinstance(t, torch.Tensor):
        return t.numel() > 0
    return True


class MoeFinalizeRoutingGradOpBuilder(OpBuilder):
    def __init__(self):
        super(MoeFinalizeRoutingGradOpBuilder, self).__init__(
            "moe_finalize_routing_grad", category="moe"
        )

    def sources(self):
        return ["csrc/moe/moe_finalize_routing_grad.cpp"]

    def schema(self) -> str:
        return (
            "moe_finalize_routing_grad(Tensor grad_y, Tensor expanded_row_idx, "
            "Tensor? expanded_x=None, Tensor? scales=None, Tensor? expert_idx=None, "
            "Tensor? bias=None, int drop_pad_mode=0, int active_num=0, "
            "int expert_num=0, int expert_capacity=0) -> (Tensor, Tensor)"
        )

    def register_meta(self):
        @impl(get_as_library(), self.name, "Meta")
        def moe_finalize_routing_grad_meta(
            grad_y,
            expanded_row_idx,
            expanded_x=None,
            scales=None,
            expert_idx=None,
            bias=None,
            drop_pad_mode=0,
            active_num=0,
            expert_num=0,
            expert_capacity=0,
        ):
            hidden = grad_y.size(1)
            if drop_pad_mode == 1:
                grad_expanded_x_shape = (expert_num, expert_capacity, hidden)
            else:
                dim0 = expanded_row_idx.numel()
                if active_num > 0 and active_num < dim0:
                    dim0 = active_num
                grad_expanded_x_shape = (dim0, hidden)

            scales_dim1 = 1
            scales_dtype = grad_y.dtype
            if _has_value(scales):
                scales_dim1 = scales.size(1)
                scales_dtype = scales.dtype

            grad_expanded_x = grad_y.new_empty(grad_expanded_x_shape)
            grad_scales = grad_y.new_empty(
                (grad_y.size(0), scales_dim1), dtype=scales_dtype
            )
            return (grad_expanded_x, grad_scales)


moe_finalize_routing_grad_op_builder = MoeFinalizeRoutingGradOpBuilder()
moe_finalize_routing_grad_op_builder._ensure_initialized()


@impl(get_as_library(), moe_finalize_routing_grad_op_builder.name, "PrivateUse1")
def _moe_finalize_routing_grad_dispatch(
    grad_y,
    expanded_row_idx,
    expanded_x=None,
    scales=None,
    expert_idx=None,
    bias=None,
    drop_pad_mode=0,
    active_num=0,
    expert_num=0,
    expert_capacity=0,
):
    op_module = moe_finalize_routing_grad_op_builder.load()
    return op_module.moe_finalize_routing_grad(
        grad_y,
        expanded_row_idx,
        expanded_x,
        scales,
        expert_idx,
        bias,
        drop_pad_mode,
        active_num,
        expert_num,
        expert_capacity,
    )


def moe_finalize_routing_grad(
    grad_y: torch.Tensor,
    expanded_row_idx: torch.Tensor,
    expanded_x: Optional[torch.Tensor] = None,
    scales: Optional[torch.Tensor] = None,
    expert_idx: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    drop_pad_mode: int = 0,
    active_num: int = 0,
    expert_num: int = 0,
    expert_capacity: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Gradient of :func:`moe_finalize_routing`, wraps aclnnMoeFinalizeRoutingV2Grad.

    Args:
        grad_y (Tensor): gradient w.r.t. the forward output, shape (R, H).
        expanded_row_idx (Tensor): row index, 1D. Note: the arrangement of
            ``expanded_row_idx`` follows the convention required by the grad op
            (flat index ``row * K + k``); it may differ from the forward layout
            (``row + k * R``). dtype: int32.
        expanded_x (Tensor, optional): forward expanded_x. Defaults to None.
        scales (Tensor, optional): forward scales, shape (R, K). Defaults to None.
        expert_idx (Tensor, optional): forward expert_idx. dtype: int32.
        bias (Tensor, optional): forward bias, shape (E, H). Defaults to None.
        drop_pad_mode (int): same as forward, must be 0 or 1. Defaults to 0.
        active_num (int): max output rows of grad_expanded_x when drop_pad_mode
            is 0; 0 means use expanded_row_idx length. Defaults to 0.
        expert_num (int): expert count, required when drop_pad_mode is 1.
        expert_capacity (int): expert capacity, required when drop_pad_mode is 1.

    Returns:
        Tuple[Tensor, Tensor]: (grad_expanded_x, grad_scales). grad_expanded_x
        has the same dtype as ``grad_y``; grad_scales has the dtype of ``scales``
        when provided, otherwise the dtype of ``grad_y``.
    """
    return torch.ops.cann_ops_transformer.moe_finalize_routing_grad(
        grad_y,
        expanded_row_idx,
        expanded_x,
        scales,
        expert_idx,
        bias,
        drop_pad_mode,
        active_num,
        expert_num,
        expert_capacity,
    )
