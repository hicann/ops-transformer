# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import torch
from torch.library import impl
from cann_ops_transformer.op_builder import OpBuilder, get_as_library


class MoeInitRoutingGradOpBuilder(OpBuilder):
    def __init__(self):
        super(MoeInitRoutingGradOpBuilder, self).__init__(
            "moe_init_routing_grad", category="moe"
        )

    def sources(self):
        return ["csrc/moe/moe_init_routing_grad.cpp"]

    def schema(self) -> str:
        return (
            "moe_init_routing_grad(Tensor grad_expanded_x, Tensor expanded_row_idx, "
            "int top_k, int drop_pad_mode=0, int active_num=0) -> Tensor"
        )

    def register_meta(self):
        @impl(get_as_library(), self.name, "Meta")
        def moe_init_routing_grad_meta(
            grad_expanded_x, expanded_row_idx, top_k, drop_pad_mode=0, active_num=0
        ):
            torch._check(
                top_k > 0, lambda: f"top_k must be greater than 0, but got {top_k}."
            )
            grad_x_dim0 = expanded_row_idx.numel() // top_k
            grad_x_dim1 = (
                grad_expanded_x.size(2)
                if drop_pad_mode == 1
                else grad_expanded_x.size(1)
            )
            return grad_expanded_x.new_empty((grad_x_dim0, grad_x_dim1))


moe_init_routing_grad_op_builder = MoeInitRoutingGradOpBuilder()
moe_init_routing_grad_op_builder._ensure_initialized()


@impl(get_as_library(), moe_init_routing_grad_op_builder.name, "PrivateUse1")
def _moe_init_routing_grad_dispatch(
    grad_expanded_x, expanded_row_idx, top_k, drop_pad_mode=0, active_num=0
):
    op_module = moe_init_routing_grad_op_builder.load()
    return op_module.moe_init_routing_grad(
        grad_expanded_x, expanded_row_idx, top_k, drop_pad_mode, active_num
    )


def moe_init_routing_grad(
    grad_expanded_x: torch.Tensor,
    expanded_row_idx: torch.Tensor,
    top_k: int,
    drop_pad_mode: int = 0,
    active_num: int = 0,
) -> torch.Tensor:
    """Gradient of :func:`moe_init_routing`, wraps aclnnMoeInitRoutingV2Grad.

    Args:
        grad_expanded_x (Tensor): gradient w.r.t. ``expanded_x``, shape
            (active_num, H) in dropless mode or (expert_num, expert_capacity, H)
            in drop mode.
        expanded_row_idx (Tensor): the ``expanded_row_idx`` output from forward,
            shape (R*K,). dtype: int32.
        top_k (int): the K value (expert_idx.shape[1] from forward).
        drop_pad_mode (int): same as forward, 0 or 1.
        active_num (int): same as forward.

    Returns:
        Tensor: ``grad_x`` of shape (R, H), same dtype as ``grad_expanded_x``.
    """
    return torch.ops.cann_ops_transformer.moe_init_routing_grad(
        grad_expanded_x,
        expanded_row_idx,
        top_k,
        drop_pad_mode,
        active_num,
    )
