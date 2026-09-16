# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import os
from typing import Tuple

import torch
from torch.library import impl

from cann_ops_transformer.op_builder.builder import OpBuilder, get_as_library


OP_NAME = "kda_input_proj"
DIM_TWO = 2


def _construct_kda_input_proj_outputs(
    x: torch.Tensor,
    weight_qkv: torch.Tensor,
    weight_beta: torch.Tensor,
    weight_gate: torch.Tensor,
    weight_g: torch.Tensor,
    device,
):
    if x.dim() != DIM_TWO:
        raise ValueError(f"x must be 2D [T, hidden], but got {x.dim()}D.")
    if x.size(0) <= 0 or x.size(1) <= 0:
        raise ValueError("All values within x's shape should be greater than 0.")
    if not (
        weight_qkv.dim() == DIM_TWO
        and weight_beta.dim() == DIM_TWO
        and weight_gate.dim() == DIM_TWO
        and weight_g.dim() == DIM_TWO
    ):
        raise ValueError("weights must be 2D matmul RHS [K, N].")
    t_size = x.size(0)
    qkv = torch.empty((t_size, weight_qkv.size(1)), dtype=torch.bfloat16, device=device)
    beta = torch.empty(
        (t_size, weight_beta.size(1)), dtype=torch.float32, device=device
    )
    gate = torch.empty(
        (t_size, weight_gate.size(1)), dtype=torch.bfloat16, device=device
    )
    g = torch.empty((t_size, weight_g.size(1)), dtype=torch.bfloat16, device=device)
    return qkv, beta, gate, g


class KdaInputProjOpBuilder(OpBuilder):
    def __init__(self):
        super().__init__(OP_NAME, category="attention")

    def sources(self):
        local_cpp = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "csrc", "kda_input_proj.cpp"
        )
        if os.path.isfile(local_cpp):
            return [local_cpp]
        return ["csrc/attention/kda_input_proj.cpp"]

    def schema(self):
        return (
            "kda_input_proj(Tensor x, Tensor weight_qkv, Tensor weight_beta, "
            "Tensor weight_gate, Tensor weight_g, Tensor weight_qkv_scale) "
            "-> (Tensor, Tensor, Tensor, Tensor)"
        )

    def register_meta(self):
        @impl(get_as_library(), self.name, "Meta")
        def kda_input_proj_meta(
            x,
            weight_qkv,
            weight_beta,
            weight_gate,
            weight_g,
            weight_qkv_scale,
        ):
            return _construct_kda_input_proj_outputs(
                x, weight_qkv, weight_beta, weight_gate, weight_g, "meta"
            )


kda_input_proj_op_builder = KdaInputProjOpBuilder()
kda_input_proj_op_builder.ensure_initialized()


@impl(get_as_library(), OP_NAME, "PrivateUse1")
def kda_input_proj(
    x: torch.Tensor,
    weight_qkv: torch.Tensor,
    weight_beta: torch.Tensor,
    weight_gate: torch.Tensor,
    weight_g: torch.Tensor,
    weight_qkv_scale: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """KdaInputProj 前处理：qkv / beta / gate / g 投影，封装 aclnnKdaInputProj。

    权重按公开接口的 matmul 右操作数解释，view shape 为 [K, N]。若末两维是列主序
    view（stride[-2]==1 且 stride[-1]==shape[-2]），aclnn 按转置权重处理。

    Args:
        x (Tensor): 隐藏层输入，shape [T, K]，dtype bfloat16。
        weight_qkv (Tensor): qkv 投影权重，view [K, N_qkv]，dtype float8_e4m3fn。
        weight_beta (Tensor): beta 投影权重，view [K, N_beta]，dtype bfloat16。
        weight_gate (Tensor): gate 投影权重，view [K, N_gate]，dtype bfloat16。
        weight_g (Tensor): g 投影权重，view [K, N_g]，dtype bfloat16。
        weight_qkv_scale (Tensor): qkv MX 量化缩放，dtype float8_e8m0fnu。

    Returns:
        Tuple[Tensor, Tensor, Tensor, Tensor]:
            qkv（bfloat16）、beta（float32）、gate（bfloat16）、g（bfloat16）。
    """
    op_module = kda_input_proj_op_builder.load()
    return op_module.kda_input_proj(
        x, weight_qkv, weight_beta, weight_gate, weight_g, weight_qkv_scale
    )
