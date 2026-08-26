#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""同一算子在四种测试路径下的 golden 编写。

Kernel/GEIR 的 golden 收到 numpy.ndarray，需手动转 torch 计算后转回 numpy；
ACLNN/E2E 的 golden 直接收到 torch.Tensor，无需转换。
"""

import numpy as np
import torch


def _moe_gating_top_k_backward_torch(
    x_norm, grad_y, expert_idx, routed_scaling_factor=1.0, eps=1e-20
):
    """
    PyTorch 实现 MoE Gating Top-K 反向传播的参考计算。

    参数均为 torch.Tensor（expert_idx 为整数型），返回 grad_x（与 x_norm 同形状）。
    与 kernel 保持一致：中间计算在 fp32 上进行，最终结果转为 grad_y 的 dtype。
    """
    x_f32 = x_norm.to(torch.float32)
    grad_y_f32 = grad_y.to(torch.float32)
    idx = expert_idx.to(torch.int64)
    # 根据选中的专家索引收集对应的 x_norm 值
    selected = torch.gather(x_f32, dim=1, index=idx)  # shape (token_count, k)
    denominator = selected.sum(dim=1, keepdim=True) + eps
    weights = selected / denominator
    grad_y_scaled = grad_y_f32 * routed_scaling_factor
    beta = (weights * grad_y_scaled).sum(dim=1, keepdim=True)
    grad_selected = (grad_y_scaled - beta) / denominator

    grad_norm = torch.zeros_like(x_f32)
    # 使用 scatter_add_ 将梯度分散到对应专家位置（支持多选，若索引重复则累加）
    grad_norm.scatter_add_(dim=1, index=idx, src=grad_selected)
    grad_x = x_f32 * (1 - x_f32) * grad_norm
    return grad_x.to(grad_y.dtype)


class MoeGatingTopKBackwardKernelSpec:
    """Kernel / GEIR 流程 — golden 收到 numpy.ndarray，third_party 收到 torch.Tensor"""

    @staticmethod
    def golden(
        x_norm,
        grad_y,
        expert_idx,
        renorm=0,
        norm_type=0,
        routed_scaling_factor=1.0,
        eps=1e-20,
        **_unused,
    ):
        # bf16 在 numpy 中是 ml_dtypes.bfloat16，torch.from_numpy 无法直接转换，
        # 统一先转 fp32 再入 torch（与旧 kernel golden 一致）
        x_t = torch.from_numpy(x_norm.astype(np.float32))
        g_t = torch.from_numpy(grad_y.astype(np.float32))
        i_t = torch.from_numpy(expert_idx.astype(np.int32))
        out = _moe_gating_top_k_backward_torch(
            x_t, g_t, i_t, routed_scaling_factor, eps
        )
        # 内部函数按 fp32 输出（kernel 模式原始 dtype 已在 numpy 侧丢失，g_t 是 fp32），
        # 所以必须在最外层用原始 numpy 的 grad_y.dtype 转回 bf16/fp16
        return [out.numpy().astype(grad_y.dtype)]

    third_party = {"torch": "torch.moe_gating_top_k_backward"}
    tolerance = {"float32": {"standard": "binary_equal"}}


class MoeGatingTopKBackwardAclnnSpec:
    """ACLNN 流程 — golden / third_party 均收到 torch.Tensor（已在设备上）"""

    @staticmethod
    def golden(
        xNorm,
        gradY,
        expertIdx,
        renorm=0,
        normType=1,
        routedScalingFactor=1.0,
        eps=1e-20,
        out=None,
        **_unused,
    ):
        grad_x = _moe_gating_top_k_backward_torch(
            xNorm, gradY, expertIdx, routedScalingFactor, eps
        )
        return [grad_x]

    third_party = {"torch": "torch.moe_gating_top_k_backward"}
    tolerance = {"float32": {"standard": "binary_equal"}}


class MoeGatingTopKBackwardTorchSpec:
    """E2E 流程 — golden 收到 torch.Tensor（CPU），third_party 收到 torch.Tensor（NPU）"""

    @staticmethod
    def golden(
        x_norm,
        grad_y,
        expertIdx,
        renorm=0,
        norm_type=1,
        routed_scaling_factor=1.0,
        eps=1e-20,
        **_unused,
    ):
        grad_x = _moe_gating_top_k_backward_torch(
            x_norm, grad_y, expertIdx, routed_scaling_factor, eps
        )
        return [grad_x]

    third_party = {"torch": "torch.moe_gating_top_k_backward"}
    tolerance = {"float32": {"standard": "binary_equal"}}


__spec__ = {
    "moe_gating_top_k_backward": MoeGatingTopKBackwardKernelSpec,
    "aclnnMoeGatingTopKBackward": MoeGatingTopKBackwardAclnnSpec,
    "torch_npu.npu_moe_gating_top_k_backward": MoeGatingTopKBackwardTorchSpec,
}
