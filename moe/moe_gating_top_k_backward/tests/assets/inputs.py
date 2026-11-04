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
"""TTK input generator for MoeGatingTopKBackward.

TTK's default random generator creates the three inputs independently.  That is
not a valid MoE forward result: ``expert_idx`` must be unique in each token row
and must select entries from the corresponding ``x_norm`` row.  This generator
preserves an already-normalized ``x_norm`` (the legacy STC cases use the
``(0.001, 0.999)`` range); for generic values it applies the requested
normalization, then derives deterministic Top-K indices.
"""

import numpy as np
import torch


__input__ = {
    "kernel": {
        "moe_gating_top_k_backward": "moe_gating_top_k_backward_inputs",
    },
    "aclnn": {
        "aclnnMoeGatingTopKBackward": "aclnn_moe_gating_top_k_backward_inputs",
    },
    "e2e": {
        "torch_npu.npu_moe_gating_top_k_backward": "e2e_moe_gating_top_k_backward_inputs",
    },
}


def _normalize_x_norm(values, norm_type):
    """Build a valid full-dimension normalization output for backward tests."""
    values = np.asarray(values, dtype=np.float32)
    if norm_type == 0:
        if np.any(values < 0.0) or np.any(values > 1.0):
            values = values - np.max(values, axis=1, keepdims=True)
            values = np.exp(values)
        else:
            values = np.maximum(values, 0.0)
        denominator = np.sum(values, axis=1, keepdims=True)
        zero_rows = denominator <= 0.0
        if np.any(zero_rows):
            values = np.where(zero_rows, 1.0, values)
            denominator = np.sum(values, axis=1, keepdims=True)
        return values / denominator
    if norm_type == 1:
        if np.any(values < 0.0) or np.any(values > 1.0):
            values = 1.0 / (1.0 + np.exp(-values))
        return values
    raise ValueError(f"Unsupported norm_type: {norm_type}")


def _normalize_x_norm_torch(values, norm_type):
    """Torch equivalent of _normalize_x_norm for ACLNN/E2E inputs."""
    values = values.to(torch.float32)
    if norm_type == 0:
        if bool((values < 0).any()) or bool((values > 1).any()):
            return torch.softmax(values, dim=1)
        values = torch.clamp_min(values, 0.0)
        denominator = values.sum(dim=1, keepdim=True)
        zero_rows = denominator <= 0.0
        values = torch.where(zero_rows, torch.ones_like(values), values)
        denominator = torch.where(
            zero_rows,
            torch.full_like(denominator, values.shape[1]),
            denominator,
        )
        return values / denominator
    if norm_type == 1:
        if bool((values < 0).any()) or bool((values > 1).any()):
            return torch.sigmoid(values)
        return values
    raise ValueError(f"Unsupported norm_type: {norm_type}")


def moe_gating_top_k_backward_inputs(
    x_norm, grad_y, expert_idx, norm_type=0, **_unused
):
    """Create mutually consistent x_norm and expert_idx inputs for TTK."""
    if x_norm.ndim != 2 or grad_y.ndim != 2 or expert_idx.ndim != 2:
        raise ValueError("MoeGatingTopKBackward expects rank-2 inputs")
    if x_norm.shape[0] != grad_y.shape[0] or grad_y.shape != expert_idx.shape:
        raise ValueError("x_norm, grad_y and expert_idx shapes are inconsistent")
    if grad_y.shape[1] > x_norm.shape[1]:
        raise ValueError("Top-K width cannot exceed expert count")

    values = _normalize_x_norm(x_norm, norm_type)
    x_norm[...] = values.astype(x_norm.dtype, copy=False)
    topk = grad_y.shape[1]
    expert_idx[...] = np.argsort(-x_norm, axis=1, kind="stable")[:, :topk].astype(
        expert_idx.dtype
    )
    return x_norm, grad_y, expert_idx


def aclnn_moe_gating_top_k_backward_inputs(
    xNorm,
    gradY,
    expertIdx,
    renorm=0,
    normType=0,
    routedScalingFactor=1.0,
    eps=1e-20,
    out=None,
    **_unused,
):
    """ACLNN 模式 input 插件。

    参数顺序与 aclnnMoeGatingTopKBackwardGetWorkspaceSize 一致（不含 workspaceSize/executor）。
    注意：aclnn 模式的 _call_custom_input 不消费返回值，必须 in-place 修改传入的
    torch.Tensor 才能生效。
    """
    topk = expertIdx.shape[1]
    values = _normalize_x_norm_torch(xNorm, normType)
    xNorm.copy_(values.to(dtype=xNorm.dtype))
    _, sorted_idx = torch.sort(values, dim=1, descending=True, stable=True)
    expertIdx.copy_(sorted_idx[:, :topk].to(expertIdx.dtype))
    return [xNorm, gradY, expertIdx, out]


def e2e_moe_gating_top_k_backward_inputs(
    x_norm,
    grad_y,
    expertIdx,
    renorm=0,
    norm_type=0,
    routed_scaling_factor=1.0,
    eps=1e-20,
    **_unused,
):
    """E2E 模式 input 插件。

    参数顺序与 torch_npu.npu_moe_gating_top_k_backward 一致。
    注意：e2e 模式的 input 插件不消费返回值，必须 in-place 修改传入的
    torch.Tensor 才能生效（framework_api/input_generation.py）。
    """
    topk = expertIdx.shape[1]
    values = _normalize_x_norm_torch(x_norm, norm_type)
    x_norm.copy_(values.to(dtype=x_norm.dtype))
    _, sorted_idx = torch.sort(values, dim=1, descending=True, stable=True)
    expertIdx.copy_(sorted_idx[:, :topk].to(expertIdx.dtype))
