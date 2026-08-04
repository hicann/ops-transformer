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
"""CPU golden for the sigmoid MoE Top-K backward kernel used by TTK."""

import numpy as np


__golden__ = {
    "kernel": {
        "moe_gating_top_k_backward": "moe_gating_top_k_backward_golden",
    },
}


def moe_gating_top_k_backward_golden(
    x_norm,
    grad_y,
    expert_idx,
    *,
    routed_scaling_factor=1.0,
    eps=1e-20,
    **_unused,
):
    """Reference the formula documented by MoeGatingTopKBackward.

    Inputs are produced by ``inputs.py``, so selected expert indices are unique
    per row. ``np.add.at`` still implements the documented scatter-sum behavior
    if this golden is used with manually supplied repeated indices.
    """
    x_norm_f32 = np.asarray(x_norm, dtype=np.float32)
    grad_y_f32 = np.asarray(grad_y, dtype=np.float32)
    indices = np.asarray(expert_idx, dtype=np.int64)
    token_count = x_norm_f32.shape[0]

    selected = np.take_along_axis(x_norm_f32, indices, axis=1)
    denominator = selected.sum(axis=1, keepdims=True) + np.float32(eps)
    weights = selected / denominator
    grad_y_scaled = grad_y_f32 * np.float32(routed_scaling_factor)
    beta = (weights * grad_y_scaled).sum(axis=1, keepdims=True)
    grad_selected = (grad_y_scaled - beta) / denominator

    grad_norm = np.zeros_like(x_norm_f32)
    np.add.at(grad_norm, (np.arange(token_count)[:, None], indices), grad_selected)
    grad_x = x_norm_f32 * (1.0 - x_norm_f32) * grad_norm
    return grad_x.astype(grad_y.dtype)
