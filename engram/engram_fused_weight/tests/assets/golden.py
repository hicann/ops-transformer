#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Golden reference for engram_fused_weight (torch)."""

__spec__ = {
    "engram_fused_weight": "EngramFusedWeightTestSpec",
    "aclnnEngramFusedWeight": "AclnnEngramFusedWeightTestSpec",
    "torch.ops.cann_ops_transformer.engram_fused_weight": "EngramFusedWeightE2ESpec",
}

import numpy as np
import torch


def _dtype_hint_from_declared(kwargs, arrays):
    """从框架 kwarg 声明推断输入 dtype ("bf16"/"fp16"), 仅用于 uint16 位型输入的解释选择。"""
    declared = None
    for key in (
        "dyn_input_dtypes",
        "stc_input_dtypes",
        "input_dtypes",
        "tensor_dtypes",
    ):
        val = kwargs.get(key)
        if val:
            declared = val
            break
    if declared:
        texts = [str(d).lower().replace(" ", "") for d in declared]
    else:
        texts = [str(a.dtype).lower() for a in arrays]
    for t in texts:
        if "bfloat16" in t or "bf16" in t:
            return "bf16"
    for t in texts:
        if "fp16" in t or "float16" in t or t == "half":
            return "fp16"
    return "bf16"


def _to_fp32(a, dtype_hint="bf16"):
    """输入 -> torch.float32 (精确无损失)。

    fp16 的 sNaN payload 在 torch 转换时会被改写, 故走 numpy 位型转换 (与 device 一致);
    BF16 位型用纯位移重解释 (payload 不变)。
    """
    if isinstance(a, torch.Tensor) or hasattr(a, "detach"):
        t = a.detach().cpu().contiguous()
        if t.dtype == torch.float16:
            return torch.from_numpy(
                t.view(torch.int16).numpy().view(np.float16).astype(np.float32)
            )
        return t.to(torch.float32)  # BF16/FLOAT: torch 原生升位精确 (低位补零)
    arr = np.ascontiguousarray(a)
    if arr.dtype == np.uint16 and dtype_hint == "fp16":
        return torch.from_numpy(arr.view(np.float16).astype(np.float32))
    if arr.dtype != np.uint16:
        if str(arr.dtype) == "bfloat16":  # ml_dtypes.bfloat16 -> 位型
            arr = arr.view(np.uint16)
        else:
            return torch.from_numpy(arr.astype(np.float32))
    return (torch.from_numpy(arr).to(torch.int32) << 16).view(
        torch.float32
    )  # BF16 位型


def _mul_fp32_ieee(x1, x2):
    """FP32 逐元素乘; NaN lane 兜底走 numpy (NaN×NaN payload 须与 device 对齐, torch 引擎选择不同)。"""
    y = x1 * x2
    mask = torch.isnan(x1) | torch.isnan(x2)
    if bool(mask.any()):
        n1 = x1.numpy()[mask.numpy()]
        n2 = x2.numpy()[mask.numpy()]
        y[mask] = torch.from_numpy(n1 * n2)
    return y


def golden_engram_fused_weight(x1, x2, dtype_hint="bf16"):
    """y = fp32(x1) * fp32(x2); uint16 位型输入按 dtype_hint 解释 (默认 bf16)。返回 torch.float32。"""
    x1 = _to_fp32(x1, dtype_hint)
    x2 = _to_fp32(x2, dtype_hint)
    if x1.shape != x2.shape:
        raise ValueError(f"x1 shape {x1.shape} != x2 shape {x2.shape}")
    if x1.numel() == 0:
        return torch.zeros(x1.shape, dtype=torch.float32)
    return _mul_fp32_ieee(x1, x2)


class EngramFusedWeightTestSpec:
    """kernel/geir 流程 (numpy 入参), 参数序同 engram_fused_weight_def.cpp。"""

    def golden(*args, **kwargs):
        x1, x2 = args[:2]
        hint = _dtype_hint_from_declared(kwargs, (x1, x2))
        return [golden_engram_fused_weight(x1, x2, hint).numpy()]

    tolerance = {"float32": {"standard": "binary_equal"}}


class AclnnEngramFusedWeightTestSpec:
    """aclnn 流程 (torch 入参), 输出 tensor 作为末位入参传入、被忽略。"""

    def golden(*args, **kwargs):
        tensors = [
            a
            for a in args
            if isinstance(a, (torch.Tensor, np.ndarray)) or hasattr(a, "detach")
        ]
        x1, x2 = tensors[0], tensors[1]
        hint = _dtype_hint_from_declared(kwargs, (x1, x2))
        return [golden_engram_fused_weight(x1, x2, hint).numpy()]

    tolerance = {"float32": {"standard": "binary_equal"}}


class EngramFusedWeightE2ESpec:
    """E2E 流程 (torch 入参), 输出由接口返回。"""

    def golden(weight_hidden, weight_embed, **kwargs):
        hint = _dtype_hint_from_declared(kwargs, (weight_hidden, weight_embed))
        return [golden_engram_fused_weight(weight_hidden, weight_embed, hint).numpy()]

    tolerance = {"float32": {"standard": "binary_equal"}}
