#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software: you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""InplacePartialRotaryMul自动微分回归：反向不得修改调用方传入的上游梯度。

背景（issue #6897）：backward曾在grad_output已连续时直接将其别名传给原地反向算子，
导致调用方上游梯度被改写、返回梯度与调用方存储别名。修复后反向在独立缓冲区上计算。
"""

import pytest
import torch

try:
    import torch_npu
except ImportError:
    torch_npu = None

try:
    from cann_ops_transformer.ops import inplace_partial_rotary_mul
except ImportError:
    inplace_partial_rotary_mul = None

_NPU_READY = (
    torch_npu is not None
    and inplace_partial_rotary_mul is not None
    and torch.npu.is_available()
)

pytestmark = pytest.mark.skipif(
    not _NPU_READY, reason="需要torch_npu与cann_ops_transformer环境"
)

# cos=0、sin=1、interleave、切片[0,2)覆盖全部最后一维时，按aclnnInplacePartialRotaryMulGrad
# 文档interleave公式：dx0 = cos1*dy1 + sin2*dy2，dx1 = cos2*dy2 - sin1*dy1
EXPECT_FROM_GRAD_12 = [2.0, -1.0]


def _make_case(partial_slice):
    """构造前向：base为叶子张量，y = base * 1.0为非叶子张量（叶子张量禁止inplace）。"""
    base = torch.zeros(1, 1, 1, 2, dtype=torch.float32, device="npu")
    base.requires_grad_(True)
    y = base * 1.0
    r1 = torch.zeros(1, 1, 1, 2, dtype=torch.float32, device="npu")  # cos = 0
    r2 = torch.ones(1, 1, 1, 2, dtype=torch.float32, device="npu")  # sin = 1
    inplace_partial_rotary_mul(
        y, r1, r2, rotary_mode="interleave", partial_slice=partial_slice
    )
    return base, y


def test_backward_keeps_contiguous_grad_output():
    """连续上游梯度：调用方buffer不被改写，返回梯度不与调用方存储别名。"""
    base, y = _make_case([0, 2])
    grad_output = torch.tensor([[[[1.0, 2.0]]]], dtype=torch.float32, device="npu")
    snapshot = grad_output.clone()
    grad = torch.autograd.grad(y, base, grad_outputs=grad_output)[0]
    assert torch.equal(grad_output, snapshot), "调用方上游梯度被反向修改"
    assert grad.data_ptr() != grad_output.data_ptr(), "返回梯度与调用方存储别名"
    assert torch.allclose(grad.flatten().cpu(), torch.tensor(EXPECT_FROM_GRAD_12))


def test_backward_keeps_non_contiguous_grad_output():
    """非连续上游梯度：同样不得被改写，梯度数值正确。"""
    base, y = _make_case([0, 2])
    holder = torch.tensor([[[[1.0, 9.0, 2.0, 9.0]]]], dtype=torch.float32, device="npu")
    grad_output = holder[..., ::2]  # shape [1,1,1,2]、最后一维stride=2，非连续，值[1,2]
    snapshot = grad_output.clone()
    grad = torch.autograd.grad(y, base, grad_outputs=grad_output)[0]
    assert torch.equal(grad_output, snapshot), "调用方上游梯度被反向修改"
    assert torch.allclose(grad.flatten().cpu(), torch.tensor(EXPECT_FROM_GRAD_12))


def test_backward_empty_slice_is_identity():
    """空切片（start == end）：前向no-op，梯度恒等直通，同样不得改写上游梯度。"""
    base, y = _make_case([1, 1])
    grad_output = torch.tensor([[[[1.0, 2.0]]]], dtype=torch.float32, device="npu")
    snapshot = grad_output.clone()
    grad = torch.autograd.grad(y, base, grad_outputs=grad_output)[0]
    assert torch.equal(grad_output, snapshot), "调用方上游梯度被反向修改"
    assert torch.allclose(grad.flatten().cpu(), torch.tensor([1.0, 2.0]))


def test_backward_engine_grad_numerics():
    """常规loss.backward()路径（引擎生成全1上游梯度）：梯度数值正确。"""
    base, y = _make_case([0, 2])
    y.sum().backward()
    # 全1上游梯度按公式得dx = [cos1 + sin2, cos2 - sin1] = [1, -1]
    assert base.grad is not None
    assert torch.allclose(base.grad.flatten().cpu(), torch.tensor([1.0, -1.0]))
