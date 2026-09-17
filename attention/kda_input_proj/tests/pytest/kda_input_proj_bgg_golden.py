# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""KdaInputProj beta / gate / g 三路投影的 CPU golden。

对应 Stage1 的 ``Matmul(beta/gate/g)`` 与 Stage2 的 ``Sigmoid``：

    beta = sigmoid(x @ w_beta)   FP32 输出，matmul 与 sigmoid 全程 FP32
    gate = x @ w_gate            BF16 输出
    g    = x @ w_g               BF16 输出

x 与三个权重都是 bf16，硬件在 FP32 上累加。gate/g 最后舍入回 bf16，beta 保持 FP32。
每一路都提供 float64 版本，用来给"这个精度是不是只由该有的舍入造成"提供基线。
"""

from __future__ import annotations

import torch


def bgg_golden(
    x: torch.Tensor, w_beta: torch.Tensor, w_gate: torch.Tensor, w_g: torch.Tensor
):
    """按硬件语义算 (beta, gate, g)：FP32 累加，gate/g 舍回 bf16。"""
    xf = x.float()
    beta = torch.sigmoid(xf @ w_beta.float())
    gate = (xf @ w_gate.float()).to(torch.bfloat16)
    g = (xf @ w_g.float()).to(torch.bfloat16)
    return beta, gate, g


def bgg_golden_hp(
    x: torch.Tensor, w_beta: torch.Tensor, w_gate: torch.Tensor, w_g: torch.Tensor
):
    """float64 高精度参考，不做任何窄化舍入。"""
    xd = x.double()
    beta = torch.sigmoid(xd @ w_beta.double())
    gate = xd @ w_gate.double()
    g = xd @ w_g.double()
    return beta, gate, g
