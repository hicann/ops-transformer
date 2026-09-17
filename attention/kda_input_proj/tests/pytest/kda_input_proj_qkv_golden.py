# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""CPU golden for the DynamicMxQuant -> QuantMatmul(qkv) path of KdaInputProj.

qkv = dequant(quant_x, x_scale) @ dequant(weight_qkv, weight_qkv_scale)，fp32 累加后转 bf16。

布局（trans_weight_qkv = false，与 Blaze ScaleANDLayoutPtn / ScaleBNDLayoutPtn 一致）:
  quant_x    fp8_e4m3fn  [T, K]
  x_scale    e8m0        [T, ceil(K/64), 2]   -> 第 k 个 32-block = x_scale[t, k//2, k%2]
  weight_qkv fp8_e4m3fn  [K, N]
  weight_qkv_scale  e8m0        [ceil(K/64), N, 2]   -> 第 k 个 32-block 的第 n 列 = weight_qkv_scale[k//2, n, k%2]
"""

from __future__ import annotations

import torch

MX_BLOCK = 32


def _expand_scale_a(x_scale: torch.Tensor, k: int) -> torch.Tensor:
    """[T, K/64, 2] e8m0 -> [T, K] fp32 乘子。"""
    t = x_scale.shape[0]
    s = x_scale.to(torch.float32).reshape(t, -1)
    return s.repeat_interleave(MX_BLOCK, dim=1)[:, :k]


def _expand_scale_b(weight_qkv_scale: torch.Tensor, k: int) -> torch.Tensor:
    """[K/64, N, 2] e8m0 -> [K, N] fp32 乘子。"""
    kg, n, _ = weight_qkv_scale.shape
    s = weight_qkv_scale.to(torch.float32).permute(0, 2, 1).reshape(kg * 2, n)
    return s.repeat_interleave(MX_BLOCK, dim=0)[:k, :]


def qkv_golden(
    quant_x, x_scale, weight_qkv, weight_qkv_scale, out_dtype=torch.bfloat16
):
    k = quant_x.shape[1]
    a = quant_x.to(torch.float32) * _expand_scale_a(x_scale, k)
    b = weight_qkv.to(torch.float32) * _expand_scale_b(weight_qkv_scale, k)
    return (a @ b).to(out_dtype)


def qkv_golden_hp(quant_x, x_scale, weight_qkv, weight_qkv_scale) -> torch.Tensor:
    """float64 参考值，用来衡量 bf16 golden 自身的舍入误差基线。"""
    k = quant_x.shape[1]
    a = quant_x.to(torch.float64) * _expand_scale_a(x_scale, k).double()
    b = weight_qkv.to(torch.float64) * _expand_scale_b(weight_qkv_scale, k).double()
    return a @ b


def bf16_ulp_diff(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """两个 bf16 张量相差的 ULP 数（符号-幅值映射成单调序数后作差）。"""
    ai = a.view(torch.int16).to(torch.int32)
    bi = b.view(torch.int16).to(torch.int32)
    ai = torch.where(ai < 0, -32768 - ai, ai)
    bi = torch.where(bi < 0, -32768 - bi, bi)
    return (ai - bi).abs()


def mean_rel_error(actual: torch.Tensor, golden_hp: torch.Tensor) -> float:
    a = actual.to(torch.float64)
    return ((a - golden_hp).abs() / golden_hp.abs().clamp_min(1e-12)).mean().item()
