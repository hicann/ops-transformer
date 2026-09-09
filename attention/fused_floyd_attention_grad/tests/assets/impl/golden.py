#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""FFAG (FusedFloydAttentionGrad) CPU Golden 参考实现

从 ATK executor_aclnnFusedFloydAttentionGrad.py 移植。
包含 floyd_attn_forward (用于 customize_inputs) 和 floyd_attn_backward (用于 golden)。
"""

import struct

import torch


def _tsoftmax(x):
    x_max = torch.max(x, dim=-1, keepdim=True)[0]
    x_sub = x.sub(x_max)
    y = torch.exp(x_sub)
    x_sum = y.sum(dim=-1, keepdim=True)
    ans = y.div(x_sum)
    return ans, x_max, x_sum


def floyd_attn_forward(Q, K1, K2, V1, V2, atten_mask, scale):
    """Forward pass — 与 ATK floyd_attn_forward 等价"""
    fp32_bits = 0xFF7FFFFF
    float_value = struct.unpack(">f", struct.pack(">I", fp32_bits))[0]
    dtype = Q.dtype
    golden_dtype = torch.float64 if dtype == torch.float32 else torch.float32

    s = torch.einsum("bhikc,bhijc->bhikj", Q, K1) + torch.einsum(
        "bhikc,bhjkc->bhikj", Q, K2
    )
    s = s.to(golden_dtype)
    s = s * scale
    if atten_mask is not None:
        s = s + atten_mask.bool() * float_value
    p, x_max, x_sum = _tsoftmax(s)

    p = p.to(dtype)
    output = torch.einsum("bhikj,bhijc->bhikc", p, V1) + torch.einsum(
        "bhikj,bhjkc->bhikc", p, V2
    )
    output = output.to(dtype)
    x_max = x_max.to(golden_dtype)
    x_sum = x_sum.to(golden_dtype)
    return output, x_max, x_sum


def floyd_attn_backward_fp16(Q, K1, K2, V1, V2, grad, atten_mask, scale):
    """原dtype精度版本的 Floyd Attention Backward

    模拟ATK CPU后端 (cpu_0) 用原dtype输入完整计算forward+backward的标杆输出。
    先用高精度(fp64)计算forward得到softmax_max/sum/atten_in，
    再把中间结果转回原dtype，最后用这些中间结果计算backward。

    与 ATK executor init_by_input_data 的逻辑一致:
      1. Q/K1/V1/K2/V2 转fp64计算forward
      2. atten_in/x_max/x_sum 转回原dtype
      3. backward用原dtype的Q/K1/V1/K2/V2/grad + 原dtype的中间结果

    ATK的三方对比:
      - cpu_benchmark (golden): fp32输入 → fp32输出 (高精度参考)
      - cpu_0 (benchmark):      fp16输入 → fp16输出 (与NPU相同dtype)
      - pyaclnn_0 (NPU):        fp16输入 → fp16输出
    """
    # 与 ATK init_by_input_data 一致: 先用fp64计算forward
    origin_dtype = Q.dtype
    golden_dtype = torch.float64
    Q_f64 = Q.to(golden_dtype)
    K1_f64 = K1.to(golden_dtype)
    V1_f64 = V1.to(golden_dtype)
    K2_f64 = K2.to(golden_dtype)
    V2_f64 = V2.to(golden_dtype)

    atten_in, x_max, x_sum = floyd_attn_forward(
        Q_f64, K1_f64, K2_f64, V1_f64, V2_f64, atten_mask, scale
    )
    x_max = x_max.repeat(1, 1, 1, 1, 8)
    x_sum = x_sum.repeat(1, 1, 1, 1, 8)

    # 转回原dtype (与 ATK init_by_input_data 一致)
    # 注意: softmax_max/sum 的原始dtype是fp32 (ATK JSON中定义), 不能用Q的dtype
    # 因为 -3.4e38 在 fp16 下会变成 -inf, 导致 backward 中 exp(s - (-inf)) = exp(+inf) = inf → nan
    x_max = x_max.to(torch.float32)
    x_sum = x_sum.to(torch.float32)
    atten_in = atten_in.to(origin_dtype)

    # 再用原dtype的Q/K/V/grad + 转回原dtype的中间结果计算backward
    return floyd_attn_backward(
        Q, K1, K2, V1, V2, grad, atten_mask, x_max, x_sum, atten_in, scale
    )


def floyd_attn_backward(
    Q, K1, K2, V1, V2, grad, atten_mask, x_max, x_sum, atten_in, scale
):
    """Backward pass — 与 ATK floyd_attn_backward 等价

    参数顺序与 aclnnFusedFloydAttentionGradGetWorkspaceSize 一致:
      query, key1, value1, key2, value2, dy, attenMask,
      softmaxMax, softmaxSum, attentionIn, scaleValue
    """
    fp32_bits = 0xFF7FFFFF
    float_value = struct.unpack(">f", struct.pack(">I", fp32_bits))[0]
    dtype = Q.dtype
    golden_dtype = torch.float64 if dtype == torch.float32 else torch.float32

    Q = Q.to(golden_dtype)
    K1 = K1.to(golden_dtype)
    V1 = V1.to(golden_dtype)
    K2 = K2.to(golden_dtype)
    V2 = V2.to(golden_dtype)
    grad = grad.to(golden_dtype)
    atten_in = atten_in.to(golden_dtype)

    s = torch.einsum("bhikc,bhijc->bhikj", Q.to(dtype), K1.to(dtype)).to(
        golden_dtype
    ) + torch.einsum("bhikc,bhjkc->bhikj", Q.to(dtype), K2.to(dtype)).to(golden_dtype)

    dp = torch.einsum("bhikc,bhijc->bhikj", grad.to(dtype), V1.to(dtype)).to(
        golden_dtype
    ) + torch.einsum("bhikc,bhjkc->bhikj", grad.to(dtype), V2.to(dtype)).to(
        golden_dtype
    )
    s = s.to(golden_dtype)
    dp = dp.to(golden_dtype)

    s = s * scale
    if atten_mask is not None:
        s = s + atten_mask.bool() * float_value
    p = torch.exp(s - x_max[:, :, :, :, 0:1]) / x_sum[:, :, :, :, 0:1]
    ds = p * (dp - (grad * atten_in).sum(dim=-1, keepdim=True)) * scale

    ds = ds.to(dtype)
    p = p.to(dtype)
    dQ = torch.einsum("bhikj,bhijc->bhikc", ds.to(dtype), K1.to(dtype)).to(
        golden_dtype
    ) + torch.einsum("bhikj,bhjkc->bhikc", ds.to(dtype), K2.to(dtype)).to(golden_dtype)
    dK1 = torch.einsum("bhikj,bhikc->bhijc", ds.to(dtype), Q.to(dtype)).to(golden_dtype)
    dK2 = torch.einsum("bhikj,bhikc->bhjkc", ds.to(dtype), Q.to(dtype)).to(golden_dtype)
    dV1 = torch.einsum("bhikj,bhikc->bhijc", p.to(dtype), grad.to(dtype)).to(
        golden_dtype
    )
    dV2 = torch.einsum("bhikj,bhikc->bhjkc", p.to(dtype), grad.to(dtype)).to(
        golden_dtype
    )

    dQ = dQ.to(dtype)
    dK1 = dK1.to(dtype)
    dK2 = dK2.to(dtype)
    dV1 = dV1.to(dtype)
    dV2 = dV2.to(dtype)
    return dQ, dK1, dV1, dK2, dV2
