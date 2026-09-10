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

"""FFA (FusedFloydAttention) CPU Golden 实现 — 从 ATK 移植

纯计算模块，不含 TTK 框架依赖。

C API 参数顺序 (aclnnFusedFloydAttentionGetWorkspaceSize):
  inputs: query, key1, value1, key2, value2, attenMaskOptional
  attr:   scaleValue (double)
  outputs: softmaxMaxOut, softmaxSumOut, attentionOutOut
"""

import struct

import torch


def _tsoftmax(x):
    """数值稳定的 softmax"""
    x_max = torch.max(x, dim=-1, keepdim=True)[0]
    x_sub = x.sub(x_max)
    y = torch.exp(x_sub)
    x_sum = y.sum(dim=-1, keepdim=True)
    ans = y.div(x_sum)
    return ans, x_max, x_sum


def floyd_attn_forward(Q, K1, K2, V1, V2, atten_mask, scale):
    """Floyd Attention 前向计算 — 与 ATK golden 完全等价

    输入 shape:
      Q:   [B, H, N, M, D]
      K1:  [B, H, N, M, D]
      V1:  [B, H, N, M, D]
      K2:  [B, H, M, M, D]
      V2:  [B, H, M, M, D]
      atten_mask: [B, H, N, 1, M] or None
      scale: float

    输出:
      softmax_max: [B, H, N, M] (float32)
      softmax_sum: [B, H, N, M] (float32)
      output: [B, H, N, M, D] (与 Q 同 dtype)
    """
    dtype = Q.dtype
    fp32_bits = 0xFF7FFFFF
    float_value = struct.unpack(">f", struct.pack(">I", fp32_bits))[0]
    golden_dtype = torch.float64 if dtype == torch.float32 else torch.float32

    Q_g = Q.to(golden_dtype)
    K1_g = K1.to(golden_dtype)
    V1_g = V1.to(golden_dtype)
    K2_g = K2.to(golden_dtype)
    V2_g = V2.to(golden_dtype)

    # s = Q·K1^T + Q·K2^T
    s = torch.einsum("bhikc,bhijc->bhikj", Q.to(dtype), K1.to(dtype)).to(
        golden_dtype
    ) + torch.einsum("bhikc,bhjkc->bhikj", Q.to(dtype), K2.to(dtype)).to(golden_dtype)
    s = s * scale

    if atten_mask is not None:
        s = s + atten_mask.bool() * float_value

    p, x_max, x_sum = _tsoftmax(s)

    # output = P·V1 + P·V2
    output = torch.einsum("bhikj,bhijc->bhikc", p.to(dtype), V1.to(dtype)).to(
        golden_dtype
    ) + torch.einsum("bhikj,bhjkc->bhikc", p.to(dtype), V2.to(dtype)).to(golden_dtype)

    return x_max.to(torch.float32), x_sum.to(torch.float32), output.to(dtype)


def floyd_attn_forward_fp16(Q, K1, K2, V1, V2, atten_mask, scale):
    """原dtype精度版本的 Floyd Attention 前向计算

    模拟ATK CPU后端 (cpu_0) 用原dtype输入完整计算的标杆输出。
    与 floyd_attn_forward 使用完全相同的代码, 但输入是原dtype而非高精度。

    ATK的三方对比:
      - cpu_benchmark (golden): fp32输入 → fp32输出 (高精度参考)
      - cpu_0 (benchmark):      fp16输入 → fp16输出 (与NPU相同dtype)
      - pyaclnn_0 (NPU):        fp16输入 → fp16输出

    本函数等价于 floyd_attn_forward(fp16输入), 产生与ATK cpu_0相同的标杆输出。
    """
    # 直接调用 floyd_attn_forward, 用原dtype输入
    # floyd_attn_forward 内部:
    #   golden_dtype = fp32 (对fp16输入)
    #   einsum用 Q.to(dtype) 计算 (fp16), 结果转fp32
    #   output.to(dtype) 最终输出fp16
    # 这与ATK cpu_0的行为完全一致
    return floyd_attn_forward(Q, K1, K2, V1, V2, atten_mask, scale)
