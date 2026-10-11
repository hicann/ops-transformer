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
__golden__ = {
    "kernel": {
        "quant_grouped_matmul_inplace_add": "quant_grouped_matmul_inplace_add_golden"
    }
}

import torch
import numpy as np


def quant_grouped_matmul_inplace_add_golden(
    x1,
    x2,
    scale2,
    group_list,
    y,
    scale1=None,
    group_list_type: int = 0,
    group_size: int = 0,
    **kwargs,
):
    scale, inplace_y, pertoken_scale = scale2, y, scale1
    x1_dtype, x2_dtype = x1.dtype, x2.dtype
    if scale is not None:
        scale_dtype = scale.dtype
    output_dtypes = kwargs["output_dtypes"]
    out_dtype = output_dtypes[0]

    # mxFP4/8
    outs = []
    group_num = len(group_list)  # 不管什么分组，分组数一定等于该group_list长度
    is_mx_quant = (
        x1_dtype in ("float4_e2m1", "float4_e1m2", "float8_e4m3fn", "float8_e5m2")
        and scale is not None
        and scale_dtype == "float8_e8m0"
    )

    # group_list_type 0: cumsum, 1: count
    if group_list_type == 1:
        group_list = np.cumsum(group_list)
    if is_mx_quant:
        M = x1.shape[-1]  # 单
        N = x2.shape[-1]  # 单
        scale_k_offset = 0
        pertoken_scale_mx = transform_tensor(pertoken_scale)
        scale_mx = transform_tensor(scale)
        for i in range(group_num):
            pre = 0 if i == 0 else group_list[i - 1]
            cur = group_list[i]
            if cur - pre == 0:  # k is 0
                out = np.zeros((M, N), dtype=out_dtype)
                outs.append(out)
                continue
            x1_g = x1[pre:cur, :]
            x2_g = x2[pre:cur, :]
            scale_k_offset = (pre // 64 + i) * 2
            cur_scale_k_offset = (cur // 64 + i + 1) * 2
            pertoken_scale_mx_g = pertoken_scale_mx[
                scale_k_offset:cur_scale_k_offset, :
            ]
            scale_mx_g = scale_mx[scale_k_offset:cur_scale_k_offset, :]
            x1_g = np.swapaxes(x1_g, -1, -2)
            pertoken_scale_mx_g = pertoken_scale_mx_g.transpose()
            # broadcast，每个数对应32个数
            pertoken_scale_mx_broadcast = np.repeat(pertoken_scale_mx_g, 32, axis=-1)
            x1_dims = len(x1_g.shape)
            x1_pad_len = pertoken_scale_mx_broadcast.shape[-1] - x1_g.shape[-1]
            x1_g = np.pad(
                x1_g,
                [(0, 0)] * (x1_dims - 1) + [(0, x1_pad_len)],
                mode="constant",
                constant_values=0,
            )
            x1_g = x1_g * pertoken_scale_mx_broadcast
            x1_g = convert_to_high_precision(x1_g, x1_dtype)
            # broadcast，每个数对应32个数
            scale_mx_broadcast = np.repeat(scale_mx_g, 32, axis=-2)
            x2_dims = len(x2_g.shape)
            x2_pad_len = scale_mx_broadcast.shape[-2] - x2_g.shape[-2]
            x2_g = np.pad(
                x2_g,
                [(0, 0)] * (x2_dims - 2) + [(0, x2_pad_len)] + [(0, 0)],
                mode="constant",
                constant_values=0,
            )
            x2_g = x2_g * scale_mx_broadcast
            # 升精度 & 转torch
            x2_g = convert_to_high_precision(x2_g, x2_dtype)
            out = single_group_mm_cal(x1_g, x2_g, None, out_dtype, True)
            outs.append(out)
    else:
        x1 = convert_to_high_precision(x1, x1_dtype)
        x2 = convert_to_high_precision(x2, x2_dtype)
        deq_scale = scale.astype(np.float32)
        deq_scale_tensor = torch.from_numpy(deq_scale)

        if pertoken_scale is not None:
            pertoken_scale = pertoken_scale.reshape(group_num, 1)
            deq_scale_tensor *= torch.from_numpy(pertoken_scale).to(torch.float32)

        deq_scale_temp = None
        # 固定k轴分组
        for i in range(group_num):
            M = x1.shape[-1]  # 单
            N = x2.shape[-1]  # 单
            pre = 0 if i == 0 else group_list[i - 1]
            cur = group_list[i]
            if cur - pre == 0:  # k is 0
                out = np.zeros((M, N), dtype=out_dtype)
                outs.append(out)
                continue
            x1_g = x1[pre:cur, :]
            x1_g = np.swapaxes(x1_g, -1, -2)

            x2_g = x2[pre:cur, :]

            deq_scale_temp = deq_scale_tensor[i]

            out = single_group_mm_cal(
                x1_g, x2_g, deq_scale_temp, out_dtype, is_mx_quant
            )
            outs.append(out)

    real_out = outs if not outs else np.concatenate(outs, axis=0)
    real_out = real_out.reshape(inplace_y.shape)
    real_out = real_out + inplace_y
    return real_out


def transform_tensor(input_tensor):
    # 转置最后两维
    transposed = np.transpose(input_tensor, axes=(0, 2, 1))
    # 重塑形状，将第一个轴和第二个轴相乘
    batch_size, height, width = transposed.shape
    result = transposed.reshape(batch_size * height, width)
    return result


def convert_to_high_precision(input_tensor, input_type):
    if input_type in (
        "float8_e4m3fn",
        "float8_e5m2",
        "float4_e2m1",
        "float4_e1m2",
        "hifloat8",
    ):
        input_tensor = torch.from_numpy(input_tensor.astype(np.float32))
    elif input_type in ("int4"):
        input_tensor = torch.from_numpy(input_tensor.astype(np.int32)).to(torch.int32)
    else:
        input_tensor = torch.from_numpy(input_tensor).to(torch.int32)
    return input_tensor


def single_group_mm_cal(x1, x2, deq_scale_tensor, out_dtype, is_mx_quant):
    out = torch.matmul(x1, x2)
    if not is_mx_quant:
        out = out * deq_scale_tensor
    out = out.numpy().astype(out_dtype)
    return out
