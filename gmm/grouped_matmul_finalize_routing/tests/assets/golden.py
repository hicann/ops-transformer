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
        "grouped_matmul_finalize_routing": "grouped_matmul_finalize_routing_golden"
    }
}

from typing import List
import numpy as np
import torch


def grouped_matmul_finalize_routing_golden(
    x,
    w,
    scale=None,
    bias=None,
    pertoken_scale=None,
    group_list=None,
    shared_input=None,
    logit=None,
    row_index=None,
    offset=None,
    dtype: int = 0,
    shared_input_weight: float = 1.0,
    shared_input_offset: int = 0,
    transpose_x: bool = False,
    transpose_w: bool = False,
    output_bs: int = 0,
    group_list_type: int = 1,
    tuning_config: List[int] = [0],
    **kwargs,
):
    x1, x2_all = x, w
    x1_dtype = x1.dtype.name
    x2_dtype = w.dtype.name
    output_dtypes = kwargs["output_dtypes"]
    out_dtype = output_dtypes[0]
    trans_b = transpose_w
    # mxFP4/8
    outs = []
    if group_list is not None:
        group_num = len(group_list)

    # group_list_type 0: cumsum, 1: count
    if group_list_type == 1 and group_list is not None:
        group_list = np.cumsum(group_list)
    M = x1.shape[0]
    N = x2_all.shape[-1]
    if pertoken_scale is not None:
        pertoken_scale_mx = pertoken_scale  # pertoken_scale为x_scale,shape为(m, ceil(k/64), 2),scale为weight_scale,shape为(G, n, ceil(k/64), 2) True或(G, ceil(k/64), n,  2) False
        m, k0, k1 = pertoken_scale_mx.shape
        pertoken_scale_mx = pertoken_scale_mx.reshape(m, k0 * k1)
        pertoken_scale_mx_broadcast = np.repeat(pertoken_scale_mx, 32, axis=-1)
    x1_dims = len(x1.shape)

    if x1_dtype == "float4_e2m1" or x1_dtype == "float4_e1m2":
        x1 = x1.astype(np.float32)

    if pertoken_scale is not None:
        x1_pad_len = pertoken_scale_mx_broadcast.shape[-1] - x1.shape[-1]

        x1 = np.pad(
            x1,
            [(0, 0)] * (x1_dims - 1) + [(0, x1_pad_len)],
            mode="constant",
            constant_values=0,
        )
        x1 = x1 * pertoken_scale_mx_broadcast
    x1 = convert_to_high_precision(x1, x1_dtype)

    for i in range(group_num):
        if scale is not None:
            scale_g = scale[i]
            # reshape scale to shape n,ceil(k,64)*2
            if trans_b is False:
                scale_g = transform_tensor(scale_g)
            else:
                n, k0, k1 = scale_g.shape
                scale_g = scale_g.reshape(n, k0 * k1)

        x2 = x2_all[i]

        if x2_dtype == "float4_e2m1" or x2_dtype == "float4_e1m2":
            x2 = x2.astype(np.float32)
        # mxFP4单独处理transpose，统一使用(M,K)和(K,N)格式处理mxFP4
        if trans_b:
            x2 = np.swapaxes(x2, -1, -2)
            if scale is not None:
                scale_g = scale_g.transpose()
        # broadcast，每个数对应32个数
        if scale is not None:
            deq_scale_mx_broadcast = np.repeat(scale_g, 32, axis=-2)
        x2_dims = len(x2.shape)
        if scale is not None:
            x2_pad_len = deq_scale_mx_broadcast.shape[-2] - x2.shape[-2]
            x2 = np.pad(
                x2,
                [(0, 0)] * (x2_dims - 2) + [(0, x2_pad_len)] + [(0, 0)],
                mode="constant",
                constant_values=0,
            )
            x2 = x2 * deq_scale_mx_broadcast
        # 升精度 & 转torch
        if i == 0:
            x1_temp = x1[: group_list[i], :]
        else:
            x1_temp = x1[group_list[i - 1] : group_list[i], :]
        x2 = convert_to_high_precision(x2, x2_dtype)
        out = single_group_mm_cal(x1_temp, x2, out_dtype)
        if bias is not None:
            out += bias[i, :].reshape(1, -1)
        outs.append(out)

    gmm_out = outs if not outs else np.concatenate(outs, axis=0)
    final_out = combine_func(
        gmm_out,
        logit,
        shared_input,
        shared_input_weight,
        row_index,
        output_bs,
        shared_input_offset,
    )
    return final_out


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


def transform_tensor(input_tensor):
    transposed = np.transpose(input_tensor, axes=(0, 2, 1))
    batch_size, height, width = transposed.shape
    result = transposed.reshape(batch_size * height, width)
    return result


def single_group_mm_cal(x1, x2, out_dtype):
    out = torch.matmul(x1, x2)
    torch.set_printoptions(threshold=torch.inf)
    # 检查是否有inf
    has_inf = torch.isinf(out).any()
    # 检查是否有nan
    has_nan = torch.isnan(out).any()

    out = out.numpy().astype(out_dtype)
    return out


def combine_func(x, logits, residual, resid_scale, source_row, output_bs, offset):
    top_k = x.shape[0] // output_bs
    remain_logits = len(logits) % top_k
    if remain_logits:
        logits = logits[: len(logits) - remain_logits]
    out = x * logits.reshape(-1, 1)
    remain_sr = len(source_row) % top_k
    if remain_sr:
        source_row = source_row[: len(source_row) - remain_sr]
    index = np.argsort(source_row)
    out = (
        out[index].reshape(output_bs, top_k, x.shape[-1]).sum(axis=1)
    )  # m / total_token = topk
    if residual is not None:
        out[offset : offset + residual.shape[0], :] += resid_scale * residual.astype(
            np.float32
        )
    return out
