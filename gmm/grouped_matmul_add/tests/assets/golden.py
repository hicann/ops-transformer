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
    "kernel": {"grouped_matmul_add": "grouped_matmul_add_golden"},
    "e2e": {"torch_npu.npu_grouped_matmul_add_": "torch_grouped_matmul_add_golden"},
}

import ml_dtypes
import torch
import numpy as np


def grouped_matmul_add_golden(
    x,
    weight,
    group_list,
    y,
    transpose_x: bool = True,
    transpose_weight: bool = False,
    group_type: int = 2,
    group_list_type: int = 0,
    **kwargs,
):
    # group_list_type 0: cumsum, 1: count
    output_dtypes = kwargs["output_dtypes"]
    out_dtype = output_dtypes[0]
    outs = []
    group_num = len(group_list)

    if group_list_type == 1:
        group_list = np.cumsum(group_list)
    x = torch.from_numpy(x.astype(np.float32))
    weight = torch.from_numpy(weight.astype(np.float32))
    for i in range(group_num):
        M = x.shape[-1]
        N = weight.shape[-1]
        pre = 0 if i == 0 else group_list[i - 1]
        cur = group_list[i]
        if cur - pre == 0:  # k is 0
            out = np.zeros((M, N), dtype=out_dtype)
            outs.append(out)
            continue
        x_g = x[pre:cur, :]
        x_g = np.swapaxes(x_g, -1, -2)
        weight_g = weight[pre:cur, :]
        out = torch.matmul(x_g, weight_g).numpy().astype(out_dtype)
        outs.append(out)

    real_out = outs if not outs else np.concatenate(outs, axis=0)
    real_out = real_out.reshape(y.shape)
    real_out = real_out + y
    return real_out


def torch_grouped_matmul_add_golden(
    self_ref,
    x,
    weight,
    group_list,
    *,
    transpose_x=None,
    transpose_weight=None,
    group_type=None,
    group_list_type=None,
    **kwargs,
):
    self_ref = torch_to_numpy(self_ref)
    x = torch_to_numpy(x)
    weight = torch_to_numpy(weight)
    group_list = torch_to_numpy(group_list)
    param_group_list = group_list
    x_dtype = x.dtype
    weight_dtype = weight.dtype
    group_tensor_new, _, group_list_cumsum = group_list_diff_cumsum_process(
        group_list_type, param_group_list
    )
    if group_list_cumsum[-1] > x.shape[0]:
        raise Exception(
            "Sum of grouplist ({}) exceeds the first dim of x[0]:({})".format(
                group_list_cumsum[-1], x.shape[0]
            )
        )

    x = transpose_last_two_dim(x, transpose_x)
    x = input_convert_to_high_precision(x, x_dtype)
    weight = transpose_last_two_dim(weight, transpose_weight)
    weight = input_convert_to_high_precision(weight, weight_dtype)
    x, w, _ = gmm_non_quant_data_split_per_expect(
        x, weight, group_type, group_list_cumsum
    )
    output = gmm_non_quant_cpu_golden_calculation(x, w)
    params_out_dtype = self_ref.dtype
    # golden结果处理
    output = output_dtype_post_process(output, params_out_dtype)
    output = golden_output_process(output, 2, group_type)
    y_input = torch.add(torch.from_numpy(self_ref), output[0])
    return y_input


# 获取Count和Comsum两种Group_List
def group_list_diff_cumsum_process(group_list_type, param_group_list, group_idx=None):
    if group_idx is None:
        group_idx = [0]
    # 计算差值是否为非递减非负数
    if group_list_type == 0:
        # Group_List为Cumsum,求相邻值之间的差值的到Count
        count_values = np.diff(param_group_list)
        count = np.insert(count_values, 0, param_group_list[0])
        cumsum = param_group_list
        group_tensor_new = torch.tensor(cumsum)

    elif group_list_type == 2:
        group_list_tensor = torch.tensor(param_group_list)
        group_idx_tensor = torch.tensor(group_idx)
        group_tensor_new = torch.stack((group_idx_tensor, group_list_tensor), dim=1).to(
            torch.int64
        )

        sorted_group_list = [
            param_group_list[i]
            for i in sorted(range(len(group_idx)), key=lambda k: group_idx[k])
        ]
        count = sorted_group_list.copy()
        cumsum = np.cumsum(sorted_group_list)

    else:
        # Group_List为Count,求和得到Cumsum
        count = param_group_list
        cumsum = np.cumsum(param_group_list)
        group_tensor_new = torch.tensor(count)

    return group_tensor_new, count, cumsum


# 处理转置
def transpose_last_two_dim(input, transpose_flag):
    if isinstance(input, torch.Tensor):
        return torch_transpose_last_two_dim(input, transpose_flag)
    elif isinstance(input, np.ndarray):
        return np_transpose_last_two_dim(input, transpose_flag)
    else:
        raise TypeError("Unsupported dtype!")


def np_transpose_last_two_dim(input, transpose_flag):
    if transpose_flag == 1:
        # 转置场景
        # 获取当前维度数
        ndim = input.ndim
        # 构造新的轴顺序：保持前面的所有维度不变，最后两个交换
        axes = list(range(ndim - 2)) + [ndim - 1, ndim - 2]
        # 进行转置
        return np.transpose(input, axes=axes)
    else:
        return input


def torch_transpose_last_two_dim(input, transpose_flag):
    if transpose_flag == 1:
        ndim = input.dim()
        # 如果维度小于2，无法转置，抛出异常
        if ndim < 2:
            raise ValueError("Tensor must have at least 2 dimensions.")
        # 构建新的维度顺序：保留前面所有维度（从 0 到 ndim - 3）, 将最后两个维度交换（-1 和 -2）
        new_dims = list(range(ndim - 2)) + [ndim - 1, ndim - 2]
        # 使用 permute 转置
        return input.permute(*new_dims)
    else:
        return input


# 升精度
def input_convert_to_high_precision(input, dtype):
    if isinstance(input, torch.Tensor):
        return torch_convert_to_high_precision(input, dtype)
    elif isinstance(input, np.ndarray):
        return np_convert_to_high_precision(input, dtype)
    else:
        raise TypeError("Unsupported dtype")


def np_convert_to_high_precision(input, dtype):
    if dtype in (
        "float8_e4m3fn",
        "float8_e5m2",
        "float4_e2m1",
        "float4_e1m2",
        "hifloat8",
        "float16",
        "bfloat16",
    ):
        return torch.from_numpy(input.astype(np.float32))
    elif dtype in ("int4",):
        return torch.from_numpy(input.astype(np.int32)).to(torch.int32)
    else:
        return torch.from_numpy(input).to(torch.int32)


def torch_convert_to_high_precision(input, dtype):
    if dtype == "int8":
        return input.to(torch.int8).numpy()
    elif dtype == "int32":
        return input.to(torch.int32).numpy()
    elif dtype in ("float8_e4m3fn", "float8_e5m2"):
        return input.to(torch.float32).numpy()
    elif dtype in ("hifloat8",):
        return input.to(torch.float32).numpy()
    elif dtype in ("bf16", "bfloat16"):
        return input.to(torch.float32).numpy()
    else:
        return input.to(torch.float32).numpy()


# 对X、W、Bias进行切分
def get_x_weight_bias(
    x_input, w_input, group_type, group_list_cumsum, biass=None, has_bias=0
):
    Xs, Ws, Bs = [], [], []
    if group_type == 0:
        # M分组
        offset = 0
        for i in range(len(group_list_cumsum)):
            Xs.append(x_input[offset : group_list_cumsum[i], :])
            Ws.append(w_input[i])
            offset = group_list_cumsum[i]

    elif group_type == 2:
        # K分组
        offset = 0
        for i in range(len(group_list_cumsum)):
            Xs.append(x_input[:, offset : group_list_cumsum[i]])
            Ws.append(w_input[offset : group_list_cumsum[i], :])
            offset = group_list_cumsum[i]
    else:
        Xs.append(x_input)
        Ws.append(w_input)
    E = len(group_list_cumsum)
    if has_bias == 1:
        Bs = np.split(biass, E, axis=0)

    return Xs, Ws, Bs


# 非量化：对输入进行切分
def gmm_non_quant_data_split_per_expect(
    x1, x2, group_type, grouplist, bias=None, has_bias=0
):
    return get_x_weight_bias(x1, x2, group_type, grouplist, bias, has_bias)


# 非量化
def gmm_non_quant_cpu_golden_calculation(x, weight, bias=None, has_bias=0):
    return gmm_non_quant_golden_calculate(x, weight, bias, has_bias)


def gmm_non_quant_golden_calculate(x, weight, bias=None, has_bias=0):
    output = [None] * len(x)
    for i in range(len(x)):
        output[i] = np.matmul(x[i], weight[i])
        if has_bias == 1:
            output[i] = torch.add(output[i], torch.from_numpy(bias[i]))
    return output


# ------------------------输出合并
def output_dtype_post_process(output, dtype):
    output_new = [None] * len(output)
    for i in range(len(output)):
        if dtype == "bfloat16":
            output_new[i] = output[i].to(torch.bfloat16)
        elif dtype == "float16":
            output_new[i] = output[i].half()
        elif dtype == "int8":
            output_new[i] = output[i].to(torch.int8)
        else:
            output_new[i] = output[i].float()
    return output_new


def golden_output_process(
    output,
    param_split_item,
    group_type,
    activation_feature_out=None,
    dyn_scale_out=None,
):
    golden = []

    if param_split_item in [2, 3]:
        if group_type == 0:
            ret = torch.cat(output, dim=-2)
            golden.append(ret)
        elif group_type == 2:
            ret = torch.stack(output, dim=0)
            golden.append(ret)
        else:
            raise TypeError("unsupported group type, please check")
        # 检查inf、nan的数量
        num_inf = torch.isinf(golden[0]).sum().item()
        num_nan = torch.isnan(golden[0]).sum().item()
        ratio = (
            100 * ((num_inf + num_nan) / (ret.shape[0] * ret.shape[1]))
            if ret.shape[0] * ret.shape[1] != 0
            else 0
        )
    else:
        golden = output
    return golden


def torch_to_numpy(x):
    if isinstance(x, torch.Tensor):
        if x.dtype == torch.float8_e5m2:
            x = x.to(torch.float32).numpy().astype(ml_dtypes.float8_e5m2)
        elif x.dtype == torch.float8_e4m3fn:
            x = x.to(torch.float32).numpy().astype(ml_dtypes.float8_e4m3fn)
        elif x.dtype == torch.bfloat16:
            x = x.to(torch.float32).numpy().astype(ml_dtypes.bfloat16)
        else:
            x = x.numpy()
    return x
