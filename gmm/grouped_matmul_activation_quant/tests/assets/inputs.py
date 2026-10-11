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
__input__ = {
    "kernel": {
        "grouped_matmul_activation_quant": "grouped_matmul_activation_quant_inputs"
    }
}

import numpy as np


N_ALIGN_SIZE = 64


def grouped_matmul_activation_quant_inputs(
    x,
    group_list,
    weight,
    weight_scale,
    bias,
    activation_type: str,
    *,
    x_scale=None,
    transpose_weight: bool = False,
    group_list_type: int = 0,
    tuning_config=None,
    quant_mode: str = "",
    y_dtype: int = 36,
    round_mode: str = "rint",
    scale_alg: int = 0,
    dst_type_max: float = 0.0,
    **kwargs,
):
    del activation_type, tuning_config, y_dtype
    del round_mode, scale_alg, dst_type_max
    if quant_mode in (None, ""):
        quant_mode = "mx"
    if quant_mode != "mx":
        raise ValueError("quant_mode 参数当前仅支持为 mx")
    if x_scale is None:
        raise ValueError("x_scale is required for mx quant mode.")

    group_list_dtype = group_list.dtype
    if "group_list_expect" in kwargs:
        group_list_new = np.array(kwargs["group_list_expect"], dtype=group_list_dtype)
    else:
        group_list_new = _make_valid_group_list(
            group_list, x.shape[0], group_list_type, group_list_dtype
        )

    group_list_cumsum = (
        np.cumsum(group_list_new) if group_list_type == 1 else group_list_new
    )
    if group_list_cumsum.size > 0 and group_list_cumsum[-1] > x.shape[0]:
        raise ValueError(
            "sum of group_list ({}) can not be greater than x.shape[0] ({})".format(
                group_list_cumsum[-1], x.shape[0]
            )
        )

    n = _get_n_size(
        _first_tensor(weight), _first_tensor(weight_scale), transpose_weight
    )
    if n % N_ALIGN_SIZE != 0:
        raise ValueError(
            "N must be a multiple of {}, actual is {}".format(N_ALIGN_SIZE, n)
        )

    return x, group_list_new, weight, weight_scale, bias, x_scale


def _first_tensor(x):
    if isinstance(x, (list, tuple)):
        return x[0] if x else None
    return x


def _make_valid_group_list(group_list, m, group_list_type, dtype):
    group_num = int(np.asarray(group_list).size)
    if group_num <= 0:
        return np.asarray(group_list, dtype=dtype)

    base = m // group_num
    remain = m % group_num
    group_sizes = np.full((group_num,), base, dtype=np.int64)
    group_sizes[:remain] += 1
    if group_list_type == 1:
        return group_sizes.astype(dtype)
    return np.cumsum(group_sizes).astype(dtype)


def _get_n_size(weight, weight_scale, transpose_weight):
    if weight_scale is not None and weight_scale.ndim >= 3:
        return weight_scale.shape[1] if transpose_weight else weight_scale.shape[-2]
    if weight is not None and weight.ndim >= 5:
        return (
            weight.shape[2] * weight.shape[3]
            if transpose_weight
            else weight.shape[1] * weight.shape[4]
        )
    if weight is not None and weight.ndim >= 3:
        return weight.shape[-2] if transpose_weight else weight.shape[-1]
    raise ValueError("can not infer N from weight or weight_scale")
