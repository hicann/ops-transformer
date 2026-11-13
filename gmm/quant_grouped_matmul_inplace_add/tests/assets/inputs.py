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
        "quant_grouped_matmul_inplace_add": "quant_grouped_matmul_inplace_add_inputs"
    }
}

import numpy as np


def quant_grouped_matmul_inplace_add_inputs(
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
    input_deq_scale = scale2
    pertoken_scale = scale1

    group_list_dtype = group_list.dtype
    if "group_list_expect" in kwargs:
        group_list_expect = kwargs[
            "group_list_expect"
        ]  # 全量化组需必传group_list_expect
        group_list_new = group_list_expect
    else:
        group_list_new = group_list
    group_list_tmp = group_list_new
    if group_list_type == 1:
        group_list_tmp = np.cumsum(group_list_new)
    if group_list_tmp[-1] > x1.shape[0]:
        raise Exception(
            "sum of grouplist: ({}) can not be greater than x1[0]: ({})".format(
                group_list_tmp[-1], x1.shape[0]
            )
        )
    return (
        x1,
        x2,
        input_deq_scale,
        np.array(group_list_new, dtype=group_list_dtype),
        y,
        pertoken_scale,
    )


def _copy_gmmia_input(target, values):
    if hasattr(target, "new_tensor"):
        target[...] = target.new_tensor(values)
    else:
        target[...] = values


def _prepare_gmmia_inputs(
    x1, x2, scale1, scale2, groups, group_list_type, *, hifloat8=False
):
    """Generate block-aligned groups and finite MX scales for all routes."""
    import math

    experts = int(groups.shape[0])
    k = x2.shape[0]
    counts = np.full(experts, (k // 64 // experts) * 64, dtype=np.int64)
    counts[-1] += k - int(counts.sum())
    _copy_gmmia_input(groups, counts if group_list_type == 1 else counts.cumsum())
    if len(scale2.shape) == 3:
        for scale in (scale1, scale2):
            shape = tuple(scale.shape)
            exponents = (np.arange(math.prod(shape)).reshape(shape) % 3) - 1
            data = (
                (exponents + 127).astype(np.uint8)
                if str(scale.dtype).endswith("uint8")
                else np.exp2(exponents)
            )
            _copy_gmmia_input(scale, data)
    if hifloat8:
        import en_dtypes

        for value in (x1, x2):
            shape = tuple(value.shape)
            numbers = (np.arange(math.prod(shape)).reshape(shape) % 9 - 4).astype(
                np.float32
            ) * 0.25
            _copy_gmmia_input(value, numbers.astype(en_dtypes.hifloat8).view(np.uint8))
