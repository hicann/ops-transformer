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
        "grouped_matmul_finalize_routing": "grouped_matmul_finalize_routing_inputs"
    }
}

from typing import List
import numpy as np
from ml_dtypes import bfloat16


def grouped_matmul_finalize_routing_inputs(
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
    x1 = x
    x2 = w
    group_list_shape = group_list
    group_list_dtype = group_list.dtype
    row_index_test = []
    for i in range(len(row_index) // group_list_shape.shape[0]):
        row_index_test.extend([i] * group_list_shape.shape[0])
    remain = len(row_index) % group_list_shape.shape[0]
    if remain:
        row_index_test.extend([0] * remain)
    row_index_test = np.array(row_index_test, dtype=np.int64)
    row_index = row_index_test
    if "group_list_expect" in kwargs:
        group_list_new = kwargs["group_list_expect"]
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
    bias_n = bias.shape[-1]
    bias = np.zeros((x2.shape[0], bias_n)).astype(
        bfloat16
    )  # shape=(e, n) 的全 0 ndarray
    return (
        x1,
        x2,
        scale,
        bias,
        pertoken_scale,
        np.array(group_list_new, dtype=group_list_dtype),
        shared_input,
        logit,
        row_index,
        None,
    )
