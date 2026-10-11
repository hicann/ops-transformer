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
    "kernel": {"grouped_matmul_add": "grouped_matmul_add_inputs"},
}

import numpy as np


def grouped_matmul_add_inputs(
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
    x1 = x
    x2 = weight
    group_list_dtype = group_list.dtype
    if "group_list_expect" in kwargs:
        group_list_expect = kwargs["group_list_expect"]
        group_list = group_list_expect
    group_list_tmp = group_list
    if group_list_type == 1:
        group_list_tmp = np.cumsum(group_list)
    if group_list_tmp[-1] > x1.shape[0]:
        raise Exception(
            "sum of grouplist: ({}) can not be greater than x1[0]: ({})".format(
                group_list_tmp[-1], x1.shape[0]
            )
        )
    return x1, x2, np.array(group_list, dtype=group_list_dtype), y
