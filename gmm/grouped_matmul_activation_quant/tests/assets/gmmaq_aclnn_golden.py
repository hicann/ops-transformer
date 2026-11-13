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
"""ACLNN reference for an NZ allocation exposed through a logical weight view."""

from pathlib import Path
from runpy import run_path

import numpy as np

_ASSETS = Path(__file__).resolve().parent
_REFERENCE = run_path(str(_ASSETS / "gmmaq_reference.py"))
grouped_matmul_activation_quant_golden = _REFERENCE[
    "grouped_matmul_activation_quant_golden"
]
_make_valid_group_list = run_path(str(_ASSETS / "inputs.py"))["_make_valid_group_list"]
as_numpy = _REFERENCE["as_numpy"]


def _nz_storage(weight, transpose):
    e, k, n = weight.shape
    rows, cols = (n, k) if transpose else (k, n)
    shape = (e, (cols + 31) // 32, (rows + 15) // 16, 16, 32)
    if hasattr(weight, "untyped_storage"):
        size = weight.untyped_storage().nbytes() // weight.element_size()
        return as_numpy(weight.as_strided((size,), (1,), 0)).reshape(shape)
    backing = weight
    while isinstance(backing.base, np.ndarray):
        backing = backing.base
    return np.asarray(backing).reshape(shape)


class AclnnGroupedMatmulActivationQuantWeightNzTestSpec:
    @staticmethod
    def customize_inputs(
        x,
        groupList,
        weight,
        weightScale,
        bias,
        xScaleOptional,
        activationType,
        groupListType,
        *args,
        ttk_group_list=None,
        **kwargs,
    ):
        groups = as_numpy(groupList)
        values = (
            ttk_group_list
            if ttk_group_list is not None
            else _make_valid_group_list(groups, x.shape[0], groupListType, groups.dtype)
        )
        groupList[...] = (
            groupList.new_tensor(values) if hasattr(groupList, "new_tensor") else values
        )

    @staticmethod
    def golden(
        x,
        groupList,
        weight,
        weightScale,
        bias,
        xScaleOptional,
        activationType,
        groupListType,
        tuningConfig,
        quantMode,
        roundMode,
        scaleAlg,
        dstTypeMax,
        y=None,
        yScale=None,
        *,
        tensor_dtypes=(),
        **kwargs,
    ):
        (w,) = weight
        (ws,) = weightScale
        transpose = (
            w.stride(-2) == 1 if hasattr(w, "stride") else w.strides[-2] == w.itemsize
        )
        scale = as_numpy(ws)
        if transpose:
            scale = scale.swapaxes(-3, -2)
        return grouped_matmul_activation_quant_golden(
            as_numpy(x),
            as_numpy(groupList),
            _nz_storage(w, transpose),
            scale,
            bias,
            activationType,
            x_scale=as_numpy(xScaleOptional),
            transpose_weight=transpose,
            group_list_type=groupListType,
            quant_mode=quantMode,
            round_mode=roundMode,
            scale_alg=scaleAlg,
            dst_type_max=dstTypeMax,
            output_dtypes=tensor_dtypes[-2:],
        )


__spec__ = {
    "aclnnGroupedMatmulActivationQuantWeightNz": "AclnnGroupedMatmulActivationQuantWeightNzTestSpec"
}
