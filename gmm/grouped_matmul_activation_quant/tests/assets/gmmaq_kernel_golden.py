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
"""Kernel reference for the physical NZ and scale descriptors in the ST table."""

from pathlib import Path
from runpy import run_path

_ASSETS = Path(__file__).resolve().parent
_REFERENCE = run_path(str(_ASSETS / "gmmaq_reference.py"))
grouped_matmul_activation_quant_golden = _REFERENCE[
    "grouped_matmul_activation_quant_golden"
]
_make_valid_group_list = run_path(str(_ASSETS / "inputs.py"))["_make_valid_group_list"]


class GroupedMatmulActivationQuantTestSpec:
    @staticmethod
    def customize_inputs(
        x,
        group_list,
        weight,
        weight_scale,
        bias,
        x_scale,
        *,
        group_list_type=0,
        ttk_group_list=None,
        **kwargs,
    ):
        group_list[...] = (
            ttk_group_list
            if ttk_group_list is not None
            else _make_valid_group_list(
                group_list, x.shape[0], group_list_type, group_list.dtype
            )
        )

        return x, group_list, weight, weight_scale, bias, x_scale

    @staticmethod
    def golden(
        x,
        group_list,
        weight,
        weight_scale,
        bias,
        x_scale,
        *,
        activation_type="gelu_tanh",
        **kwargs,
    ):
        # The kernel's y_dtype controls quantization even when TTK promotes outputs for comparison.
        kwargs.pop("output_dtypes", None)
        return grouped_matmul_activation_quant_golden(
            x,
            group_list,
            weight,
            weight_scale,
            bias,
            activation_type,
            x_scale=x_scale,
            **kwargs,
        )


__spec__ = {"grouped_matmul_activation_quant": "GroupedMatmulActivationQuantTestSpec"}
