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
"""Torch extension reference and device-side preparation of FRACTAL_NZ weight."""

from pathlib import Path
from runpy import run_path

import en_dtypes
import numpy as np

_ASSETS = Path(__file__).resolve().parent
_REFERENCE = run_path(str(_ASSETS / "gmmaq_reference.py"))
grouped_matmul_activation_quant_golden = _REFERENCE[
    "grouped_matmul_activation_quant_golden"
]
_make_valid_group_list = run_path(str(_ASSETS / "inputs.py"))["_make_valid_group_list"]
as_numpy = _REFERENCE["as_numpy"]


def _decode_scale(tensor):
    return as_numpy(tensor).view(en_dtypes.float8_e8m0).astype(np.float32)


class GmmaqTorchTestSpec:
    @staticmethod
    def customize_inputs(
        x,
        group_list,
        weight,
        weight_scale,
        activation_type,
        *,
        group_list_type=0,
        ttk_group_list=None,
        **kwargs,
    ):
        group_list[...] = group_list.new_tensor(
            ttk_group_list
            if ttk_group_list is not None
            else _make_valid_group_list(
                as_numpy(group_list), x.shape[0], group_list_type, np.int64
            )
        )

    @staticmethod
    def npu_preprocess(
        x,
        group_list,
        weight,
        weight_scale,
        activation_type,
        *,
        ttk_transpose_weight=False,
        **kwargs,
    ):
        import torch_npu

        (source,) = weight
        if ttk_transpose_weight:
            # TTK generates dense TensorList members. Restore the API's transpose views.
            source = source.transpose(-1, -2).contiguous()
            weight_scale[0] = (
                weight_scale[0].transpose(-3, -2).contiguous().transpose(-3, -2)
            )
        converted = torch_npu.npu_format_cast(source, 29)
        weight[0] = converted.transpose(-1, -2) if ttk_transpose_weight else converted

    @staticmethod
    def golden(
        x,
        group_list,
        weight,
        weight_scale,
        activation_type,
        *,
        bias=None,
        x_scale,
        group_list_type=0,
        y_dtype=None,
        scale_alg=0,
        **kwargs,
    ):
        return grouped_matmul_activation_quant_golden(
            as_numpy(x),
            as_numpy(group_list),
            as_numpy(weight[0]),
            _decode_scale(weight_scale[0]),
            bias,
            activation_type,
            x_scale=_decode_scale(x_scale),
            group_list_type=group_list_type,
            y_dtype=y_dtype
            if y_dtype is not None
            else (35 if str(x.dtype).endswith("e5m2") else 36),
            scale_alg=scale_alg,
        )

    @staticmethod
    def pre_compare(y, y_scale, expected_y, expected_scale):
        if y_scale.dtype in (np.int8, np.uint8):
            # Integer carrier outputs are compared as E8M0 encodings.
            expected_scale = expected_scale.astype(en_dtypes.float8_e8m0).view(
                y_scale.dtype
            )
        return y, y_scale, expected_y, expected_scale


__spec__ = {
    "torch.ops.cann_ops_transformer.grouped_matmul_activation_quant": "GmmaqTorchTestSpec"
}
