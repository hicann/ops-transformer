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
"""GMMIA kernel inputs use physical K-major matrices."""

from pathlib import Path
from runpy import run_path

_ASSETS = Path(__file__).resolve().parent
_REF = run_path(str(_ASSETS / "gmmia_reference.py"))
_as_array, _inplace = _REF["as_array"], _REF["inplace"]
_prepare = run_path(str(_ASSETS / "inputs.py"))["_prepare_gmmia_inputs"]


class KernelSpec:
    @staticmethod
    def customize_inputs(
        x1,
        x2,
        scale2,
        group_list,
        y,
        scale1=None,
        *,
        group_list_type=0,
        ttk_group_list=None,
        **kwargs,
    ):
        _prepare(x1, x2, scale1, scale2, group_list, group_list_type)
        if ttk_group_list is not None:
            group_list[...] = ttk_group_list
        return x1, x2, scale2, group_list, y, scale1

    @staticmethod
    def golden(
        x1, x2, scale2, group_list, y, scale1=None, *, group_list_type=0, **kwargs
    ):
        tensors = (x1, x2, scale1, scale2, group_list, y)
        return _inplace(*map(_as_array, tensors), group_list_type)


__spec__ = {"quant_grouped_matmul_inplace_add": "KernelSpec"}
