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
"""GMMIA ACLNN inputs expose a transposed x1 view and a mutable yRef."""

from pathlib import Path
from runpy import run_path

_ASSETS = Path(__file__).resolve().parent
_REF = run_path(str(_ASSETS / "gmmia_reference.py"))
_as_array, _inplace = _REF["as_array"], _REF["inplace"]
_INPUTS = run_path(str(_ASSETS / "inputs.py"))
_prepare = _INPUTS["_prepare_gmmia_inputs"]
_copy_input = _INPUTS["_copy_gmmia_input"]


class AclnnSpec:
    @staticmethod
    def customize_inputs(
        x1,
        x2,
        scale1Optional,
        scale2,
        groupList,
        yRef,
        groupListType,
        groupSize,
        ttk_group_list=None,
        **kwargs,
    ):
        _prepare(x1, x2, scale1Optional, scale2, groupList, groupListType)
        if ttk_group_list is not None:
            _copy_input(groupList, ttk_group_list)

    @staticmethod
    def golden(
        x1,
        x2,
        scale1Optional,
        scale2,
        groupList,
        yRef,
        groupListType,
        groupSize,
        **kwargs,
    ):
        scale1 = _as_array(scale1Optional)
        if scale1 is not None and scale1.ndim == 3:
            scale1 = scale1.swapaxes(0, 1)
        return _inplace(
            _as_array(x1).T,
            _as_array(x2),
            scale1,
            _as_array(scale2),
            _as_array(groupList),
            _as_array(yRef),
            groupListType,
        )


__spec__ = {"aclnnQuantGroupedMatmulInplaceAdd": "AclnnSpec"}
