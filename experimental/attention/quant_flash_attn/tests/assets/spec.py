#!/usr/bin/python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import importlib.util
from pathlib import Path


ASSET_IMPL_DIR = Path(__file__).with_name("impl")


def load_impl_module(stem):
    path = ASSET_IMPL_DIR / f"{stem}.py"
    spec = importlib.util.spec_from_file_location(
        f"qfa_mxfp4_assets_impl_{stem}_{abs(hash(path))}", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


golden_module = load_impl_module("golden")
inputs_module = load_impl_module("inputs")
compare_module = load_impl_module("compare")


class QfaMxfp4Spec:
    """quant_flash_attn (MXFP4) 测试规范."""

    golden = golden_module.cpu_qfa_mxfp4
    customize_inputs = inputs_module.generate_qfa_mxfp4_inputs
    compare = compare_module.compare

    tolerance = {
        "bfloat16": {
            "standard": "stat_rel_err",
            "rtol": 0.0078125,
            "ptol": 0.005,
            "atol": 0.0001,
        },
        "float16": {
            "standard": "stat_rel_err",
            "rtol": 0.005,
            "ptol": 0.005,
            "atol": 0.000025,
        },
    }


__spec__ = {
    "qfa_mxfp4_wrapper.npu_qfa_mxfp4": "QfaMxfp4Spec",
}
