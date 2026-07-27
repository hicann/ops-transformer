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
        f"qfa_assets_impl_{stem}_{abs(hash(path))}", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


golden_module = load_impl_module("golden")
inputs_module = load_impl_module("inputs")
compare_module = load_impl_module("compare")
graph_module = load_impl_module("graph")


class QfaMxfp8Spec:
    """quant_flash_attn (MXFP8) 测试规范。"""

    golden = golden_module.cpu_qfa_mxfp8
    customize_inputs = inputs_module.generate_qfa_mxfp8_inputs
    compare = compare_module.compare

    torch_graph = graph_module.QuantFlashAttnAclGraph

    tolerance = {
        "float16": {
            "rtol": 0.005,
            "ptol": 0.005,
            "atol": 0.000025,
        },
        "bfloat16": {
            "rtol": 0.0078125,
            "ptol": 0.005,
            "atol": 0.0001,
        },
    }


class QfaMetadataSpec:
    """quant_flash_attn_metadata 独立测试规范 (T4 metadata/main 分离)。

    复用 QfaMxfp8Spec 的 golden / customize_inputs / compare / tolerance:
    输入生成与 golden 计算路径与合并测试一致, 仅 wrapper 分流到 metadata-only op。
    """

    golden = golden_module.cpu_qfa_mxfp8
    customize_inputs = inputs_module.generate_qfa_mxfp8_inputs
    compare = compare_module.compare

    tolerance = {
        "float16": {
            "rtol": 0.005,
            "ptol": 0.005,
            "atol": 0.000025,
        },
        "bfloat16": {
            "rtol": 0.0078125,
            "ptol": 0.005,
            "atol": 0.0001,
        },
    }


class QfaMainSpec:
    """quant_flash_attn 主算子独立测试规范 (T4 metadata/main 分离)。

    run_main 内部重建 metadata 后调主算子; golden / customize_inputs / compare
    与合并测试一致, 仅 wrapper 分流到 main-only 路径。
    """

    golden = golden_module.cpu_qfa_mxfp8
    customize_inputs = inputs_module.generate_qfa_mxfp8_inputs
    compare = compare_module.compare

    tolerance = {
        "float16": {
            "rtol": 0.005,
            "ptol": 0.005,
            "atol": 0.000025,
        },
        "bfloat16": {
            "rtol": 0.0078125,
            "ptol": 0.005,
            "atol": 0.0001,
        },
    }


__spec__ = {
    "qfa_mxfp8_wrapper.npu_qfa_mxfp8": "QfaMxfp8Spec",
    "qfa_metadata_wrapper.run_metadata": "QfaMetadataSpec",
    "qfa_main_wrapper.run_main": "QfaMainSpec",
}
