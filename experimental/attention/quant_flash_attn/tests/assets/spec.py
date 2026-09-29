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

_impl_cache = {}


def load_impl_module(stem):
    if stem not in _impl_cache:
        path = ASSET_IMPL_DIR / f"{stem}.py"
        spec = importlib.util.spec_from_file_location(
            f"qfa_mxfp4_assets_impl_{stem}_{abs(hash(path))}", path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _impl_cache[stem] = module
    return _impl_cache[stem]


def _golden_dispatch(*args, **kwargs):
    return load_impl_module("golden").cpu_qfa_mxfp4(*args, **kwargs)


def _inputs_dispatch(*args, **kwargs):
    return load_impl_module("inputs").generate_qfa_mxfp4_inputs(*args, **kwargs)


class QuantFlashAttnMxfp4Spec:
    """quant_flash_attn (MXFP4, quant_mode=5) 测试规范."""

    golden = staticmethod(_golden_dispatch)
    customize_inputs = staticmethod(_inputs_dispatch)

    @staticmethod
    def compare(*outputs, **kwargs):
        return load_impl_module("compare").compare(*outputs, **kwargs)

    @staticmethod
    def npu_preprocess(*args, **kwargs):
        return load_impl_module("npu_preprocess").run(*args, **kwargs)

    torch_graph = load_impl_module("graph").QuantFlashAttnMxfp4AclGraph

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


class QuantFlashAttnMxfp4MetadataSpec:
    """quant_flash_attn_metadata 生成器的 TestSpec."""

    customize_inputs = load_impl_module(
        "metadata_inputs"
    ).generate_quant_flash_attn_metadata_inputs


__spec__ = {
    "torch.ops.cann_ops_transformer.quant_flash_attn": "QuantFlashAttnMxfp4Spec",
    "torch.ops.cann_ops_transformer.quant_flash_attn_metadata": (
        "QuantFlashAttnMxfp4MetadataSpec"
    ),
}
