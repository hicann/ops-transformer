#!/usr/bin/python
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

"""TestSpec adapter for QuantCompressor assets."""

import importlib.util
from pathlib import Path


ASSET_IMPL_DIR = Path(__file__).with_name("impl")


def load_impl_module(stem):
    path = ASSET_IMPL_DIR / f"{stem}.py"
    spec = importlib.util.spec_from_file_location(
        f"quant_compressor_assets_impl_{stem}_{abs(hash(path))}", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


golden_module = load_impl_module("golden")
inputs_module = load_impl_module("inputs")
compare_module = load_impl_module("compare")


class QuantCompressorSpec:
    golden = golden_module.cpu_quant_compressor
    customize_inputs = inputs_module.generate_quant_compressor_inputs
    tolerance = {
        "float16": {"standard": "stat_rel_err"},
        "bfloat16": {"standard": "stat_rel_err"},
    }

    def compare(*outputs, **kwargs):
        ctx = golden_module.get_golden_context()
        kwargs["cmp_kv_mask"] = ctx.get("cmp_kv_mask")
        kwargs["update_kv"] = ctx.get("update_kv")
        kwargs["update_score"] = ctx.get("update_score")
        kwargs["start_pos_list"] = ctx.get("start_pos_list")
        kwargs["seqused_list"] = ctx.get("seqused_list")
        kwargs["cu_seqlens_list"] = ctx.get("cu_seqlens_list")
        kwargs["cmp_ratio"] = ctx.get("cmp_ratio")
        kwargs["is_th"] = ctx.get("is_th")
        return compare_module.compare(*outputs, **kwargs)


class KernelQuantCompressorSpec:
    golden = golden_module.kernel_quant_compressor_golden
    customize_inputs = inputs_module.kernel_quant_compressor_input
    tolerance = {
        "float16": {"standard": "stat_rel_err"},
        "bfloat16": {"standard": "stat_rel_err"},
    }

    def compare(*outputs, **kwargs):
        ctx = golden_module.get_golden_context()
        kwargs["cmp_kv_mask"] = ctx.get("cmp_kv_mask")
        kwargs["update_kv"] = ctx.get("update_kv")
        kwargs["update_score"] = ctx.get("update_score")
        kwargs["start_pos_list"] = ctx.get("start_pos_list")
        kwargs["seqused_list"] = ctx.get("seqused_list")
        kwargs["cu_seqlens_list"] = ctx.get("cu_seqlens_list")
        kwargs["cmp_ratio"] = ctx.get("cmp_ratio")
        kwargs["is_th"] = ctx.get("is_th")
        return compare_module.compare(*outputs, **kwargs)


class AclnnQuantCompressorSpec:
    golden = golden_module.aclnn_quant_compressor_golden
    customize_inputs = inputs_module.aclnn_quant_compressor_input
    tolerance = {
        "float16": {"standard": "stat_rel_err"},
        "bfloat16": {"standard": "stat_rel_err"},
    }

    def compare(*outputs, **kwargs):
        ctx = golden_module.get_golden_context()
        kwargs["cmp_kv_mask"] = ctx.get("cmp_kv_mask")
        kwargs["update_kv"] = ctx.get("update_kv")
        kwargs["update_score"] = ctx.get("update_score")
        kwargs["start_pos_list"] = ctx.get("start_pos_list")
        kwargs["seqused_list"] = ctx.get("seqused_list")
        kwargs["cu_seqlens_list"] = ctx.get("cu_seqlens_list")
        kwargs["cmp_ratio"] = ctx.get("cmp_ratio")
        kwargs["is_th"] = ctx.get("is_th")
        return compare_module.compare_aclnn(*outputs, **kwargs)


__spec__ = {
    "quant_compressor": "KernelQuantCompressorSpec",
    "cann_ops_transformer.ops.quant_compressor": "QuantCompressorSpec",
    "aclnnQuantCompressor": "AclnnQuantCompressorSpec",
}
