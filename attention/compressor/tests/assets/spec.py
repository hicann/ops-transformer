#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import importlib.util
from pathlib import Path


ASSET_IMPL_DIR = Path(__file__).with_name("impl")


def load_impl_module(stem):
    path = ASSET_IMPL_DIR / f"{stem}.py"
    spec = importlib.util.spec_from_file_location(
        f"compressor_assets_impl_{stem}_{abs(hash(path))}", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


golden_module = load_impl_module("golden")
inputs_module = load_impl_module("inputs")
compare_module = load_impl_module("compare")
graph_module = load_impl_module("graph")


class CompressorSpec:
    golden = golden_module.cpu_compressor
    customize_inputs = inputs_module.generate_compressor_inputs
    torch_graph = graph_module.CompressorGraphNetwork
    tolerance = {
        "float16": {
            "standard": "cross_check",
            "level": "L0",
            "mare_ratio": 10.0,
            "mere_ratio": 2.0,
            "rmse_ratio": 2.0,
            "small_value": 2**-10,
            "small_value_atol": 2**-16,
        },
        "bfloat16": {
            "standard": "cross_check",
            "level": "L0",
            "mare_ratio": 10.0,
            "mere_ratio": 2.0,
            "rmse_ratio": 2.0,
            "small_value": 2**-10,
            "small_value_atol": 2**-16,
        },
    }

    def compare(*outputs, compare_context=None, **kwargs):
        compare_method = compare_module._get_compare_method()
        golden_module.rebuild_golden_context_from_compare_context(
            compare_context, api_kind="e2e"
        )
        ctx = golden_module.get_golden_context()
        kwargs["cmp_kv_mask"] = ctx.get("cmp_kv_mask")
        kwargs["update_kv"] = ctx.get("update_kv")
        kwargs["update_score"] = ctx.get("update_score")
        kwargs["start_pos_list"] = ctx.get("start_pos_list")
        kwargs["seqused_list"] = ctx.get("seqused_list")
        kwargs["cu_seqlens_list"] = ctx.get("cu_seqlens_list")
        kwargs["cmp_ratio"] = ctx.get("cmp_ratio")
        kwargs["is_th"] = ctx.get("is_th")
        kwargs["data_type"] = ctx.get("data_type")
        bench_outputs = ctx.get("bench_outputs")
        if (
            compare_method == "cross_check"
            and bench_outputs is not None
            and compare_module.is_cross_check_available()
        ):
            return compare_module._run_ttk_cross_check_e2e(
                outputs, kwargs, bench_outputs, CompressorSpec.tolerance
            )
        return compare_module.compare(*outputs, **kwargs)


class AclnnCompressorSpec:
    golden = golden_module.aclnn_compressor_golden
    customize_inputs = inputs_module.aclnn_compressor_input
    tolerance = {
        "float16": {
            "standard": "cross_check",
            "level": "L0",
            "mare_ratio": 10.0,
            "mere_ratio": 2.0,
            "rmse_ratio": 2.0,
            "small_value": 2**-10,
            "small_value_atol": 2**-16,
        },
        "bfloat16": {
            "standard": "cross_check",
            "level": "L0",
            "mare_ratio": 10.0,
            "mere_ratio": 2.0,
            "rmse_ratio": 2.0,
            "small_value": 2**-10,
            "small_value_atol": 2**-16,
        },
    }

    def compare(*outputs, compare_context=None, **kwargs):
        compare_method = compare_module._get_compare_method()
        golden_module.rebuild_golden_context_from_compare_context(
            compare_context, api_kind="aclnn"
        )
        ctx = golden_module.get_golden_context()
        kwargs["cmp_kv_mask"] = ctx.get("cmp_kv_mask")
        kwargs["update_kv"] = ctx.get("update_kv")
        kwargs["update_score"] = ctx.get("update_score")
        kwargs["start_pos_list"] = ctx.get("start_pos_list")
        kwargs["seqused_list"] = ctx.get("seqused_list")
        kwargs["cu_seqlens_list"] = ctx.get("cu_seqlens_list")
        kwargs["cmp_ratio"] = ctx.get("cmp_ratio")
        kwargs["is_th"] = ctx.get("is_th")
        kwargs["gradEnabled"] = ctx.get("gradEnabled")
        kwargs["mid_result_mask"] = ctx.get("mid_result_mask")
        kwargs["data_type"] = ctx.get("data_type")
        bench_outputs = ctx.get("bench_outputs")
        if (
            compare_method == "cross_check"
            and bench_outputs is not None
            and compare_module.is_cross_check_available()
        ):
            return compare_module._run_ttk_cross_check_aclnn(
                outputs, kwargs, bench_outputs, AclnnCompressorSpec.tolerance
            )
        return compare_module.compare_aclnn(*outputs, **kwargs)


__spec__ = {
    "cann_ops_transformer.ops.compressor": "CompressorSpec",
    "aclnnCompressor": "AclnnCompressorSpec",
}
