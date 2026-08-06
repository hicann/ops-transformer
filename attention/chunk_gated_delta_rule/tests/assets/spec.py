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

"""TestSpec adapter for ChunkGatedDeltaRule assets."""

import importlib.util
from pathlib import Path


ASSET_IMPL_DIR = Path(__file__).with_name("impl")


def load_impl_module(stem):
    path = ASSET_IMPL_DIR / f"{stem}.py"
    spec = importlib.util.spec_from_file_location(
        f"cgdr_assets_impl_{stem}_{abs(hash(path))}", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


golden_module = load_impl_module("golden")
inputs_module = load_impl_module("inputs")
compare_module = load_impl_module("compare")


class ChunkGatedDeltaRuleSpec:
    golden = golden_module.cpu_chunk_gated_delta_rule
    customize_inputs = inputs_module.generate_cgdr_inputs
    tolerance = {
        "bfloat16": {"standard": "cross_check", "level": "L1"},
    }

    def compare(*outputs, **kwargs):
        ctx = golden_module.get_golden_context()
        kwargs["bench_out"] = ctx.get("bench_out")
        kwargs["bench_state"] = ctx.get("bench_state")
        return compare_module.compare(*outputs, **kwargs)


class AclnnChunkGatedDeltaRuleSpec:
    golden = golden_module.aclnn_chunk_gated_delta_rule_golden
    customize_inputs = inputs_module.aclnn_chunk_gated_delta_rule_input
    tolerance = {
        "bfloat16": {"standard": "cross_check", "level": "L1"},
    }

    def compare(*outputs, **kwargs):
        ctx = golden_module.get_golden_context()
        kwargs["bench_out"] = ctx.get("bench_out")
        kwargs["bench_state"] = ctx.get("bench_state")
        return compare_module.compare(*outputs, **kwargs)


__spec__ = {
    "torch_npu.npu_chunk_gated_delta_rule": "ChunkGatedDeltaRuleSpec",
    "aclnnChunkGatedDeltaRule": "AclnnChunkGatedDeltaRuleSpec",
}
