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

"""TestSpec adapter for SparseFlashAttention assets."""

import importlib.util
import sys
from pathlib import Path


ASSET_IMPL_DIR = Path(__file__).with_name("impl")


def load_impl_module(stem):
    name = f"sfa_ttk_{stem}"
    if name in sys.modules:
        return sys.modules[name]
    path = ASSET_IMPL_DIR / f"{stem}.py"
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot create import spec for {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    except Exception as exc:
        sys.modules.pop(name, None)
        raise RuntimeError(
            "Failed to load SparseFlashAttention assets module; "
            f"stage=impl/{stem}; module={path.resolve()}; "
            f"original error: {type(exc).__name__}: {exc}"
        ) from exc
    return module


golden_module = load_impl_module("golden")
inputs_module = load_impl_module("inputs")
compare_module = load_impl_module("compare")


# cross_check 三方精度标准（L0）：NPU 对标杆的 mare/mere/rmse 不得超过
# golden 对标杆的 10/2/2 倍，小值域错误点数不超过 2 倍。
CROSS_CHECK_TOLERANCE = {
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


def _sfa_compare(*outputs, compare_context=None, **kwargs):
    testcase_name = (
        getattr(compare_context, "testcase_name", None)
        if compare_context is not None
        else None
    )
    return compare_module.dispatch(
        *outputs,
        bench_outputs=golden_module.peek_bench(testcase_name),
        spec_tolerance=CROSS_CHECK_TOLERANCE,
    )


class SparseFlashAttentionSpec:
    golden = golden_module.cpu_sparse_flash_attention
    customize_inputs = inputs_module.generate_sfa_inputs
    tolerance = CROSS_CHECK_TOLERANCE

    compare = staticmethod(_sfa_compare)


class SparseFlashAttentionAclnnV2Spec:
    """ACLNN 直调版 spec：golden / input 使用 aclnn C 签名顺序的 wrapper。
    golden 与 e2e 共用同一份 pytest CPU 参考实现（按 testcase_name 从 CASE_DATA 取回），
    仅参数布局不同：tensors 在前、9 个 camelCase others 在后，且包含输出 tensors。
    """

    golden = golden_module.cpu_sparse_flash_attention_aclnn
    customize_inputs = inputs_module.generate_sfa_inputs_aclnn_v2
    tolerance = CROSS_CHECK_TOLERANCE
    compare = staticmethod(_sfa_compare)


class SparseFlashAttentionAclnnSpec(SparseFlashAttentionAclnnV2Spec):
    customize_inputs = inputs_module.generate_sfa_inputs_aclnn


__spec__ = {
    "torch_npu.npu_sparse_flash_attention": "SparseFlashAttentionSpec",
    "aclnnSparseFlashAttention": "SparseFlashAttentionAclnnSpec",
    "aclnnSparseFlashAttentionV2": "SparseFlashAttentionAclnnV2Spec",
}
