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

"""Two-party TestSpec adapter for KvQuantSparseFlashAttentionV2 assets."""

import importlib.util
import sys
from pathlib import Path

ASSET_IMPL_DIR = Path(__file__).with_name("impl")
PACKAGE = "_kv_quant_sparse_flash_attention_v2_ttk_impl"
if PACKAGE not in sys.modules:
    package_spec = importlib.util.spec_from_file_location(
        PACKAGE,
        ASSET_IMPL_DIR / "__init__.py",
        # Standalone plugins also import the operator-local pytest golden.
        submodule_search_locations=[
            str(ASSET_IMPL_DIR),
            str(ASSET_IMPL_DIR.parents[1] / "pytest"),
        ],
    )
    package = importlib.util.module_from_spec(package_spec)
    sys.modules[PACKAGE] = package
    package_spec.loader.exec_module(package)


def load_impl_module(stem):
    return importlib.import_module(f"{PACKAGE}.{stem}")


golden_module = load_impl_module("golden")
inputs_module = load_impl_module("inputs")
compare_module = load_impl_module("compare")


class KvQuantSparseFlashAttentionV2Spec:
    golden = golden_module.cpu_kv_quant_sparse_flash_attention_v2
    customize_inputs = inputs_module.generate_qsfa_v2_inputs
    compare = staticmethod(compare_module.compare)
    tolerance = {
        dtype: {"standard": "close"} for dtype in ("float16", "bfloat16", "float32")
    }


__spec__ = {
    "torch.ops.cann_ops_transformer.kv_quant_sparse_flash_attention": "KvQuantSparseFlashAttentionV2Spec",
    "torch_npu.npu_kv_quant_sparse_flash_attention_v2": "KvQuantSparseFlashAttentionV2Spec",
    "qsfa_v2_ttk_ops.kv_quant_sparse_flash_attention_v2_ttk": "KvQuantSparseFlashAttentionV2Spec",
}
