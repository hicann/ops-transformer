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

"""TestSpec adapter for QuantSparseFlashMla TTK assets."""

import importlib.util
import sys
from pathlib import Path

ASSET_IMPL_DIR = Path(__file__).with_name("impl")


def load_impl_module(stem):
    name = f"qsmla_ttk_{stem}"
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
            "Failed to load QuantSparseFlashMla assets module; "
            f"stage=impl/{stem}; module={path.resolve()}; "
            f"original error: {type(exc).__name__}: {exc}"
        ) from exc
    return module


npu_preprocess_module = load_impl_module("npu_preprocess")
golden_module = load_impl_module("golden")
inputs_module = load_impl_module("inputs")
metadata_inputs_module = load_impl_module("metadata_inputs")
compare_module = load_impl_module("compare")


class QuantSparseFlashMlaSpec:
    golden = golden_module.cpu_quant_sparse_flash_mla
    customize_inputs = inputs_module.generate_quant_sparse_flash_mla_inputs
    npu_preprocess = npu_preprocess_module.run
    tolerance = {
        "bfloat16": {"standard": "stat_rel_err"},
        "float16": {"standard": "stat_rel_err"},
    }

    compare = staticmethod(compare_module.compare)


class AclnnQuantSparseFlashMlaSpec(QuantSparseFlashMlaSpec):
    golden = golden_module.cpu_aclnn_quant_sparse_flash_mla
    customize_inputs = inputs_module.generate_aclnn_quant_sparse_flash_mla_inputs
    npu_preprocess = npu_preprocess_module.run_aclnn


class QuantSparseFlashMlaMetadataSpec:
    customize_inputs = (
        metadata_inputs_module.generate_quant_sparse_flash_mla_metadata_inputs
    )


__spec__ = {
    "torch.ops.cann_ops_transformer.quant_sparse_flash_mla": "QuantSparseFlashMlaSpec",
    "torch.ops.cann_ops_transformer.quant_sparse_flash_mla_metadata": (
        "QuantSparseFlashMlaMetadataSpec"
    ),
    "aclnnQuantSparseFlashMla": "AclnnQuantSparseFlashMlaSpec",
}
