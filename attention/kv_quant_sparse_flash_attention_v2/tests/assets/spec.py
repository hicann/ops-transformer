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

"""TestSpec adapter for KvQuantSparseFlashAttentionV2 assets."""

import importlib.util
import sys
from pathlib import Path


ASSET_IMPL_DIR = Path(__file__).with_name("impl")


def load_impl_module(stem):
    name = f"qsfa_v2_ttk_{stem}"
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
            "Failed to load KvQuantSparseFlashAttentionV2 assets module; "
            f"stage=impl/{stem}; module={path.resolve()}; "
            f"original error: {type(exc).__name__}: {exc}"
        ) from exc
    return module


golden_module = load_impl_module("golden")
inputs_module = load_impl_module("inputs")
compare_module = load_impl_module("compare")


class KvQuantSparseFlashAttentionV2Spec:
    golden = golden_module.cpu_kv_quant_sparse_flash_attention_v2
    customize_inputs = inputs_module.generate_qsfa_v2_inputs
    tolerance = {
        "float16": {"standard": "stat_rel_err"},
        "bfloat16": {"standard": "stat_rel_err"},
    }

    compare = staticmethod(compare_module.compare)


__spec__ = {
    "torch.ops.cann_ops_transformer.kv_quant_sparse_flash_attention": "KvQuantSparseFlashAttentionV2Spec",
    "torch_npu.npu_kv_quant_sparse_flash_attention_v2": "KvQuantSparseFlashAttentionV2Spec",
    "qsfa_v2_ttk_ops.kv_quant_sparse_flash_attention_v2_ttk": "KvQuantSparseFlashAttentionV2Spec",
}
