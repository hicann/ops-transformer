#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""TestSpec adapter for quant_block_sparse_attn TTK assets.

This spec is decoupled from TTK internals - no direct imports from ttk.* modules.
All impl modules are loaded lazily to avoid import-time failures when pytest golden
module is not available.
"""

import importlib.util
from pathlib import Path

from bsa_ttk_ops import QuantBlockSparseAttnGraph

ASSET_IMPL_DIR = Path(__file__).with_name("impl")

_impl_cache = {}


def _load_impl_module(stem):
    """Lazy-load impl modules to avoid import-time failures."""
    if stem not in _impl_cache:
        path = ASSET_IMPL_DIR / f"{stem}.py"
        spec = importlib.util.spec_from_file_location(
            f"bsa_assets_impl_{stem}_{abs(hash(path))}", path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _impl_cache[stem] = module
    return _impl_cache[stem]


class QuantBlockSparseAttnSpec:
    """TestSpec for quant_block_sparse_attn operator.

    Attributes are loaded lazily on first access to defer import errors.
    """

    @staticmethod
    def golden(*args, **kwargs):
        return _load_impl_module("golden").cpu_quant_block_sparse_attn(*args, **kwargs)

    @staticmethod
    def customize_inputs(*args, **kwargs):
        return _load_impl_module("inputs").customize_inputs(*args, **kwargs)

    # TTK constructs this module before torch.compile, so QBSA metadata and
    # auxiliary tensors are prepared outside ACLGraph GLOBAL capture.
    torch_graph = QuantBlockSparseAttnGraph

    tolerance = {
        "bfloat16": {"standard": "stat_rel_err"},
        "float32": {"standard": "stat_rel_err"},
    }

    @staticmethod
    def compare(*outputs, **kwargs):
        return _load_impl_module("compare").compare(*outputs, **kwargs)


__spec__ = {
    "bsa_ttk_ops.quant_block_sparse_attn": "QuantBlockSparseAttnSpec",
}
