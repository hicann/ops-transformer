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

"""TestSpec adapter for FlashAttn TTK assets.

All impl modules are loaded lazily to avoid import-time failures.
"""

import importlib.util
from pathlib import Path

ASSET_IMPL_DIR = Path(__file__).with_name("impl")

_impl_cache = {}


def _load_impl_module(stem):
    """Lazy-load impl modules to avoid import-time failures."""
    if stem not in _impl_cache:
        path = ASSET_IMPL_DIR / f"{stem}.py"
        spec = importlib.util.spec_from_file_location(
            f"fa_assets_impl_{stem}_{abs(hash(path))}", path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _impl_cache[stem] = module
    return _impl_cache[stem]


class FlashAttnSpec:
    """TestSpec for FlashAttn operator.

    Attributes are loaded lazily on first access to defer import errors.
    """

    @staticmethod
    def golden(*args, **kwargs):
        return _load_impl_module("golden").cpu_flash_attn(*args, **kwargs)

    @staticmethod
    def customize_inputs(*args, **kwargs):
        return _load_impl_module("inputs").customize_inputs(*args, **kwargs)

    tolerance = {
        "float16": {"standard": "stat_rel_err"},
        "bfloat16": {"standard": "stat_rel_err"},
    }

    @staticmethod
    def compare(*outputs, **kwargs):
        return _load_impl_module("compare").compare(*outputs, **kwargs)

    torch_graph = _load_impl_module("graph").FlashAttnAclGraph

    npu_preprocess = _load_impl_module("npu_preprocess").run


class FlashAttnMetadataSpec:
    """TestSpec for the FlashAttn metadata generator.

    Only customized inputs are provided; there is no standalone test suite.
    """

    customize_inputs = _load_impl_module(
        "metadata_inputs"
    ).generate_flash_attn_metadata_inputs


__spec__ = {
    "flash_attn_ttk_ops.flash_attn_ttk": "FlashAttnSpec",
    "torch.ops.cann_ops_transformer.flash_attn_metadata": "FlashAttnMetadataSpec",
}
