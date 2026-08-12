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

"""TestSpec adapter for stem_indexer TTK assets."""

import importlib.util
from pathlib import Path

ASSET_IMPL_DIR = Path(__file__).with_name("impl")

_impl_cache = {}


def _load_impl_module(stem):
    if stem not in _impl_cache:
        path = ASSET_IMPL_DIR / f"{stem}.py"
        spec = importlib.util.spec_from_file_location(
            f"si_assets_impl_{stem}_{abs(hash(path))}", path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _impl_cache[stem] = module
    return _impl_cache[stem]


class StemIndexerSpec:
    """TestSpec for stem_indexer operator."""

    @staticmethod
    def golden(*args, **kwargs):
        return _load_impl_module("golden").cpu_stem_indexer(*args, **kwargs)

    @staticmethod
    def customize_inputs(*args, **kwargs):
        return _load_impl_module("inputs").customize_inputs(*args, **kwargs)

    tolerance = {
        "int32": {"standard": "binary_equal"},
    }

    @staticmethod
    def compare(*outputs, compare_context=None, **kwargs):
        return _load_impl_module("compare").compare(
            *outputs, compare_context=compare_context, **kwargs
        )


__spec__ = {
    "stem_indexer_ttk_ops.stem_indexer": "StemIndexerSpec",
}
