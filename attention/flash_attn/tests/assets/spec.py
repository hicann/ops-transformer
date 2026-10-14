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
import sys
from pathlib import Path

ASSET_IMPL_DIR = Path(__file__).with_name("impl")

_impl_cache = {}


def _load_cross_check_module():
    """Lazily load cross_check by a stable module name (shared with impl)."""
    name = "fa_ttk_flash_attn_cross_check"
    if name in sys.modules:
        return sys.modules[name]
    path = ASSET_IMPL_DIR / "cross_check.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


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


def _call_main_hook(stem, function, *args, **kwargs):
    """Bind positional torch.ops arguments using the installed operator schema."""
    import torch
    import cann_ops_transformer  # noqa: F401

    parameters = torch.ops.cann_ops_transformer.flash_attn.default._schema.arguments
    if len(args) > len(parameters):
        raise TypeError("too many positional FlashAttn arguments")
    arguments = dict(kwargs)
    for parameter, value in zip(parameters, args):
        if parameter.name in arguments:
            raise TypeError(f"multiple values for FlashAttn argument {parameter.name}")
        arguments[parameter.name] = value
    return getattr(_load_impl_module(stem), function)(**arguments)


class FlashAttnSpec:
    """TestSpec for FlashAttn operator.

    Attributes are loaded lazily on first access to defer import errors.
    """

    @staticmethod
    def golden(*args, **kwargs):
        return _call_main_hook("golden", "cpu_flash_attn", *args, **kwargs)

    @staticmethod
    def customize_inputs(*args, **kwargs):
        return _call_main_hook("inputs", "customize_inputs", *args, **kwargs)

    tolerance = {
        "float16": {"standard": "stat_rel_err"},
        "bfloat16": {"standard": "stat_rel_err"},
    }

    @staticmethod
    def compare(*outputs, compare_context=None, **kwargs):
        cross_check = _load_cross_check_module()
        if cross_check.enabled():
            return cross_check.compare(
                *outputs,
                testcase_name=getattr(compare_context, "testcase_name", None),
            )
        return _load_impl_module("compare").compare(*outputs)

    torch_graph = _load_impl_module("graph").FlashAttnAclGraph

    @staticmethod
    def npu_preprocess(*args, **kwargs):
        return _call_main_hook("npu_preprocess", "run", *args, **kwargs)


class FlashAttnMetadataSpec:
    """TestSpec for the FlashAttn metadata generator.

    Only customized inputs are provided; there is no standalone test suite.
    """

    customize_inputs = _load_impl_module(
        "metadata_inputs"
    ).generate_flash_attn_metadata_inputs


__spec__ = {
    "torch.ops.cann_ops_transformer.flash_attn": "FlashAttnSpec",
    "flash_attn_ttk_ops.flash_attn_ttk": "FlashAttnSpec",
    "torch.ops.cann_ops_transformer.flash_attn_metadata": "FlashAttnMetadataSpec",
}
