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

"""Traditional input entry for ACLNN mode (aclnnCompressor)."""

import importlib.util
from pathlib import Path

_ASSETS_DIR = Path(__file__).resolve().parent
_IMPL_DIR = _ASSETS_DIR / "impl"


def _load_impl(stem):
    path = _IMPL_DIR / f"{stem}.py"
    spec = importlib.util.spec_from_file_location(
        f"compressor_assets_{stem}_{abs(hash(path))}", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_inputs_impl = _load_impl("inputs")

aclnn_compressor_input = _inputs_impl.aclnn_compressor_input

__input__ = {
    "aclnn": {
        "aclnnCompressor": "aclnn_compressor_input",
    }
}
