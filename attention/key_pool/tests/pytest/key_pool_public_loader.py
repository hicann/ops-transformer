# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from pathlib import Path

import torch_npu


def load_key_pool_public_api():
    """Load only the KeyPool wheel modules, without importing unrelated operators."""
    loaded = sys.modules.get("cann_ops_transformer.ops.attention.key_pool")
    if loaded is not None:
        return loaded

    package_spec = importlib.util.find_spec("cann_ops_transformer")
    if package_spec is None or package_spec.origin is None:
        raise ImportError("cann_ops_transformer is not installed")

    package_root = Path(package_spec.origin).resolve().parent
    package = types.ModuleType("cann_ops_transformer")
    package.__file__ = str(package_root / "__init__.py")
    package.__package__ = "cann_ops_transformer"
    package.__path__ = [str(package_root)]
    sys.modules["cann_ops_transformer"] = package

    ops = types.ModuleType("cann_ops_transformer.ops")
    ops.__file__ = str(package_root / "ops" / "__init__.py")
    ops.__package__ = "cann_ops_transformer.ops"
    ops.__path__ = [str(package_root / "ops")]
    sys.modules["cann_ops_transformer.ops"] = ops
    package.ops = ops

    module = importlib.import_module("cann_ops_transformer.ops.attention.key_pool")
    ops.key_pool = module.key_pool
    # Keep the historical module path available to existing test/spec loaders.
    sys.modules["cann_ops_transformer.ops.key_pool"] = module
    torch_npu.key_pool = module.key_pool
    return module
