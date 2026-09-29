# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import sys
import types

import torch

from .indexer_prologue_qw import indexer_prologue_qw as _indexer_prologue_qw_impl  # noqa: F401

# Share the ds41 module with the other experimental operators. The disk
# folder is indexer_prologue_qw_dsl, so ops discovery will not export the
# public name unless we attach it here.
_ops_mod = None
for _key in ("cann_ops_transformer.ops", "cann_ops_transformer_custom.ops"):
    _ops_mod = sys.modules.get(_key)
    if _ops_mod is not None:
        break
if _ops_mod is None:
    raise ImportError(
        "cann_ops_transformer.ops is not loaded; import it before "
        "indexer_prologue_qw_dsl"
    )

parent_pkg = _ops_mod.__name__
module_name = f"{parent_pkg}.ds41"
ds41 = sys.modules.get(module_name)
if ds41 is None:
    ds41 = types.ModuleType(module_name)
    ds41.__package__ = parent_pkg
    sys.modules[module_name] = ds41

indexer_prologue_qw = torch.ops.cann_ops_transformer.ds41.indexer_prologue_qw
ds41.indexer_prologue_qw = indexer_prologue_qw
_ops_mod.ds41 = ds41
_ops_mod.indexer_prologue_qw = indexer_prologue_qw

__all__ = ["ds41", "indexer_prologue_qw"]
