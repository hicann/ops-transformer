# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import sys
import types

import torch

from .quant_sparse_lightning_indexer import (
    quant_sparse_lightning_indexer,
    quant_sparse_lightning_indexer_metadata,
)

# Share the ds41 module with the other experimental operators.
parent_pkg = sys.modules["cann_ops_transformer.ops"].__name__
module_name = f"{parent_pkg}.ds41"
ds41 = sys.modules.get(module_name)
if ds41 is None:
    ds41 = types.ModuleType(module_name)
    ds41.__package__ = parent_pkg
    sys.modules[module_name] = ds41

ds41.quant_sparse_lightning_indexer = (
    torch.ops.cann_ops_transformer.ds41.quant_sparse_lightning_indexer
)
ds41.quant_sparse_lightning_indexer_metadata = (
    torch.ops.cann_ops_transformer.ds41.quant_sparse_lightning_indexer_metadata
)

__all__ = ["ds41"]
