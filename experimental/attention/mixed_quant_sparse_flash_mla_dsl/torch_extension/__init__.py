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

from .mixed_quant_sparse_flash_mla import (
    mixed_quant_sparse_flash_mla,
    mixed_quant_sparse_flash_mla_metadata,
)

# 与 compressor_v2 共用 ds41 模块，保留其他算子已经注册的入口。
parent_pkg = sys.modules["cann_ops_transformer.ops"].__name__
module_name = f"{parent_pkg}.ds41"
ds41 = sys.modules.get(module_name)
if ds41 is None:
    ds41 = types.ModuleType(module_name)
    ds41.__package__ = parent_pkg
    sys.modules[module_name] = ds41

ds41.mixed_quant_sparse_flash_mla = (
    torch.ops.cann_ops_transformer.ds41.mixed_quant_sparse_flash_mla
)
ds41.mixed_quant_sparse_flash_mla_metadata = (
    torch.ops.cann_ops_transformer.ds41.mixed_quant_sparse_flash_mla_metadata
)

__all__ = ["ds41"]
