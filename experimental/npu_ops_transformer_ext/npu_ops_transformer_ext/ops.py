#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from typing import Optional

import torch
from torch import Tensor

__all__ = ["grouped_matmul_quant"]


def grouped_matmul_quant(
    x: Tensor,
    quantized_weight: Tensor,
    weight_scale: Tensor,
    weight_offset: Tensor,
    group_list: Optional[Tensor],
    scale_group_size: int,
) -> Tensor:
    return torch.ops.npu_ops_transformer_ext.grouped_matmul_quant.default(
        x,
        quantized_weight,
        weight_scale,
        weight_offset,
        group_list,
        scale_group_size,
    )


# def dummy(x: Tensor) -> Tensor:
#     return torch.ops.npu_ops_transformer_ext.dummy.default(x)
