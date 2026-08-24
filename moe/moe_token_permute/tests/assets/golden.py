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

import numpy
import torch

__spec__ = {
    "aclnnMoeTokenPermute": "AclnnMoeTokenPermuteTestSpec",
}


class AclnnMoeTokenPermuteTestSpec:
    @staticmethod
    def golden(
        tokens,
        indices,
        numOutTokens=0,
        paddedMode=False,
        permuteTokensOut=None,
        sortedIndicesOut=None,
        *args,
        **kwargs,
    ):
        x_dtype = tokens.dtype
        input_x = tokens.to(torch.float32)
        expert_idx = indices

        active_num = numOutTokens
        expert_num = 256
        drop_pad_mode = 0

        num_rows = input_x.shape[0]
        hidden_size = input_x.shape[-1]
        k = expert_idx.shape[-1] if len(expert_idx.shape) == 2 else 1
        sorted_row_idx = torch.argsort(expert_idx.reshape(-1), dim=-1, stable=True)
        sorted_expert_idx = torch.sort(
            expert_idx.reshape(-1), dim=-1, stable=True
        ).values

        expanded_row_idx = torch.zeros_like(sorted_row_idx, dtype=torch.int32)
        expanded_row_idx[sorted_row_idx] = torch.arange(
            sorted_row_idx.shape[0], dtype=torch.int32, device=sorted_row_idx.device
        )

        if active_num == 0:
            active_num = num_rows * k
        else:
            active_num = min(active_num, num_rows * k)
        selected_indices = sorted_row_idx[:active_num] // k
        expanded_x = input_x[selected_indices]

        return [expanded_x.to(x_dtype), expanded_row_idx]

    tolerance = {
        "float32": {"standard": "stat_rel_err"},
        "float16": {"standard": "stat_rel_err"},
        "bfloat16": {"standard": "stat_rel_err"},
    }
