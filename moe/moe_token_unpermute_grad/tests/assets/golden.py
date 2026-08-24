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
    "aclnnMoeTokenUnpermuteGrad": "AclnnMoeTokenUnpermuteGradTestSpec",
}


class AclnnMoeTokenUnpermuteGradTestSpec:
    @staticmethod
    def golden(
        permuteTokensOptional=None,
        unpermutedTokensGrad=None,
        sortedIndices=None,
        probsOptional=None,
        paddedMode=False,
        restoreShapeOptional=None,
        permutedTokensGradOut=None,
        probsGradOut=None,
        *args,
        **kwargs,
    ):
        grad_y = unpermutedTokensGrad
        expanded_row_idx = sortedIndices
        expanded_x = permuteTokensOptional
        scales = probsOptional

        grad_y_dtype = grad_y.dtype
        grad_y = grad_y.to(torch.float32)

        row, hidden = grad_y.shape
        row_topk = expanded_row_idx.shape[0]

        topk = 1
        if scales is not None:
            topk = scales.shape[1]

        active_num = expanded_x.shape[0]
        expandedX_dim_0 = row_topk
        if active_num > 0 and active_num < row_topk:
            expandedX_dim_0 = active_num

        grad_y_expanded = (
            grad_y.unsqueeze(1).expand(row, topk, hidden).reshape(row_topk, hidden)
        )
        expanded_row_idx_int = expanded_row_idx.to(torch.int64)
        _, indices = torch.sort(expanded_row_idx_int, dim=-1)

        if scales is None:
            if expandedX_dim_0 < row_topk:
                indices = indices[:expandedX_dim_0]
            grad_expanded_x = grad_y_expanded.index_select(0, indices)
            return [grad_expanded_x.to(grad_y_dtype), None]
        else:
            scales_dtype = scales.dtype
            scales_flat = scales.reshape(row_topk).to(torch.float32)
            scales_expanded = scales_flat.unsqueeze(1).expand(-1, hidden)

            if expandedX_dim_0 < row_topk:
                indices = indices[:expandedX_dim_0]
            grad_expanded_x = grad_y_expanded.index_select(
                0, indices
            ) * scales_expanded.index_select(0, indices)

            expanded_x_f = expanded_x.to(torch.float32)
            zeros = torch.zeros((1, hidden), dtype=grad_y.dtype)
            if expandedX_dim_0 < row_topk:
                expanded_x_f = torch.cat((expanded_x_f, zeros), dim=0)
                expanded_row_idx_int = torch.where(
                    expanded_row_idx_int >= expandedX_dim_0,
                    torch.tensor(expandedX_dim_0),
                    expanded_row_idx_int,
                )
            add_result = expanded_x_f.index_select(0, expanded_row_idx_int)
            grad_scales = torch.sum(add_result * grad_y_expanded, dim=1)

            return [grad_expanded_x.to(grad_y_dtype), grad_scales.to(scales_dtype)]

    @staticmethod
    def customize_inputs(*args, **kwargs):
        sortedIndices = args[2]
        n = sortedIndices.shape[0]
        sortedIndices.copy_(torch.randperm(n, dtype=torch.int32))

    tolerance = {
        "float32": {"standard": "stat_rel_err"},
        "float16": {"standard": "stat_rel_err"},
        "bfloat16": {"standard": "stat_rel_err"},
    }
