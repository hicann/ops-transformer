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
    "aclnnMoeTokenPermuteGrad": "AclnnMoeTokenPermuteGradTestSpec",
}


class AclnnMoeTokenPermuteGradTestSpec:
    @staticmethod
    def golden(
        permutedOutputGrad,
        sortedIndices,
        numTopk=1,
        paddedMode=False,
        out=None,
        *args,
        **kwargs,
    ):
        x_dtype = permutedOutputGrad.dtype
        grad_expanded_x = permutedOutputGrad.to(torch.float32)
        expanded_row_idx = sortedIndices

        A = grad_expanded_x.shape[0]
        H = grad_expanded_x.shape[1]
        BSK = expanded_row_idx.shape[0]
        BS = BSK // numTopk

        if A <= BSK:
            _, indices = torch.sort(expanded_row_idx)
            indices = indices[:A]
            grad_expanded_x_tmp = grad_expanded_x.reshape(A, H)
            grad_x = torch.ops.aten.index_select_backward(
                grad_expanded_x_tmp, [BS, H], 0, indices // numTopk
            )
        else:
            value, indices = torch.sort(expanded_row_idx)
            grad_expanded_x_tmp = grad_expanded_x.reshape(A, H)
            grad_expanded_x_tmp = torch.index_select(grad_expanded_x_tmp, 0, value)
            grad_x = torch.ops.aten.index_select_backward(
                grad_expanded_x_tmp, [BS, H], 0, indices // numTopk
            )

        return [grad_x.to(x_dtype)]

    @staticmethod
    def customize_inputs(*args, **kwargs):
        sortedIndices = args[1]
        n = sortedIndices.shape[0]
        sortedIndices.copy_(torch.randperm(n, dtype=torch.int32))

    tolerance = {
        "float32": {"standard": "stat_rel_err"},
        "float16": {"standard": "stat_rel_err"},
        "bfloat16": {"standard": "stat_rel_err"},
    }
