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
    "aclnnMoeTokenUnpermute": "AclnnMoeTokenUnpermuteTestSpec",
}


def _moe_finalize_routing_v2_np_torch(
    expanded_x, expanded_row_idx, scales, drop_pad_mode=2
):
    bsk = expanded_row_idx.shape[0]
    h = expanded_x.shape[-1]
    if h == 0:
        return torch.zeros((0, 0), dtype=expanded_x.dtype)
    expanded_x = expanded_x.reshape(-1, h)
    K = 1
    if scales is not None:
        K = scales.shape[-1]
    num_rows = bsk // K

    out = torch.zeros((num_rows, h), dtype=torch.float32)

    for i in range(num_rows):
        for k in range(K):
            if drop_pad_mode == 0 or drop_pad_mode == 1:
                expanded_row_idx_idx = k * num_rows + i
            else:
                expanded_row_idx_idx = i * K + k
            expanded_row_idx_value = expanded_row_idx[expanded_row_idx_idx]
            if drop_pad_mode == 1 or drop_pad_mode == 3:
                if expanded_row_idx_value == -1:
                    continue
            else:
                if expanded_row_idx_value >= expanded_x.shape[0]:
                    continue
            dst_row = expanded_x[expanded_row_idx[expanded_row_idx_idx], :].to(
                torch.float32
            )
            if scales is not None:
                dst_row = dst_row * scales[i, k].to(torch.float32)
            out[i, :] = out[i, :] + dst_row

    return out


class AclnnMoeTokenUnpermuteTestSpec:
    @staticmethod
    def golden(
        permutedTokens,
        sortedIndices,
        probsOptional=None,
        paddedMode=False,
        restoreShapeOptional=None,
        out=None,
        *args,
        **kwargs,
    ):
        expanded_x_dtype = permutedTokens.dtype
        expanded_x = permutedTokens.to(torch.float32)
        expanded_row_idx = sortedIndices
        scales = probsOptional
        if scales is not None:
            scales = scales.to(torch.float32)

        res = _moe_finalize_routing_v2_np_torch(
            expanded_x, expanded_row_idx, scales, drop_pad_mode=2
        )
        return [res.to(expanded_x_dtype)]

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
