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
    "moe_init_routing": "MoeInitRoutingTestSpec",
    "aclnnMoeInitRouting": "AclnnMoeInitRoutingTestSpec",
}


class AclnnMoeInitRoutingTestSpec:
    @staticmethod
    def golden(
        x,
        rowIdx,
        expertIdx,
        activeNum,
        expandedXOut=None,
        expandedRowIdxOut=None,
        expandedExpertIdxOut=None,
        *args,
        **kwargs,
    ):
        ori_dtype = x.dtype
        num_rows = x.shape[0]
        k = expertIdx.shape[-1]

        sort_expert_idx = torch.argsort(expertIdx.reshape(-1), dim=-1, stable=True)
        expanded_expert_idx = torch.sort(
            expertIdx.reshape(-1), dim=-1, stable=True
        ).values

        expanded_dst_to_src_row = rowIdx.reshape(-1).gather(0, sort_expert_idx)
        expanded_row_idx = torch.zeros_like(expanded_dst_to_src_row, dtype=torch.int32)
        expanded_row_idx[expanded_dst_to_src_row] = torch.arange(
            expanded_dst_to_src_row.shape[0], dtype=torch.int32
        )

        active_num = min(activeNum, num_rows) * k
        selected_indices = expanded_dst_to_src_row[:active_num] % num_rows
        expanded_x = x[selected_indices, :]

        return [
            expanded_x.to(ori_dtype),
            expanded_row_idx,
            expanded_expert_idx.to(torch.int32),
        ]

    @staticmethod
    def customize_inputs(x, rowIdx, expertIdx, activeNum, *args, **kwargs):
        n = rowIdx.numel()
        new_row = (
            torch.arange(0, n, 1, dtype=torch.int32)
            .reshape(rowIdx.shape[1], rowIdx.shape[0])
            .t()
            .contiguous()
        )
        rowIdx.copy_(new_row)

    tolerance = {
        "float32": {"standard": "stat_rel_err"},
        "float16": {"standard": "stat_rel_err"},
        "bfloat16": {"standard": "stat_rel_err"},
    }


class MoeInitRoutingTestSpec:
    @staticmethod
    def golden(x, row_idx, expert_idx, *, active_num, **kwargs):
        ori_dtype = x.dtype
        num_rows = x.shape[0]
        k = expert_idx.shape[-1]

        sort_expert_idx = numpy.argsort(
            expert_idx.reshape((-1,)), axis=-1, kind="stable"
        )
        expanded_expert_idx = numpy.sort(
            expert_idx.reshape((-1,)), axis=-1, kind="stable"
        )

        expanded_dst_to_src_row = numpy.take_along_axis(
            row_idx.reshape((-1,)), sort_expert_idx, axis=-1
        )
        expanded_row_idx = numpy.zeros(expanded_dst_to_src_row.shape, dtype=numpy.int32)
        expanded_row_idx[expanded_dst_to_src_row] = numpy.arange(
            expanded_dst_to_src_row.shape[-1], dtype=numpy.int32
        )

        active_num = min(active_num, num_rows) * k
        expanded_x = x[expanded_dst_to_src_row[:active_num] % num_rows, :]

        return [
            expanded_x.astype(ori_dtype),
            expanded_row_idx,
            expanded_expert_idx.astype(numpy.int32),
        ]

    @staticmethod
    def customize_inputs(x, row_idx, expert_idx, **kwargs):
        n = row_idx.size
        row_idx = (
            numpy.arange(0, n, 1, dtype=row_idx.dtype)
            .reshape(row_idx.shape[1], row_idx.shape[0])
            .transpose(1, 0)
        )
        return (x, row_idx, expert_idx)

    tolerance = {
        "float32": {"standard": "stat_rel_err"},
        "float16": {"standard": "stat_rel_err"},
        "bfloat16": {"standard": "stat_rel_err"},
    }
