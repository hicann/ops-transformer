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
    "moe_finalize_routing_v2": "MoeFinalizeRoutingV2TestSpec",
}


class MoeFinalizeRoutingV2TestSpec:
    @staticmethod
    def golden(
        expanded_x,
        expanded_row_idx,
        x1,
        x2,
        bias,
        scales,
        expert_idx,
        x=None,
        a1=None,
        a2=None,
        v=None,
        *,
        drop_pad_mode=0,
        zero_expert_range=None,
        copy_expert_range=None,
        constant_expert_range=None,
        k=1,
        **kwargs,
    ):
        if zero_expert_range is None:
            zero_expert_range = []
        if copy_expert_range is None:
            copy_expert_range = []
        if constant_expert_range is None:
            constant_expert_range = []

        ori_dtype = expanded_x.dtype
        out_type = numpy.float32
        if expanded_x.dtype == numpy.float32:
            out_type = numpy.float64

        bsk = expanded_row_idx.shape[0]
        h = expanded_x.shape[-1]
        if h == 0:
            return [numpy.array([], dtype=out_type)]
        expanded_x = expanded_x.reshape(-1, h)
        K = k
        if scales is not None:
            K = scales.shape[1]
        num_rows = bsk // K

        out = numpy.zeros((num_rows, h), dtype=out_type)
        if x1 is not None:
            out = out + x1
        if x2 is not None:
            out = out + x2

        for i in range(num_rows):
            for kk in range(K):
                if drop_pad_mode == 0 or drop_pad_mode == 1:
                    expanded_row_idx_idx = kk * num_rows + i
                else:
                    expanded_row_idx_idx = i * K + kk
                expanded_row_idx_value = expanded_row_idx[expanded_row_idx_idx]
                if drop_pad_mode == 1 or drop_pad_mode == 3:
                    if expanded_row_idx_value == -1:
                        continue
                else:
                    if expanded_row_idx_value >= expanded_x.shape[0]:
                        continue

                if x is not None and expert_idx is not None:
                    zero_start, zero_end = zero_expert_range
                    copy_start, copy_end = copy_expert_range
                    const_start, const_end = constant_expert_range
                    expert_id = expert_idx[i, kk]
                    if zero_start <= expert_id < zero_end:
                        continue
                    elif copy_start <= expert_id < copy_end:
                        dst_row = x[i, :].astype(out_type)
                    elif (
                        x is not None
                        and v is not None
                        and a1 is not None
                        and a2 is not None
                        and const_start <= expert_id < const_end
                    ):
                        row_idx = expert_id - const_start
                        dst_row = (
                            a1[row_idx, :] * x[i, :] + a2[row_idx, :] * v[row_idx, :]
                        ).astype(out_type)
                    else:
                        dst_row = expanded_x[
                            expanded_row_idx[expanded_row_idx_idx], :
                        ].astype(out_type)
                else:
                    dst_row = expanded_x[
                        expanded_row_idx[expanded_row_idx_idx], :
                    ].astype(out_type)
                if bias is not None and expert_idx is not None:
                    expert_id = expert_idx[i, kk]
                    if expert_id < 0 or expert_id >= bias.shape[0]:
                        continue
                    dst_row = dst_row + bias[expert_id, :].astype(out_type)
                if scales is not None:
                    dst_row = dst_row * scales[i, kk].astype(out_type)
                out[i, :] = out[i, :] + dst_row

        if ori_dtype == numpy.float32:
            return [out.astype(numpy.float32)]
        return [out.astype(ori_dtype)]

    @staticmethod
    def customize_inputs(
        expanded_x,
        expanded_row_idx,
        x1,
        x2,
        bias,
        scales,
        expert_idx,
        x=None,
        a1=None,
        a2=None,
        v=None,
        **kwargs,
    ):
        drop_pad_mode = kwargs.get("drop_pad_mode", 0)
        expand_n = expanded_row_idx.shape[0]
        expanded_row_idx = numpy.random.choice(
            expand_n, expand_n, replace=False
        ).astype(numpy.int32)
        if drop_pad_mode == 1 or drop_pad_mode == 3:
            e = expanded_x.shape[0]
            c = expanded_x.shape[1] if len(expanded_x.shape) > 2 else expand_n
            expanded_row_idx[expanded_row_idx >= e * c] = -1
        if bias is not None:
            e = bias.shape[0]
            expert_idx = (
                numpy.random.randint(0, e, size=expert_idx.size)
                .reshape(expert_idx.shape)
                .astype(numpy.int32)
            )
        return (
            expanded_x,
            expanded_row_idx,
            x1,
            x2,
            bias,
            scales,
            expert_idx,
            x,
            a1,
            a2,
            v,
        )

    tolerance = {
        "float32": {"standard": "stat_rel_err"},
        "float16": {"standard": "stat_rel_err"},
        "bfloat16": {"standard": "stat_rel_err"},
    }
