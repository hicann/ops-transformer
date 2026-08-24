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
    "moe_init_routing_v2_grad": "MoeInitRoutingV2GradTestSpec",
    "aclnnMoeInitRoutingV2Grad": "AclnnMoeInitRoutingV2GradTestSpec",
}


class AclnnMoeInitRoutingV2GradTestSpec:
    @staticmethod
    def golden(
        gradExpandedX,
        expandedRowIdx,
        topK,
        dropPadMode=0,
        activeNum=0,
        out=None,
        *args,
        **kwargs,
    ):
        x_dtype = gradExpandedX.dtype
        grad_expanded_x = gradExpandedX.to(torch.float32)
        if grad_expanded_x.dim() > 2:
            grad_expanded_x = grad_expanded_x.reshape(-1, grad_expanded_x.shape[-1])
        A = grad_expanded_x.shape[0]
        H = grad_expanded_x.shape[1]
        BSK = expandedRowIdx.shape[0]
        BS = BSK // topK

        grad_x = torch.zeros((BS, H), dtype=torch.float32)
        eri = expandedRowIdx.to(torch.int64)
        for element in range(BS):
            rows = []
            for k in range(topK):
                idx = element * topK + k
                if idx < BSK:
                    src_row = eri[idx].item()
                    if 0 <= src_row < A:
                        rows.append(grad_expanded_x[src_row])
            if rows:
                grad_x[element] = torch.stack(rows).sum(dim=0)

        return [grad_x.to(x_dtype)]

    @staticmethod
    def customize_inputs(
        gradExpandedX, expandedRowIdx, topK, dropPadMode, activeNum, *args, **kwargs
    ):
        if gradExpandedX.dim() > 2:
            A = int(numpy.prod(gradExpandedX.shape[:-1]))
        else:
            A = gradExpandedX.shape[0]
        BSK = expandedRowIdx.shape[0]
        if dropPadMode == 1:
            new_eri = torch.randint(0, A, (BSK,), dtype=torch.int32)
            if activeNum > 0 and activeNum < BSK:
                new_eri[activeNum:] = -1
        else:
            new_eri = torch.randint(0, A, (BSK,), dtype=torch.int32)
        expandedRowIdx.copy_(new_eri)

    tolerance = {
        "float32": {"standard": "stat_rel_err"},
        "float16": {"standard": "stat_rel_err"},
        "bfloat16": {"standard": "stat_rel_err"},
    }


class MoeInitRoutingV2GradTestSpec:
    @staticmethod
    def golden(
        grad_expanded_x,
        expanded_row_idx,
        *,
        top_k,
        drop_pad_mode=0,
        active_num=0,
        **kwargs,
    ):
        ori_dtype = grad_expanded_x.dtype
        grad_expanded_x = grad_expanded_x.astype(numpy.float32)
        if grad_expanded_x.ndim > 2:
            grad_expanded_x = grad_expanded_x.reshape(-1, grad_expanded_x.shape[-1])
        A = grad_expanded_x.shape[0]
        H = grad_expanded_x.shape[1]
        BSK = expanded_row_idx.shape[0]
        BS = BSK // top_k

        grad_x = numpy.zeros((BS, H), dtype=numpy.float32)
        for element in range(BS):
            rows = []
            for k in range(top_k):
                idx = element * top_k + k
                if idx < BSK:
                    src_row = int(expanded_row_idx[idx])
                    if 0 <= src_row < A:
                        rows.append(grad_expanded_x[src_row])
            if rows:
                grad_x[element] = numpy.sum(
                    numpy.stack(rows).astype(numpy.float32), axis=0
                )

        return [grad_x.astype(ori_dtype)]

    @staticmethod
    def customize_inputs(grad_expanded_x, expanded_row_idx, **kwargs):
        if grad_expanded_x.ndim > 2:
            A = int(numpy.prod(grad_expanded_x.shape[:-1]))
        else:
            A = grad_expanded_x.shape[0]
        BSK = expanded_row_idx.shape[0]
        drop_pad_mode = kwargs.get("drop_pad_mode", 0)
        active_num = kwargs.get("active_num", 0)
        if drop_pad_mode == 1:
            new_eri = numpy.random.randint(0, A, size=BSK, dtype=numpy.int32)
            if active_num > 0 and active_num < BSK:
                new_eri[active_num:] = -1
        else:
            new_eri = numpy.random.randint(0, A, size=BSK, dtype=numpy.int32)
        return (grad_expanded_x, new_eri)

    tolerance = {
        "float32": {"standard": "stat_rel_err"},
        "float16": {"standard": "stat_rel_err"},
        "bfloat16": {"standard": "stat_rel_err"},
    }
