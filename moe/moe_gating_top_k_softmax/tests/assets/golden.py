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
    "moe_gating_top_k_softmax": "MoeGatingTopKSoftmaxTestSpec",
    "aclnnMoeGatingTopKSoftmax": "AclnnMoeGatingTopKSoftmaxTestSpec",
}


def _softmax_numpy(x, axis=-1):
    if "float16" in x.dtype.name:
        x = x.astype(numpy.float32)
    x_max = x.max(axis=axis, keepdims=True)
    x_sub = x - x_max
    y = numpy.exp(x_sub)
    x_sum = y.sum(axis=axis, keepdims=True)
    return y / x_sum


def _softmax_torch(x, axis=-1):
    if x.dtype == torch.float16:
        x = x.to(torch.float32)
    x_max = x.max(dim=axis, keepdim=True).values
    x_sub = x - x_max
    y = torch.exp(x_sub)
    x_sum = y.sum(dim=axis, keepdim=True)
    return y / x_sum


class AclnnMoeGatingTopKSoftmaxTestSpec:
    @staticmethod
    def golden(
        x,
        finishedOptional=None,
        k=1,
        yOut=None,
        expertIdxOut=None,
        rowIdxOut=None,
        *args,
        **kwargs,
    ):
        ori_dtype = x.dtype
        gating = x.to(torch.float32)

        if gating.dim() == 3:
            gating = gating.reshape(-1, gating.shape[-1])
            if finishedOptional is not None:
                finishedOptional = finishedOptional.flatten()

        num_expert = gating.shape[-1]
        softmax = _softmax_torch(gating, -1)

        indices = torch.argsort(softmax, dim=-1, descending=True, stable=True)
        indices = indices[:, :k].to(torch.int64)
        out = torch.gather(softmax, -1, indices)

        if finishedOptional is not None:
            finished = finishedOptional.to(torch.bool).reshape(-1, 1).expand(-1, k)
            indices = torch.where(
                finished, torch.full_like(indices, num_expert), indices
            )

        rows = out.shape[0]
        row_idx = (
            torch.arange(rows * k, dtype=torch.int32)
            .reshape(k, rows)
            .transpose(0, 1)
            .contiguous()
        )

        out = out.to(ori_dtype)

        return [out, indices.to(torch.int32), row_idx]

    tolerance = {
        "float32": {"standard": "stat_rel_err"},
        "float16": {"standard": "stat_rel_err"},
        "bfloat16": {"standard": "stat_rel_err"},
    }


class MoeGatingTopKSoftmaxTestSpec:
    @staticmethod
    def golden(x, finished, *, k, **kwargs):
        ori_dtype = x.dtype
        gating = x.astype(numpy.float32)

        if len(gating.shape) == 3:
            gating = gating.reshape(-1, gating.shape[-1])
            if finished is not None:
                finished = finished.flatten()

        num_expert = gating.shape[-1]
        softmax = _softmax_numpy(gating, -1)

        indices = numpy.argsort(-softmax, axis=-1, kind="stable")
        indices = indices[:, :k]
        out = numpy.take_along_axis(softmax, indices, axis=-1)

        if finished is not None:
            finished_2d = finished.reshape(finished.shape[0], 1)
            finished_2d = numpy.tile(finished_2d, (1, k))
            indices = numpy.where(finished_2d, num_expert, indices)

        rows = out.shape[0]
        row_idx = (
            numpy.arange(rows * k, dtype=numpy.int32).reshape(k, rows).transpose(1, 0)
        )

        out = out.astype(ori_dtype)

        return [out, indices.astype(numpy.int32), row_idx]

    tolerance = {
        "float32": {"standard": "stat_rel_err"},
        "float16": {"standard": "stat_rel_err"},
        "bfloat16": {"standard": "stat_rel_err"},
    }
