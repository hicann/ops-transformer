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
    "moe_gating_top_k": "MoeGatingTopKTestSpec",
    "aclnnMoeGatingTopK": "AclnnMoeGatingTopKTestSpec",
    "aclnnMoeGatingTopKV2": "AclnnMoeGatingTopKV2TestSpec",
}


def _softmax_torch(x, axis=-1):
    if x.dtype == torch.float16:
        x = x.to(torch.float32)
    x_max = x.max(dim=axis, keepdim=True).values
    x_sub = x - x_max
    y = torch.exp(x_sub)
    x_sum = y.sum(dim=axis, keepdim=True)
    return y / x_sum


def _softmax_numpy(x, axis=-1):
    if "float16" in x.dtype.name:
        x = x.astype(numpy.float32)
    x_max = x.max(axis=axis, keepdims=True)
    x_sub = x - x_max
    y = numpy.exp(x_sub)
    x_sum = y.sum(axis=axis, keepdims=True)
    return y / x_sum


class AclnnMoeGatingTopKTestSpec:
    @staticmethod
    def golden(
        x,
        biasOptional,
        k=1,
        kGroup=1,
        groupCount=1,
        groupSelectMode=0,
        renorm=0,
        normType=0,
        outFlag=False,
        routedScalingFactor=1.0,
        eps=1e-20,
        yOut=None,
        expertIdxOut=None,
        outOut=None,
        *args,
        **kwargs,
    ):
        inputIdsOptional = kwargs.get("inputIdsOptional", None)
        tid2eidOptional = kwargs.get("tid2eidOptional", None)
        ori_dtype = x.dtype
        x = x.to(torch.float32)
        if biasOptional is not None:
            biasOptional = biasOptional.to(torch.float32)

        if normType == 0:
            x = _softmax_torch(x, -1)
        elif normType == 1:
            x = 1 / (1 + numpy.exp(-x.numpy()))
            x = torch.from_numpy(x).to(torch.float32)
        elif normType == 2:
            x = torch.sqrt(torch.nn.functional.softplus(x))

        original_x = x
        if biasOptional is not None:
            x = x + biasOptional

        hashFlag = inputIdsOptional is not None and tid2eidOptional is not None
        if hashFlag:
            indices = tid2eidOptional[inputIdsOptional].to(torch.int64)
        else:
            if groupCount > 1:
                x_reshaped = x.reshape(x.shape[0], groupCount, -1)
                if groupSelectMode == 0:
                    group_x = torch.amax(x_reshaped, dim=-1)
                else:
                    top2 = torch.topk(x_reshaped, 2, dim=-1).values
                    group_x = top2.sum(dim=-1)
                _, group_indices = torch.sort(
                    group_x, dim=-1, descending=True, stable=True
                )
                group_indices = group_indices[:, :kGroup]

                mask = torch.ones((x_reshaped.shape[0], groupCount), dtype=torch.bool)
                mask[torch.arange(x_reshaped.shape[0])[:, None], group_indices] = False
                x = torch.where(mask.unsqueeze(-1), float("-inf"), x_reshaped)
                x = x.reshape(x.shape[0], -1)

            _, indices = torch.sort(x, dim=-1, stable=True, descending=True)
            indices = indices[:, :k].to(torch.int64)

        y = torch.gather(original_x, 1, indices)

        if normType != 0 or renorm != 0:
            y = y / (y.sum(dim=-1, keepdim=True) + eps)
        y = y * routedScalingFactor

        if outFlag:
            out = original_x
        else:
            out = None

        return [y.to(ori_dtype), None, out]

    tolerance = {
        "float32": {"standard": "stat_rel_err"},
        "float16": {"standard": "stat_rel_err"},
        "bfloat16": {"standard": "stat_rel_err"},
    }


class AclnnMoeGatingTopKV2TestSpec:
    @staticmethod
    def golden(
        x,
        biasOptional,
        inputIdsOptional=None,
        tid2eidOptional=None,
        k=1,
        kGroup=1,
        groupCount=1,
        groupSelectMode=0,
        renorm=0,
        normType=0,
        outFlag=False,
        routedScalingFactor=1.0,
        eps=1e-20,
        yOut=None,
        expertIdxOut=None,
        outOut=None,
        *args,
        **kwargs,
    ):
        ori_dtype = x.dtype
        x = x.to(torch.float32)
        if biasOptional is not None:
            biasOptional = biasOptional.to(torch.float32)

        if normType == 0:
            x = _softmax_torch(x, -1)
        elif normType == 1:
            x = 1 / (1 + torch.exp(-x))
        elif normType == 2:
            x = torch.sqrt(torch.nn.functional.softplus(x))

        original_x = x
        if biasOptional is not None:
            x = x + biasOptional

        hashFlag = inputIdsOptional is not None and tid2eidOptional is not None
        if hashFlag:
            indices = tid2eidOptional[inputIdsOptional].to(torch.int64)
        else:
            if groupCount > 1:
                x_reshaped = x.reshape(x.shape[0], groupCount, -1)
                if groupSelectMode == 0:
                    group_x = torch.amax(x_reshaped, dim=-1)
                else:
                    top2 = torch.topk(x_reshaped, 2, dim=-1).values
                    group_x = top2.sum(dim=-1)
                _, group_indices = torch.sort(
                    group_x, dim=-1, descending=True, stable=True
                )
                group_indices = group_indices[:, :kGroup]

                mask = torch.ones((x_reshaped.shape[0], groupCount), dtype=torch.bool)
                mask[torch.arange(x_reshaped.shape[0])[:, None], group_indices] = False
                x = torch.where(mask.unsqueeze(-1), float("-inf"), x_reshaped)
                x = x.reshape(x.shape[0], -1)

            _, indices = torch.sort(x, dim=-1, stable=True, descending=True)
            indices = indices[:, :k].to(torch.int64)

        y = torch.gather(original_x, 1, indices)

        if normType != 0 or renorm != 0:
            y = y / (y.sum(dim=-1, keepdim=True) + eps)
        y = y * routedScalingFactor

        if outFlag:
            out = original_x
        else:
            out = None

        return [y.to(ori_dtype), None, out]

    tolerance = {
        "float32": {"standard": "stat_rel_err"},
        "float16": {"standard": "stat_rel_err"},
        "bfloat16": {"standard": "stat_rel_err"},
    }


class MoeGatingTopKTestSpec:
    @staticmethod
    def golden(
        x,
        bias,
        input_ids,
        tid2eid,
        *,
        k,
        k_group=1,
        group_count=1,
        group_select_mode=0,
        renorm=0,
        norm_type=0,
        out_flag=False,
        routed_scaling_factor=1.0,
        eps=1e-20,
        **kwargs,
    ):
        ori_dtype = x.dtype
        x = x.astype(numpy.float32)
        if bias is not None:
            bias = bias.astype(numpy.float32)

        if norm_type == 0:
            x = _softmax_numpy(x, -1)
        elif norm_type == 1:
            x = 1 / (1 + numpy.exp(-x))
        elif norm_type == 2:
            x = numpy.sqrt(numpy.log1p(numpy.exp(x)))

        original_x = x
        if bias is not None:
            x = x + bias

        hashFlag = input_ids is not None and tid2eid is not None
        if hashFlag:
            indices = tid2eid[input_ids]
        else:
            if group_count > 1:
                x = x.reshape(x.shape[0], group_count, -1)
                if group_select_mode == 0:
                    group_x = numpy.amax(x, axis=-1)
                else:
                    group_x = numpy.partition(x, -2, axis=-1)[..., -2:].sum(axis=-1)
                indices = numpy.argsort(-group_x, axis=-1, kind="stable")[:, :k_group]

                mask = numpy.ones((x.shape[0], group_count), dtype=bool)
                mask[numpy.arange(x.shape[0])[:, None], indices] = False
                x = numpy.where(mask[..., None], float("-inf"), x)
                x = x.reshape(x.shape[0], -1)

            _, indices = torch.sort(
                torch.from_numpy(x), dim=-1, stable=True, descending=True
            )
            indices = numpy.asarray(indices[:, :k])

        y = numpy.take_along_axis(original_x, indices, axis=1)

        if norm_type != 0 or renorm != 0:
            y = y / (numpy.sum(y, axis=-1, keepdims=True) + eps)
        y = y * routed_scaling_factor

        if out_flag:
            out = original_x.astype(numpy.float32)
        else:
            out = None

        return [y.astype(ori_dtype), None, out]

    tolerance = {
        "float32": {"standard": "stat_rel_err"},
        "float16": {"standard": "stat_rel_err"},
        "bfloat16": {"standard": "stat_rel_err"},
    }
