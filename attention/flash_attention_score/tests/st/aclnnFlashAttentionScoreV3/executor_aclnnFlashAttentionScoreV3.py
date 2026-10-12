#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
import torch
from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.tasks.backends.lib_interface.acl_wrapper import AclTensor
import logging
import ctypes
from einops import rearrange
import numpy as np


def tsoftmax(x):
    softmax_max = torch.max(x, dim=-1, keepdims=True)[0]
    x_sub = x.sub(softmax_max)
    y = torch.exp(x_sub)
    softmax_sum = y.sum(dim=-1, keepdims=True)
    softmax_res = y.div(softmax_sum)
    return softmax_res, softmax_max, softmax_sum


def tforward(
    q, k, v, drop_mask, atten_mask, pse, headNum, inputLayout, scale, keep_prob
):
    if inputLayout == "BSH":
        # (B, S, N*D) => (B, N, S, D)
        B = q.shape[0]
        S1 = q.shape[1]
        N1 = headNum
        D1 = q.shape[2] // N1
        S2 = v.shape[1]
        N2 = k.shape[2] // D1
        D2 = v.shape[2] // N2
        q = q.view(B, S1, N1, D1).permute(0, 2, 1, 3).contiguous()
        k = k.view(B, S2, N2, D1).permute(0, 2, 1, 3).contiguous()
        v = v.view(B, S2, N2, D2).permute(0, 2, 1, 3).contiguous()
    elif inputLayout == "SBH":
        # (S, B, H) => (B, N, S, D)
        B = q.shape[1]
        S1 = q.shape[0]
        N1 = headNum
        D1 = q.shape[2] // N1
        S2 = v.shape[0]
        N2 = k.shape[2] // D1
        D2 = v.shape[2] // N2
        q = q.view(S1, B, N1, D1).permute(1, 2, 0, 3).contiguous()
        k = k.view(S2, B, N2, D1).permute(1, 2, 0, 3).contiguous()
        v = v.view(S2, B, N2, D2).permute(1, 2, 0, 3).contiguous()
    elif inputLayout == "BSND":
        # (B, S, N, D) => (B, N, S, D)
        q = q.permute(0, 2, 1, 3).contiguous()
        k = k.permute(0, 2, 1, 3).contiguous()
        v = v.permute(0, 2, 1, 3).contiguous()
    q_ori_dtype = q.dtype
    if q_ori_dtype == torch.float64:
        gtype = torch.float64
    elif q_ori_dtype != torch.float32:
        gtype = torch.float32
    else:
        gtype = torch.float64
    q = q.to(gtype)
    k = k.to(gtype)
    v = v.to(gtype)
    atten_mask = atten_mask.to(gtype)
    if pse is None or len(pse.shape) == 0:
        qk = torch.matmul(q, k.permute(0, 1, 3, 2)).mul(scale)
    else:
        qk = (torch.matmul(q, k.permute(0, 1, 3, 2)) + pse).mul(scale)
    if atten_mask is None or len(atten_mask.shape) == 0:
        qk = qk
    else:
        qk = qk + atten_mask * (-40000.0)
    softmax_res, softmax_max, softmax_sum = tsoftmax(qk)
    if len(atten_mask.shape) != 0:
        softmax_res[atten_mask.bool().broadcast_to(softmax_res.shape)] = 0
    if drop_mask is None or len(drop_mask.shape) == 0:
        drop_res = softmax_res
    else:
        drop_res = softmax_res * drop_mask * (1.0 / keep_prob)
    attention_out = torch.matmul(drop_res, v)
    softmax_max = softmax_max.repeat(1, 1, 1, 8)
    softmax_sum = softmax_sum.repeat(1, 1, 1, 8)
    if inputLayout == "BSND":
        attention_out = attention_out.permute(0, 2, 1, 3).contiguous()
    elif inputLayout == "BSH":
        attention_out = (
            attention_out.permute(0, 2, 1, 3).reshape(B, S1, -1).contiguous()
        )
    elif inputLayout == "SBH":
        attention_out = (
            attention_out.permute(2, 0, 1, 3).reshape(S1, B, -1).contiguous()
        )

    if q_ori_dtype == torch.float32 and gtype == torch.float64:
        softmax_sum = softmax_sum.to(q_ori_dtype)
        softmax_max = softmax_max.to(q_ori_dtype)
        attention_out = attention_out.to(q_ori_dtype)
    elif q_ori_dtype == torch.float16 or q_ori_dtype == torch.bfloat16:
        attention_out = attention_out.to(q_ori_dtype)
    return (
        softmax_max,
        softmax_sum,
        torch.Tensor(0).to(softmax_sum.dtype),
        attention_out,
    )


def aclnn_op_func_fa_cpu(input_data: InputDataset):
    input_dtype = input_data.kwargs["query"].dtype
    query = input_data.kwargs["query"]
    key = input_data.kwargs["key"]
    value = input_data.kwargs["value"]
    real_shift = input_data.kwargs["real_shift"]
    drop_mask = input_data.kwargs["drop_mask"]
    atten_mask = input_data.kwargs["atten_mask"]
    headNum = input_data.kwargs["headNum"]
    inputLayout = input_data.kwargs["inputLayout"]
    scaleValue = input_data.kwargs["scaleValue"]
    keep_prob = input_data.kwargs["keepProb"]

    return tforward(
        query,
        key,
        value,
        drop_mask,
        atten_mask,
        real_shift,
        headNum,
        inputLayout,
        scaleValue,
        keep_prob,
    )


@register("executor_flash_attention_score_v3")
class aclnnFlashAttentionScoreApi(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(aclnnFlashAttentionScoreApi, self).__init__(task_result)

    def get_cpp_func_signature_type(self):
        return "aclnnStatus aclnnFlashAttentionScoreV3GetWorkspaceSize(\
            const aclTensor *query, const aclTensor *key, const aclTensor *value, const aclTensor *realShiftOptional,\
            const aclTensor *dropMaskOptional, const aclTensor *paddingMaskOptional, const aclTensor *attenMaskOptional,const aclTensor *sinkOptional,\
            const aclIntArray *prefixOptional, const aclIntArray *qStartIdxOptional, const aclIntArray *kvStartIdxOptional,\
            double scaleValue, double keepProb, int64_t preTokens, int64_t nextTokens, int64_t headNum, char *inputLayout,\
            int64_t innerPrecise, int64_t sparseMode, int64_t pseType, const aclTensor *softmaxMaxOut,\
            const aclTensor *softmaxSumOut, const aclTensor *softmaxOutOut, const aclTensor *attentionOutOut,\
            uint64_t *workspaceSize, aclOpExecutor **executor)"

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        output = aclnn_op_func_fa_cpu(input_data)
        return output


@register("aclnn_flash_attention_score_v3")
class FlashAttentionScoreApi(AclnnBaseApi):
    def __call__(self):
        super().__call__()

    def init_by_input_data(self, input_data: InputDataset):
        input_args, output_packages = super().init_by_input_data(input_data)
        AclTensorPtr = ctypes.POINTER(AclTensor)
        null_void_ptr = ctypes.c_void_p(None)
        null_tensor_ptr = ctypes.cast(null_void_ptr, AclTensorPtr)
        input_args[4] = null_tensor_ptr
        return input_args, output_packages
