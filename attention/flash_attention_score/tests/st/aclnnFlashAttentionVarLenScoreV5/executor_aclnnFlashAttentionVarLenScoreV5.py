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


def aclnn_op_func_fa_varLen_cpu(input_data: InputDataset):
    input_dtype = input_data.kwargs["query"].dtype
    q = input_data.kwargs["query"]
    k = input_data.kwargs["key"]
    v = input_data.kwargs["value"]
    real_shift = input_data.kwargs["real_shift"]
    drop_mask = input_data.kwargs["drop_mask"]
    atten_mask = input_data.kwargs["atten_mask"]
    actual_seq_qlen = input_data.kwargs["actual_seq_qlen"]
    actual_seq_kvlen = input_data.kwargs["actual_seq_kvlen"]
    headNum = input_data.kwargs["headNum"]
    scaleValue = input_data.kwargs["scaleValue"]
    keep_prob = input_data.kwargs["keepProb"]

    B = len(actual_seq_qlen)
    softmax_max = torch.empty(0)
    softmax_sum = torch.empty(0)
    attention_out = torch.zeros_like(q)
    seqlens_list_q = []
    seqlens_list_kv = []
    cu_seqlens_q = [0] + actual_seq_qlen
    cu_seqlens_kv = [0] + actual_seq_kvlen

    for i in range(B):
        if i == 0:
            seqlens_list_q.append(actual_seq_qlen[0])
            seqlens_list_kv.append(actual_seq_kvlen[0])
        else:
            seqlens_list_q.append(actual_seq_qlen[i] - actual_seq_qlen[i - 1])
            seqlens_list_kv.append(actual_seq_kvlen[i] - actual_seq_kvlen[i - 1])

    for i in range(B):
        if seqlens_list_q[i] != 0 and seqlens_list_kv[i] != 0:
            qi = q[cu_seqlens_q[i] : cu_seqlens_q[i + 1]]
            ki = k[cu_seqlens_kv[i] : cu_seqlens_kv[i + 1]]
            vi = v[cu_seqlens_kv[i] : cu_seqlens_kv[i + 1]]
            qi = qi.unsqueeze(0).permute(0, 2, 1, 3)
            ki = ki.unsqueeze(0).permute(0, 2, 1, 3)
            vi = vi.unsqueeze(0).permute(0, 2, 1, 3)
            drop_maski = drop_mask
            atten_maski = atten_mask[: seqlens_list_q[i], : seqlens_list_kv[i]]
            psei = real_shift
            softmax_maxi, softmax_sumi, softmax_res, attention_outi = tforward(
                qi,
                ki,
                vi,
                drop_maski,
                atten_maski,
                psei,
                headNum,
                "BNSD",
                scaleValue,
                keep_prob,
            )
            softmax_maxi = softmax_maxi.broadcast_to(1, headNum, seqlens_list_q[i], 8)
            softmax_maxi = softmax_maxi.contiguous().view(-1)
            softmax_sumi = softmax_sumi.broadcast_to(1, headNum, seqlens_list_q[i], 8)
            softmax_sumi = softmax_sumi.contiguous().view(-1)
            softmax_max = torch.cat([softmax_max, softmax_maxi], dim=0)
            softmax_sum = torch.cat([softmax_sum, softmax_sumi], dim=0)
            attention_out[cu_seqlens_q[i] : cu_seqlens_q[i + 1]] = (
                attention_outi.squeeze(0).permute(1, 0, 2)
            )
        else:
            softmax_maxi = torch.zeros(1, N1, seqlens_list_q[i], 8)
            softmax_maxi = softmax_maxi.contiguous().view(-1)
            softmax_sumi = torch.zeros(1, N1, seqlens_list_q[i], 8)
            softmax_sumi = softmax_sumi.contiguous().view(-1)
            softmax_max = torch.cat([softmax_max, softmax_maxi], dim=0)
            softmax_sum = torch.cat([softmax_sum, softmax_sumi], dim=0)

    softmax_max = softmax_max.view(q.shape[0], q.shape[1], 8)
    softmax_sum = softmax_sum.view(q.shape[0], q.shape[1], 8)

    return softmax_max, softmax_sum, softmax_res, attention_out


@register("executor_flash_attention_varLen_score_v5")
class aclnnFlashAttentionVarLenScoreApi(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(aclnnFlashAttentionVarLenScoreApi, self).__init__(task_result)

    def get_cpp_func_signature_type(self):
        return "aclnnStatus aclnnFlashAttentionVarLenScoreV5GetWorkspaceSize(\
            const aclTensor *query, const aclTensor *queryRope, const aclTensor *key, const aclTensor *keyRope,\
            const aclTensor *value, const aclTensor *realShiftOptional, const aclTensor *dropMaskOptional,\
            const aclTensor *paddingMaskOptional, const aclTensor *attenMaskOptional, const aclTensor *sinkOptional,\
            const aclIntArray *prefixOptional, const aclIntArray *actualSeqQLenOptional,\
            const aclIntArray *actualSeqKvLenOptional, const aclIntArray *qStartIdxOptional,\
            const aclIntArray *kvStartIdxOptional, double scaleValue, double keepProb, int64_t preTokens, int64_t nextTokens,\
            int64_t headNum, char *inputLayout, int64_t innerPrecise, int64_t sparseMode, int64_t pseType, char *softmaxOutLayout,\
            const aclTensor *softmaxMaxOut, const aclTensor *softmaxSumOut, const aclTensor *softmaxOutOut,\
            const aclTensor *attentionOutOut, uint64_t *workspaceSize, aclOpExecutor **executor)"

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        output = aclnn_op_func_fa_varLen_cpu(input_data)
        return output


@register("aclnn_flash_attention_varLen_score_v5")
class FlashAttentionScoreVarLenApi(AclnnBaseApi):
    def __call__(self):
        super().__call__()

    def init_by_input_data(self, input_data: InputDataset):
        input_args, output_packages = super().init_by_input_data(input_data)
        AclTensorPtr = ctypes.POINTER(AclTensor)
        null_void_ptr = ctypes.c_void_p(None)
        null_tensor_ptr = ctypes.cast(null_void_ptr, AclTensorPtr)
        input_args[1] = null_tensor_ptr
        input_args[3] = null_tensor_ptr
        input_args[5] = null_tensor_ptr
        input_args[6] = null_tensor_ptr

        return input_args, output_packages
