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
import logging
import numpy as np


def softmax(a):
    # data = a.numpy()
    row_max = np.max(a, axis=-1, keepdims=True)
    data = a - row_max
    exp_data = np.exp(data.astype(np.float32))
    row_sum = np.sum(exp_data, axis=-1, keepdims=True)
    soft_res = exp_data / row_sum
    return soft_res


def msd(a, b, offset, scale):  # 1 * headdim*headnum
    # step1: preprocess
    a = a.astype(np.float32)  # keep
    amax = np.max(np.abs(a), axis=1) * 1.001
    asum = np.sum(a, axis=1)
    amodamax = a / amax
    a1 = np.floor(128 * amodamax)
    a2 = np.floor(128**2 * amodamax - a1 * 128)
    # step2: matmul
    c1 = np.dot(a1.astype(np.float32), b.astype(np.float32))
    c2 = np.dot(a2.astype(np.float32), b.astype(np.float32))
    # step3: postPress
    c = (amax * (c1 / 128 + c2 / 128**2) + asum * offset.astype(np.float32)) * scale
    return c


def flash_attention(query, key, value, pse, mask, antiquant_scale, antiquant_offset):
    is_float16 = query.dtype == np.float16
    num_heads = query.shape[1]
    key = np.transpose(key, (0, 2, 1))
    qk = np.matmul(query.astype(np.float32), key)
    # 伪量化QK
    if len(antiquant_scale) >= 1:
        for n in range(num_heads):
            # 伪量化参数初始化
            temp_scale = np.ones((query.shape[1], key.shape[2])).astype(np.float32)
            temp_offset = np.zeros((query.shape[1], key.shape[2])).astype(np.float32)
            temp_scale[:, :] = antiquant_scale[0]
            if len(antiquant_offset) >= 1:
                temp_offset[:, :] = antiquant_offset[0]
            qk[n, :, :] = msd(
                query[n, :, :].astype(np.float32), key[n, :, :], temp_offset, temp_scale
            )

    if len(pse) >= 1:
        qk += pse
    if len(mask) >= 1:
        qk *= 1 - mask
    p = softmax(qk)
    if is_float16:
        p = p.astype(np.float16)
    pv = np.matmul(p.astype(np.float32), value)
    # 伪量化 PV
    if len(antiquant_scale) >= 1:
        for n in range(num_heads):
            # 伪量化参数初始化
            temp_scale = np.ones((p.shape[1], value.shape[2])).astype(np.float32)
            temp_offset = np.zeros((p.shape[1], value.shape[2])).astype(np.float32)
            temp_scale[:, :] = antiquant_scale[1]
            if len(antiquant_offset) >= 1:
                temp_offset[:, :] = antiquant_offset[1]
            pv[n, :, :] = msd(
                p[n, :, :].astype(np.float32), value[n, :, :], temp_offset, temp_scale
            )

    if is_float16:
        pv = pv.astype(np.float16)
    return pv


def compute(
    query,  # (batch, num_heads, query_seq, head_dim)
    key,  # (batch, num_key_value_heads, key_value_seq, head_dim)
    value,  # (batch, num_key_value_heads, key_value_seq, head_dim)
    actual_query_seqlen,
    actual_kv_seqlen,
    pse_shift,
    atten_mask,
    antiquant_scale,
    antiquant_offset,
    num_heads=1,
    input_layout="BNSD",
):
    # 获取输入输出相关shape参数
    batch = query.shape[0]
    if input_layout == "BNSD":
        query_seq = query.shape[2]
        key_value_seq = key.shape[2]
        head_dim = query.shape[3]
        num_key_value_heads = key.shape[1]
    elif input_layout == "BSND":
        query_seq = query.shape[1]
        key_value_seq = key.shape[1]
        head_dim = query.shape[3]
        num_key_value_heads = key.shape[2]
    elif input_layout == "BSH":
        query_seq = query.shape[1]
        key_value_seq = key.shape[1]
        head_dim = query.shape[2] // num_heads
        num_key_value_heads = key.shape[2] // head_dim
    else:
        return

    # 将输入整理为 BNSD 的数据排布格式
    if input_layout == "BSH":
        query = query.reshape((batch, query_seq, num_heads, head_dim))
        key = key.reshape((batch, key_value_seq, num_key_value_heads, head_dim))
        value = value.reshape((batch, key_value_seq, num_key_value_heads, head_dim))
    if input_layout == "BSH" or input_layout == "BSND":
        query_bnsd = np.transpose(query, (0, 2, 1, 3))
        key_bnsd = np.transpose(key, (0, 2, 1, 3))
        value_bnsd = np.transpose(value, (0, 2, 1, 3))
    else:
        query_bnsd = query
        key_bnsd = key
        value_bnsd = value

    output = np.zeros(shape=(batch, num_heads, query_seq, head_dim), dtype=query.dtype)

    # 实现actual_seqlen功能
    if batch != 1 and len(actual_query_seqlen) == 1:
        arr_query_seqlen = np.array(actual_query_seqlen, dtype=np.int32)
        expanded = np.broadcast_to(arr_query_seqlen, (batch,))  # 创建广播视图
        actual_query_seqlen = tuple(expanded)  # 转换为tuple
    if batch != 1 and len(actual_kv_seqlen) == 1:
        arr_kv_seqlen = np.array(actual_kv_seqlen, dtype=np.int32)
        expanded = np.broadcast_to(arr_kv_seqlen, (batch,))  # 创建广播视图
        actual_kv_seqlen = tuple(expanded)  # 转换为tuple

    # 处理可选输入mask，并将mask广播到BNSD
    pse_flag = 1 - (len(pse_shift) == 0)
    mask_flag = 1 - (len(atten_mask) == 0)
    mask_bn1s = np.repeat(atten_mask, repeats=num_heads, axis=1) if mask_flag else []
    for b in range(batch):
        temp_query = query_bnsd[b, :, : actual_query_seqlen[0], :]
        temp_key = key_bnsd[b, :, : actual_kv_seqlen[b], :]
        temp_value = value_bnsd[b, :, : actual_kv_seqlen[b], :]
        temp_pse = pse_shift[b, :, :, : actual_kv_seqlen[b]] if pse_flag else []
        temp_mask = atten_mask[b, :, :, : actual_kv_seqlen[b]] if mask_flag else []
        temp_output = flash_attention(
            temp_query,
            temp_key,
            temp_value,
            temp_pse,
            temp_mask,
            antiquant_scale,
            antiquant_offset,
        )

        np.copyto(output[b, :, : actual_query_seqlen[b], :], temp_output)

    if input_layout == "BSH" or input_layout == "BSND":
        output = np.transpose(output, (0, 2, 1, 3))
    if input_layout == "BSH":
        output = output.reshape((batch, query_seq, num_heads * head_dim))

    return torch.from_numpy(output)


def aclnn_op_func_ifa_cpu(input_data: InputDataset):
    input_dtype = input_data.kwargs["query"].dtype
    if input_dtype == torch.bfloat16:
        query = input_data.kwargs["query"].to(dtype=torch.float32).numpy()
        key = input_data.kwargs["key"][0].to(dtype=torch.float32).numpy()
        value = input_data.kwargs["value"][0].to(dtype=torch.float32).numpy()
    else:
        query = input_data.kwargs["query"].numpy()
        key = input_data.kwargs["key"][0].numpy()
        value = input_data.kwargs["value"][0].numpy()
    numHeads = input_data.kwargs["numHeads"]
    input_layout = input_data.kwargs["inputLayout"]

    act_q_seq = [1]
    act_kv_seq = input_data.kwargs["actualSeqLengths"]

    pse = (
        input_data.kwargs["pseShift"].to(dtype=torch.float16).numpy()
        if input_data.kwargs["pseShift"] is not None
        else []
    )
    mask = (
        input_data.kwargs["attenMask"].to(dtype=torch.uint8).numpy()
        if input_data.kwargs["attenMask"] is not None
        else []
    )

    antiquant_scale = (
        input_data.kwargs["antiquantScale"].to(dtype=torch.float16).numpy()
        if input_data.kwargs["antiquantScale"] is not None
        else []
    )
    antiquant_offset = (
        input_data.kwargs["antiquantOffset"].to(dtype=torch.float16).numpy()
        if input_data.kwargs["antiquantOffset"] is not None
        else []
    )

    return compute(
        query,
        key,
        value,
        act_q_seq,
        act_kv_seq,
        pse,
        mask,
        antiquant_scale,
        antiquant_offset,
        num_heads=numHeads,
        input_layout=input_layout,
    ).to(dtype=input_dtype)


@register("executor_incre_flash_attention_v4")
class fusedInferAttentionScoreApi(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(fusedInferAttentionScoreApi, self).__init__(task_result)

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        output = aclnn_op_func_ifa_cpu(input_data)
        # print("golden output is: ", output)
        return output


@register("executor_aclnn_incre_flash_attention_v4")
class aclnnFusedInferAttentionScoreApi(AclnnBaseApi):
    def __init__(self, task_result: TaskResult, backend):
        super(aclnnFusedInferAttentionScoreApi, self).__init__(task_result, backend)

    def init_by_input_data(self, input_data: InputDataset):
        input_args = []  # 算子的入参列表
        input_args, output_packages = super().init_by_input_data(input_data)
        output_packages = []  # 算子的出参数据包列表
        input_args.pop()
        output_packages.append(input_args[-1])
        return input_args, output_packages

    def __call__(self):
        self.backend.aclnn_x_get_workspace_size()
        self.backend.aclnn_x()

    def after_call(self, output_packages):
        output = []
        for output_pack in output_packages:
            output.append(self.acl_tensor_to_torch(output_pack))
        # print("aclnn output is: ", output)
        return output
