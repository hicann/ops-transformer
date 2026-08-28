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
import os
import torch
import logging
import math
import numpy as np
import copy
import time
import random
import ctypes
import copy
import tensorflow as tf
from functools import wraps
from enum import Enum
from array import array
from ml_dtypes import bfloat16
from dataclasses import dataclass
from collections import defaultdict
from dataclasses import dataclass
from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.tasks.backends.lib_interface.acl_wrapper import AclIntArray, pointer


# public function
def get_np_dtype(type_str):
    type_dict = {
        "fp64": np.float64,
        "fp32": np.float32,
        "fp16": np.float16,
        "int64": np.int64,
        "int32": np.int32,
        "int16": np.int16,
        "int8": np.int8,
        "uint64": np.uint64,
        "uint32": np.uint32,
        "uint16": np.uint16,
        "uint8": np.uint8,
        "bool": np.bool_,
        "complex64": np.complex64,
        "complex128": np.complex128,
        "complex32": np.float16,
        "bf16": tf.bfloat16.as_numpy_dtype,
        "bfloat16": tf.bfloat16.as_numpy_dtype,
        "float4_e2m1": np.uint8,
        "float4_e1m2": np.uint8,
        "float8_e8m0": np.uint8,
        "hifloat8": np.uint8,
        "fp4_e1m2": np.uint8,
        "fp4_e2m1": np.uint8,
        "fp8_e8m0": np.uint8,
        "float32": np.float32,
        "float16": np.float16,
        "qint8": np.int8,
        "qint32": np.int32,
        "quint8": np.uint8,
        "qint16": np.int16,
        "uint1": np.uint8,
        "quint16": np.uint16,
        "fp4_e2m1_as_fp32": np.float32,
    }
    if type_str == "int4":
        from ml_dtypes import int4

        return int4
    elif type_str == "float8_e5m2":
        from ml_dtypes import float8_e5m2

        return float8_e5m2
    elif type_str == "float8_e4m3fn":
        from ml_dtypes import float8_e4m3fn

        return float8_e4m3fn
    else:
        return type_dict[type_str]


def get_pt_dtype(type_str):
    type_dict = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "fp64": torch.float64,
        "int8": torch.int8,
        "int16": torch.int16,
        "int32": torch.int32,
        "int64": torch.int64,
        "uint8": torch.uint8,
        "bool": torch.bool,
        "complex64": torch.complex64,
        "complex128": torch.complex128,
        "bf16": torch.bfloat16,
        "uint1": torch.uint8,
    }
    if type_str == "hifloat8":
        import torch_npu

        return torch_npu.hifloat8
    return type_dict[type_str]


def cvt_hifuint8_to_float(x, over_mode=True):
    x = int(x)
    if x == 0:
        return float(0)
    elif x == 128:
        if over_mode:
            return np.nan
        else:
            return float(0)
    elif x == 239:
        if over_mode:
            return -np.inf
        else:
            return -32768
    elif x == 111:
        if over_mode:
            return np.inf
        else:
            return 32768
    else:
        if x >= 128:
            sign = -1.0
        else:
            sign = 1.0
        dot_4_bits = x & 120  # b01111000 = 120
        dot_4_value = dot_4_bits >> 3
        if dot_4_value >= 12:
            # b1100 =12 D4
            exponet = x & 30  # b00011110 = 30
            exponet_int = exponet >> 1
            if exponet_int >= 8:
                # b1000 = 8
                exponet_value = -exponet_int
            else:
                exponet_value = exponet_int + 8

            fra_int = x & 1  # b00000001
            m_value = 1.0 + fra_int * 0.5
        elif dot_4_value >= 8:
            # b1000 =8 D3
            exponet = x & 28  # b00011100 = 28
            exponet_int = exponet >> 2
            if exponet_int >= 4:
                # b100 = 4
                exponet_value = -exponet_int
            else:
                exponet_value = exponet_int + 4
            fra_int = x & 3  # b00000011
            m_value = 1.0 + fra_int * 0.25
        elif dot_4_value >= 4:
            # b0100 =8 D2
            exponet = x & 24  # b00011000 = 24
            exponet_int = exponet >> 3
            if exponet_int >= 2:
                # b10 = 2
                exponet_value = -exponet_int
            else:
                exponet_value = exponet_int + 2
            fra_int = x & 7  # b00000111
            m_value = 1.0 + fra_int * 0.125
        elif dot_4_value >= 2:
            # b0010 =2 D1
            exponet = x & 8  # b00001000 = 8
            exponet_sign = exponet >> 3
            if exponet_sign >= 1:
                # b10 = 2
                exponet_value = -1
            else:
                exponet_value = 1
            fra_int = x & 7  # b00000111
            m_value = 1.0 + fra_int * 0.125
        elif dot_4_value == 1:
            # d0
            exponet_value = 0
            fra_int = x & 7  # b00000111
            m_value = 1.0 + fra_int * 0.125
        elif dot_4_value == 0:
            # dml
            m_value = 1
            exponet_value = (x & 7) - 23  # b00000111 = 7
        else:
            print("error,dot error")
            m_value = 0.0
            exponet_value = 0
        return sign * pow(2.0, exponet_value) * m_value


def cvt_fp4_e1m2_to_bfloat16(x):
    Fp4e1m2ToBf16 = {
        "0": 0.0,
        "1": 0.25,
        "2": 0.5,
        "3": 0.75,
        "4": 1.0,
        "5": 1.25,
        "6": 1.5,
        "7": 1.75,
        "8": -0.0,
        "9": -0.25,
        "10": -0.5,
        "11": -0.75,
        "12": -1.0,
        "13": -1.25,
        "14": -1.5,
        "15": -1.75,
    }
    x = int(x)
    first_fp4val = x & 0x0F
    first_fp4str = str(first_fp4val)
    return Fp4e1m2ToBf16[first_fp4str]


def new_trans_np_fp4_e2m1_tensor_to_bfloat16(in_tensor):
    shape_tensor = in_tensor.shape
    multi_shape = np.prod(shape_tensor)
    in_tensor = in_tensor.reshape(multi_shape)
    bfloat16_tensor = np.zeros(multi_shape)
    for i in range(multi_shape):
        bfloat16_tensor[i] = cvt_fp4_e2m1_to_bfloat16(in_tensor[i])
    return bfloat16_tensor.reshape(shape_tensor)


def new_trans_np_fp4_e1m2_tensor_to_bfloat16(in_tensor):
    shape_tensor = in_tensor.shape
    multi_shape = np.prod(shape_tensor)
    in_tensor = in_tensor.reshape(multi_shape)
    bfloat16_tensor = np.zeros(multi_shape)
    for i in range(multi_shape):
        bfloat16_tensor[i] = cvt_fp4_e1m2_to_bfloat16(in_tensor[i])
    return bfloat16_tensor.reshape(shape_tensor)


def trans_np_hifuint8_tensor_to_float32(in_tensor):
    shape_tensor = in_tensor.shape
    multi_shape = np.prod(shape_tensor)
    out_tensor = np.zeros(multi_shape).astype(np.float32)
    in_tensor = in_tensor.reshape(multi_shape)
    for i in range(multi_shape):
        out_tensor[i] = cvt_hifuint8_to_float(in_tensor[i])
    out_tensor = out_tensor.reshape(shape_tensor).astype(np.float32)
    return out_tensor


# IFA
def _create_mask(m_shape, pre_tokens, next_tokens):
    next_masks = np.triu(
        np.ones(m_shape, dtype="uint8"), k=1 + int(next_tokens)
    )  # 生成下三角全是0的矩阵
    pre_mask = np.tril(
        np.ones(m_shape, dtype="uint8"), k=-1 - int(pre_tokens)
    )  # 生成上三角全是0的矩阵
    atten_masks = pre_mask + next_masks

    return atten_masks


def _create_mask_band(
    m_shape,
    pre_tokens,
    next_tokens,
    actualSeqLengths,
    actualSeqLengthsKV,
    actualprefixKV,
    prefix_kvs,
    batch,
    numheads,
    kvs_list,
    m_dtype,
):
    mask_s_q = m_shape[0]
    mask_s_kv = m_shape[1]

    pre_tokens_list = []
    next_tokens_list = []
    re_mask_batch = []
    for i in range(batch):
        if len(actualSeqLengths) == 0:
            S1 = mask_s_q
        else:
            S1 = actualSeqLengths[i]
        if len(actualSeqLengthsKV) != 0:
            S2 = actualSeqLengthsKV[i] + actualprefixKV
        elif len(kvs_list) > 1:
            S2 = kvs_list[i] + actualprefixKV
        else:
            S2 = mask_s_kv - prefix_kvs + actualprefixKV
        pre_tokens_new = S1 - S2 + pre_tokens
        pre_tokens_list.append(pre_tokens_new)

        next_tokens_new = S2 - S1 + next_tokens
        next_tokens_list.append(next_tokens_new)
        atten_masks = _create_mask(m_shape, pre_tokens_new, next_tokens_new)
        re_mask_batch.append(atten_masks)
    return re_mask_batch, pre_tokens_list, next_tokens_list


def get_attention_mask_batch_num(npu_m_shape, q_bnsd_shape):
    batch, numhead = None, None
    if len(npu_m_shape) == 2:
        s1 = npu_m_shape[0]
        s2 = npu_m_shape[1]
        return batch, numhead, s1, s2
    if len(npu_m_shape) == 3:
        batch = npu_m_shape[0]
        s1 = npu_m_shape[1]
        s2 = npu_m_shape[2]
        return batch, numhead, s1, s2
    if len(npu_m_shape) == 4:
        batch = npu_m_shape[0]
        numhead = npu_m_shape[1]
        s1 = npu_m_shape[2]
        s2 = npu_m_shape[3]
        return batch, numhead, s1, s2


def _create_mask_right_down(
    m_shape,
    pre_tokens,
    next_tokens,
    actualSeqLengths,
    actualSeqLengthsKV,
    actualprefixKV,
    prefix_kvs,
    batch,
    numheads,
    kvs_list,
    m_dtype,
):
    mask_s_q = m_shape[0]
    mask_s_kv = m_shape[1]

    next_tokens_list = []
    re_mask_batch = []
    for i in range(batch):
        if len(actualSeqLengths) == 0:
            S1 = mask_s_q
        else:
            S1 = actualSeqLengths[i]
        if len(actualSeqLengthsKV) != 0:
            S2 = actualSeqLengthsKV[i] + actualprefixKV
        elif len(kvs_list) > 1:
            S2 = kvs_list[i] + actualprefixKV
        else:
            S2 = mask_s_kv - prefix_kvs + actualprefixKV
        next_tokens = S2 - S1
        next_tokens_list.append(next_tokens)
        atten_masks = _create_mask(m_shape, pre_tokens, next_tokens)
        re_mask_batch.append(atten_masks)
    return re_mask_batch, next_tokens_list


def _random_fill_tensor(tensor, shape, random_number, value=0):
    for i in range(0, random_number):
        point = []
        for k in range(0, len(shape)):
            point.append(random.randint(1, shape[k]) - 1)
        tensor[point[0], point[1]] = value
    return tensor


def _t_broadcastKV_sigle(numHeads, numKeyValueHeads, kv_tensor):
    factor = numHeads // numKeyValueHeads
    kv_shape = kv_tensor.shape
    B = kv_shape[0]
    S = kv_shape[2]
    D = kv_shape[3]
    kv_res = torch.zeros([B, numHeads, S, D])
    for i in range(numHeads):
        j = i // factor
        kv_res[:, i : i + 1, :, :] = kv_tensor[:, j : j + 1, :, :]
    return kv_res, kv_res.shape


def _np_broadcastKV_sigle(numHeads, numKeyValueHeads, kv_tensor, dtype):
    factor = numHeads // numKeyValueHeads
    kv_shape = kv_tensor.shape
    B = kv_shape[0]
    S = kv_shape[2]
    D = kv_shape[3]
    kv_res = np.zeros([B, numHeads, S, D], dtype=dtype)
    for i in range(numHeads):
        j = i // factor
        kv_res[:, i : i + 1, :, :] = kv_tensor[:, j : j + 1, :, :]
    return kv_res, kv_res.shape


def _create_random_mask_by_spars(
    cpu_m_shape,
    npu_m_shape,
    m_dtype,
    pre_tokens,
    next_tokens,
    actualSeqLengths,
    actualSeqLengthsKV,
    actualprefixKV,
    prefix_kvs,
    kvs_list,
    batch=1,
    numheads=1,
    sp_mode=0,
    random_ones=0,
):
    # mask shape [sq,skv]  #mshape  npu  fshape cpu
    print(
        f"[_create_random_mask_by_spars] full_m_shape:{cpu_m_shape} m_shape:{npu_m_shape} datype:{m_dtype} pret:{pre_tokens} nextt:{next_tokens} sp_mode:{sp_mode}"
    )
    if sp_mode == 0:
        cpu_mask, npu_mask = _create_mask_no_sparse(
            cpu_m_shape,
            npu_m_shape,
            pre_tokens,
            next_tokens,
            batch,
            numheads,
            m_dtype,
            random_ones,
        )
        return cpu_mask, npu_mask.astype(m_dtype), pre_tokens, next_tokens
    if sp_mode == 1:
        print("[_create_random_mask_by_spars] sp_mode is 1 return all zero mask")
        pre_tokens = 214748647
        next_tokens = 214748647
        cpu_mask, npu_mask = _create_mask_no_sparse(
            cpu_m_shape,
            npu_m_shape,
            pre_tokens,
            next_tokens,
            batch,
            numheads,
            m_dtype,
            random_ones,
        )
        return cpu_mask, npu_mask.astype(m_dtype), pre_tokens, next_tokens

    if sp_mode == 2:
        pre_tokens = 214748647
        next_tokens = 0
        print(
            f"[_create_random_mask_by_spars] sp_mode is 2 npu mask shape:{npu_m_shape}"
        )
        npu_mask = np.triu(np.ones(npu_m_shape), k=1)
        cpu_mask = _create_mask_left_up(
            cpu_m_shape, pre_tokens, next_tokens, batch, numheads, m_dtype
        )
        return cpu_mask, npu_mask.astype(m_dtype), pre_tokens, next_tokens
    if sp_mode == 3:  # rightdown
        pre_tokens = 214748647
        print(
            f"[_create_random_mask_by_spars] sp_mode is 3 npu mask shape:{npu_m_shape}"
        )
        npu_mask = np.triu(np.ones(npu_m_shape), k=1)
        cpu_mask, next_tokens_new = _create_mask_right_down(
            cpu_m_shape,
            pre_tokens,
            next_tokens,
            actualSeqLengths,
            actualSeqLengthsKV,
            actualprefixKV,
            prefix_kvs,
            batch,
            numheads,
            kvs_list,
            m_dtype,
        )
        return cpu_mask, npu_mask, pre_tokens, next_tokens_new
    if sp_mode == 4:
        npu_mask = np.triu(np.ones(npu_m_shape), k=1)
        cpu_mask, pre_tokens_new, next_tokens_new = _create_mask_band(
            cpu_m_shape,
            pre_tokens,
            next_tokens,
            actualSeqLengths,
            actualSeqLengthsKV,
            actualprefixKV,
            prefix_kvs,
            batch,
            numheads,
            kvs_list,
            m_dtype,
        )
        return cpu_mask, npu_mask.astype(m_dtype), pre_tokens_new, next_tokens_new


def find_output_value(input_list, output_value_list):
    temp_list = []
    is_list_or_tuple = False
    for item in input_list:
        if not isinstance(item, list) and not isinstance(item, tuple):
            output_value_list.append(item)
        else:
            for elment in item:
                temp_list.append(elment)
    if len(temp_list):
        is_list_or_tuple = True
    return is_list_or_tuple, temp_list


def dequant(x, deqscale, relu_weight):
    deqscale = np.uint64(deqscale)
    deqscale = np.frombuffer(deqscale, dtype=np.float32)
    deqscale = deqscale[:1][0]
    if relu_weight is None:
        relu_weight = deqscale
    else:
        relu_weight = np.full(deqscale.shape, relu_weight)
    res_cal = np.zeros(x.shape, np.float16)
    for n in range(x.shape[0]):
        for c in range(x.shape[1]):
            for h in range(x.shape[2]):
                for w in range(x.shape[3]):
                    x[n, c, h, w] = x[n, c, h, w].astype(np.float32)
                    if x[n, c, h, w] >= 0:
                        res_cal[n, c, h, w] = x[n, c, h, w] * deqscale
                    else:
                        res_cal[n, c, h, w] = x[n, c, h, w] * relu_weight
    return res_cal


def s8_saturation(inputdata):
    if inputdata > 127:
        inputdata = 127
    elif inputdata < -128:
        inputdata = -128
    outdata = np.int8(inputdata)
    return outdata


def s9_saturation(inputdata):
    if inputdata > 255:
        inputdata = 255
    elif inputdata < -256:
        inputdata = -256
    return inputdata


def quant_pc(x, qscale, qoffset):
    print(f"qscale:{qscale.shape}")
    print(f"x:{x.shape}")
    s8_res_cal = np.zeros(x.shape, np.int8)
    for n in range(x.shape[0]):
        for c in range(x.shape[1]):
            for h in range(x.shape[2]):
                for w in range(x.shape[3]):
                    s8_res_cal[n, c, h, w] = s8_saturation(
                        np.round(
                            s9_saturation(
                                np.half(x[n, c, h, w]) * np.half(qscale[0, c, 0, w])
                            )
                            + np.half(qoffset[0, c, 0, w])
                        )
                    )
    return s8_res_cal


def ieee_754_conversion(sign, exponent_raw, mantissa, exp_len=8, mant_len=7):
    """Convert binary data into the floating point value"""
    sign_mult = -1 if sign == 1 else 1
    exponent = exponent_raw - (2 ** (exp_len - 1) - 1)
    mant_mult = 1
    for b in range(mant_len - 1, -1, -1):
        if mantissa & (2**b):
            mant_mult += 1 / (2 ** (mant_len - b))

    return sign_mult * (2**exponent) * mant_mult


def trans_tensor_fp8_e8m0_to_bf16(tensor):
    # import pdb;pdb.set_trace()
    from ml_dtypes import bfloat16

    # 创建一个与输入tensor形状相同的新tensor
    new_tensor = np.zeros_like(tensor).astype(bfloat16)
    # 使用nditer遍历tensor中的每个元素
    with np.nditer(tensor, flags=["multi_index"], op_flags=["readwrite"]) as it:
        for x in it:
            # 将元素传递给trans函数进行处理
            # new_value = new_value.astype(bfloat16)

            # x = float(x)
            # new_value = 2 ** (x - 127)
            new_value = ieee_754_conversion(0, int(x), 0)
            # 将处理后的结果回填到新tensor中
            new_tensor[it.multi_index] = new_value
    return new_tensor


def antiquant(
    k, v, scale, offset, pto_flag, lowprecision_flag, kv_dtype="int8", q_dtype="float16"
):
    N = k.shape[1]
    S = v.shape[2]
    if kv_dtype in ["fp4_e1m2", "fp4_e2m1", "float4_e1m2", "float4_e2m1"]:
        if lowprecision_flag:
            if q_dtype == "float16":
                scale_bf16 = trans_tensor_fp8_e8m0_to_bf16(scale)
                scale = scale_bf16.astype(np.float16)
            elif q_dtype == "bfloat16":
                scale_bf16 = trans_tensor_fp8_e8m0_to_bf16(scale)
                scale = scale_bf16
            else:
                print(f"[ERROR]Invalid input q dtype: {q_dtype}")
                exit(1)
        else:
            scale_bf16 = trans_tensor_fp8_e8m0_to_bf16(scale)
            scale = scale_bf16.astype(np.float16)
        scale1 = scale[0]
        scale2 = scale[1]
        if kv_dtype in ["fp4_e1m2", "float4_e1m2"]:
            if lowprecision_flag:
                if q_dtype == "float16":
                    key = new_trans_np_fp4_e1m2_tensor_to_bfloat16(k).astype(np.float16)
                    value = new_trans_np_fp4_e1m2_tensor_to_bfloat16(v).astype(
                        np.float16
                    )
                elif q_dtype == "bfloat16":
                    key = new_trans_np_fp4_e1m2_tensor_to_bfloat16(k)
                    value = new_trans_np_fp4_e1m2_tensor_to_bfloat16(v)
                else:
                    print(f"[ERROR]Invalid input q dtype: {q_dtype}")
                    exit(1)
            else:
                key = new_trans_np_fp4_e1m2_tensor_to_bfloat16(k).astype(np.float16)
                value = new_trans_np_fp4_e1m2_tensor_to_bfloat16(v).astype(np.float16)
        else:
            if lowprecision_flag:
                if q_dtype == "float16":
                    key = new_trans_np_fp4_e2m1_tensor_to_bfloat16(k).astype(np.float16)
                    value = new_trans_np_fp4_e2m1_tensor_to_bfloat16(v).astype(
                        np.float16
                    )
                elif q_dtype == "bfloat16":
                    key = new_trans_np_fp4_e2m1_tensor_to_bfloat16(k)
                    value = new_trans_np_fp4_e2m1_tensor_to_bfloat16(v)
                else:
                    print(f"[ERROR]Invalid input q dtype: {q_dtype}")
                    exit(1)
            else:
                key = new_trans_np_fp4_e2m1_tensor_to_bfloat16(k).astype(np.float16)
                value = new_trans_np_fp4_e2m1_tensor_to_bfloat16(v).astype(np.float16)
        g = scale.shape[4]
        grp_size = 32
        for nIdx in range(N):
            for sIdx in range(S):
                for gIdx in range(g):
                    # copy from
                    key_cur = key[
                        :,
                        nIdx : (nIdx + 1),
                        sIdx : (sIdx + 1),
                        gIdx * grp_size : (gIdx + 1) * grp_size,
                    ]
                    # broardcast
                    scale_k = scale1[
                        :, nIdx : (nIdx + 1), sIdx : (sIdx + 1), gIdx : (gIdx + 1)
                    ]
                    scale_k = np.full((1, grp_size), scale_k)
                    # mul
                    key_cur = key_cur * scale_k
                    # copy to
                    key[
                        :,
                        nIdx : (nIdx + 1),
                        sIdx : (sIdx + 1),
                        gIdx * grp_size : (gIdx + 1) * grp_size,
                    ] = key_cur

                for gIdx in range(g):
                    # copy from
                    value_cur = value[
                        :,
                        nIdx : (nIdx + 1),
                        sIdx : (sIdx + 1),
                        gIdx * grp_size : (gIdx + 1) * grp_size,
                    ]
                    # broardcast
                    scale_v = scale2[
                        :, nIdx : (nIdx + 1), sIdx : (sIdx + 1), gIdx : (gIdx + 1)
                    ]
                    scale_v = np.full((1, grp_size), scale_v)
                    # mul
                    value_cur = value_cur * scale_v
                    # copy to
                    value[
                        :,
                        nIdx : (nIdx + 1),
                        sIdx : (sIdx + 1),
                        gIdx * grp_size : (gIdx + 1) * grp_size,
                    ] = value_cur
    else:
        if lowprecision_flag:
            if q_dtype == "float16":
                scale = scale.astype(np.float16)
                offset = offset.astype(np.float16)
            elif q_dtype == "bfloat16":
                scale = scale.astype(tf.bfloat16.as_numpy_dtype)
                offset = offset.astype(tf.bfloat16.as_numpy_dtype)
            else:
                print(f"[ERROR]Invalid input q dtype: {q_dtype}")
                exit(1)
        else:
            scale = scale.astype(np.float32)
            offset = offset.astype(np.float32)
        if kv_dtype == "int8":
            if lowprecision_flag:
                if q_dtype == "float16":
                    key = k.astype(np.float16)
                    value = v.astype(np.float16)
                elif q_dtype == "bfloat16":
                    key = k.astype(tf.bfloat16.as_numpy_dtype)
                    value = v.astype(tf.bfloat16.as_numpy_dtype)
                else:
                    print(f"[ERROR]Invalid input q dtype: {q_dtype}")
                    exit(1)
            else:
                key = k.astype(np.float32)
                value = v.astype(np.float32)
        elif kv_dtype in ["int4", "int32"]:
            if lowprecision_flag:
                if q_dtype == "float16":
                    key = k.astype(np.float16)
                    value = v.astype(np.float16)
                elif q_dtype == "bfloat16":
                    key = k.astype(np.float16).astype(tf.bfloat16.as_numpy_dtype)
                    value = v.astype(np.float16).astype(tf.bfloat16.as_numpy_dtype)
                else:
                    print(f"[ERROR]Invalid input q dtype: {q_dtype}")
                    exit(1)
            else:
                scale = scale.astype(np.float16)
                offset = offset.astype(np.float16)
                key = k.astype(np.float16)
                value = v.astype(np.float16)
        elif kv_dtype == "hifloat8":
            # uint8->fp16，精度转换
            if lowprecision_flag:
                if q_dtype == "float16":
                    key = trans_np_hifuint8_tensor_to_float32(k).astype(np.float16)
                    value = trans_np_hifuint8_tensor_to_float32(v).astype(np.float16)
                elif q_dtype == "bfloat16":
                    key = trans_np_hifuint8_tensor_to_float32(k).astype(
                        tf.bfloat16.as_numpy_dtype
                    )
                    value = trans_np_hifuint8_tensor_to_float32(v).astype(
                        tf.bfloat16.as_numpy_dtype
                    )
                else:
                    print(f"[ERROR]Invalid input q dtype: {q_dtype}")
                    exit(1)
            else:
                key = trans_np_hifuint8_tensor_to_float32(k).astype(np.float16)
                value = trans_np_hifuint8_tensor_to_float32(v).astype(np.float16)
        elif kv_dtype in ["float8_e5m2", "float8_e4m3fn"]:
            if lowprecision_flag:
                if q_dtype == "float16":
                    key = k.astype(np.float16)
                    value = v.astype(np.float16)
                elif q_dtype == "bfloat16":
                    key = k.astype(np.float16).astype(tf.bfloat16.as_numpy_dtype)
                    value = v.astype(np.float16).astype(tf.bfloat16.as_numpy_dtype)
                else:
                    print(f"[ERROR]Invalid input q dtype: {q_dtype}")
                    exit(1)
            else:
                key = k.astype(np.float16)
                value = v.astype(np.float16)
        else:
            print(f"[ERROR]Invalid input kv dtype: {kv_dtype}")
            exit(1)

        if pto_flag:
            scale1 = scale[0:1, :, :]
            scale2 = scale[1:2, :, :]
            offset1 = offset[0:1, :, :]
            offset2 = offset[1:2, :, :]
            for nIdx in range(N):
                key_cur = key[:, nIdx : (nIdx + 1), :, :]
                scale1_cur = np.expand_dims(scale1, axis=-1)
                offset1_cur = np.expand_dims(offset1, axis=-1)
                key_cur = np.add(key_cur, offset1_cur)
                key_cur = np.multiply(key_cur, scale1_cur)
                key[:, nIdx : (nIdx + 1), :, :] = key_cur

                value_cur = value[:, nIdx : (nIdx + 1), :, :]
                scale2_cur = np.expand_dims(scale2, axis=-1)
                offset2_cur = np.expand_dims(offset2, axis=-1)
                value_cur = np.add(value_cur, offset2_cur)
                value_cur = np.multiply(value_cur, scale2_cur)
                value[:, nIdx : (nIdx + 1), :, :] = value_cur
        else:
            # 2,n,1,d -> 1,n,1,d
            scale1 = scale[0:1, :, :, :]
            scale2 = scale[1:2, :, :, :]
            offset1 = offset[0:1, :, :, :]
            offset2 = offset[1:2, :, :, :]
            for nIdx in range(N):
                for sIdx in range(S):
                    key_cur = key[:, nIdx : (nIdx + 1), sIdx : (sIdx + 1), :]
                    key_cur = key_cur + offset1[:, nIdx : (nIdx + 1), :, :]
                    key_cur = key_cur * scale1[:, nIdx : (nIdx + 1), :, :]
                    key[:, nIdx : (nIdx + 1), sIdx : (sIdx + 1), :] = key_cur

                    value_cur = value[:, nIdx : (nIdx + 1), sIdx : (sIdx + 1), :]
                    value_cur = value_cur + offset2[:, nIdx : (nIdx + 1), :, :]
                    value_cur = value_cur * scale2[:, nIdx : (nIdx + 1), :, :]
                    value[:, nIdx : (nIdx + 1), sIdx : (sIdx + 1), :] = value_cur
    return key, value


def kv_split_antiquant_tensor(
    k,
    v,
    k_scale,
    k_offset,
    v_scale,
    v_offset,
    k_anti_mode,
    v_anti_mode,
    kv_dtype="int8",
    q_dtype="float16",
):
    lowprecision_flag = False if q_dtype == "float32" else True
    if kv_dtype in ["fp4_e1m2", "fp4_e2m1", "float4_e1m2", "float4_e2m1"]:
        raise NotImplementedError

    else:
        if lowprecision_flag:
            if q_dtype == "bfloat16":
                k_scale = k_scale.type(torch.bfloat16)
                k_offset = k_offset.type(torch.bfloat16)
                v_scale = v_scale.type(torch.bfloat16)
                v_offset = v_offset.type(torch.bfloat16)
            elif q_dtype == "float16":
                k_scale = k_scale.type(torch.float16)
                k_offset = k_offset.type(torch.float16)
                v_scale = v_scale.type(torch.float16)
                v_offset = v_offset.type(torch.float16)
            else:
                print(f"[ERROR]Invalid input q dtype: {q_dtype}")
                exit(1)
        else:
            k_scale = k_scale.type(torch.float32)
            k_offset = k_offset.type(torch.float32)
            v_scale = v_scale.type(torch.float32)
            v_offset = v_offset.type(torch.float32)
        if kv_dtype == "int8":
            if lowprecision_flag:
                if q_dtype == "bfloat16":
                    key = k.type(torch.bfloat16)
                    value = v.type(torch.bfloat16)
                elif q_dtype == "float16":
                    key = k.type(torch.float16)
                    value = v.type(torch.float16)
                else:
                    print(f"[ERROR]Invalid input q dtype: {q_dtype}")
                    exit(1)
            else:
                key = k.type(torch.float32)
                value = v.type(torch.float32)
        else:
            raise NotImplementedError

    key = (key + k_offset) * k_scale
    value = (value + v_offset) * v_scale
    if q_dtype == "float16":
        key = key.type(torch.float16)
        value = value.type(torch.float16)
    elif q_dtype == "bfloat16":
        key = key.type(torch.bfloat16)
        value = value.type(torch.bfloat16)
    elif q_dtype == "float32":
        key = key.type(torch.float32)
        value = value.type(torch.float32)
    else:
        raise NotImplementedError
    return key, value


def kv_split_antiquant(
    k,
    v,
    k_scale,
    k_offset,
    v_scale,
    v_offset,
    k_anti_mode,
    v_anti_mode,
    lowprecision_flag,
    kv_dtype="int8",
    q_dtype="float16",
):
    N = k.shape[1]
    S = k.shape[2]
    if kv_dtype in ["fp4_e1m2", "fp4_e2m1", "float4_e1m2", "float4_e2m1"]:
        k_scale_bf16 = trans_tensor_fp8_e8m0_to_bf16(k_scale)
        v_scale_bf16 = trans_tensor_fp8_e8m0_to_bf16(v_scale)
        if lowprecision_flag:
            if q_dtype == "float16":
                k_scale = k_scale_bf16.astype(np.float16)
                v_scale = v_scale_bf16.astype(np.float16)
            elif q_dtype == "bfloat16":
                k_scale = k_scale_bf16
                v_scale = v_scale_bf16
            else:
                print(f"[ERROR]Invalid input q dtype: {q_dtype}")
                exit(1)
        else:
            k_scale = k_scale_bf16.astype(np.float16)
            v_scale = v_scale_bf16.astype(np.float16)

        if kv_dtype in ["fp4_e1m2", "float4_e1m2"]:
            if lowprecision_flag:
                if q_dtype == "float16":
                    key = new_trans_np_fp4_e1m2_tensor_to_bfloat16(k).astype(np.float16)
                    value = new_trans_np_fp4_e1m2_tensor_to_bfloat16(v).astype(
                        np.float16
                    )
                elif q_dtype == "bfloat16":
                    key = new_trans_np_fp4_e1m2_tensor_to_bfloat16(k)
                    value = new_trans_np_fp4_e1m2_tensor_to_bfloat16(v)
                else:
                    print(f"[ERROR]Invalid input q dtype: {q_dtype}")
                    exit(1)
            else:
                key = new_trans_np_fp4_e1m2_tensor_to_bfloat16(k).astype(np.float16)
                value = new_trans_np_fp4_e1m2_tensor_to_bfloat16(v).astype(np.float16)
        else:
            if lowprecision_flag:
                if q_dtype == "float16":
                    key = new_trans_np_fp4_e2m1_tensor_to_bfloat16(k).astype(np.float16)
                    value = new_trans_np_fp4_e2m1_tensor_to_bfloat16(v).astype(
                        np.float16
                    )
                elif q_dtype == "bfloat16":
                    key = new_trans_np_fp4_e2m1_tensor_to_bfloat16(k)
                    value = new_trans_np_fp4_e2m1_tensor_to_bfloat16(v)
                else:
                    print(f"[ERROR]Invalid input q dtype: {q_dtype}")
                    exit(1)
            else:
                key = new_trans_np_fp4_e2m1_tensor_to_bfloat16(k).astype(np.float16)
                value = new_trans_np_fp4_e2m1_tensor_to_bfloat16(v).astype(np.float16)

    else:
        if lowprecision_flag:
            if q_dtype == "float16":
                k_scale = np.array(k_scale, dtype=(np.float16))
                k_offset = np.array(k_offset, dtype=(np.float16))
                v_scale = np.array(v_scale, dtype=(np.float16))
                v_offset = np.array(v_offset, dtype=(np.float16))
            elif q_dtype == "bfloat16":
                k_scale = k_scale.astype(tf.bfloat16.as_numpy_dtype)
                k_offset = k_offset.astype(tf.bfloat16.as_numpy_dtype)
                v_scale = v_scale.astype(tf.bfloat16.as_numpy_dtype)
                v_offset = v_offset.astype(tf.bfloat16.as_numpy_dtype)
            else:
                print(f"[ERROR]Invalid input q dtype: {q_dtype}")
                exit(1)
        else:
            k_scale = np.array(k_scale, dtype=np.float32)
            k_offset = np.array(k_offset, dtype=np.float32)
            v_scale = np.array(v_scale, dtype=np.float32)
            v_offset = np.array(v_offset, dtype=np.float32)

        if kv_dtype == "int8":
            if lowprecision_flag:
                if q_dtype == "float16":
                    key = k.astype(np.float16)
                    value = v.astype(np.float16)
                elif q_dtype == "bfloat16":
                    key = k.astype(tf.bfloat16.as_numpy_dtype)
                    value = v.astype(tf.bfloat16.as_numpy_dtype)
                else:
                    print(f"[ERROR]Invalid input q dtype: {q_dtype}")
                    exit(1)
            else:
                key = k.astype(np.float32)
                value = v.astype(np.float32)
        elif kv_dtype in ["int4", "int32"]:
            if lowprecision_flag:
                if q_dtype == "float16":
                    key = k.astype(np.float16)
                    value = v.astype(np.float16)
                elif q_dtype == "bfloat16":
                    key = k.astype(np.float16).astype(tf.bfloat16.as_numpy_dtype)
                    value = v.astype(np.float16).astype(tf.bfloat16.as_numpy_dtype)
                else:
                    print(f"[ERROR]Invalid input q dtype: {q_dtype}")
                    exit(1)
            else:
                k_scale = k_scale.astype(np.float16)
                k_offset = k_offset.astype(np.float16)
                v_scale = v_scale.astype(np.float16)
                v_offset = v_offset.astype(np.float16)
                key = k.astype(np.float16)
                value = v.astype(np.float16)
        elif kv_dtype == "hifloat8":
            # uint8->fp16，精度转换
            if lowprecision_flag:
                if q_dtype == "float16":
                    key = trans_np_hifuint8_tensor_to_float32(k).astype(np.float16)
                    value = trans_np_hifuint8_tensor_to_float32(v).astype(np.float16)
                elif q_dtype == "bfloat16":
                    key = trans_np_hifuint8_tensor_to_float32(k).astype(
                        tf.bfloat16.as_numpy_dtype
                    )
                    value = trans_np_hifuint8_tensor_to_float32(v).astype(
                        tf.bfloat16.as_numpy_dtype
                    )
                else:
                    print(f"[ERROR]Invalid input q dtype: {q_dtype}")
                    exit(1)
            else:
                key = trans_np_hifuint8_tensor_to_float32(k).astype(np.float16)
                value = trans_np_hifuint8_tensor_to_float32(v).astype(np.float16)
        elif kv_dtype in ["float8_e5m2", "float8_e4m3fn"]:
            if lowprecision_flag:
                if q_dtype == "float16":
                    key = k.astype(np.float16)
                    value = v.astype(np.float16)
                elif q_dtype == "bfloat16":
                    key = k.astype(np.float16).astype(tf.bfloat16.as_numpy_dtype)
                    value = v.astype(np.float16).astype(tf.bfloat16.as_numpy_dtype)
                else:
                    print(f"[ERROR]Invalid input q dtype: {q_dtype}")
                    exit(1)
            else:
                key = k.astype(np.float16)
                value = v.astype(np.float16)
        else:
            print(f"[ERROR]Invalid input kv dtype: {kv_dtype}")
            exit(1)

    if k_anti_mode in ["1", "4"]:
        scale1 = k_scale[0:1, :, :]  # 11S
        offset1 = k_offset[0:1, :, :]
        for nIdx in range(N):
            key_cur = key[:, nIdx : (nIdx + 1), :, :]
            # scale: 11s -> 11s1
            scale1_cur = np.expand_dims(scale1, axis=-1)
            offset1_cur = np.expand_dims(offset1, axis=-1)
            key_cur = np.add(key_cur, offset1_cur)
            key_cur = np.multiply(key_cur, scale1_cur)
            key[:, nIdx : (nIdx + 1), :, :] = key_cur
    elif k_anti_mode in ["0", "2"]:
        # 2,n,1,d -> 1,n,1,d
        scale1 = k_scale[0:1, :, :, :]
        offset1 = k_offset[0:1, :, :, :]
        for nIdx in range(N):
            for sIdx in range(S):
                key_cur = key[:, nIdx : (nIdx + 1), sIdx : (sIdx + 1), :]
                key_cur = key_cur + offset1[:, nIdx : (nIdx + 1), :, :]
                key_cur = key_cur * scale1[:, nIdx : (nIdx + 1), :, :]
                key[:, nIdx : (nIdx + 1), sIdx : (sIdx + 1), :] = key_cur
    elif k_anti_mode in ["3", "5"]:
        # 计算, scale为1NS, key为1nsd
        scale1 = k_scale[0:1, :, :]
        offset1 = k_offset[0:1, :, :]
        for nIdx in range(N):
            # key_cur：11sd
            key_cur = key[:, nIdx : (nIdx + 1), :, :]
            # scale/offset：11s
            scale1_cur = scale1[:, nIdx : (nIdx + 1), :]
            offset1_cur = offset1[:, nIdx : (nIdx + 1), :]
            # scale/offset：11s1
            scale1_cur = np.expand_dims(scale1_cur, axis=-1)
            offset1_cur = np.expand_dims(offset1_cur, axis=-1)
            # (11sd + 11s1) * 11s1
            key_cur = np.add(key_cur, offset1_cur)
            key_cur = np.multiply(key_cur, scale1_cur)
            key[:, nIdx : (nIdx + 1), :, :] = key_cur
    elif k_anti_mode == "6":
        g = k_scale.shape[4]
        k_scale = k_scale[0]
        grp_size = 32
        for nIdx in range(N):
            for sIdx in range(S):
                for gIdx in range(g):
                    # copy from
                    key_cur = key[
                        :,
                        nIdx : (nIdx + 1),
                        sIdx : (sIdx + 1),
                        gIdx * grp_size : (gIdx + 1) * grp_size,
                    ]
                    # broardcast
                    scale_k = k_scale[
                        :, nIdx : (nIdx + 1), sIdx : (sIdx + 1), gIdx : (gIdx + 1)
                    ]
                    scale_k = np.full((1, grp_size), scale_k)
                    # mul
                    key_cur = key_cur * scale_k
                    # copy to
                    key[
                        :,
                        nIdx : (nIdx + 1),
                        sIdx : (sIdx + 1),
                        gIdx * grp_size : (gIdx + 1) * grp_size,
                    ] = key_cur
    if v_anti_mode in ["1", "4"]:
        scale2 = v_scale[0:1, :, :]
        offset2 = v_offset[0:1, :, :]
        for nIdx in range(N):
            value_cur = value[:, nIdx : (nIdx + 1), :, :]
            scale2_cur = np.expand_dims(scale2, axis=-1)
            offset2_cur = np.expand_dims(offset2, axis=-1)
            value_cur = np.add(value_cur, offset2_cur)
            value_cur = np.multiply(value_cur, scale2_cur)
            value[:, nIdx : (nIdx + 1), :, :] = value_cur
    elif v_anti_mode in ["0", "2"]:
        # 2,n,1,d -> 1,n,1,d
        scale2 = v_scale[0:1, :, :, :]
        offset2 = v_offset[0:1, :, :, :]
        for nIdx in range(N):
            for sIdx in range(S):
                value_cur = value[:, nIdx : (nIdx + 1), sIdx : (sIdx + 1), :]
                value_cur = value_cur + offset2[:, nIdx : (nIdx + 1), :, :]
                value_cur = value_cur * scale2[:, nIdx : (nIdx + 1), :, :]
                value[:, nIdx : (nIdx + 1), sIdx : (sIdx + 1), :] = value_cur
    elif v_anti_mode in ["3", "5"]:
        # 计算, scale为1SN, value为bnsd
        scale1 = v_scale[0:1, :, :]
        offset1 = v_offset[0:1, :, :]
        for nIdx in range(N):
            # value_cur：11sd
            value_cur = value[:, nIdx : (nIdx + 1), :, :]
            # scale/offset：11s
            scale1_cur = scale1[:, nIdx : (nIdx + 1), :]
            offset1_cur = offset1[:, nIdx : (nIdx + 1), :]
            # scale/offset：11s1
            scale1_cur = np.expand_dims(scale1_cur, axis=-1)
            offset1_cur = np.expand_dims(offset1_cur, axis=-1)
            # (11sd + 11s1) * 11s1
            value_cur = np.add(value_cur, offset1_cur)
            value_cur = np.multiply(value_cur, scale1_cur)
            value[:, nIdx : (nIdx + 1), :, :] = value_cur
    elif v_anti_mode == "6":
        g = v_scale.shape[4]
        v_scale = v_scale[0]
        grp_size = 32
        for nIdx in range(N):
            for sIdx in range(S):
                for gIdx in range(g):
                    # copy from
                    value_cur = value[
                        :,
                        nIdx : (nIdx + 1),
                        sIdx : (sIdx + 1),
                        gIdx * grp_size : (gIdx + 1) * grp_size,
                    ]
                    # broardcast
                    scale_v = v_scale[
                        :, nIdx : (nIdx + 1), sIdx : (sIdx + 1), gIdx : (gIdx + 1)
                    ]
                    scale_v = np.full((1, grp_size), scale_v)
                    # mul
                    value_cur = value_cur * scale_v
                    # copy to
                    value[
                        :,
                        nIdx : (nIdx + 1),
                        sIdx : (sIdx + 1),
                        gIdx * grp_size : (gIdx + 1) * grp_size,
                    ] = value_cur
    return key, value


def _trans_antiparam_to_1bs(tensor):
    if len(tensor.shape) == 2:
        tensor_new = tensor.reshape(1, tensor.shape[0], tensor.shape[1])
    else:
        tensor_new = tensor

    return tensor_new


def softmaxv1(x):
    x = x.astype(np.float32)
    x_max = x.max(axis=-1, keepdims=True)
    x_sub = x - x_max
    y = np.exp(x_sub)
    x_sum = y.sum(axis=-1, keepdims=True)
    ans = y / x_sum
    return ans, x_max, x_sum


def softmax(x):
    # this func is only used by quant_dequant
    x = x.astype(np.float32)
    x_max = x.max(axis=-1, keepdims=True)
    x_sub = x - x_max
    y = np.exp(x_sub)
    x_sum = y.sum(axis=-1, keepdims=True)
    ans = y
    # ans = ans.astype(np.float16)
    return ans, x_sum, x_max


def softmax_flash(x, max_front=None, sum_front=None, axis=None, update=False):
    if update == False:
        x = x.astype(np.float32)
        x_max = x.max(axis=-1, keepdims=True)
        x_sub = x - x_max
        y = np.exp(x_sub)
        x_sum = y.sum(axis=-1, keepdims=True)
        ans = y / x_sum
        ans = ans.astype(np.float16)
        x_max = x_max.astype(np.float16)
        x_sum = x_sum.astype(np.float16)
        exp_max = None
        return x_max, x_sum, ans, exp_max
    else:
        x = x.astype(np.float16)
        max_front = max_front.astype(np.float16)
        sum_front = sum_front.astype(np.float16)
        x_max = np.max(np.concatenate((max_front, x), axis=-1), axis=-1, keepdims=True)
        x_exp = np.exp(x - x_max)
        x_sum = np.sum(x_exp, axis=-1, keepdims=True)
        exp_max = np.exp(max_front - x_max)
        x_sum = exp_max * sum_front + x_sum
        exp_max = exp_max * sum_front / x_sum
        out = x_exp / x_sum
        out = out.astype(np.float16)
        exp_max = exp_max.astype(np.float16)
        return x_max, x_sum, out, exp_max


def sum_ceil(lst, a):
    res = 0
    for num in lst:
        res += math.ceil(num / a)
    return res


def get_h(shape, layout):
    if layout == "BNSD":
        h = shape[1] * shape[3]
    elif layout == "BSND":
        h = shape[2] * shape[3]
    elif layout == "BSH":
        h = shape[2]
    elif layout == "TND":
        h = shape[1] * shape[2]
    else:
        h = 0
    return h


def get_sinner(ifa_param):
    from tbe.common.utils.op_tiling import do_op_tiling
    from tbe.common.platform import set_current_compile_soc_info

    set_current_compile_soc_info("Ascend910B1")
    print("=============>ASCEND_OPP_PATH:", os.environ.get("ASCEND_OPP_PATH"))
    q_shape = ifa_param["q_shape"]
    q_dtype = trans_input_dtype(ifa_param["q_dtype"])
    m_shape = ifa_param["m_shape"]
    m_dtype = ifa_param["m_dtype"]
    p_shape = ifa_param["p_shape"]
    p_dtype = ifa_param["p_dtype"]
    k_shape = ifa_param["k_shape_list"][0]
    v_shape = ifa_param["v_shape_list"][0]
    k_shape[0] = v_shape[0] = q_shape[0]
    out_shape = ifa_param["out_shape"]
    out_dtype = ifa_param["out_dtype"]

    op_type = "FusedInferAttentionScore"
    compile_info = {}
    # qkv
    inputs = [
        {
            "shape": q_shape,
            "ori_shape": q_shape,
            "format": "ND",
            "ori_format": "ND",
            "dtype": q_dtype,
            "range": [(0, None)],
            "param_name": "query",
        },
        {
            "shape": k_shape,
            "ori_shape": k_shape,
            "format": "ND",
            "ori_format": "ND",
            "dtype": "int8",
            "range": [(0, None)],
            "param_name": "key",
        },
        {
            "shape": v_shape,
            "ori_shape": v_shape,
            "format": "ND",
            "ori_format": "ND",
            "dtype": "int8",
            "range": [(0, None)],
            "param_name": "value",
        },
    ]
    # pse_shift
    if ifa_param["p_flag"]:
        inputs.extend(
            [
                {
                    "shape": p_shape,
                    "ori_shape": p_shape,
                    "format": "ND",
                    "ori_format": "ND",
                    "dtype": p_dtype,
                    "range": [(0, None)],
                    "param_name": "pse_shift",
                }
            ]
        )
    else:
        inputs.extend([None])
    # atten_mask
    if ifa_param["m_flag"]:
        inputs.extend(
            [
                {
                    "shape": m_shape,
                    "ori_shape": m_shape,
                    "format": "ND",
                    "ori_format": "ND",
                    "dtype": m_dtype,
                    "range": [(0, None)],
                    "param_name": "atten_mask",
                }
            ]
        )
    else:
        inputs.extend([None])
    # actual_seq_lengths
    inputs.extend([None])
    # actual_seq_lengths_kv
    if ifa_param["as_flag"]:
        act_seq = ifa_param["actualSeqLengths_raw"]
        inputs.extend(
            [
                {
                    "shape": [len(act_seq)],
                    "ori_shape": [len(act_seq)],
                    "format": "ND",
                    "ori_format": "ND",
                    "range": [(0, None)],
                    "const_value": act_seq,
                    "dtype": "int64",
                    "param_name": "actual_seq_lengths_kv",
                }
            ]
        )
    else:
        inputs.extend([None])
    # dequant_scale1 quant_scale1 dequant_scale2
    if ifa_param["in_quant_flag"]:
        inputs.extend(
            [
                {
                    "shape": [1],
                    "ori_shape": [1],
                    "format": "ND",
                    "ori_format": "ND",
                    "dtype": "uint64",
                    "range": [(0, None)],
                    "param_name": "dequant_scale_1",
                },
                {
                    "shape": [1],
                    "ori_shape": [1],
                    "format": "ND",
                    "ori_format": "ND",
                    "dtype": "float32",
                    "range": [(0, None)],
                    "param_name": "quant_scale_1",
                },
                {
                    "shape": [1],
                    "ori_shape": [1],
                    "format": "ND",
                    "ori_format": "ND",
                    "dtype": "uint64",
                    "range": [(0, None)],
                    "param_name": "dequant_scale_2",
                },
            ]
        )
    else:
        inputs.extend([None, None, None])
    # quant_scale2 quant_offset2
    if ifa_param["out_quant_flag"]:
        quantScale2_shape = ifa_param["quantScale2_shape"]
        quantOffset2_shape = ifa_param["quantOffset2_shape"]
        quantScale2_dtype = ifa_param["quantScale2_dtype"]
        quantOffset2_dtype = ifa_param["quantOffset2_dtype"]
        inputs.extend(
            [
                {
                    "shape": quantScale2_shape,
                    "ori_shape": quantScale2_shape,
                    "format": "ND",
                    "ori_format": "ND",
                    "dtype": quantScale2_dtype,
                    "range": [(0, None)],
                    "param_name": "quant_scale_2",
                }
            ]
        )
        if ifa_param["out_quant_offset_flag"]:
            inputs.extend(
                [
                    {
                        "shape": quantOffset2_shape,
                        "ori_shape": quantOffset2_shape,
                        "format": "ND",
                        "ori_format": "ND",
                        "dtype": quantOffset2_dtype,
                        "range": [(0, None)],
                        "param_name": "quant_offset_2",
                    }
                ]
            )
        else:
            inputs.extend([None])
    else:
        inputs.extend([None, None])
    # antiquant_scale antiquant_offset
    if ifa_param["kv_quant_flag"]:
        anti_quant_shape = ifa_param["antiquantscale_shape_raw"]
        anti_quant_dtype = ifa_param["antiquantscale_dtype"]
        inputs.extend(
            [
                {
                    "shape": anti_quant_shape,
                    "ori_shape": anti_quant_shape,
                    "format": "ND",
                    "ori_format": "ND",
                    "dtype": anti_quant_dtype,
                    "range": [(0, None)],
                    "param_name": "antiquant_scale",
                }
            ]
        )
        if ifa_param["kv_quant_offset_flag"]:
            inputs.extend(
                [
                    {
                        "shape": anti_quant_shape,
                        "ori_shape": anti_quant_shape,
                        "format": "ND",
                        "ori_format": "ND",
                        "dtype": anti_quant_dtype,
                        "range": [(0, None)],
                        "param_name": "antiquant_offset",
                    }
                ]
            )
        else:
            inputs.extend([None])
    else:
        inputs.extend([None, None])

    # block_table
    if ifa_param["pa_flag"]:
        bt_shape = ifa_param["bt_shape"]
        bt_dtype = ifa_param["bt_dtype"]
        inputs.extend(
            [
                {
                    "shape": bt_shape,
                    "ori_shape": bt_shape,
                    "format": "ND",
                    "ori_format": "ND",
                    "dtype": bt_dtype,
                    "range": [(0, None)],
                    "param_name": "block_table",
                }
            ]
        )
        bs = int(ifa_param["blocksize"])
        act_seq = ifa_param["actualSeqLengths"]
        kvcache_shape_1 = sum_ceil(act_seq, bs)
        kvcache_shape_2 = bs
        kvcache_shape_3 = get_h(k_shape, ifa_param["inputLayout"])
        inputs[1]["shape"] = inputs[1]["ori_shape"] = [
            kvcache_shape_1,
            kvcache_shape_2,
            kvcache_shape_3,
        ]
        inputs[2]["shape"] = inputs[2]["ori_shape"] = [
            kvcache_shape_1,
            kvcache_shape_2,
            kvcache_shape_3,
        ]
    else:
        inputs.extend([None])
    # query_padding_size
    inputs.extend([None])
    # kv_padding_size
    if ifa_param["padding_flag"]:
        inputs.extend(
            [
                {
                    "shape": [1],
                    "ori_shape": [1],
                    "format": "ND",
                    "ori_format": "ND",
                    "dtype": "int64",
                    "range": [(0, None)],
                    "param_name": "kv_padding_size",
                }
            ]
        )
    else:
        inputs.extend([None])

    outputs = [
        {
            "shape": out_shape,
            "ori_shape": out_shape,
            "format": "ND",
            "ori_format": "ND",
            "dtype": out_dtype,
            "range": [(0, None)],
            "param_name": "attention_out",
        }
    ]

    attrs = [
        {"name": "num_heads", "dtype": "int", "value": ifa_param["numHeads"]},
        {"name": "scale_value", "dtype": "float", "value": ifa_param["scaleValue"]},
        {"name": "pre_tokens", "dtype": "int", "value": ifa_param["pre_tokens"]},
        {"name": "next_tokens", "dtype": "int", "value": ifa_param["next_tokens"]},
        {"name": "input_layout", "dtype": "str", "value": ifa_param["inputLayout"]},
        {
            "name": "num_key_value_heads",
            "dtype": "int",
            "value": ifa_param["numKeyValueHeads"],
        },
        {"name": "sparse_mode", "dtype": "int", "value": ifa_param["sparse_mode"]},
        {"name": "inner_precise", "dtype": "int", "value": ifa_param["innerprecise"]},
        {"name": "block_size", "dtype": "int", "value": ifa_param["blocksize"]},
        {
            "name": "antiquant_mode",
            "dtype": "int",
            "value": int(ifa_param["antiquant_mode"]),
        },
        {"name": "softmax_lse_flag", "dtype": "bool", "value": False},
    ]

    print(f"inputs:{inputs}")
    print(f"outputs:{outputs}")
    print(f"attrs:{attrs}")
    ret = do_op_tiling(op_type, compile_info, inputs, outputs, attrs=attrs)
    tiling_key = ret.get("tiling_key")
    cache_tiling = ret.get("tiling_data")
    tiling_data_pr = np.frombuffer(cache_tiling, dtype=np.int32)
    idx_singleProcessSInnerSize = 254
    singleProcessSInnerSize = tiling_data_pr[idx_singleProcessSInnerSize]
    print(
        f"[INFO]get_sinner result --> tiling_key:{tiling_key}, singleProcessSInnerSize:{singleProcessSInnerSize}"
    )
    return singleProcessSInnerSize


def _t_broadcastKV_sigle(numHeads, numKeyValueHeads, kv_tensor, input_dtype):
    factor = numHeads // numKeyValueHeads
    kv_shape = kv_tensor.shape
    B = kv_shape[0]
    S = kv_shape[2]
    D = kv_shape[3]
    kv_res = np.zeros([B, numHeads, S, D], dtype=kv_tensor.dtype)
    for i in range(numHeads):
        j = i // factor
        kv_res[:, i : i + 1, :, :] = kv_tensor[:, j : j + 1, :, :]
    return kv_res, kv_res.shape


def trans_uint64_2_float32(input):
    output = np.uint64(input)
    output = np.frombuffer(output, dtype=np.float32)
    output = output[:1][0]
    print(f"uint64:{input}, float32:{output}")
    return output


def torch_to_numpy(tensor):
    n_tensor = None
    print(tensor.dtype)
    if type(tensor) == "<class 'torch.Tensor'>":
        if tensor.dtype in ["torch.float16", "torch.bool"]:
            n_tensor = tensor.numpy()
        elif tensor.dtype in ["torch.bfloat16"]:
            n_tensor = tensor.numpy()
    return n_tensor


def _n_trans_shape_to_bnsd(tensor, shape, layout, headnums=None, act_seq=None):
    if layout in ["BSH", "BSH_NBSD"]:
        if headnums is None:
            return tensor, shape
        B = shape[0]
        S = shape[1]
        H = shape[2]
        N = headnums
        D = H // N
        tensor = tensor.reshape(B, S, N, D).transpose(0, 2, 1, 3)
        return tensor, [B, N, S, D]
    elif layout in ["BSND", "BSND_NBSD"]:
        B = shape[0]
        S = shape[1]
        N = shape[2]
        D = shape[3]
        tensor = tensor.transpose(0, 2, 1, 3)
        return tensor, [B, N, S, D]
    elif layout in ["TND", "TND_NTD"]:
        T = shape[0]
        N = shape[1]
        D = shape[2]
        B = len(act_seq)
        S = max(act_seq)
        new_tensor = np.zeros((B, N, S, D), dtype=tensor.dtype)
        t_start = 0
        for b_index in range(B):
            act_s = act_seq[b_index]
            t_end = t_start + act_s
            if act_s == 0:
                continue
            for n_index in range(N):
                new_tensor[b_index, n_index, 0:act_s, :] = tensor[
                    t_start:t_end, n_index, :
                ]
            t_start += act_s
        return new_tensor, [B, N, S, D]
    else:
        return tensor, shape


def _t_bsh_to_bnsd(tensor, bsh_shape, headnums, batch=1):
    if len(bsh_shape) == 2:
        B = batch
        S = bsh_shape[0] // B
        H = bsh_shape[1]
        N = headnums
        D = H // N
        return tensor.reshape(B, S, N, D).transpose(0, 2, 1, 3), [B, N, S, D]
    elif len(bsh_shape) == 3:
        B = bsh_shape[0]
        S = bsh_shape[1]
        H = bsh_shape[2]
        N = headnums
        D = H // N
        return tensor.reshape(B, S, N, D).transpose(0, 2, 1, 3), [B, N, S, D]
    else:
        return tensor, bsh_shape


def trans_bnsd_to_layout(tensor, shape, layout, act_q=None):
    if layout == "BSH":
        shape[2] = tensor.shape[1] * tensor.shape[3]
        output = tensor.permute(0, 2, 1, 3).contiguous().view(shape)
        return output
    elif layout == "BSND":
        output = tensor.permute(0, 2, 1, 3).contiguous()
        return output
    elif layout in ["BSND_NBSD", "BNSD_NBSD", "BSH_NBSD"]:
        output = tensor.permute(1, 0, 2, 3).contiguous()
        return output
    elif layout in ["TND", "TND_NTD"]:
        T = sum(act_q)
        B = tensor.shape[0]
        N = tensor.shape[1]
        D = tensor.shape[3]
        output = torch.zeros(size=(T, N, D), dtype=tensor.dtype)
        t_start = 0
        for b_index in range(B):
            act_s = act_q[b_index]
            t_end = t_start + act_s
            if act_s == 0:
                continue
            for n_index in range(N):
                output[t_start:t_end, n_index, :] = tensor[b_index, n_index, :act_s, :]
            t_start += act_s
        if layout == "TND_NTD":
            output = output.permute(1, 0, 2).contiguous()
        return output
    else:
        return tensor


def trans_bnsd_to_bsh(tensor, shape):
    if len(shape) == 4:
        B = shape[0]
        N = shape[1]
        S = shape[2]
        D = shape[3]
        H = N * D
        return tensor.transpose(0, 2, 1, 3).reshape(B, S, H)
    if len(shape) == 3:
        B = shape[0]
        S = shape[1]
        H = shape[2]
        return tensor.transpose(0, 2, 1, 3).reshape(B, S, H)
    else:
        return tensor


def _t_broadcast_mask_n(m_tensor, m_shape, N, B):
    print(f"broadcast_mask_n:mask shape:{m_shape}")
    if len(m_shape) == 4:
        # b11s
        m_res = torch.zeros([B, N, m_shape[2], m_shape[3]])
        for i in range(B):
            for j in range(N):
                m_res[i : i + 1, j : j + 1, :, :] = m_tensor[i : i + 1, :, :, :]
        print(f"after broadcast_mask_n:mask shape:{m_res.shape}")
        return m_res
    elif len(m_shape) == 3:
        # b1s
        m_res = torch.zeros([B, N, m_shape[1], m_shape[2]])
        for i in range(B):
            for j in range(N):
                m_res[i : i + 1, j : j + 1, :, :] = m_tensor[i : i + 1, :, :]
        print(f"after broadcast_mask_n:mask shape:{m_res.shape}")
        return m_res
    elif len(m_shape) == 2:
        # bs
        m_res = torch.zeros([B, N, 1, m_shape[1]])
        for i in range(B):
            for j in range(N):
                m_res[i : i + 1, j : j + 1, 0, :] = m_tensor[i : i + 1, :]
        print(f"after broadcast_mask_n:mask shape:{m_res.shape}")
        return m_res
    else:
        print("[ERROR] m_tensor shape is invalid!")
        exit(1)


def _trans_antiparam_to_2bnsg(shape, tensor, layout, numKeyValueHeads):
    # bsh: (2, B, S, H/32) -> (2, B, N, S, D/32)
    if layout in ["BSH", "BSH_NBSD"]:
        B = shape[1]
        S = shape[2]
        G = shape[3]
        N = numKeyValueHeads
        G2 = int(G / N)
        print(
            f"[INFO]_trans_antiparam_to_2bnsg: layout={layout}, b={B}, s={S} ,n={numKeyValueHeads},g(h/32)={G}"
        )
        new_tensor = tensor.reshape(2, B, S, N, G2).transpose(0, 1, 3, 2, 4)
        return new_tensor
    # bsh: (2, B, S, N, D/32) -> (2, B, N, S, D/32)
    elif layout in ["BSND", "BSND_NBSD"]:
        print(f"[INFO]_trans_antiparam_to_2bnsg: layout={layout}")
        new_tensor = tensor.transpose(0, 1, 3, 2, 4)
        return new_tensor
    else:
        print(f"[INFO]_trans_antiparam_to_2bnsg: layout={layout}")
        return tensor


def _trans_antiparam_to_1bnsg(shape, tensor, layout, numKeyValueHeads):
    # bsh: (1, B, S, H/32) -> (1, B, N, S, D/32)
    if layout in ["BSH", "BSH_NBSD"]:
        B = shape[1]
        S = shape[2]
        G = shape[3]
        N = numKeyValueHeads
        G2 = int(G / N)
        print(
            f"[INFO]_trans_antiparam_to_1bnsg: layout={layout}, b={B}, s={S} ,n={numKeyValueHeads},g(h/32)={G}"
        )
        new_tensor = tensor.reshape(1, B, S, N, G2).transpose(0, 1, 3, 2, 4)
        return new_tensor
    # bsh: (1, B, S, N, D/32) -> (1, B, N, S, D/32)
    elif layout in ["BSND", "BSND_NBSD"]:
        print(f"[INFO]_trans_antiparam_to_1bnsg: layout={layout}")
        new_tensor = tensor.transpose(0, 1, 3, 2, 4)
        return new_tensor
    else:
        print(f"[INFO]_trans_antiparam_to_1bnsg: layout={layout}")
        return tensor


def broadcast_kv_dequant_tensor_fp8e8m0(tensor, numKeyValueHeads, numheads):
    B = tensor.shape[1]
    S = tensor.shape[3]
    G = tensor.shape[4]
    if numKeyValueHeads == numheads:
        tensor.astype(np.uint8)
        return tensor
    else:
        factor = numheads // numKeyValueHeads
        shape = tensor.shape
        D = shape[3]
        tensor_new = np.zeros([2, B, numheads, S, G], dtype=np.uint8)
        for i in range(numheads):
            j = i // factor
            tensor_new[:, :, i : i + 1, :, :] = tensor[:, :, j : j + 1, :, :]
        return tensor_new


def broadcast_kv_split_antiquant_tensor_fp8e8m0(tensor, numKeyValueHeads, numheads):
    B = tensor.shape[1]
    S = tensor.shape[3]
    G = tensor.shape[4]
    if numKeyValueHeads == numheads:
        tensor.astype(np.uint8)
        return tensor
    else:
        factor = numheads // numKeyValueHeads
        shape = tensor.shape
        D = shape[3]
        tensor_new = np.zeros([1, B, numheads, S, G], dtype=np.uint8)
        for i in range(numheads):
            j = i // factor
            tensor_new[:, :, i : i + 1, :, :] = tensor[:, :, j : j + 1, :, :]
        return tensor_new


def _t_softmax(x):
    x_max = torch.max(x, dim=-1, keepdims=True)[0]
    x_sub = x.sub(x_max)
    y = torch.exp(x_sub)
    x_sum = y.sum(dim=-1, keepdims=True)
    ans = y.div(x_sum)
    return ans, x_max, x_sum


def AS_fp32(int_vals):
    arr = np.array(int_vals, dtype=np.int32)
    uint32 = arr.view(np.uint32)
    s = (uint32 >> 31) & 0x1
    e = (uint32 >> 23) & 0xFF
    m = uint32 & 0x007FFFFF
    fp32_bits = (s << 31) | (e << 23) | m
    return fp32_bits.view(np.float32).tolist()


def _t_indequant_ds_act(ifa_param):
    batch_index = ifa_param["batch_id"]
    q_tensor = ifa_param["q_tensor_cur"]
    k_tensor = ifa_param["k_sub_tensor"]
    v_tensor = ifa_param["v_sub_tensor"]
    q_rope_tensor = ifa_param["q_rope_tensor_cur"]
    k_rope_tensor = ifa_param["k_rope_tensor_cur"]
    scalar = ifa_param["scaleValue"]
    act_seq = ifa_param["act_seq"]
    act_seq_q = ifa_param["act_seq_q"]
    lse_flag = ifa_param["softmax_lse_flag"]
    print(
        f"q_shape:{q_tensor.shape} k_shape:{k_tensor.shape} v_shape:{v_tensor.shape} scalar:{scalar} act:{act_seq} act_q:{act_seq_q}"
    )

    dequantScale1 = ifa_param["dequantScale1_cur"]
    dequantScale2 = ifa_param["dequantScale2"]
    print(f"dequantScale2: {dequantScale2.shape}")

    if lse_flag:
        lse = np.zeros((q_tensor.shape[0], q_tensor.shape[1], q_tensor.shape[2], 1))
    else:
        lse = None

    if ifa_param["m_flag"]:
        mask_tensor = ifa_param["mask_cur"].numpy()

    kv_smax = k_tensor.shape[2]
    if act_seq is not None and act_seq != -1:
        kv_smax = act_seq

    s2Inner = np.minimum(512, kv_smax)
    loop = np.ceil(kv_smax / s2Inner).astype(np.int32)
    tail = kv_smax - (loop - 1) * s2Inner
    print(f"s2Inner:{s2Inner} kv_smax:{kv_smax} loop:{loop} tail:{tail}")

    rowSum = np.zeros(
        shape=(q_tensor.shape[0], q_tensor.shape[1], q_tensor.shape[2], 1),
        dtype=np.float32,
    )
    rowMax = np.full(
        shape=(q_tensor.shape[0], q_tensor.shape[1], q_tensor.shape[2], 1),
        fill_value=-np.inf,
        dtype=np.float32,
    )
    O_flash = np.zeros(
        shape=(
            q_tensor.shape[0],
            q_tensor.shape[1],
            q_tensor.shape[2],
            q_tensor.shape[3],
        ),
        dtype=np.float16,
    )
    pse_cur = None
    mask_cur = None
    for i in range(loop):
        kv_start_pos = i * s2Inner
        kv_end_pos = (i + 1) * s2Inner
        if i == (loop - 1):
            kv_end_pos = i * s2Inner + tail
        k_cur = k_tensor[:, :, kv_start_pos:kv_end_pos, :]
        v_cur = v_tensor[:, :, kv_start_pos:kv_end_pos, :]
        k_rope_cur = k_rope_tensor[:, :, kv_start_pos:kv_end_pos, :]
        if ifa_param["p_flag"]:
            pse_cur = ifa_param["pse_cur"][:, :, :, kv_start_pos:kv_end_pos]
        if ifa_param["m_flag"]:
            mask_cur = mask_tensor[:, :, :, kv_start_pos:kv_end_pos]

        bias = 2**21 + 150 * 2**23
        qkNopeRes = np.matmul(
            q_tensor.astype(np.int32), k_cur.transpose(0, 1, 3, 2).astype(np.int32)
        )
        qkNopeRes = qkNopeRes + bias

        bias2 = np.zeros_like(qkNopeRes, dtype=np.float32) - 2**21 - 2**23
        qkNopeRes_fp32 = (AS_fp32(qkNopeRes) + bias2).astype(np.float32)

        qkRopeRes = np.matmul(
            q_rope_tensor.astype(np.float32),
            k_rope_cur.transpose(0, 1, 3, 2).astype(np.float32),
        )

        qkBmmRes = qkNopeRes_fp32 + qkRopeRes

        qkBmmRes = qkBmmRes * dequantScale1 * scalar

        if pse_cur is not None:
            qkBmmRes += pse_cur
        if mask_cur is not None:
            qkBmmRes[mask_cur.astype(np.bool_)] = -1000000000000

        rowmax_tmp = qkBmmRes.max(axis=3, keepdims=True)
        replace_idx = np.where(rowMax < rowmax_tmp)
        rowMax_old = copy.deepcopy(rowMax)
        rowMax[replace_idx] = rowmax_tmp[replace_idx]
        update_mul = np.exp(rowMax_old - rowMax)

        softmaxres = np.exp(qkBmmRes - rowMax).astype(np.float32)
        rowSum_A = softmaxres.sum(axis=3, keepdims=True).astype(np.float32)
        rowMax_A = (np.exp(rowmax_tmp - rowMax) + np.finfo(float).eps).astype(
            np.float32
        )
        rowSum = update_mul * rowSum + rowSum_A

        quantScale = 127 / rowMax_A

        softmaxres = (softmaxres * quantScale).astype(np.float32)

        if ifa_param["q_dtype"] != "float32":
            softmaxres = np.round(softmaxres.astype(np.float16)).astype(np.int8)
        else:
            softmaxres = np.round(softmaxres).astype(np.int8)

        bmm2Res = np.matmul(softmaxres.astype(np.int32), k_cur.astype(np.int32))
        bmm2Res = (bmm2Res / np.float32(1024)).astype(np.float16) * rowMax_A.astype(
            np.float16
        )

        O_flash = O_flash.astype(np.float16) * update_mul.astype(np.float16) + bmm2Res

    O = (
        O_flash.astype(np.float32)
        * np.float32(1024)
        / rowSum
        / np.float32(127)
        * np.float32(dequantScale2)
    )

    # 这里处理行无效,要使用整个kv_s的mask,即mask_tensor
    if ifa_param["m_flag"]:
        if mask_tensor is not None:
            if ifa_param["sparse_mode"] == 3 or act_seq_q > 1:
                for i in range(mask_tensor.shape[2]):
                    if mask_tensor[:, :, i, :].all() == 1:
                        print(f"mask i: {i} all is 1")
                        O[:, :, i, :] = 0
                        if lse is not None:
                            lse[:, :, i, :] = np.inf

    return O, lse


def _t_indequant_ds_act_golden(ifa_param):
    print("def _t_indequant_ds_act_golden(ifa_param): enter")

    q_tensor = ifa_param["q_tensor_cur"].astype(np.float64)
    k_tensor = ifa_param["k_sub_tensor"].astype(np.float64)
    v_tensor = ifa_param["v_sub_tensor"].astype(np.float64)
    q_rope_tensor = ifa_param["q_rope_tensor_cur"].astype(np.float64)
    k_rope_tensor = ifa_param["k_rope_tensor_cur"].astype(np.float64)
    scalar = np.array(ifa_param["scaleValue"]).astype(np.float64)
    act_seq = ifa_param["act_seq"]
    act_seq_q = ifa_param["act_seq_q"]
    lse_flag = ifa_param["softmax_lse_flag"]
    print(
        f"q_shape:{q_tensor.shape} k_shape:{k_tensor.shape} v_shape:{v_tensor.shape} scalar:{scalar} act:{act_seq} act_q:{act_seq_q}"
    )

    dequantScale1 = ifa_param["dequantScale1_cur"].astype(np.float64)  # [1, 128, 1, 1]
    dequantScale2 = ifa_param["dequantScale2"].astype(np.float64)  # float
    scaleQ = dequantScale1 / dequantScale2
    scaleK = dequantScale2

    q_tensor = q_tensor * scaleQ
    k_tensor = k_tensor * scaleK
    v_tensor = v_tensor * scaleK

    # q_rope_tensor = q_rope_tensor * scaleQ * scaleK

    q_tensor_all = np.concatenate([q_tensor, q_rope_tensor], axis=3)
    k_tensor_all = np.concatenate([k_tensor, k_rope_tensor], axis=3)

    # dequantScale1 = ScaleQ * ScaleK
    # dequantScale2 = ScaleK

    if lse_flag:
        lse = np.zeros((q_tensor.shape[0], q_tensor.shape[1], q_tensor.shape[2], 1))
    else:
        lse = None

    kv_smax = k_tensor.shape[2]
    if act_seq is not None and act_seq != -1:
        kv_smax = act_seq

    s2Inner = np.minimum(512, kv_smax)
    loop = np.ceil(kv_smax / s2Inner).astype(np.int32)
    tail = kv_smax - (loop - 1) * s2Inner
    print(f"s2Inner:{s2Inner} kv_smax:{kv_smax} loop:{loop} tail:{tail}")

    rowSum = np.zeros(
        shape=(q_tensor.shape[0], q_tensor.shape[1], q_tensor.shape[2], 1),
        dtype=np.float64,
    )
    rowMax = np.full(
        shape=(q_tensor.shape[0], q_tensor.shape[1], q_tensor.shape[2], 1),
        fill_value=-np.inf,
        dtype=np.float64,
    )
    O_flash = np.zeros(
        shape=(
            q_tensor.shape[0],
            q_tensor.shape[1],
            q_tensor.shape[2],
            q_tensor.shape[3],
        ),
        dtype=np.float64,
    )
    pse_cur = None
    mask_cur = None

    for i in range(loop):
        if i < loop - 1:
            seq_start = i * s2Inner
            seq_end = (i + 1) * s2Inner
        else:
            seq_start = i * s2Inner
            seq_end = i * s2Inner + tail
        k_cur = k_tensor_all[:, :, seq_start:seq_end, :]
        v_cur = v_tensor[:, :, seq_start:seq_end, :]
        if ifa_param["p_flag"]:
            pse_cur = ifa_param["pse_cur"][:, :, :, seq_start:seq_end]
        if ifa_param["m_flag"]:
            mask_tensor = ifa_param["mask_cur"].numpy()
            mask_cur = mask_tensor[:, :, :, seq_start:seq_end]

        qkBmmRes = np.matmul(q_tensor_all, k_cur.transpose(0, 1, 3, 2)).astype(
            np.float64
        )
        qkBmmRes = (qkBmmRes * scalar).astype(np.float64)

        if pse_cur is not None:
            qkBmmRes += pse_cur
        if mask_cur is not None:
            qkBmmRes[mask_cur.astype(np.bool_)] = -1000000000000

        rowmax_tmp = qkBmmRes.max(axis=3, keepdims=True)
        replace_idx = np.where(rowMax < rowmax_tmp)
        rowMax_old = copy.deepcopy(rowMax)
        rowMax[replace_idx] = rowmax_tmp[replace_idx]
        update_mul = np.exp(rowMax_old - rowMax)

        softmaxres = np.exp(qkBmmRes - rowMax).astype(np.float64)
        rowSum_A = softmaxres.sum(axis=3, keepdims=True).astype(np.float64)
        rowMax_A = (np.exp(rowmax_tmp - rowMax) + np.finfo(float).eps).astype(
            np.float64
        )
        rowSum = update_mul * rowSum + rowSum_A

        quantScale = 127 / rowMax_A

        softmaxres = (softmaxres * quantScale).astype(np.float64)
        softmaxres = np.round(softmaxres).astype(np.int8)

        bmm2Res = np.matmul(softmaxres.astype(np.float64), v_cur.astype(np.float64))
        bmm2Res = bmm2Res.astype(np.float64) * rowMax_A.astype(np.float64)

        O_flash = O_flash * update_mul + bmm2Res

    O = O_flash / rowSum / 127

    if ifa_param["m_flag"]:
        if mask_tensor is not None:
            if ifa_param["sparse_mode"] == 3 or act_seq_q > 1:
                for i in range(mask_tensor.shape[2]):
                    if mask_tensor[:, :, i, :].all() == 1:
                        O[:, :, i, :] = 0
                        if lse is not None:
                            lse[:, :, i, :] = np.inf

    return O, lse


def _t_ifaattention_act(ifa_param, device="cpu"):
    print("def _t_ifaattention_act(ifa_param):")
    # 低精度标记位，默认走高精度流程，lowprecision_flag = False
    lowprecision_flag = False

    # 如果跑one命令或者innerprecise=1且跑bm命令时，改为走低精度流程，lowprecision_flag = True
    if ifa_param["action_type"] == "one" or (
        ifa_param["action_type"] == "bm" and ifa_param["innerprecise"] == 1
    ):
        lowprecision_flag = True
        print("低精度模式...")
    q_tensor = ifa_param["q_tensor_cur"]
    k_tensor = ifa_param["k_sub_tensor"]
    v_tensor = ifa_param["v_sub_tensor"]
    m_tensor = ifa_param["mask_cur"]
    m_dtype = ifa_param["m_dtype"]
    p_tensor = ifa_param["pse_cur"]
    scalar = ifa_param["scaleValue"]
    act_seq = ifa_param["act_seq"]
    act_seq_q = ifa_param["act_seq_q"]
    sinner = ifa_param["sinner"]
    lse_flag = ifa_param["softmax_lse_flag"]
    if ifa_param["padding_flag"]:
        s_begin = int(k_tensor.shape[2] - act_seq - ifa_param["padding_size"])
        s_end = int(k_tensor.shape[2] - ifa_param["padding_size"])
        print(f"left padding--- s_begin:{s_begin}, s_end:{s_end}")

    print(
        f"q_shape:{q_tensor.shape} k_shape:{k_tensor.shape} v_shape:{v_tensor.shape} scalar:{scalar} act:{act_seq} act_q:{act_seq_q}"
    )
    # matmul_dtype 赋值
    matmul_dtype = np.float32
    if ifa_param["in_quant_flag"]:
        # 输入int8 场景，mm使用int32
        matmul_dtype = np.int32
    # elif ifa_param['out_quant_flag']:
    # 输出int8场景，mm使用float16
    #    matmul_dtype = np.float16
    else:
        # 非量化场景，mm使用float32
        matmul_dtype = np.float32

    kv_smax = k_tensor.shape[2]
    if act_seq is None or act_seq == -1:
        k_cur = k_tensor
        v_cur = v_tensor
    else:
        kv_smax = act_seq
        if ifa_param["padding_flag"]:
            k_cur = k_tensor[:, :, s_begin:s_end, :]
            v_cur = v_tensor[:, :, s_begin:s_end, :]
            if ifa_param["kv_quant_pto_flag"]:
                ifa_param["antiquantScale_cur"] = ifa_param["antiquantScale_cur"][
                    :, :, s_begin:s_end
                ]
                ifa_param["antiquantOffset_cur"] = ifa_param["antiquantOffset_cur"][
                    :, :, s_begin:s_end
                ]
            if ifa_param["k_antiquant_pto_flag"] or ifa_param["k_antiquant_ptopa_flag"]:
                ifa_param["k_antiquantScale_cur"] = ifa_param["k_antiquantScale_cur"][
                    :, :, s_begin:s_end
                ]
                ifa_param["k_antiquantOffset_cur"] = ifa_param["k_antiquantOffset_cur"][
                    :, :, s_begin:s_end
                ]
            if ifa_param["v_antiquant_pto_flag"] or ifa_param["v_antiquant_ptopa_flag"]:
                ifa_param["v_antiquantScale_cur"] = ifa_param["v_antiquantScale_cur"][
                    :, :, s_begin:s_end
                ]
                ifa_param["v_antiquantOffset_cur"] = ifa_param["v_antiquantOffset_cur"][
                    :, :, s_begin:s_end
                ]

            if (
                ifa_param["k_antiquant_ptoh_flag"]
                or ifa_param["k_antiquant_ptohpa_flag"]
            ):
                ifa_param["k_antiquantScale_cur"] = ifa_param["k_antiquantScale_cur"][
                    :, :, s_begin:s_end
                ]
                ifa_param["k_antiquantOffset_cur"] = ifa_param["k_antiquantOffset_cur"][
                    :, :, s_begin:s_end
                ]
            if (
                ifa_param["v_antiquant_ptoh_flag"]
                or ifa_param["v_antiquant_ptohpa_flag"]
            ):
                ifa_param["v_antiquantScale_cur"] = ifa_param["v_antiquantScale_cur"][
                    :, :, s_begin:s_end
                ]
                ifa_param["v_antiquantOffset_cur"] = ifa_param["v_antiquantOffset_cur"][
                    :, :, s_begin:s_end
                ]

            if ifa_param["kv_quant_ptog_flag"]:
                ifa_param["antiquantScale_cur"] = ifa_param["antiquantScale_cur"][
                    :, :, :, s_begin:s_end, :
                ]
                ifa_param["antiquantOffset_cur"] = ifa_param["antiquantOffset_cur"][
                    :, :, :, s_begin:s_end, :
                ]
            if ifa_param["k_antiquant_ptog_flag"]:
                ifa_param["k_antiquantScale_cur"] = ifa_param["k_antiquantScale_cur"][
                    :, :, :, s_begin:s_end, :
                ]
                ifa_param["k_antiquantOffset_cur"] = ifa_param["k_antiquantOffset_cur"][
                    :, :, :, s_begin:s_end, :
                ]
            if ifa_param["v_antiquant_ptog_flag"]:
                ifa_param["v_antiquantScale_cur"] = ifa_param["v_antiquantScale_cur"][
                    :, :, :, s_begin:s_end, :
                ]
                ifa_param["v_antiquantOffset_cur"] = ifa_param["v_antiquantOffset_cur"][
                    :, :, :, s_begin:s_end, :
                ]
        else:
            k_cur = k_tensor[:, :, :act_seq, :]
            v_cur = v_tensor[:, :, :act_seq, :]
            if ifa_param["kv_quant_pto_flag"]:
                ifa_param["antiquantScale_cur"] = ifa_param["antiquantScale_cur"][
                    :, :, :act_seq
                ]
                ifa_param["antiquantOffset_cur"] = ifa_param["antiquantOffset_cur"][
                    :, :, :act_seq
                ]
            if ifa_param["k_antiquant_pto_flag"] or ifa_param["k_antiquant_ptopa_flag"]:
                ifa_param["k_antiquantScale_cur"] = ifa_param["k_antiquantScale_cur"][
                    :, :, :act_seq
                ]
                ifa_param["k_antiquantOffset_cur"] = ifa_param["k_antiquantOffset_cur"][
                    :, :, :act_seq
                ]
            if ifa_param["v_antiquant_pto_flag"] or ifa_param["v_antiquant_ptopa_flag"]:
                ifa_param["v_antiquantScale_cur"] = ifa_param["v_antiquantScale_cur"][
                    :, :, :act_seq
                ]
                ifa_param["v_antiquantOffset_cur"] = ifa_param["v_antiquantOffset_cur"][
                    :, :, :act_seq
                ]

            if (
                ifa_param["k_antiquant_ptoh_flag"]
                or ifa_param["k_antiquant_ptohpa_flag"]
            ):
                ifa_param["k_antiquantScale_cur"] = ifa_param["k_antiquantScale_cur"][
                    :, :, :act_seq
                ]
                ifa_param["k_antiquantOffset_cur"] = ifa_param["k_antiquantOffset_cur"][
                    :, :, :act_seq
                ]
            if (
                ifa_param["v_antiquant_ptoh_flag"]
                or ifa_param["v_antiquant_ptohpa_flag"]
            ):
                ifa_param["v_antiquantScale_cur"] = ifa_param["v_antiquantScale_cur"][
                    :, :, :act_seq
                ]
                ifa_param["v_antiquantOffset_cur"] = ifa_param["v_antiquantOffset_cur"][
                    :, :, :act_seq
                ]

            if ifa_param["kv_quant_ptog_flag"]:
                ifa_param["antiquantScale_cur"] = ifa_param["antiquantScale_cur"][
                    :, :, :, :act_seq, :
                ]
                ifa_param["antiquantOffset_cur"] = ifa_param["antiquantOffset_cur"][
                    :, :, :, :act_seq, :
                ]
            if ifa_param["k_antiquant_ptog_flag"]:
                ifa_param["k_antiquantScale_cur"] = ifa_param["k_antiquantScale_cur"][
                    :, :, :, :act_seq, :
                ]
                ifa_param["k_antiquantOffset_cur"] = ifa_param["k_antiquantOffset_cur"][
                    :, :, :, :act_seq, :
                ]
            if ifa_param["v_antiquant_ptog_flag"]:
                ifa_param["v_antiquantScale_cur"] = ifa_param["v_antiquantScale_cur"][
                    :, :, :, :act_seq, :
                ]
                ifa_param["v_antiquantOffset_cur"] = ifa_param["v_antiquantOffset_cur"][
                    :, :, :, :act_seq, :
                ]

    if ifa_param["in_quant_flag"] and ifa_param["deep_seek_flag"]:
        if ifa_param["action_type"] == "bm_output_gold":
            # 走golden流程，生成高精度cpu golden数据。
            print("bmm2Res, lse = _t_indequant_ds_act_golden(ifa_param)")
            bmm2Res, lse = _t_indequant_ds_act_golden(ifa_param)
        else:
            # 走DaTo流程，生成cpu仿npu计算的golden数据。
            print("bmm2Res, lse = _t_indequant_ds_act(ifa_param)")
            bmm2Res, lse = _t_indequant_ds_act(ifa_param)
    else:
        if ifa_param["k_antiquant_flag"] and ifa_param["v_antiquant_flag"]:
            k_cur, v_cur = kv_split_antiquant(
                k_cur,
                v_cur,
                ifa_param["k_antiquantScale_cur"],
                ifa_param["k_antiquantOffset_cur"],
                ifa_param["v_antiquantScale_cur"],
                ifa_param["v_antiquantOffset_cur"],
                ifa_param["k_antiquant_mode"],
                ifa_param["v_antiquant_mode"],
                lowprecision_flag,
                ifa_param["k_dtype"],
                ifa_param["q_dtype"],
            )

        elif ifa_param["kv_quant_flag"]:
            k_cur, v_cur = antiquant(
                k_cur,
                v_cur,
                ifa_param["antiquantScale_cur"],
                ifa_param["antiquantOffset_cur"],
                ifa_param["kv_quant_pto_flag"],
                lowprecision_flag,
                ifa_param["k_dtype"],
                ifa_param["q_dtype"],
            )

        qkBmmRes = np.matmul(
            q_tensor, k_cur.astype(np.float32).transpose(0, 1, 3, 2), dtype=matmul_dtype
        )
        if ifa_param["in_quant_flag"] and not ifa_param["deep_seek_flag"]:
            qkBmmRes = dequant(qkBmmRes, ifa_param["dequantScale1"], None)
        qkEleRes = qkBmmRes * scalar

        if ifa_param["p_flag"]:
            if act_seq is None or act_seq == -1:
                pse_cur = ifa_param["pse_cur"]
            else:
                if ifa_param["padding_flag"]:
                    pse_cur = ifa_param["pse_cur"][:, :, :, s_begin:s_end]
                else:
                    pse_cur = ifa_param["pse_cur"][:, :, :, :act_seq]
            qkEleRes = qkEleRes + pse_cur
        mask_cur = None
        if ifa_param["m_flag"]:
            mask_tensor = ifa_param["mask_cur"].numpy()
            if act_seq is None:
                mask_cur = mask_tensor
            else:
                if ifa_param["padding_flag"]:
                    mask_cur = mask_tensor[:, :, :, s_begin:s_end]
                else:
                    mask_cur = mask_tensor[:, :, :, :act_seq]

            if m_dtype == "float16":
                qkEleRes = qkEleRes + mask_cur * -10000
            else:
                qkEleRes[mask_cur.astype(np.bool_)] = -1000000000000
        if ifa_param["in_quant_flag"] and not ifa_param["deep_seek_flag"]:
            bmm2Res = np.zeros(q_tensor.shape, dtype=np.float16)  # B, S, N, D
            s_loop_times = int((kv_smax + sinner - 1) / sinner)
            if s_loop_times > 1:
                N = q_tensor.shape[1]
                for nIdx in range(N):
                    for sinner_loop_times in range(s_loop_times):
                        sinner_tail = sinner
                        # 最后一次loop
                        if sinner_loop_times == (kv_smax - 1) // sinner:
                            if kv_smax == sinner:
                                sinner_tail = sinner
                            else:
                                sinner_tail = kv_smax % sinner
                        qk_cur_loop = qkEleRes[
                            :,
                            nIdx : (nIdx + 1),
                            :,
                            (sinner_loop_times * sinner) : (
                                (sinner_loop_times) * sinner + sinner_tail
                            ),
                        ]  # 1, 1, 1, Sinner
                        v_cur_loop = v_cur[
                            :,
                            nIdx : (nIdx + 1),
                            (sinner_loop_times * sinner) : (
                                (sinner_loop_times) * sinner + sinner_tail
                            ),
                            :,
                        ]  # 1, 1, Sinner, 1
                        # 处理第一次loop
                        if sinner_loop_times == 0:
                            softmax_max, softmax_sum, softmax_res, exp_max_res = (
                                softmax_flash(qk_cur_loop)
                            )
                            softmax_res = quant(
                                softmax_res, ifa_param["quantScale1"], 0
                            )
                            bmm2Res_1 = np.matmul(
                                softmax_res, v_cur_loop, dtype=matmul_dtype
                            )
                            bmm2Res_1 = dequant(
                                bmm2Res_1, ifa_param["dequantScale2"], None
                            )
                            max_front = softmax_max
                            sum_front = softmax_sum
                            o_front = bmm2Res_1
                        # 处理后续场景
                        else:
                            softmax_max, softmax_sum, softmax_res, exp_max_res = (
                                softmax_flash(
                                    qk_cur_loop,
                                    max_front=max_front,
                                    sum_front=sum_front,
                                    update=True,
                                )
                            )
                            softmax_res = quant(
                                softmax_res, ifa_param["quantScale1"], 0
                            )
                            o_front_update = o_front.astype(
                                np.float32
                            ) * exp_max_res.astype(np.float32)  # Mul
                            o_front_update = o_front_update.astype(np.float16)
                            bmm2Res_1 = np.matmul(
                                softmax_res, v_cur_loop, dtype=matmul_dtype
                            )
                            bmm2Res_1 = dequant(
                                bmm2Res_1, ifa_param["dequantScale2"], None
                            )
                            bmm2Res_1 = o_front_update.astype(
                                np.float32
                            ) + bmm2Res_1.astype(np.float32)
                            max_front = softmax_max
                            sum_front = softmax_sum
                            o_front = bmm2Res_1.astype(np.float16)
                    # 按n拼接
                    bmm2Res[:, nIdx : (nIdx + 1), :, :] = o_front
            # 量化非切分场景
            else:
                print("[INFO]输入量化+无sinner切分！")
                softmax_res = softmax(qkEleRes)
                softmax_res = quant(softmax_res, ifa_param["quantScale1"], 0)
                bmm2Res = np.matmul(softmax_res, v_cur, dtype=matmul_dtype)
                bmm2Res = dequant(bmm2Res, ifa_param["dequantScale2"], None)
        # 非输入量化场景
        else:
            if kv_smax <= 512 and ifa_param["softmaxV1_flag"] == True:
                softmax_res, softmax_sum, softmax_max = softmaxv1(qkEleRes)
                if ifa_param["q_dtype"] == "float16":
                    bmm2Res = np.matmul(
                        softmax_res.astype(np.float16).astype(np.float32),
                        v_cur,
                        dtype=matmul_dtype,
                    )
                elif ifa_param["q_dtype"] == "bfloat16":
                    bmm2Res = np.matmul(
                        softmax_res.astype(tf.bfloat16.as_numpy_dtype).astype(
                            np.float32
                        ),
                        v_cur,
                        dtype=matmul_dtype,
                    )
                else:
                    bmm2Res = np.matmul(softmax_res, v_cur, dtype=matmul_dtype)
            else:
                softmax_res, softmax_sum, softmax_max = softmax(qkEleRes)
                if ifa_param["q_dtype"] == "float16":
                    bmm2Res = (
                        np.matmul(
                            softmax_res.astype(np.float16).astype(np.float32),
                            v_cur,
                            dtype=matmul_dtype,
                        )
                        / softmax_sum
                    )
                elif ifa_param["q_dtype"] == "bfloat16":
                    bmm2Res = (
                        np.matmul(
                            softmax_res.astype(tf.bfloat16.as_numpy_dtype).astype(
                                np.float32
                            ),
                            v_cur,
                            dtype=matmul_dtype,
                        )
                        / softmax_sum
                    )
                else:
                    bmm2Res = (
                        np.matmul(softmax_res, v_cur, dtype=matmul_dtype) / softmax_sum
                    )
            if lse_flag:
                lse = np.log(softmax_sum) + softmax_max
            else:
                lse = None
            if mask_cur is not None:
                if ifa_param["sparse_mode"] == 3 or act_seq_q > 1:
                    for i in range(mask_cur.shape[2]):
                        if mask_cur[:, :, i, :].all() == 1:
                            bmm2Res[:, :, i, :] = 0
                for i in range(mask_cur.shape[2]):
                    if mask_cur[:, :, i, :].all() == 1:
                        if lse is not None:
                            lse[:, :, i, :] = np.inf
    return bmm2Res, lse


def str_to_bool_list(s: str):
    bool_list = []
    for item in s:
        if item == 0:
            bool_list.append(False)
        else:
            bool_list.append(True)
    return bool_list


def trans_input_dtype(input_dtype):
    if input_dtype == "fp16":
        return "float16"
    elif input_dtype == "int8":
        return "int8"
    elif input_dtype == "uint8":
        return "uint8"
    elif input_dtype == "bf16":
        return "bfloat16"
    elif input_dtype == "bool":
        return "bool"
    elif input_dtype == "int32":
        return "int32"
    elif input_dtype == "fp32":
        return "float32"
    elif input_dtype == "int4":
        return "int4"
    else:
        return input_dtype


def split_tensor_shape_by_b(input_list):
    # [[3,N,S,D]]-->[[1,N,S,D],[1,N,S,D],[1,N,S,D]]
    list_len = input_list[0][0]
    list_new = []
    for i in range(0, list_len):
        list_new_item = []
        list_new_item.append(1)
        list_new_item.append(input_list[0][1])
        list_new_item.append(input_list[0][2])
        list_new_item.append(input_list[0][3])
        list_new.append(list_new_item)
    return list_new


def split_tensor_by_b(input_list):
    # tensor:[[3,N,S,D]]-->[[1,N,S,D],[1,N,S,D],[1,N,S,D]]
    input_tensor = input_list[0]
    split_data = np.split(input_tensor, input_tensor.shape[0])
    return split_data


def _trans_antiparam_to_2n1d(shape, tensor, layout, numKeyValueHeads, d):
    # pertensor
    if shape == [2]:
        new_tensor = np.zeros((2, numKeyValueHeads, 1, d))
        new_tensor[0, :, :, :] = tensor[0]
        new_tensor[1, :, :, :] = tensor[1]
    # per channel
    elif len(shape) == 2:
        d_num = int(shape[1] / numKeyValueHeads)
        print(
            f"[INFO]_trans_2h_to_2n1d: layout={layout}, h={shape[1]}, n={numKeyValueHeads} ,d(h/n)={d_num}"
        )
        new_tensor = tensor.reshape(2, 1, numKeyValueHeads, d_num).transpose(0, 2, 1, 3)
    elif len(shape) == 3:
        print(f"[INFO]_trans_2nd_to_2n1d : layout={layout}, n={shape[1]} ,d={shape[2]}")
        new_tensor = tensor.reshape(2, 1, shape[1], shape[2]).transpose(0, 2, 1, 3)
    elif len(shape) == 4:
        if shape[1] == 1:
            print(
                f"[INFO]_trans_21nd_to_2n1d : layout={layout}, n={shape[2]} ,d={shape[3]}"
            )
            new_tensor = tensor.transpose(0, 2, 1, 3)
        else:
            new_tensor = tensor
    else:
        print(f"[ERROR]_trans_antiparam_to_2n1d : len(shape)({len(shape)}) > 4")
        new_tensor = tensor
    return new_tensor


def _trans_antiparam_to_1n1d(shape, tensor, layout, numKeyValueHeads, d, mode):
    if mode == "0":
        # per tensor
        if shape == [1]:
            new_tensor = np.zeros((1, numKeyValueHeads, 1, d))
            new_tensor[0, :, :, :] = tensor[0]
        # per channel
        elif len(shape) == 1:
            h = shape[0]
            d_num = int(h / numKeyValueHeads)
            print(
                f"[INFO]_trans_h_to_1n1d: layout={layout}, h={h}, n={numKeyValueHeads} ,d(h/n)={d_num}"
            )
            new_tensor = tensor.reshape(1, 1, numKeyValueHeads, d_num).transpose(1, 2)
        elif len(shape) == 2:
            if shape[0] == 1:
                h = shape[1]
                d_num = int(h / numKeyValueHeads)
                print(
                    f"[INFO]_trans_1h_to_1n1d: layout={layout}, h={h}, n={numKeyValueHeads} ,d(h/n)={d_num}"
                )
                new_tensor = tensor.reshape(1, 1, numKeyValueHeads, d_num).transpose(
                    1, 2
                )
            else:
                print(
                    f"[INFO]_trans_nd_to_1n1d : layout={layout}, n={shape[0]} ,d={shape[1]}"
                )
                new_tensor = tensor.reshape(1, 1, shape[0], shape[1]).transpose(1, 2)
        elif len(shape) == 3:
            if shape[0] == 1:
                new_tensor = tensor.reshape(1, 1, shape[1], shape[2]).transpose(1, 2)
            else:
                print(
                    f"[INFO]_trans_n1d_to_1n1d : layout={layout}, n={shape[0]} ,d={shape[2]}"
                )
                new_tensor = tensor.reshape(1, 1, shape[0], shape[2]).transpose(1, 2)
        elif len(shape) == 4:
            if shape[1] == 1:
                print(
                    f"[INFO]_trans_11nd_to_1n1d : layout={layout}, n={shape[2]} ,d={shape[3]}"
                )
                new_tensor = tensor.transpose(1, 2)
            else:
                new_tensor = tensor
        else:
            print(f"[ERROR]_trans_antiparam_to_1n1d : len(shape)({len(shape)}) > 4")
            new_tensor = tensor
        return new_tensor
    elif mode == "2":
        # pertensor-head
        print(f"shape is n:{numKeyValueHeads}")
        new_tensor = np.zeros((1, numKeyValueHeads, 1, d))
        for i in range(numKeyValueHeads):
            new_tensor[:, i, :, :] = tensor[i]
        return new_tensor
    else:
        print("[INFO]_trans_antiparam_to_1n1d : pass！！")
        return tensor


def broadcast_kv_dequant_tensor(tensor, numKeyValueHeads, numheads, dtype=np.float16):
    if numKeyValueHeads == numheads:
        return tensor
    else:
        factor = numheads // numKeyValueHeads
        shape = tensor.shape
        D = shape[3]
        if "bfloat16" in str(dtype):
            tensor_new = np.zeros([2, numheads, 1, D], dtype=tf.bfloat16.as_numpy_dtype)
        else:
            tensor_new = np.zeros([2, numheads, 1, D], dtype=dtype)
        for i in range(numheads):
            j = i // factor
            tensor_new[:, i : i + 1, :, :] = tensor[:, j : j + 1, :, :]
        return tensor_new


def broadcast_kv_split_dequant_tensor(
    tensor, numKeyValueHeads, numheads, dtype=np.float16
):
    if numKeyValueHeads == numheads:
        return tensor
    else:
        factor = numheads // numKeyValueHeads
        shape = tensor.shape
        D = shape[3]
        if "bfloat16" in str(dtype):
            tensor_new = np.zeros([1, numheads, 1, D], dtype=tf.bfloat16.as_numpy_dtype)
        else:
            tensor_new = np.zeros([1, numheads, 1, D], dtype=dtype)
        for i in range(numheads):
            j = i // factor
            tensor_new[:, i : i + 1, :, :] = tensor[:, j : j + 1, :, :]
        return tensor_new


# trans bns(kvn) -> bns(qn)
def broadcast_kv_split_dequant_tensor_ptoh(
    tensor, numKeyValueHeads, numheads, dtype=np.float32
):
    if numKeyValueHeads == numheads:
        return tensor
    else:
        factor = numheads // numKeyValueHeads
        shape = tensor.shape
        B = shape[0]
        G = shape[1]
        S = shape[2]
        tensor_new = np.zeros([B, numheads, S], dtype=dtype)
        for i in range(numheads):
            j = i // factor
            tensor_new[:, i : i + 1, :] = tensor[:, j : j + 1, :]
        return tensor_new


def _t_increattention_bnsd(ifa_param):
    batch_value = ifa_param["q_shape_bnsd"][0]
    n = ifa_param["q_shape_bnsd"][1]
    qs = ifa_param["q_shape_bnsd"][2]
    k_tensor_list = ifa_param["k_tensor_list_bnsd"]
    v_tensor_list = ifa_param["v_tensor_list_bnsd"]
    k_shape_list = ifa_param["k_shape_list_bnsd"]
    v_shape_list = ifa_param["v_shape_list_bnsd"]
    actseqlens_q = ifa_param["actualSeqLengths_q"]
    actseqlens = ifa_param["actualSeqLengths"]
    actseqlens_size = len(actseqlens)
    print(
        f"B:{batch_value} k tensor number:{len(k_tensor_list)} v tensor number:{len(v_tensor_list)}  actseqlens_q:{actseqlens_q} actseqlens:{actseqlens}"
    )
    # deepseek场景下，output的d轴需要复原
    out_shape = ifa_param["q_shape_bnsd"].copy()
    out_shape[3] = v_shape_list[0][3]
    if ifa_param["deep_seek_flag"]:
        if ifa_param["inputLayout"] in ["BNSD", "BSND", "BSND_NBSD", "BNSD_NBSD"]:
            out_shape[3] = ifa_param["out_shape"][3]
        elif ifa_param["inputLayout"] in ["TND", "TND_NTD"]:
            out_shape[3] = ifa_param["out_shape"][2]
        else:
            out_shape[3] = int(ifa_param["out_shape"][2] / ifa_param["numHeads"])
    y = np.zeros(out_shape, dtype=np.float32)
    lse = np.full([batch_value, n, qs, 1], np.inf)
    # 连续场景预处理
    if ifa_param["continue_flag"]:
        # 处理连续场景：将单个tensor shape依据B值拆成列表
        k_shape_list = split_tensor_shape_by_b(k_shape_list)
        v_shape_list = split_tensor_shape_by_b(v_shape_list)
        # 处理连续场景：将单个tensor依据B值拆成列表
        k_tensor_list = split_tensor_by_b(k_tensor_list)
        v_tensor_list = split_tensor_by_b(v_tensor_list)

    for b_index in range(batch_value):
        prefix_len = 0
        ifa_param["batch_id"] = b_index
        if ifa_param["prefix_flag"]:
            prefix_len = ifa_param["prefix_act_lens"]
        # 处理实际参与计算的kv s
        if ifa_param["as_flag"]:
            ifa_param["act_seq"] = act_seq = actseqlens[b_index] + prefix_len
        else:
            ifa_param["act_seq"] = act_seq = k_shape_list[b_index][2] + prefix_len

        if ifa_param["as_q_flag"]:
            ifa_param["act_seq_q"] = act_seq_q = actseqlens_q[b_index]
        else:
            ifa_param["act_seq_q"] = act_seq_q = qs

        ifa_param["k_sub_shape"] = k_sub_shape = k_shape_list[b_index]
        ifa_param["v_sub_shape"] = v_sub_shape = v_shape_list[b_index]
        if act_seq == 0 or act_seq_q == 0 or 0 in k_sub_shape or 0 in v_sub_shape:
            continue
        ifa_param["k_sub_tensor"] = k_tensor_list[b_index]
        ifa_param["v_sub_tensor"] = v_tensor_list[b_index]
        ifa_param["q_tensor_cur"] = ifa_param["q_tensor_bnsd"][
            b_index : (b_index + 1), :, :act_seq_q, :
        ]
        if ifa_param["in_quant_flag"]:
            ifa_param["dequantScale1_cur"] = ifa_param["dequantScale1"][
                b_index : (b_index + 1), :, :act_seq_q, :
            ]
            ifa_param["q_rope_tensor_cur"] = ifa_param["q_rope_bnsd_tensor"][
                b_index : (b_index + 1), :, :act_seq_q, :
            ]
            ifa_param["k_rope_tensor_cur"] = ifa_param["k_rope_bnsd_tensor"][
                b_index : (b_index + 1), :, :, :
            ]

        # prefix拼接
        if ifa_param["prefix_flag"]:
            k_sub_tensor_temp = np.concatenate(
                (ifa_param["k_prefix_tensor"], ifa_param["k_sub_tensor"]), axis=2
            )
            v_sub_tensor_temp = np.concatenate(
                (ifa_param["v_prefix_tensor"], ifa_param["v_sub_tensor"]), axis=2
            )
            ifa_param["k_sub_tensor"] = k_sub_tensor_temp
            ifa_param["v_sub_tensor"] = v_sub_tensor_temp
            print(ifa_param["k_sub_tensor"].shape)

        # 判断attenmask是否为空
        if not ifa_param["m_flag"]:
            ifa_param["mask_cur"] = None
        else:
            ifa_param["mask_cur"] = ifa_param["m_tensor"][
                b_index : (b_index + 1), :, :act_seq_q, :
            ]
        # 判断pse是否为空,如果非空,检查pse第一维是否为1：如果格式为1n1s,则直接传入下层计算;如果格式为bn1s,则按B拆分后进入下层。
        if not ifa_param["p_flag"]:
            pse_cur = None
        elif ifa_param["p_tensor"].shape[0] == 1:
            pse_cur = ifa_param["p_tensor"]
        else:
            pse_cur = ifa_param["p_tensor"][b_index : (b_index + 1), :, :, :]
        ifa_param["pse_cur"] = pse_cur
        # 伪量化Per token处理
        if ifa_param["kv_quant_pto_flag"]:
            ifa_param["antiquantScale_cur"] = ifa_param["antiquantScale"][
                :, b_index : (b_index + 1), :
            ]
            ifa_param["antiquantOffset_cur"] = ifa_param["antiquantOffset"][
                :, b_index : (b_index + 1), :
            ]
        # ptog伪量化参数切分
        elif ifa_param["kv_quant_ptog_flag"]:
            ifa_param["antiquantScale_cur"] = ifa_param["antiquantScale"][
                :, b_index : (b_index + 1), :, :, :
            ]
            ifa_param["antiquantOffset_cur"] = ifa_param["antiquantOffset"][
                :, b_index : (b_index + 1), :, :, :
            ]
        elif ifa_param["kv_quant_flag"]:
            ifa_param["antiquantScale_cur"] = ifa_param["antiquantScale"]
            ifa_param["antiquantOffset_cur"] = ifa_param["antiquantOffset"]
        else:
            pass
        # kv分离伪量化处理
        if ifa_param["k_antiquant_pto_flag"] or ifa_param["k_antiquant_ptopa_flag"]:
            ifa_param["k_antiquantScale_cur"] = ifa_param["k_antiquantScale"][
                :, b_index : (b_index + 1), :
            ]

            ifa_param["k_antiquantOffset_cur"] = ifa_param["k_antiquantOffset"][
                :, b_index : (b_index + 1), :
            ]

        elif ifa_param["k_antiquant_ptoh_flag"] or ifa_param["k_antiquant_ptohpa_flag"]:
            ifa_param["k_antiquantScale_cur"] = ifa_param["k_antiquantScale"][
                b_index : (b_index + 1), :, :
            ]
            ifa_param["k_antiquantOffset_cur"] = ifa_param["k_antiquantOffset"][
                b_index : (b_index + 1), :, :
            ]
        elif ifa_param["k_antiquant_ptog_flag"]:
            ifa_param["k_antiquantScale_cur"] = ifa_param["k_antiquantScale"][
                :, b_index : (b_index + 1), :, :, :
            ]
            ifa_param["k_antiquantOffset_cur"] = ifa_param["k_antiquantOffset"][
                :, b_index : (b_index + 1), :, :, :
            ]
        elif ifa_param["k_antiquant_flag"]:
            ifa_param["k_antiquantScale_cur"] = ifa_param["k_antiquantScale"]
            ifa_param["k_antiquantOffset_cur"] = ifa_param["k_antiquantOffset"]
        else:
            pass
        if ifa_param["v_antiquant_pto_flag"] or ifa_param["v_antiquant_ptopa_flag"]:
            ifa_param["v_antiquantScale_cur"] = ifa_param["v_antiquantScale"][
                :, b_index : (b_index + 1), :
            ]
            ifa_param["v_antiquantOffset_cur"] = ifa_param["v_antiquantOffset"][
                :, b_index : (b_index + 1), :
            ]
        elif ifa_param["v_antiquant_ptoh_flag"] or ifa_param["v_antiquant_ptohpa_flag"]:
            ifa_param["v_antiquantScale_cur"] = ifa_param["v_antiquantScale"][
                b_index : (b_index + 1), :, :
            ]
            ifa_param["v_antiquantOffset_cur"] = ifa_param["v_antiquantOffset"][
                b_index : (b_index + 1), :, :
            ]
        elif ifa_param["v_antiquant_ptog_flag"]:
            ifa_param["v_antiquantScale_cur"] = ifa_param["v_antiquantScale"][
                :, b_index : (b_index + 1), :, :, :
            ]
            ifa_param["v_antiquantOffset_cur"] = ifa_param["v_antiquantOffset"][
                :, b_index : (b_index + 1), :, :, :
            ]
        elif ifa_param["v_antiquant_flag"]:
            ifa_param["v_antiquantScale_cur"] = ifa_param["v_antiquantScale"]
            ifa_param["v_antiquantOffset_cur"] = ifa_param["v_antiquantOffset"]
        else:
            pass

        if ifa_param["as_q_flag"]:
            (
                y[b_index : (b_index + 1), :, :act_seq_q, :],
                lse[b_index : (b_index + 1), :, :act_seq_q, :],
            ) = _t_ifaattention_act(ifa_param)
        else:
            (
                y[b_index : (b_index + 1), :, :act_seq_q, :],
                lse[b_index : (b_index + 1), :, :, :],
            ) = _t_ifaattention_act(ifa_param)

    if ifa_param["out_quant_flag"]:
        if ifa_param["out_quant_pc_flag"]:
            y = quant_pc(y, ifa_param["quantScale2"], ifa_param["quantOffset2"])
        else:
            y = quant(y, ifa_param["quantScale2"][0], ifa_param["quantOffset2"][0])
    return y, lse


def get_slopes(n_heads):
    n = 2 ** math.floor(math.log2(n_heads))
    m_0 = 2.0 ** (-8.0 / n)
    m = torch.pow(m_0, torch.arange(1, 1 + n))
    if n < n_heads:
        m_hat_0 = 2.0 ** (-4.0 / n)
        m_hat = torch.pow(m_hat_0, torch.arange(1, 1 + 2 * (n_heads - n), 2))
        m = torch.cat([m, m_hat])
    return m


def is_0_tensor(tensor_list):
    for i in tensor_list:
        if 0 in i:
            return True
    return False


def trans_to_1n1d(tensor, shape, d):
    if len(shape) == 4:
        # 1n1d->1n1d
        return tensor
    elif len(shape) == 2:
        # nd->1n1d
        n = shape[0]
        d = shape[1]
        new_tensor = tensor.reshape(1, n, 1, d)
        return new_tensor
    elif len(shape) == 1:
        # h->1n1d
        h = shape[0]
        if h % d != 0:
            print(f"[ERROR]trans_to_1n1d: h={h}, d={d}. Wrong input !")
            return None
        n = int(h / d)
        new_tensor = tensor.reshape(1, n, 1, d)
        return new_tensor
    else:
        print("[ERROR]trans_to_1n1d: Unknown input shape!")
        return None


def get_kvs_list(shape_list, layout):
    kvs_list = []
    for shape in shape_list:
        if layout == "BNSD":
            kvs = shape[2]
        else:
            kvs = shape[1]
        kvs_list.append(kvs)
    return kvs_list


def cut_padding_size(tensor, list, padding_size):
    shape = tensor.shape
    ms = shape[-1]
    for i in range(len(list)):
        cut_len = int(ms - padding_size - list[i])
        tensor[i : (i + 1), ..., :cut_len] = 1
    return tensor


# 融合表格参数获取函数
def get_param_fus(torch_tensor_list, params):
    ifa_param = {}
    ifa_param["normal_flag"] = True
    # ===参数获取===
    ifa_param["flag_list"] = str_to_bool_list(params["flaglist"])

    # >> attr info
    ifa_param["actualSeqLengths_q_raw"] = actualSeqLengths_q = params[
        "actualseqlengths"
    ]
    ifa_param["actualSeqLengths_raw"] = actualSeqLengths = params["actualseqlengthskv"]
    ifa_param["numHeads"] = numHeads = params["numheads"]
    ifa_param["scaleValue"] = scaleValue = params["scalevalue"]
    ifa_param["inputLayout"] = inputLayout = params["inputlayout"]

    # >> q info
    ifa_param["q_shape"] = q_shape = params["shape_input"][0]
    ifa_param["q_dtype"] = q_dtype = trans_input_dtype(params["dtype_input"][0])
    ifa_param["q_tensor"] = torch_tensor_list[0]

    if inputLayout in ["TND", "TND_NTD"]:
        ifa_param["b"] = len(actualSeqLengths_q)
    else:
        ifa_param["b"] = ifa_param["q_shape"][0]

    # >> kv info
    # k和v的位置通过b计算
    k1_shape = params["shape_input"][1]
    kb1 = k1_shape[0]
    # 如果第一个K的B=1,则默认kv列表长度为b;如果第一个K的B!=1,则默认kv列表长度为1
    if kb1 == 1:
        k_shape_num = v_shape_num = ifa_param["b"]
    else:
        k_shape_num = v_shape_num = 1

    ifa_param["k_num"] = k_shape_num
    ifa_param["k_start_index"] = k_start_index = 1
    ifa_param["k_end_index"] = k_end_index = int(k_shape_num)
    ifa_param["v_start_index"] = v_start_index = int(k_shape_num) + 1
    ifa_param["v_end_index"] = v_end_index = int(k_shape_num + v_shape_num)

    print(
        f"k_start_index:{k_start_index} k_end_index:{k_end_index} v_start_index:{v_start_index} v_end_index:{v_end_index}"
    )

    ifa_param["k_shape_list"] = k_ori_shape_list = params["shape_input"][
        k_start_index : k_end_index + 1
    ]
    ifa_param["v_shape_list"] = v_ori_shape_list = params["shape_input"][
        v_start_index : v_end_index + 1
    ]
    ifa_param["k_dtype"] = k_dtype = trans_input_dtype(params["dtype_input"][1])
    ifa_param["v_dtype"] = v_dtype = trans_input_dtype(
        params["dtype_input"][v_start_index]
    )
    ifa_param["k_tensor_list"] = torch_tensor_list[k_start_index : k_end_index + 1]
    ifa_param["v_tensor_list"] = torch_tensor_list[v_start_index : v_end_index + 1]

    # >> out info
    ifa_param["out_shape"] = q_shape
    ifa_param["out_dtype"] = trans_input_dtype(params["dtype_output"][0])

    # >> p info
    ifa_param["p_shape"] = p_shape = params["shape_input"][v_end_index + 1]
    ifa_param["p_dtype"] = trans_input_dtype(params["dtype_input"][v_end_index + 1])
    ifa_param["p_tensor"] = p_tensor = torch_tensor_list[v_end_index + 1]

    # >> m info
    ifa_param["m_shape"] = m_shape = params["shape_input"][v_end_index + 2]
    ifa_param["m_dtype"] = trans_input_dtype(params["dtype_input"][v_end_index + 2])
    ifa_param["m_tensor"] = m_tensor = torch_tensor_list[v_end_index + 2]

    # kv prefix
    ifa_param["k_prefix_shape"] = k_prefix_shape = params["shape_input"][
        v_end_index + 17
    ]
    ifa_param["k_prefix_dtype"] = k_prefix_dtype = trans_input_dtype(
        params["dtype_input"][v_end_index + 17]
    )
    ifa_param["k_prefix_tensor"] = k_prefix_tensor = torch_tensor_list[v_end_index + 17]

    ifa_param["v_prefix_shape"] = v_prefix_shape = params["shape_input"][
        v_end_index + 18
    ]
    ifa_param["v_prefix_dtype"] = v_prefix_dtype = trans_input_dtype(
        params["dtype_input"][v_end_index + 18]
    )

    ifa_param["v_prefix_tensor"] = v_prefix_tensor = torch_tensor_list[v_end_index + 18]

    # >> q rope info
    ifa_param["q_rope_shape"] = params["shape_input"][v_end_index + 21]
    ifa_param["q_rope_dtype"] = trans_input_dtype(
        params["dtype_input"][v_end_index + 21]
    )
    ifa_param["q_rope_tensor"] = torch_tensor_list[v_end_index + 21]

    # >> k rope info
    ifa_param["k_rope_shape"] = params["shape_input"][v_end_index + 22]
    ifa_param["k_rope_dtype"] = trans_input_dtype(
        params["dtype_input"][v_end_index + 22]
    )
    ifa_param["k_rope_tensor"] = torch_tensor_list[v_end_index + 22]
    ifa_param["k_rope_cache_shape"] = params["shape_input"][v_end_index + 23]

    # 当numKeyValueHeads传入0时，处理为与numHeads相等
    numKeyValueHeads = (
        params["numkeyvalueheads"] if params["numkeyvalueheads"] != 0 else numHeads
    )
    ifa_param["numKeyValueHeads"] = numKeyValueHeads
    ifa_param["blocksize"] = params["blocksize"]
    ifa_param["innerprecise"] = params["innerprecise"]
    ifa_param["antiquant_mode"] = str(params["antiquant_mode"])
    # kv antiquant分离
    ifa_param["k_antiquant_mode"] = str(params["k_antiquant_mode"])
    ifa_param["v_antiquant_mode"] = str(params["v_antiquant_mode"])

    ifa_param["pre_tokens"] = params["pretokens"]
    ifa_param["next_tokens"] = params["nexttokens"]
    ifa_param["sparse_mode"] = params["sparsemode"]
    ifa_param["softmax_lse_flag"] = params["softmax_lse_flag"]

    # >> quant info
    ifa_param["dequantScale1"] = params["range_input"][v_end_index + 3][0]
    ifa_param["quantScale1"] = params["range_input"][v_end_index + 4][0]
    ifa_param["dequantScale2"] = params["range_input"][v_end_index + 5][0]

    # V4
    ifa_param["query_quant_mode"] = str(params["query_quant_mode"])
    ifa_param["dequant_scale_query"] = params["range_input"][v_end_index + 24][0]

    ifa_param["quantScale2_shape"] = params["shape_input"][v_end_index + 6]
    ifa_param["quantScale2"] = torch_tensor_list[v_end_index + 6]

    ifa_param["quantOffset2_shape"] = params["shape_input"][v_end_index + 7]
    ifa_param["quantOffset2"] = torch_tensor_list[v_end_index + 7]

    # >> kv_s list
    ifa_param["kv_s_list"] = get_kvs_list(
        ifa_param["k_shape_list"], ifa_param["inputLayout"]
    )

    # ===flag_list判断===
    ifa_param["p_flag"] = False
    ifa_param["m_flag"] = False
    ifa_param["as_flag"] = False
    ifa_param["as_q_flag"] = False
    ifa_param["pa_flag"] = False
    ifa_param["in_quant_flag"] = ifa_param["out_quant_flag"] = ifa_param[
        "kv_quant_flag"
    ] = False
    ifa_param["out_quant_offset_flag"] = ifa_param["kv_quant_offset_flag"] = ifa_param[
        "kv_quant_pto_flag"
    ] = False
    ifa_param["k_antiquant_flag"] = ifa_param["v_antiquant_flag"] = False
    ifa_param["k_antiquant_offset_flag"] = ifa_param["v_antiquant_offset_flag"] = False
    ifa_param["k_antiquant_pc_flag"] = ifa_param["v_antiquant_pc_flag"] = False
    ifa_param["k_antiquant_pto_flag"] = ifa_param["v_antiquant_pto_flag"] = False
    ifa_param["k_antiquant_pth_flag"] = ifa_param["v_antiquant_pth_flag"] = False
    ifa_param["k_antiquant_ptoh_flag"] = ifa_param["v_antiquant_ptoh_flag"] = False
    ifa_param["k_antiquant_ptopa_flag"] = ifa_param["v_antiquant_ptopa_flag"] = False
    ifa_param["k_antiquant_ptohpa_flag"] = ifa_param["v_antiquant_ptohpa_flag"] = False
    ifa_param["k_antiquant_ptog_flag"] = ifa_param["v_antiquant_ptog_flag"] = False
    ifa_param["kv_quant_ptog_flag"] = False
    ifa_param["continue_flag"] = False
    ifa_param["padding_flag"] = False
    ifa_param["out_quant_pc_flag"] = False
    ifa_param["deep_seek_flag"] = False
    ifa_param["prefix_flag"] = ifa_param["prefix_act_flag"] = False
    ifa_param["tnd_flag"] = False
    if inputLayout in ["TND", "TND_NTD"]:
        ifa_param["tnd_flag"] = True
    flag_list = str_to_bool_list(params["flaglist"])
    kv_batch_value = ifa_param["k_shape_list"][0][0]
    if ifa_param["k_dtype"] in ["fp4_e2m1", "fp4_e1m2", "float4_e1m2", "float4_e2m1"]:
        ifa_param["kv_quant_ptog_flag"] = True
    # 判断是否是连续场景
    if kv_batch_value >= 1 and len(ifa_param["k_shape_list"]) == 1:
        ifa_param["continue_flag"] = True
    if not flag_list[0] or not flag_list[1] or not flag_list[2] or not flag_list[24]:
        ifa_param["normal_flag"] = False
        print("[WARNING]:异常-->  q/k/v/out不输入场景！")
    if 0 in q_shape:
        ifa_param["normal_flag"] = False
        print("[WARNING]:异常-->  q为空场景！")
    if (
        0 in k_ori_shape_list
        or len(k_ori_shape_list) == 0
        or is_0_tensor(k_ori_shape_list)
    ):
        ifa_param["normal_flag"] = False
        print("[WARNING]:异常-->  k为空场景！")
    if (
        0 in v_ori_shape_list
        or len(v_ori_shape_list) == 0
        or is_0_tensor(v_ori_shape_list)
    ):
        ifa_param["normal_flag"] = False
        print("[WARNING]:异常-->  v为空场景！")
    if flag_list[3]:
        ifa_param["p_flag"] = True
    if flag_list[4]:
        ifa_param["m_flag"] = True
    if flag_list[5]:
        ifa_param["as_q_flag"] = True
    if flag_list[6]:
        ifa_param["as_flag"] = True
    if flag_list[7] and flag_list[8] and flag_list[9]:
        ifa_param["in_quant_flag"] = True
    if flag_list[17] and flag_list[19] and flag_list[29] and flag_list[26]:
        ifa_param["in_quant_flag"] = True
    if flag_list[10]:
        ifa_param["out_quant_flag"] = True
        if flag_list[11]:
            ifa_param["out_quant_offset_flag"] = True
        else:
            # 当不传入quantOffset2时，也需要生成一个和scale一样大小的全0 Offset，用于后续计算
            ifa_param["quantOffset2"] = np.zeros(
                ifa_param["quantScale2_shape"], np.float32
            )
            ifa_param["quantOffset2_shape"] = ifa_param["quantScale2_shape"]
        if ifa_param["quantScale2_shape"] != [1]:
            ifa_param["out_quant_pc_flag"] = True

    # 判断kv 分离的antiquant是否存在
    if flag_list[17]:
        ifa_param["k_antiquant_flag"] = True
        if flag_list[18]:
            ifa_param["k_antiquant_offset_flag"] = True
        if ifa_param["k_antiquant_mode"] == "0":
            ifa_param["k_antiquant_pc_flag"] = True
        if ifa_param["k_antiquant_mode"] == "1":
            ifa_param["k_antiquant_pto_flag"] = True
        if ifa_param["k_antiquant_mode"] == "2":
            ifa_param["k_antiquant_pth_flag"] = True
        if ifa_param["k_antiquant_mode"] == "3":
            ifa_param["k_antiquant_ptoh_flag"] = True
        if ifa_param["k_antiquant_mode"] == "4":
            ifa_param["k_antiquant_ptopa_flag"] = True
        if ifa_param["k_antiquant_mode"] == "5":
            ifa_param["k_antiquant_ptohpa_flag"] = True
        if ifa_param["k_antiquant_mode"] == "6":
            ifa_param["k_antiquant_ptog_flag"] = True
            ifa_param["kv_quant_ptog_flag"] = False
    if flag_list[19]:
        ifa_param["v_antiquant_flag"] = True
        if flag_list[20]:
            ifa_param["v_antiquant_offset_flag"] = True
        if ifa_param["v_antiquant_mode"] == "0":
            ifa_param["v_antiquant_pc_flag"] = True
        if ifa_param["v_antiquant_mode"] == "1":
            ifa_param["v_antiquant_pto_flag"] = True
        if ifa_param["v_antiquant_mode"] == "2":
            ifa_param["v_antiquant_pth_flag"] = True
        if ifa_param["v_antiquant_mode"] == "3":
            ifa_param["v_antiquant_ptoh_flag"] = True
        if ifa_param["v_antiquant_mode"] == "4":
            ifa_param["v_antiquant_ptopa_flag"] = True
        if ifa_param["v_antiquant_mode"] == "5":
            ifa_param["v_antiquant_ptohpa_flag"] = True
        if ifa_param["v_antiquant_mode"] == "6":
            ifa_param["v_antiquant_ptog_flag"] = True
            ifa_param["kv_quant_ptog_flag"] = False
    if flag_list[12] and not flag_list[17] and not flag_list[19]:
        ifa_param["kv_quant_flag"] = True
        if flag_list[13]:
            ifa_param["kv_quant_offset_flag"] = True
        if ifa_param["antiquant_mode"] == "1":
            ifa_param["kv_quant_pto_flag"] = True
    if flag_list[14]:
        ifa_param["pa_flag"] = True
        ifa_param["bt_shape"] = params["shape_input"][12]
        ifa_param["bt_dtype"] = trans_input_dtype(params["dtype_input"][12])
        ifa_param["block_table"] = torch_tensor_list[12]
        ifa_param["k_cache_shape"] = params["shape_input"][21]
        ifa_param["v_cache_shape"] = params["shape_input"][22]
        if 0 in ifa_param["k_cache_shape"]:
            ifa_param["normal_flag"] = False

    if (
        flag_list[16]
        and ifa_param["as_flag"]
        and ifa_param["continue_flag"]
        and not ifa_param["pa_flag"]
    ):
        ifa_param["padding_flag"] = True
        ifa_param["padding_size"] = params["range_input"][v_end_index + 12][0]
        if ifa_param["padding_size"] < 0:
            ifa_param["padding_size"] = 0
    if flag_list[21] and flag_list[22]:
        ifa_param["prefix_flag"] = True
    elif not flag_list[21] and not flag_list[22]:
        ifa_param["prefix_flag"] = False
    else:
        ifa_param["normal_flag"] = False
        print("[WARNING]:异常--> prefix未成对出现！")
    if flag_list[23]:
        ifa_param["prefix_act_flag"] = True
        ifa_param["prefix_act_lens"] = params["prefix_act_lens"][0]

    if flag_list[26]:
        ifa_param["deep_seek_flag"] = True
        ifa_param["v_shape_list"] = v_ori_shape_list = params["shape_input"][
            k_start_index : k_end_index + 1
        ]
        ifa_param["v_dtype"] = v_dtype = trans_input_dtype(params["dtype_input"][1])
        ifa_param["v_tensor_list"] = torch_tensor_list[k_start_index : k_end_index + 1]

    # 获取kv量化参数
    if ifa_param["kv_quant_flag"]:
        ifa_param["antiquantscale_shape_raw"] = antiquantscale_shape_raw = params[
            "shape_input"
        ][v_end_index + 8]
        if 0 in ifa_param["antiquantscale_shape_raw"]:
            ifa_param["normal_flag"] = False
            print("[WARNING]:异常-->  antiquantscale为空场景！")
        ifa_param["antiquantscale_dtype"] = trans_input_dtype(
            params["dtype_input"][v_end_index + 8]
        )
        ifa_param["antiquantscale_tensor_raw"] = torch_tensor_list[v_end_index + 8]
        if ifa_param["kv_quant_offset_flag"]:
            ifa_param["antiquantoffset_shape_raw"] = antiquantoffset_shape_raw = params[
                "shape_input"
            ][v_end_index + 9]
            if 0 in antiquantoffset_shape_raw:
                ifa_param["normal_flag"] = False
                print("[WARNING]:异常-->  antiquantoffset为空场景！")
            ifa_param["antiquantoffset_tensor_raw"] = torch_tensor_list[v_end_index + 9]
        else:
            ifa_param["antiquantoffset_shape_raw"] = antiquantscale_shape_raw
            ifa_param["antiquantoffset_tensor_raw"] = torch.zeros(
                antiquantscale_shape_raw
            ).numpy()
    # 获取kv分离的量化参数
    ifa_param["k_rope_antiquantscale_shape_raw"] = None
    ifa_param["k_rope_antiquantscale_dtype_raw"] = None
    ifa_param["k_rope_antiquantscale_tensor_raw"] = None
    if ifa_param["k_antiquant_flag"]:
        ifa_param["k_antiquantscale_shape_raw"] = k_antiquantscale_shape_raw = params[
            "shape_input"
        ][v_end_index + 13]
        if 0 in ifa_param["k_antiquantscale_shape_raw"]:
            ifa_param["normal_flag"] = False
            print("[WARNING]:异常-->  k_antiquantscale为空场景！")
        ifa_param["k_antiquantscale_dtype"] = trans_input_dtype(
            params["dtype_input"][v_end_index + 13]
        )
        ifa_param["k_antiquantscale_tensor_raw"] = torch_tensor_list[v_end_index + 13]
        if ifa_param["k_antiquant_offset_flag"]:
            ifa_param["k_antiquantoffset_shape_raw"] = k_antiquantoffset_shape_raw = (
                params["shape_input"][v_end_index + 14]
            )
            ifa_param["k_antiquantoffset_dtype"] = trans_input_dtype(
                params["dtype_input"][v_end_index + 14]
            )
            if 0 in k_antiquantoffset_shape_raw:
                ifa_param["normal_flag"] = False
                print("[WARNING]:异常-->  k_antiquantoffset为空场景！")
            ifa_param["k_antiquantoffset_tensor_raw"] = torch_tensor_list[
                v_end_index + 14
            ]
        else:
            ifa_param["k_antiquantoffset_shape_raw"] = k_antiquantscale_shape_raw
            ifa_param["k_antiquantoffset_dtype"] = ifa_param["k_antiquantscale_dtype"]
            ifa_param["k_antiquantoffset_tensor_raw"] = torch.zeros(
                k_antiquantscale_shape_raw
            ).numpy()
        # rope quant scale
        if len(ifa_param["flag_list"]) > 28:
            ifa_param["k_rope_antiquantscale_shape_raw"] = params["shape_input"][
                v_end_index + 24
            ]
            ifa_param["k_rope_antiquantscale_dtype_raw"] = trans_input_dtype(
                params["dtype_input"][v_end_index + 24]
            )
            ifa_param["k_rope_antiquantscale_tensor_raw"] = torch_tensor_list[
                v_end_index + 24
            ]
            # 创建一个空的rope offset
            ifa_param["k_rope_antiquantoffset_shape_raw"] = params["shape_input"][
                v_end_index + 24
            ]
            ifa_param["k_rope_antiquantoffset_dtype_raw"] = trans_input_dtype(
                params["dtype_input"][v_end_index + 24]
            )
            ifa_param["k_rope_antiquantoffset_tensor_raw"] = torch.zeros(
                ifa_param["k_rope_antiquantoffset_shape_raw"]
            ).numpy()

    if ifa_param["v_antiquant_flag"]:
        if ifa_param["flag_list"][26]:
            ifa_param["v_antiquantscale_shape_raw"] = copy.deepcopy(
                ifa_param["k_antiquantscale_shape_raw"]
            )
            ifa_param["v_antiquantscale_dtype"] = copy.deepcopy(
                ifa_param["k_antiquantoffset_dtype"]
            )
            ifa_param["v_antiquantscale_tensor_raw"] = copy.deepcopy(
                ifa_param["k_antiquantscale_tensor_raw"]
            )

            ifa_param["v_antiquantoffset_shape_raw"] = copy.deepcopy(
                ifa_param["k_antiquantoffset_shape_raw"]
            )
            ifa_param["v_antiquantoffset_dtype"] = copy.deepcopy(
                ifa_param["k_antiquantoffset_dtype"]
            )
            ifa_param["v_antiquantoffset_tensor_raw"] = copy.deepcopy(
                ifa_param["k_antiquantoffset_tensor_raw"]
            )
        else:
            ifa_param["v_antiquantscale_shape_raw"] = v_antiquantscale_shape_raw = (
                params["shape_input"][v_end_index + 15]
            )
            if 0 in ifa_param["v_antiquantscale_shape_raw"]:
                ifa_param["normal_flag"] = False
                print("[WARNING]:异常-->  v_antiquantscale为空场景！")
            ifa_param["v_antiquantscale_dtype"] = trans_input_dtype(
                params["dtype_input"][v_end_index + 15]
            )
            ifa_param["v_antiquantscale_tensor_raw"] = torch_tensor_list[
                v_end_index + 15
            ]
            if ifa_param["v_antiquant_offset_flag"]:
                ifa_param["v_antiquantoffset_shape_raw"] = (
                    v_antiquantoffset_shape_raw
                ) = params["shape_input"][v_end_index + 16]
                ifa_param["v_antiquantoffset_dtype"] = trans_input_dtype(
                    params["dtype_input"][v_end_index + 16]
                )
                if 0 in v_antiquantoffset_shape_raw:
                    ifa_param["normal_flag"] = False
                    print("[WARNING]:异常-->  v_antiquantoffset为空场景！")
                ifa_param["v_antiquantoffset_tensor_raw"] = torch_tensor_list[
                    v_end_index + 16
                ]
            else:
                ifa_param["v_antiquantoffset_shape_raw"] = v_antiquantscale_shape_raw
                ifa_param["v_antiquantoffset_dtype"] = ifa_param[
                    "v_antiquantscale_dtype"
                ]
                ifa_param["v_antiquantoffset_tensor_raw"] = torch.zeros(
                    v_antiquantscale_shape_raw
                ).numpy()

    # ND+rope： 当kvs<=512用softmaxV1 ;transpose/nz+rope ：golden计算都用softmaxV2
    ifa_param["softmaxV1_flag"] = False
    # if ifa_param['deep_seek_flag'] and ifa_param['pa_flag']:
    #     if inputLayout in ["TND", "BSH", "BNSD", "BSND"]:
    #         if len(ifa_param['k_cache_shape']) != 5:
    #             ifa_param["softmaxV1_flag"] = True
    return ifa_param


def cut_kvs_by_mask(tensor, mask):
    new_tensor = tensor.clone()
    b = tensor.shape[0]
    smax = tensor.shape[1]
    new_slist = []
    mask = mask.squeeze()
    for maskBatch in range(b):
        s = 0
        m_tensor_batch = mask[maskBatch : maskBatch + 1, :]
        for maskIndex in range(smax):
            maskOfsindex = m_tensor_batch[0][maskIndex]
            if not maskOfsindex:
                new_tensor[maskBatch : maskBatch + 1, s : (s + 1), :, :] = tensor[
                    maskBatch : maskBatch + 1, maskIndex : (maskIndex + 1), :, :
                ]
                s += 1
        new_slist.append(s)
    return new_tensor, new_slist


def get_min_list(list1, list2):
    new_list = [min(x, y) for x, y in zip(list1, list2)]
    return new_list


def left_padding_handler(tensor, act_list, padding_size):
    new_tensor = tensor.clone()
    b = tensor.shape[0]
    smax = tensor.shape[1]
    for bindex in range(b):
        act_s = act_list[bindex]
        start_index = smax - padding_size - act_s
        end_index = smax - padding_size
        new_tensor[bindex : bindex + 1, 0:act_s, :, :] = tensor[
            bindex : bindex + 1, start_index:end_index, :, :
        ]
    return new_tensor


def left_padding_handler_mask(tensor, act_list, padding_size):
    new_tensor = tensor.clone()
    b = tensor.shape[0]
    smax = tensor.shape[1]
    for bindex in range(b):
        act_s = act_list[bindex]
        start_index = smax - padding_size - act_s
        end_index = smax - padding_size
        new_tensor[bindex : bindex + 1, 0:act_s] = tensor[
            bindex : bindex + 1, start_index:end_index
        ]
        new_tensor[bindex : bindex + 1, act_s:] = 1
    return new_tensor


# 将int32类型的数据,拆成Int4,存进int8
def trans_int32_2_int4(input_int32):
    # 将Int32类型的数据按bit位平均拆成8份，每份长度为4bit
    parts = [(input_int32 >> i) & 0xF for i in range(0, 32, 4)]
    output_int4 = []
    for part in parts:
        # 将每份数据构造成一个Int8类型的数据
        # 符号位为第一个bit的值
        sign = (part >> 3) & 0x1
        # 剩余的值作为int8最后3个bit的值
        value = part & 0x7
        if sign == 1:
            # 如果符号位为1，则需要将value转换成负数
            value = -((value ^ 0x7) + 1)
        output_int4.append(value)
    return output_int4


# 将int32的tensor拆分，存入Int8类型的tensor
def trans_int32_2_int4_tensor_bnsd(tensor_int32, shape_int32):
    shape_int4 = [shape_int32[0], shape_int32[1], shape_int32[2], shape_int32[3] * 8]
    tensor_int4 = np.zeros(shape_int4).astype(np.int8)
    for Bid in range(shape_int32[0]):
        for Nid in range(shape_int32[1]):
            for Sid in range(shape_int32[2]):
                for Did in range(shape_int32[3]):
                    int4_data_list = trans_int32_2_int4(
                        int(
                            tensor_int32[
                                Bid : Bid + 1,
                                Nid : Nid + 1,
                                Sid : Sid + 1,
                                Did : Did + 1,
                            ]
                        )
                    )
                    for i in range(8):
                        tensor_int4[
                            Bid : Bid + 1,
                            Nid : Nid + 1,
                            Sid : Sid + 1,
                            8 * Did + i : 8 * Did + i + 1,
                        ] = int4_data_list[i]
    return tensor_int4, shape_int4


# pertoken_pa 场景，将scale/offset从BB转为1BS
def trans_bb_2_1bs(input_tensor, tables, batch):
    block_size = input_tensor.shape[1]
    max_seq = tables.shape[1] * block_size
    max_block_per_seq = math.ceil(max_seq / block_size)
    output_tensor = np.zeros((1, batch, max_seq), dtype=np.float32)
    for bindex in range(batch):
        table = tables[bindex]
        for block_base_index in range(max_block_per_seq):
            block_index = table[block_base_index]
            if block_index == -1:
                break
            if (block_base_index + 1) * block_size > max_seq:
                lens = max_seq - block_base_index * block_size
                output_tensor[0, bindex, block_base_index * block_size : max_seq] = (
                    input_tensor[block_index][0:lens]
                )
            else:
                output_tensor[
                    0,
                    bindex,
                    block_base_index * block_size : (block_base_index + 1) * block_size,
                ] = input_tensor[block_index]
    return output_tensor


# perheadtoken_pa 场景，将scale/offset从BnB转为1BnS
def trans_bb_2_1bns(input_tensor, tables, batch):
    num_kv_head = input_tensor.shape[1]
    block_size = input_tensor.shape[2]
    max_seq = tables.shape[1] * block_size
    max_block_per_seq = math.ceil(max_seq / block_size)
    output_tensor = np.zeros((batch, num_kv_head, max_seq), dtype=np.float32)

    for bindex in range(batch):
        table = tables[bindex]
        for block_base_index in range(max_block_per_seq):
            block_index = table[block_base_index]
            if block_index == -1:
                break
            if (block_base_index + 1) * block_size > max_seq:
                lens = max_seq - block_base_index * block_size
                output_tensor[bindex, :, block_base_index * block_size : max_seq] = (
                    input_tensor[block_index][:][0:lens]
                )
            else:
                output_tensor[
                    bindex,
                    :,
                    block_base_index * block_size : (block_base_index + 1) * block_size,
                ] = input_tensor[block_index]
    return output_tensor


def concat_tensor(tensor1, shape1, tensor2, shape2, n, tnd_flag=False):
    if len(shape1) != len(shape2):
        print(
            f"[ERROR]concat_tensor: 相加的两个tensor 维数不同! shape1 = {shape1}, shape2 =  {shape2}"
        )
        return None, None
    elif len(shape1) == 2:
        # 量化参数1h
        if shape1[0] != shape2[0]:
            print(
                f"[ERROR]concat_tensor: 相加的两个tensor 维度非法! shape1 = {shape1}, shape2 =  {shape2}"
            )
            return None, None
        d1 = int(shape1[1] / n)
        d2 = int(shape2[1] / n)
        tensor1_1nd = tensor1.reshape(shape1[0], n, d1)
        tensor2_1nd = tensor2.reshape(shape2[0], n, d2)
        concatenated_tensor = np.concatenate((tensor1_1nd, tensor2_1nd), axis=2)
        concatenated_shape = [shape1[0], shape1[1] + shape2[1]]
        concatenated_tensor = concatenated_tensor.reshape(concatenated_shape)
    elif len(shape1) == 3:
        if shape1[0] != shape2[0] or shape1[1] != shape2[1]:
            print(
                f"[ERROR]concat_tensor: 相加的两个tensor 维度非法! shape1 = {shape1}, shape2 =  {shape2}"
            )
            return None, None
        if tnd_flag:
            concatenated_tensor = np.concatenate((tensor1, tensor2), axis=2)
            concatenated_shape = [shape1[0], shape1[1], shape1[2] + shape2[2]]
        else:
            d1 = int(shape1[2] / n)
            d2 = int(shape2[2] / n)
            tensor1_bsnd = tensor1.reshape(shape1[0], shape1[1], n, d1)
            tensor2_bsnd = tensor2.reshape(shape2[0], shape2[1], n, d2)
            concatenated_tensor = np.concatenate((tensor1_bsnd, tensor2_bsnd), axis=3)
            concatenated_shape = [shape1[0], shape1[1], shape1[2] + shape2[2]]
            concatenated_tensor = concatenated_tensor.reshape(concatenated_shape)
    elif len(shape1) == 4:
        if shape1[0] != shape2[0] or shape1[1] != shape2[1] or shape1[2] != shape2[2]:
            print(
                f"[ERROR]concat_tensor: 相加的两个tensor 维度非法! shape1 = {shape1}, shape2 =  {shape2}"
            )
            return None, None
        concatenated_tensor = np.concatenate((tensor1, tensor2), axis=3)
        concatenated_shape = [shape1[0], shape1[1], shape1[2], shape1[3] + shape2[3]]
    else:
        print(f"[ERROR]concat_tensor: shape1 维度非法! shape1 = {shape1}")
        return None, None
    return concatenated_tensor, concatenated_shape


def trans_tnd_actseq(list):
    list_len = len(list)
    list_new = []
    list_new.append(list[0])
    for i in range(list_len - 1):
        new_item = list[i + 1] - list[i]
        if new_item >= 0:
            list_new.append(new_item)
        else:
            print(
                f"[ERROR]trans_tnd_actseq: Wrong input actseq:{list}, in loop {i}, item {new_item} < 0"
            )
    print(f"[INFO]before trans: {list}")
    print(f"[INFO]after trans: {list_new}")
    return list_new


def deepseek_preprocessing(ifa_param):
    numHeads = ifa_param["numHeads"]
    numKeyValueHeads = ifa_param["numKeyValueHeads"]
    # 备份拼接前的信息
    ifa_param["q_tensor_old"] = copy.deepcopy(ifa_param["q_tensor"])
    ifa_param["q_shape_old"] = copy.deepcopy(ifa_param["q_shape"])
    ifa_param["k_tensor_list_old"] = copy.deepcopy(ifa_param["k_tensor_list"])
    ifa_param["k_shape_list_old"] = copy.deepcopy(ifa_param["k_shape_list"])
    # 非全量化场景1.将Q与QROPE拼接2.将K与KROPE拼接3.伪量化场景，k_antiscale与k_rope_antiscale拼接
    if not ifa_param["in_quant_flag"]:
        q_new_tensor, q_new_shape = concat_tensor(
            ifa_param["q_tensor"],
            ifa_param["q_shape"],
            ifa_param["q_rope_tensor"],
            ifa_param["q_rope_shape"],
            numHeads,
            ifa_param["tnd_flag"],
        )
        if q_new_tensor is not None:
            ifa_param["q_tensor"] = q_new_tensor
            ifa_param["q_shape"] = q_new_shape
        else:
            print("[ERROR]q tensor的deepseek预处理异常，输出空tensor！")
            # return torch.zeros(out_shape)
        if ifa_param["k_num"] == 1:
            k_new_tensor, k_new_shape = concat_tensor(
                ifa_param["k_tensor_list"][0],
                ifa_param["k_shape_list"][0],
                ifa_param["k_rope_tensor"],
                ifa_param["k_rope_shape"],
                numKeyValueHeads,
                ifa_param["tnd_flag"],
            )
            if k_new_tensor is not None:
                ifa_param["k_tensor_list"][0] = k_new_tensor
                ifa_param["k_shape_list"][0] = k_new_shape
            else:
                print("[ERROR]k tensor的deepseek预处理异常，输出空tensor！")
                # return torch.zeros(out_shape)
        else:
            print("[ERROR]k tensor 长度不为1, deepseek预处理异常，输出空tensor！")
            # return torch.zeros(out_shape)
        # rope scale处理
        if ifa_param["k_rope_antiquantscale_shape_raw"] is not None:
            k_antiquantscale_shape_raw0 = ifa_param["k_antiquantscale_shape_raw"]
            k_antiquantscale_tensor_raw0 = ifa_param["k_antiquantscale_tensor_raw"]
            k_rope_antiquantscale_shape_raw = ifa_param[
                "k_rope_antiquantscale_shape_raw"
            ]
            k_rope_antiquantscale_tensor_raw = ifa_param[
                "k_rope_antiquantscale_tensor_raw"
            ]
            k_scale_new_tensor, k_scale_new_shape = concat_tensor(
                k_antiquantscale_tensor_raw0,
                k_antiquantscale_shape_raw0,
                k_rope_antiquantscale_tensor_raw,
                k_rope_antiquantscale_shape_raw,
                numKeyValueHeads,
                ifa_param["tnd_flag"],
            )

            # offset是空的，但是也要拼接成大小相同的tensor
            k_antiquantoffset_shape_raw0 = ifa_param["k_antiquantoffset_shape_raw"]
            k_antiquantoffset_tensor_raw0 = ifa_param["k_antiquantoffset_tensor_raw"]
            k_rope_antiquantoffset_shape_raw = ifa_param[
                "k_rope_antiquantoffset_shape_raw"
            ]
            k_rope_antiquantoffset_tensor_raw = ifa_param[
                "k_rope_antiquantoffset_tensor_raw"
            ]
            k_offset_new_tensor, k_offset_new_shape = concat_tensor(
                k_antiquantoffset_tensor_raw0,
                k_antiquantoffset_shape_raw0,
                k_rope_antiquantoffset_tensor_raw,
                k_rope_antiquantoffset_shape_raw,
                numKeyValueHeads,
                ifa_param["tnd_flag"],
            )
            if k_scale_new_tensor is not None:
                ifa_param["k_antiquantscale_tensor_raw"] = k_scale_new_tensor
                ifa_param["k_antiquantscale_shape_raw"] = k_scale_new_shape
                ifa_param["k_antiquantoffset_tensor_raw"] = k_offset_new_tensor
                ifa_param["k_antiquantoffset_shape_raw"] = k_offset_new_shape
            else:
                print(
                    "[ERROR]k antiquant scale tensor的deepseek预处理异常，输出空tensor！"
                )
                # return torch.zeros(out_shape)
    return ifa_param


def trans_bnsd_to_dequant_layout(tensor, shape, layout, act_q=None):
    B = tensor.shape[0]
    N = tensor.shape[1]
    if layout in ["TND", "TND_NTD"]:
        T = sum(act_q)
        output = torch.zeros(size=(T, N), dtype=tensor.dtype)
        t_start = 0
        for b_index in range(B):
            act_s = act_q[b_index]
            t_end = t_start + act_s
            if act_s == 0:
                continue
            for n_index in range(N):
                output[t_start:t_end, n_index : n_index + 1] = tensor[
                    b_index, n_index, :act_s
                ]
            t_start += act_s
        return output
    else:
        S = shape[2]
        return tensor.permute(0, 2, 1, 3).reshape(B, S, N)


def trans_dequant_layout_to_bnsd(tensor, shape, layout, act_seq=None):
    print(
        f"def trans_dequant_layout_to_bsnd(tensor, shape, layout({layout}), act_q=None):"
    )
    print("tensor.shape : ", tensor.shape)
    if layout in ["BSND", "BSND_NBSD", "BSH", "BSH_NBSD"]:
        B = tensor.shape[0]
        S = tensor.shape[1]
        N = tensor.shape[2]
        return tensor.reshape([B, S, N, 1]).transpose(0, 2, 1, 3)
    elif layout in ["TND", "TND_NTD"]:
        T = shape[0]
        N = shape[1]
        B = len(act_seq)
        S = max(act_seq)
        new_tensor = np.zeros((B, N, S), dtype=tensor.dtype)
        t_start = 0
        for b_index in range(B):
            act_s = act_seq[b_index]
            t_end = t_start + act_s
            if act_s == 0:
                continue
            for n_index in range(N):
                new_tensor[b_index, n_index, 0:act_s] = tensor[t_start:t_end, n_index]
            t_start += act_s
        return new_tensor.reshape([B, N, S, 1])
    else:
        print("error inputlayout")


def trans_bnsd_to_input_layout(tensor, shape, layout, act_q=None):
    if layout in ["BSH", "BSH_NBSD"]:
        output = tensor.permute(0, 2, 1, 3).contiguous().view(shape)
        return output
    elif layout in ["BSND", "BSND_NBSD"]:
        output = tensor.permute(0, 2, 1, 3).contiguous()
        return output
    elif layout in ["TND", "TND_NTD"]:
        T = sum(act_q)
        B = tensor.shape[0]
        N = tensor.shape[1]
        D = tensor.shape[3]
        output = torch.zeros(size=(T, N, D), dtype=tensor.dtype)
        t_start = 0
        for b_index in range(B):
            act_s = act_q[b_index]
            t_end = t_start + act_s
            if act_s == 0:
                continue
            for n_index in range(N):
                output[t_start:t_end, n_index, :] = tensor[b_index, n_index, :act_s, :]
            t_start += act_s
        return output
    else:
        return tensor


def deepseek_ds_pa_preprocessing(ifa_param, params):
    k_np_dtype = ifa_param["k_np_dtype"]
    v_np_dtype = ifa_param["v_np_dtype"]
    k_ori_shape_list = ifa_param["k_shape_list_old"]
    numKeyValueHeads = ifa_param["numKeyValueHeads"]
    numHeads = ifa_param["numHeads"]
    kv_layout = ifa_param["kv_layout"]
    actualSeqLengths = ifa_param["actualSeqLengths"]
    actualSeqLengths_q = ifa_param["actualSeqLengths_q"]
    # k_shape_bnsd_list = ifa_param['k_shape_bnsd_list']
    blockSize = ifa_param["blocksize"]
    blockTableShape = ifa_param["bt_shape"]
    block_table = ifa_param["block_table"]
    inputLayout = ifa_param["inputLayout"]
    k_cache = np.zeros(ifa_param["k_cache_shape"]).astype(k_np_dtype)
    v_cache = np.zeros(ifa_param["v_cache_shape"]).astype(v_np_dtype)
    k_rope_cache = np.zeros(ifa_param["k_rope_cache_shape"]).astype(k_np_dtype)
    k_dtype = ifa_param["k_dtype"]
    # 对备份的拼接前信息进行预处理
    x = 0
    k_tensor_bnsd_list = []
    for i in range(0, ifa_param["k_num"]):
        k_tensor_bnsd_list = []
        k_tensor, k_bnsd_shape = _n_trans_shape_to_bnsd(
            ifa_param["k_tensor_list_old"][i],
            k_ori_shape_list[x],
            kv_layout,
            numKeyValueHeads,
            actualSeqLengths,
        )
        x = x + 1
        k_tensor_bnsd_list.append(k_tensor)
        # k_shape_bnsd_list.append(k_bnsd_shape)
        # k int32预处理
        if ifa_param["k_dtype"] == "int32":
            k_tensor, k_bnsd_shape = trans_int32_2_int4_tensor_bnsd(
                k_tensor, k_bnsd_shape
            )

        if k_bnsd_shape[2] > ifa_param["ks_max"]:
            ifa_param["ks_max"] = k_bnsd_shape[2]
    k_rope_bnsd_tensor, k_rope_bnsd_shape = _n_trans_shape_to_bnsd(
        ifa_param["k_rope_tensor"],
        ifa_param["k_rope_shape"],
        kv_layout,
        numKeyValueHeads,
        actualSeqLengths,
    )
    if ifa_param["k_dtype"] == "int32":
        k_rope_bnsd_tensor, k_rope_bnsd_shape = trans_int32_2_int4_tensor_bnsd(
            k_rope_bnsd_tensor, k_rope_bnsd_shape
        )
    # gen k cache
    # 获取qd
    q_bnsd_tensor, q_bnsd_shape = _n_trans_shape_to_bnsd(
        ifa_param["q_tensor_old"],
        ifa_param["q_shape_old"],
        inputLayout,
        numHeads,
        actualSeqLengths_q,
    )
    if len(ifa_param["k_cache_shape"]) != len(ifa_param["k_rope_cache_shape"]):
        print("[WARNING]:k_cache_shape与k_rope_cache_shape 维数不符，输出空tensor！")
        # return torch.zeros(out_shape)
    if len(ifa_param["k_cache_shape"]) == 3:
        # trans kv to bsh(此处使用的tensor, 没有经过n的扩展)
        k_tensor_bsh_raw = trans_bnsd_to_bsh(
            k_tensor_bnsd_list[0], k_tensor_bnsd_list[0].shape
        )
        # kv paddIng
        kvh = numKeyValueHeads * q_bnsd_shape[3]
        k_tensor_bsh = np.zeros(
            (q_bnsd_shape[0], blockTableShape[1] * blockSize, kvh)
        ).astype(k_np_dtype)
        print(f"[INFO]k_tensor_bbh: {k_tensor_bsh.shape}")
        k_tensor_bsh[:, : k_tensor_bsh_raw.shape[1], :] = k_tensor_bsh_raw[:, :, :]
        for b in range(q_bnsd_shape[0]):
            for block_i, kv_cache_blk_id in enumerate(block_table[b]):
                block_offset = block_i * blockSize
                if kv_cache_blk_id == -1:
                    continue
                else:
                    k_cache[kv_cache_blk_id, 0:blockSize, :] = k_tensor_bsh[
                        b, block_offset : (block_offset + blockSize), :
                    ]

        k_rope_tensor_bsh_raw = trans_bnsd_to_bsh(k_rope_bnsd_tensor, k_rope_bnsd_shape)

        kropeh = numKeyValueHeads * k_rope_bnsd_shape[3]
        k_rope_tensor_bsh = np.zeros(
            (q_bnsd_shape[0], blockTableShape[1] * blockSize, kropeh)
        ).astype(k_np_dtype)
        print(f"[INFO]k_rope_tensor_bbh: {k_rope_tensor_bsh.shape}")
        k_rope_tensor_bsh[:, : k_rope_tensor_bsh_raw.shape[1], :] = (
            k_rope_tensor_bsh_raw[:, :, :]
        )
        for b in range(q_bnsd_shape[0]):
            for block_i, kv_cache_blk_id in enumerate(block_table[b]):
                block_offset = block_i * blockSize
                if kv_cache_blk_id == -1:
                    continue
                else:
                    k_rope_cache[kv_cache_blk_id, 0:blockSize, :] = k_rope_tensor_bsh[
                        b, block_offset : (block_offset + blockSize), :
                    ]
    elif len(ifa_param["k_cache_shape"]) == 4:
        kvn = k_tensor_bnsd_list[0].shape[1]
        kvs = k_tensor_bnsd_list[0].shape[2]
        kvd = k_tensor_bnsd_list[0].shape[3]
        k_tensor_bnsd = np.zeros(
            (q_bnsd_shape[0], kvn, blockTableShape[1] * blockSize, kvd)
        ).astype(k_np_dtype)
        print(f"[INFO]k_tensor_bnbd: {k_tensor_bnsd.shape}")
        k_tensor_bnsd[:, :, :kvs, :] = k_tensor_bnsd_list[0][:, :, :, :]

        for b in range(q_bnsd_shape[0]):
            for block_i, kv_cache_blk_id in enumerate(block_table[b]):
                block_offset = block_i * blockSize
                if kv_cache_blk_id == -1:
                    continue
                else:
                    for n in range(q_bnsd_shape[1]):
                        k_cache[kv_cache_blk_id, n : n + 1, 0:blockSize, :] = (
                            k_tensor_bnsd[
                                b,
                                n : n + 1,
                                block_offset : (block_offset + blockSize),
                                :,
                            ]
                        )
        kropen = k_rope_bnsd_shape[1]
        kropes = k_rope_bnsd_shape[2]
        kroped = k_rope_bnsd_shape[3]
        k_rope_tensor_bnsd = np.zeros(
            (q_bnsd_shape[0], kropen, blockTableShape[1] * blockSize, kroped)
        ).astype(k_np_dtype)
        print(f"[INFO]k_rope_tensor_bnbd: {k_rope_tensor_bnsd.shape}")
        k_rope_tensor_bnsd[:, :, :kropes, :] = k_rope_bnsd_tensor[:, :, :, :]

        for b in range(q_bnsd_shape[0]):
            for block_i, kv_cache_blk_id in enumerate(block_table[b]):
                block_offset = block_i * blockSize
                if kv_cache_blk_id == -1:
                    continue
                else:
                    for n in range(q_bnsd_shape[1]):
                        k_rope_cache[kv_cache_blk_id, n : n + 1, 0:blockSize, :] = (
                            k_rope_tensor_bnsd[
                                b,
                                n : n + 1,
                                block_offset : (block_offset + blockSize),
                                :,
                            ]
                        )
    elif len(ifa_param["k_cache_shape"]) == 5:
        kvn = k_tensor_bnsd_list[0].shape[1]
        kvs = k_tensor_bnsd_list[0].shape[2]
        kvd = k_tensor_bnsd_list[0].shape[3]
        k_tensor_bnsd = np.zeros(
            (q_bnsd_shape[0], kvn, blockTableShape[1] * blockSize, kvd)
        ).astype(k_np_dtype)
        print(f"[INFO]k_tensor_bnbd: {k_tensor_bnsd.shape}")
        k_tensor_bnsd[:, :, :kvs, :] = k_tensor_bnsd_list[0][:, :, :, :]
        k_cache_tensor_bnbd = np.zeros((k_cache.shape[0], kvn, blockSize, kvd)).astype(
            k_np_dtype
        )

        for b in range(q_bnsd_shape[0]):
            for block_i, kv_cache_blk_id in enumerate(block_table[b]):
                block_offset = block_i * blockSize
                if kv_cache_blk_id == -1:
                    continue
                else:
                    for n in range(q_bnsd_shape[1]):
                        k_cache_tensor_bnbd[
                            kv_cache_blk_id, n : n + 1, 0:blockSize, :
                        ] = k_tensor_bnsd[
                            b, n : n + 1, block_offset : (block_offset + blockSize), :
                        ]
        # [block_num, N, block_size, D] -> [block_num, N, D/16, block_size, 16]
        print(f"[INFO]k_cache_tensor_bnbd BNBD: {k_cache_tensor_bnbd.shape}")
        if k_dtype == "int8":
            base = 32
        else:
            base = 16
        k_cache = k_cache_tensor_bnbd.reshape(
            k_cache_tensor_bnbd.shape[0],
            k_cache_tensor_bnbd.shape[1],
            k_cache_tensor_bnbd.shape[2],
            k_cache_tensor_bnbd.shape[3] // base,
            base,
        ).transpose(0, 1, 3, 2, 4)
        print(f"[INFO]k_cache NZ: {k_cache.shape}")

        kropen = k_rope_bnsd_shape[1]
        kropes = k_rope_bnsd_shape[2]
        kroped = k_rope_bnsd_shape[3]
        k_rope_tensor_bnsd = np.zeros(
            (q_bnsd_shape[0], kropen, blockTableShape[1] * blockSize, kroped)
        ).astype(k_np_dtype)
        print(f"[INFO]k_rope_tensor_bnbd: {k_rope_tensor_bnsd.shape}")
        k_rope_tensor_bnsd[:, :, :kropes, :] = k_rope_bnsd_tensor[:, :, :, :]
        k_rope_cache_tensor_bnbd = np.zeros(
            (k_rope_cache.shape[0], kropen, blockSize, kroped)
        ).astype(k_np_dtype)
        print(f"[INFO]k_rope_cache_tensor_bnbd: {k_rope_cache_tensor_bnbd.shape}")

        for b in range(q_bnsd_shape[0]):
            for block_i, kv_cache_blk_id in enumerate(block_table[b]):
                block_offset = block_i * blockSize
                if kv_cache_blk_id == -1:
                    continue
                else:
                    for n in range(q_bnsd_shape[1]):
                        k_rope_cache_tensor_bnbd[
                            kv_cache_blk_id, n : n + 1, 0:blockSize, :
                        ] = k_rope_tensor_bnsd[
                            b, n : n + 1, block_offset : (block_offset + blockSize), :
                        ]
        # [block_num, N, block_size, ROPE_D] -> [block_num, N, ROPE_D/16, block_size, 16]
        print(f"[INFO]k_rope_cache_tensor_bnbd BNBD: {k_rope_cache_tensor_bnbd.shape}")
        k_rope_cache = k_rope_cache_tensor_bnbd.reshape(
            k_rope_cache_tensor_bnbd.shape[0],
            k_rope_cache_tensor_bnbd.shape[1],
            k_rope_cache_tensor_bnbd.shape[2],
            k_rope_cache_tensor_bnbd.shape[3] // base,
            base,
        ).transpose(0, 1, 3, 2, 4)
        print(f"[INFO]k_rope_cache NZ: {k_rope_cache.shape}")
    else:
        print(f"[ERROR]Wrong input k_cache_shape: {ifa_param['k_cache_shape']}")
        # return torch.zeros(out_shape)
    # 将kv cache 生成新的bin文件
    k_cache_index = 21
    v_cache_index = 22
    k_rope_cache_index = 25


def deepseek_inquant_ds_pa_preprocessing(ifa_param, params):
    # 生成blocktable
    k_np_dtype = np.int8
    v_np_dtype = np.int8
    k_rope_dtype = get_np_dtype(params["dtype_input"][24])
    k_tensor_bnsd_list = ifa_param["k_tensor_list_bnsd"]
    numKeyValueHeads = ifa_param["numKeyValueHeads"]
    q_bnsd_shape = ifa_param["q_bnsd_shape"]
    blockSize = ifa_param["blocksize"]
    blockTableShape = ifa_param["bt_shape"]
    block_table = ifa_param["block_table"]
    k_rope_bnsd_tensor = ifa_param["k_rope_bnsd_tensor"]
    k_rope_bnsd_shape = ifa_param["k_rope_bnsd_shape"]
    print(
        f"[PageAtten]Input Kdtype:{params['dtype_input'][1]} Vdtype:{params['dtype_input'][2]} KRopeType:{params['dtype_input'][24]}"
    )
    k_cache = np.zeros(ifa_param["k_cache_shape"]).astype(k_np_dtype)
    v_cache = np.zeros(ifa_param["v_cache_shape"]).astype(v_np_dtype)
    print("[INFO]:PA + Deepseek！")
    k_rope_cache = np.zeros(ifa_param["k_rope_cache_shape"]).astype(k_rope_dtype)
    if len(ifa_param["k_cache_shape"]) != len(ifa_param["k_rope_cache_shape"]):
        print("[WARNING]:k_cache_shape与k_rope_cache_shape 维数不符，输出空tensor！")
    if len(ifa_param["k_cache_shape"]) == 3:
        # trans kv to bsh(此处使用的tensor, 没有经过n的扩展)
        k_tensor_bsh_raw = trans_bnsd_to_bsh(
            k_tensor_bnsd_list[0], k_tensor_bnsd_list[0].shape
        )
        # kv paddIng
        kvh = numKeyValueHeads * q_bnsd_shape[3]
        k_tensor_bsh = np.zeros(
            (q_bnsd_shape[0], blockTableShape[1] * blockSize, kvh)
        ).astype(k_np_dtype)
        print(f"[INFO]k_tensor_bbh: {k_tensor_bsh.shape}")
        k_tensor_bsh[:, : k_tensor_bsh_raw.shape[1], :] = k_tensor_bsh_raw[:, :, :]
        for b in range(q_bnsd_shape[0]):
            for block_i, kv_cache_blk_id in enumerate(block_table[b]):
                block_offset = block_i * blockSize
                if kv_cache_blk_id == -1:
                    continue
                else:
                    k_cache[kv_cache_blk_id, 0:blockSize, :] = k_tensor_bsh[
                        b, block_offset : (block_offset + blockSize), :
                    ]

        k_rope_tensor_bsh_raw = trans_bnsd_to_bsh(k_rope_bnsd_tensor, k_rope_bnsd_shape)

        kropeh = numKeyValueHeads * k_rope_bnsd_shape[3]
        k_rope_tensor_bsh = np.zeros(
            (q_bnsd_shape[0], blockTableShape[1] * blockSize, kropeh)
        ).astype(k_rope_dtype)
        print(f"[INFO]k_rope_tensor_bbh: {k_rope_tensor_bsh.shape}")
        k_rope_tensor_bsh[:, : k_rope_tensor_bsh_raw.shape[1], :] = (
            k_rope_tensor_bsh_raw[:, :, :]
        )
        for b in range(q_bnsd_shape[0]):
            for block_i, kv_cache_blk_id in enumerate(block_table[b]):
                block_offset = block_i * blockSize
                if kv_cache_blk_id == -1:
                    continue
                else:
                    k_rope_cache[kv_cache_blk_id, 0:blockSize, :] = k_rope_tensor_bsh[
                        b, block_offset : (block_offset + blockSize), :
                    ]
    elif len(ifa_param["k_cache_shape"]) == 4:
        kvn = k_tensor_bnsd_list[0].shape[1]
        kvs = k_tensor_bnsd_list[0].shape[2]
        kvd = k_tensor_bnsd_list[0].shape[3]
        k_tensor_bnsd = np.zeros(
            (q_bnsd_shape[0], kvn, blockTableShape[1] * blockSize, kvd)
        ).astype(k_np_dtype)
        print(f"[INFO]k_tensor_bnbd: {k_tensor_bnsd.shape}")
        k_tensor_bnsd[:, :, :kvs, :] = k_tensor_bnsd_list[0][:, :, :, :]

        for b in range(q_bnsd_shape[0]):
            for block_i, kv_cache_blk_id in enumerate(block_table[b]):
                block_offset = block_i * blockSize
                if kv_cache_blk_id == -1:
                    continue
                else:
                    for n in range(q_bnsd_shape[1]):
                        k_cache[kv_cache_blk_id, n : n + 1, 0:blockSize, :] = (
                            k_tensor_bnsd[
                                b,
                                n : n + 1,
                                block_offset : (block_offset + blockSize),
                                :,
                            ]
                        )
        kropen = k_rope_bnsd_shape[1]
        kropes = k_rope_bnsd_shape[2]
        kroped = k_rope_bnsd_shape[3]
        k_rope_tensor_bnsd = np.zeros(
            (q_bnsd_shape[0], kropen, blockTableShape[1] * blockSize, kroped)
        ).astype(k_rope_dtype)
        print(f"[INFO]k_rope_tensor_bnbd: {k_rope_tensor_bnsd.shape}")
        k_rope_tensor_bnsd[:, :, :kropes, :] = k_rope_bnsd_tensor[:, :, :, :]

        for b in range(q_bnsd_shape[0]):
            for block_i, kv_cache_blk_id in enumerate(block_table[b]):
                block_offset = block_i * blockSize
                if kv_cache_blk_id == -1:
                    continue
                else:
                    for n in range(q_bnsd_shape[1]):
                        k_rope_cache[kv_cache_blk_id, n : n + 1, 0:blockSize, :] = (
                            k_rope_tensor_bnsd[
                                b,
                                n : n + 1,
                                block_offset : (block_offset + blockSize),
                                :,
                            ]
                        )
    elif len(ifa_param["k_cache_shape"]) == 5:
        kvn = k_tensor_bnsd_list[0].shape[1]
        kvs = k_tensor_bnsd_list[0].shape[2]
        kvd = k_tensor_bnsd_list[0].shape[3]
        k_tensor_bnsd = np.zeros(
            (q_bnsd_shape[0], kvn, blockTableShape[1] * blockSize, kvd),
            dtype=k_np_dtype,
        )
        k_tensor_bnsd[:, :, :kvs, :] = k_tensor_bnsd_list[0][:, :, :, :]
        k_cache_tensor_bnbd = np.zeros((k_cache.shape[0], kvn, blockSize, kvd)).astype(
            k_np_dtype
        )

        for b in range(q_bnsd_shape[0]):
            for block_i, kv_cache_blk_id in enumerate(block_table[b]):
                block_offset = block_i * blockSize
                if kv_cache_blk_id == -1:
                    continue
                else:
                    for n in range(q_bnsd_shape[1]):
                        k_cache_tensor_bnbd[
                            kv_cache_blk_id, n : n + 1, 0:blockSize, :
                        ] = k_tensor_bnsd[
                            b, n : n + 1, block_offset : (block_offset + blockSize), :
                        ]
        basek = 32
        basekrope = 16
        k_cache = k_cache_tensor_bnbd.reshape(
            k_cache_tensor_bnbd.shape[0],
            k_cache_tensor_bnbd.shape[1],
            k_cache_tensor_bnbd.shape[2],
            k_cache_tensor_bnbd.shape[3] // basek,
            basek,
        ).transpose(0, 1, 3, 2, 4)
        print(f"[INFO]k_cache NZ: {k_cache.shape}")
        kropen = k_rope_bnsd_shape[1]
        kropes = k_rope_bnsd_shape[2]
        kroped = k_rope_bnsd_shape[3]
        k_rope_tensor_bnsd = np.zeros(
            (q_bnsd_shape[0], kropen, blockTableShape[1] * blockSize, kroped)
        ).astype(k_rope_dtype)
        print(
            f"[INFO]k_rope_tensor_bnbd: {k_rope_tensor_bnsd.shape} {kropes} {k_rope_bnsd_tensor.shape}"
        )
        k_rope_tensor_bnsd[:, :, :kropes, :] = k_rope_bnsd_tensor[:, :, :, :]
        k_rope_cache_tensor_bnbd = np.zeros(
            (k_rope_cache.shape[0], kropen, blockSize, kroped)
        ).astype(k_rope_dtype)
        print(f"[INFO]k_rope_cache_tensor_bnbd: {k_rope_cache_tensor_bnbd.shape}")

        for b in range(q_bnsd_shape[0]):
            for block_i, kv_cache_blk_id in enumerate(block_table[b]):
                block_offset = block_i * blockSize
                if kv_cache_blk_id == -1:
                    continue
                else:
                    for n in range(q_bnsd_shape[1]):
                        k_rope_cache_tensor_bnbd[
                            kv_cache_blk_id, n : n + 1, 0:blockSize, :
                        ] = k_rope_tensor_bnsd[
                            b, n : n + 1, block_offset : (block_offset + blockSize), :
                        ]
        k_rope_cache = k_rope_cache_tensor_bnbd.reshape(
            k_rope_cache_tensor_bnbd.shape[0],
            k_rope_cache_tensor_bnbd.shape[1],
            k_rope_cache_tensor_bnbd.shape[2],
            k_rope_cache_tensor_bnbd.shape[3] // basekrope,
            basekrope,
        ).transpose(0, 1, 3, 2, 4)
        print(f"[INFO]k_rope_cache NZ: {k_rope_cache.shape}")
    else:
        print(f"[ERROR]Wrong input k_cache_shape: {ifa_param['k_cache_shape']}")
        # return torch.zeros(out_shape)
    # 将kv cache 生成新的bin文件
    k_cache_index = 21
    v_cache_index = 22
    k_rope_cache_index = 25
    if self.params["is_preprocess"]:
        self.params["input_data"].kwargs["k_cache"] = k_cache
        self.params["input_data"].kwargs["v_cache"] = v_cache
        self.params["input_data"].kwargs["k_rope_cache"] = k_rope_cache


def deepseek_inquant_preprocessing(ifa_param, params):
    numHeads = ifa_param["numHeads"]
    numKeyValueHeads = ifa_param["numKeyValueHeads"]
    actualSeqLengths_q = ifa_param["actualSeqLengths_q"]
    actualSeqLengths = ifa_param["actualSeqLengths"]
    inputLayout = ifa_param["inputLayout"]
    kv_layout = ifa_param["kv_layout"]
    q_rope_bnsd_tensor, q_rope_bnsd_shape = _n_trans_shape_to_bnsd(
        ifa_param["q_rope_tensor"],
        ifa_param["q_rope_shape"],
        inputLayout,
        numHeads,
        actualSeqLengths_q,
    )
    ifa_param["q_rope_bnsd_tensor"] = q_rope_bnsd_tensor
    ifa_param["q_rope_shape_bnsd"] = q_rope_bnsd_shape

    k_rope_bnsd_tensor, k_rope_bnsd_shape = _n_trans_shape_to_bnsd(
        ifa_param["k_rope_tensor"],
        ifa_param["k_rope_shape"],
        kv_layout,
        numKeyValueHeads,
        actualSeqLengths,
    )
    ifa_param["k_rope_bnsd_tensor"] = k_rope_bnsd_tensor
    ifa_param["k_rope_bnsd_shape"] = k_rope_bnsd_shape
    q_bnsd_tensor = ifa_param["q_tensor_bnsd"]
    q_bnsd_shape = ifa_param["q_shape_bnsd"]
    k_tensor_list = ifa_param["k_tensor_list_bnsd"]

    v_tensor_list = ifa_param["v_tensor_list_bnsd"]

    rowMax_Qn = np.abs(q_bnsd_tensor).max(axis=(3), keepdims=True).astype(np.float32)
    tenMax_Kn = np.abs(k_tensor_list[0]).max(keepdims=False).astype(np.float32)
    scaleQ = rowMax_Qn / 127
    scaleK = tenMax_Kn / 127

    dequantScale1 = scaleQ * scaleK

    q_int8_bnsd_tensor = np.round(q_bnsd_tensor / scaleQ).astype(np.int8)
    k_int8_bnsd_tensor = np.round(k_tensor_list[0] / scaleK).astype(np.int8)
    k_tensor_list[0] = k_int8_bnsd_tensor
    v_tensor_list[0] = k_int8_bnsd_tensor
    q_bnsd_tensor = q_int8_bnsd_tensor
    qmin_datarange = int(q_int8_bnsd_tensor.min())
    qmax_datarange = int(q_int8_bnsd_tensor.max())
    kmin_datarange = int(k_int8_bnsd_tensor.min())
    kmax_datarange = int(k_int8_bnsd_tensor.max())

    q_int8_bnsd_tensor = torch.from_numpy(q_int8_bnsd_tensor).to(torch.int8)
    k_int8_bnsd_tensor = torch.from_numpy(k_int8_bnsd_tensor).to(torch.int8)

    ifa_param["q_tensor_bnsd"] = q_bnsd_tensor
    print(
        "ifa_param['q_tensor_bnsd'] = q_bnsd_tensor, q_bnsd_tensor : ",
        q_bnsd_tensor.shape,
        q_bnsd_tensor.dtype,
    )

    ifa_param["dequantScale1"] = dequantScale1
    print(
        "ifa_param['dequantScale1'] = dequantScale1, dequantScale1 : ",
        dequantScale1.shape,
        dequantScale1.dtype,
    )
    ifa_param["dequantScale2"] = np.array([scaleK])
    print(
        "ifa_param['dequantScale2'] = np.array([scaleK]), np.array([scaleK]) : ",
        np.array([scaleK]),
        np.array([scaleK]).dtype,
    )

    dequantScale1 = torch.from_numpy(dequantScale1).to(torch.float32)
    dequantScale2 = torch.from_numpy(np.array([scaleK])).to(torch.float32)

    dequantScale1 = trans_bnsd_to_dequant_layout(
        dequantScale1, dequantScale1.shape, inputLayout, actualSeqLengths_q
    )
    q_shape = ifa_param["q_shape"]
    k_shape = ifa_param["k_shape_list"][0]
    q_int8_npu = trans_bnsd_to_input_layout(
        q_int8_bnsd_tensor, q_shape, inputLayout, actualSeqLengths_q
    )
    k_int8_npu = trans_bnsd_to_input_layout(
        k_int8_bnsd_tensor, k_shape, kv_layout, actualSeqLengths
    )

    dequantScaleQ = torch.from_numpy(scaleQ).to(torch.float32)
    dequantScaleQ = trans_bnsd_to_dequant_layout(
        dequantScaleQ, dequantScaleQ.shape, inputLayout, actualSeqLengths_q
    )

    ifa_param["k_tensor_list_bnsd"] = k_tensor_list
    ifa_param["v_tensor_list_bnsd"] = v_tensor_list

    print("k_tensor_list : ", k_tensor_list[0].shape, k_tensor_list[0].dtype)
    print("v_tensor_list : ", v_tensor_list[0].shape, v_tensor_list[0].dtype)

    return ifa_param


def deepseek_inquant_preprocessing_for_recovery(ifa_param, torch_tensor_list, params):
    print(
        "def deepseek_inquant_preprocessing_for_recovery(ifa_param, torch_tensor_list, params): enter"
    )

    numHeads = ifa_param["numHeads"]
    numKeyValueHeads = ifa_param["numKeyValueHeads"]
    actualSeqLengths_q = ifa_param["actualSeqLengths_q"]
    actualSeqLengths = ifa_param["actualSeqLengths"]
    inputLayout = ifa_param["inputLayout"]
    kv_layout = ifa_param["kv_layout"]
    dequantScale1 = torch_tensor_list[27] * torch_tensor_list[17]
    dequantScale1_bnsd = trans_dequant_layout_to_bnsd(
        dequantScale1,
        dequantScale1.shape,
        ifa_param["inputLayout"],
        ifa_param["actualSeqLengths_q"],
    )

    index = 28
    rope_shape = ifa_param["q_rope_shape"]
    if len(params["shape_input"]) == index:
        from libs.generate_tensor_data import gen_tensor_data

        params["shape_input"].append(params["shape_input"][23])
        params["dtype_input"].append("fp32")
        params["range_input"].append(params["range_input"][23])

        input_data_i = gen_tensor_data(params, index)

        q_rope_bnsd_tensor, q_rope_bnsd_shape = _n_trans_shape_to_bnsd(
            input_data_i,
            ifa_param["q_rope_shape"],
            inputLayout,
            numHeads,
            actualSeqLengths_q,
        )

        q_rope_bnsd_tensor_quant = np.nan_to_num(
            q_rope_bnsd_tensor.astype(np.float32) / dequantScale1_bnsd, nan=0.0
        )

        g_q_rope = torch.from_numpy(q_rope_bnsd_tensor.astype(np.float32)).to(
            torch.float32
        )
        npu_q_rope = torch.from_numpy(q_rope_bnsd_tensor_quant).to(torch.bfloat16)

        g_q_rope = trans_bnsd_to_input_layout(
            g_q_rope, rope_shape, inputLayout, actualSeqLengths_q
        )
        npu_q_rope = trans_bnsd_to_input_layout(
            npu_q_rope, rope_shape, inputLayout, actualSeqLengths_q
        )

    else:
        q_rope_bnsd_tensor, q_rope_bnsd_shape = _n_trans_shape_to_bnsd(
            torch_tensor_list[28], rope_shape, inputLayout, numHeads, actualSeqLengths_q
        )
    ifa_param["q_rope_bnsd_tensor"] = q_rope_bnsd_tensor
    ifa_param["q_rope_shape_bnsd"] = q_rope_bnsd_shape

    k_rope_bnsd_tensor, k_rope_bnsd_shape = _n_trans_shape_to_bnsd(
        ifa_param["k_rope_tensor"],
        ifa_param["k_rope_shape"],
        kv_layout,
        numKeyValueHeads,
        actualSeqLengths,
    )
    ifa_param["k_rope_bnsd_tensor"] = k_rope_bnsd_tensor
    ifa_param["k_rope_bnsd_shape"] = k_rope_bnsd_shape
    ifa_param["dequantScale1"] = dequantScale1_bnsd
    ifa_param["dequantScale2"] = torch_tensor_list[17]

    return ifa_param


def aclnn_op_func_ifa_cpu(torch_tensor_list, params):
    print(f"aclnn_op_func: {type(torch_tensor_list[0])}")
    print("cpu执行中...")
    ifa_param = get_param_fus(torch_tensor_list, params)

    if not ifa_param["normal_flag"]:
        return torch.zeros(ifa_param["out_shape"])
    ifa_param["action_type"] = params["action_type"]

    # ===参数预处理===
    actualSeqLengths_q = ifa_param["actualSeqLengths_q_raw"]
    actualSeqLengths = ifa_param["actualSeqLengths_raw"]
    inputLayout = ifa_param["inputLayout"]
    numHeads = ifa_param["numHeads"]
    numKeyValueHeads = ifa_param["numKeyValueHeads"]
    p_shape = ifa_param["p_shape"]
    k_start_index = ifa_param["k_start_index"]
    k_end_index = ifa_param["k_end_index"]
    v_start_index = ifa_param["v_start_index"]
    v_end_index = ifa_param["v_end_index"]
    out_shape = ifa_param["out_shape"]
    k_ori_shape_list = copy.deepcopy(ifa_param["k_shape_list"])
    v_ori_shape_list = copy.deepcopy(ifa_param["v_shape_list"])
    k_dtype = ifa_param["k_dtype"]
    v_dtype = ifa_param["v_dtype"]
    m_tensor = ifa_param["m_tensor"]
    m_shape = ifa_param["m_shape"]
    sparse_mode = ifa_param["sparse_mode"]
    if ifa_param["tnd_flag"] and ifa_param["pa_flag"]:
        kv_layout = "BNSD"
    else:
        kv_layout = inputLayout
    ifa_param["kv_layout"] = kv_layout

    # >> actualSeqLengths预处理：actualSeqLengths为单值场景，如果长度为1且b不为1，则将actualSeqLengths扩展为b个单值的列表
    # >> 仅非tnd场景，或tnd+pa场景，需要处理
    if not ifa_param["tnd_flag"] or (ifa_param["tnd_flag"] and ifa_param["pa_flag"]):
        # actq
        if len(actualSeqLengths_q) == 1 and len(actualSeqLengths_q) != ifa_param["b"]:
            actualSeqLengthsq_item = actualSeqLengths_q[0]
            for b_count in range(ifa_param["b"] - 1):
                actualSeqLengths_q.append(actualSeqLengthsq_item)
        if len(actualSeqLengths_q) > ifa_param["b"]:
            actualSeqLengths_q = actualSeqLengths_q[: ifa_param["b"]]
        # actkv
        if len(actualSeqLengths) == 1 and len(actualSeqLengths) != ifa_param["b"]:
            actualSeqLengths_item = actualSeqLengths[0]
            for b_count in range(ifa_param["b"] - 1):
                actualSeqLengths.append(actualSeqLengths_item)
        # >> actualSeqLengths预处理：actualSeqLengths长度超过b
        if len(actualSeqLengths) > ifa_param["b"]:
            actualSeqLengths = actualSeqLengths[: ifa_param["b"]]
    if ifa_param["tnd_flag"]:
        # 将tnd格式下的act seq转成普通的act seq
        actualSeqLengths_q = trans_tnd_actseq(actualSeqLengths_q)
        if not ifa_param["pa_flag"]:
            actualSeqLengths = trans_tnd_actseq(actualSeqLengths)
        print("[INFO]trans_tnd_actseq end.")

    ifa_param["actualSeqLengths"] = actualSeqLengths
    ifa_param["actualSeqLengths_q"] = actualSeqLengths_q
    # deepseek 预处理： 1.将Q与QROPE拼接 2.将K与KROPE拼接 3.伪量化场景，k_antiscale与k_rope_antiscale拼接
    if ifa_param["deep_seek_flag"]:
        ifa_param = deepseek_preprocessing(ifa_param)
    # >> q 预处理：将q转换为bnsd
    q_bnsd_tensor, q_bnsd_shape = _n_trans_shape_to_bnsd(
        ifa_param["q_tensor"],
        ifa_param["q_shape"],
        inputLayout,
        numHeads,
        actualSeqLengths_q,
    )
    ifa_param["q_bnsd_shape"] = q_bnsd_shape

    qs = q_bnsd_shape[2]
    ifa_param["q_tensor_bnsd"] = q_bnsd_tensor
    ifa_param["q_shape_bnsd"] = q_bnsd_shape
    ifa_param["q_d"] = q_bnsd_shape[3]

    # >> kv预处理：1、将kv list 转换为bnsd 2、GQA场景，将kvn扩展为qn
    k_tensor_list = []
    v_tensor_list = []
    k_shape_list = []
    v_shape_list = []
    k_tensor_bnsd_list = []
    v_tensor_bnsd_list = []
    k_shape_bnsd_list = []
    v_shape_bnsd_list = []
    ifa_param["ks_max"] = 0

    for i in range(0, ifa_param["k_num"]):
        k_tensor, k_bnsd_shape = _n_trans_shape_to_bnsd(
            ifa_param["k_tensor_list"][i],
            ifa_param["k_shape_list"][i],
            kv_layout,
            numKeyValueHeads,
            actualSeqLengths,
        )
        k_tensor_bnsd_list.append(k_tensor)
        k_shape_bnsd_list.append(k_bnsd_shape)
        # k int32预处理
        if ifa_param["k_dtype"] == "int32":
            k_tensor, k_bnsd_shape = trans_int32_2_int4_tensor_bnsd(
                k_tensor, k_bnsd_shape
            )

        if numKeyValueHeads != numHeads:
            if ifa_param["deep_seek_flag"] and ifa_param["in_quant_flag"]:
                pass
            else:
                k_tensor, k_bnsd_shape = _t_broadcastKV_sigle(
                    numHeads, numKeyValueHeads, k_tensor, k_dtype
                )
        if k_bnsd_shape[2] > ifa_param["ks_max"]:
            ifa_param["ks_max"] = k_bnsd_shape[2]
        k_tensor_list.append(k_tensor)
        k_shape_list.append(k_bnsd_shape)

    x = 0
    for i in range(0, ifa_param["k_num"]):
        v_tensor, v_bnsd_shape = _n_trans_shape_to_bnsd(
            ifa_param["v_tensor_list"][i],
            v_ori_shape_list[x],
            kv_layout,
            numKeyValueHeads,
            actualSeqLengths,
        )
        x = x + 1
        v_tensor_bnsd_list.append(v_tensor)
        v_shape_bnsd_list.append(v_bnsd_shape)
        # v int32预处理
        if ifa_param["v_dtype"] == "int32":
            v_tensor, v_bnsd_shape = trans_int32_2_int4_tensor_bnsd(
                v_tensor, v_bnsd_shape
            )

        if numKeyValueHeads != numHeads:
            if ifa_param["deep_seek_flag"] and ifa_param["in_quant_flag"]:
                pass
            else:
                v_tensor, v_bnsd_shape = _t_broadcastKV_sigle(
                    numHeads, numKeyValueHeads, v_tensor, v_dtype
                )
        v_tensor_list.append(v_tensor)
        v_shape_list.append(v_bnsd_shape)

    ifa_param["k_tensor_list_bnsd"] = k_tensor_list
    ifa_param["k_shape_list_bnsd"] = k_shape_list
    ifa_param["v_tensor_list_bnsd"] = v_tensor_list
    ifa_param["v_shape_list_bnsd"] = v_shape_list
    # 处理deepseek全量化：1.将qkv从FP16转成INT8 2.生成dequantscale1,dequantscale2
    if ifa_param["deep_seek_flag"] and ifa_param["in_quant_flag"]:
        if ifa_param["action_type"] in ["bm_output_gold", "bm_output"]:
            ifa_param = deepseek_inquant_preprocessing_for_recovery(
                ifa_param, torch_tensor_list, params
            )
        else:
            ifa_param = deepseek_inquant_preprocessing(ifa_param, params)

    # >> left padding 异常场景处理：起点/终点异常，返回全0
    if ifa_param["padding_flag"]:
        max_act_seq = max(actualSeqLengths)
        kv_s = ifa_param["ks_max"]
        if kv_s - ifa_param["padding_size"] - max_act_seq < 0:
            print("[WARNING]:paddingsize 溢出，输出空tensor！")
            return torch.zeros(out_shape)
    action_type = params["action_type"]
    # >> pse 处理
    if ifa_param["p_flag"]:
        p_tensor = torch_tensor_list[3]

    # >> m预处理：1、将m扩展为BN1S  2、padding场景下，偏移部分设置为1 3、针对FP16格式，将tensor转成0/1
    m_dtype = get_pt_dtype(params["dtype_input"][v_end_index + 2])
    m_tensor = torch.tensor(m_tensor, dtype=m_dtype)

    # qs大于1
    if sparse_mode == 3:
        batch = q_bnsd_shape[0]
        numheads = q_bnsd_shape[1]
        npu_m_shape_s = m_shape
        if sparse_mode == 0 or sparse_mode == 1:
            batch, numheads, ns1, ns2 = get_attention_mask_batch_num(
                m_shape, q_bnsd_shape
            )  # 获取输入attentionmask的batch 和numhead
            npu_m_shape_s = [ns1, ns2]
        kvs = k_shape_list[0][2]
        cpu_m_shape = [qs, kvs]  # cpu
        preTokens = ifa_param["pre_tokens"]
        nextTokens = ifa_param["next_tokens"]
        actualprefixKV = 0
        prefix_kvs = 0
        kvs_list = [kvs]
        cpu_m_tensor, npu_m_tensor, preTokens, nextTokens = (
            _create_random_mask_by_spars(
                cpu_m_shape,
                npu_m_shape_s,
                m_dtype,
                preTokens,
                nextTokens,
                ifa_param["actualSeqLengths_q"],
                actualSeqLengths,
                actualprefixKV,
                prefix_kvs,
                kvs_list,
                batch,
                numheads,
                sparse_mode,
            )
        )

        npu_m_tensor = torch.from_numpy(npu_m_tensor).to(m_dtype)

        if action_type in ["bm_output", "bm_output_gold"]:
            pass
        if sparse_mode == 0 or sparse_mode == 1:
            pass
        else:
            cpu_m_tensor = np.array(cpu_m_tensor)
            cpu_m_tensor = torch.from_numpy(cpu_m_tensor)
            ifa_param["m_tensor"] = _t_broadcast_mask_n(
                cpu_m_tensor, cpu_m_tensor.shape, numHeads, q_bnsd_shape[0]
            )
    else:
        if ifa_param["m_flag"]:
            m_rewrite_flag = False
            if ifa_param["padding_flag"]:
                m_tensor = cut_padding_size(
                    m_tensor, actualSeqLengths, ifa_param["padding_size"]
                )
                m_rewrite_flag = True
            if ifa_param["m_dtype"] == "float16":
                m_tensor = torch.where(
                    m_tensor < 0.8,
                    torch.tensor(0, dtype=torch.float16),
                    torch.tensor(1, dtype=torch.float16),
                )
                m_rewrite_flag = True
            ifa_param["m_tensor"] = _t_broadcast_mask_n(
                m_tensor, m_shape, numHeads, m_shape[0]
            )

    # >> 后量化参数处理：1、给dtype赋值 2、如果是perchannel模式，要将后量化参数统一转换成1n1d格式
    if ifa_param["out_quant_flag"]:
        ifa_param["quantScale2_dtype"] = trans_input_dtype(
            params["dtype_input"][v_end_index + 6]
        )
        ifa_param["quantOffset2_dtype"] = trans_input_dtype(
            params["dtype_input"][v_end_index + 7]
        )
    if ifa_param["out_quant_pc_flag"]:
        ifa_param["quantScale2"] = trans_to_1n1d(
            ifa_param["quantScale2"],
            ifa_param["quantScale2_shape"],
            ifa_param["q_shape_bnsd"][3],
        )
        if ifa_param["quantScale2"] is None:
            return torch.zeros(out_shape)
        ifa_param["quantOffset2"] = trans_to_1n1d(
            ifa_param["quantOffset2"],
            ifa_param["quantOffset2_shape"],
            ifa_param["q_shape_bnsd"][3],
        )
        if ifa_param["quantOffset2"] is None:
            return torch.zeros(out_shape)
        ifa_param["quantScale2_dtype"] = trans_input_dtype(
            params["dtype_input"][v_end_index + 6]
        )
        ifa_param["quantOffset2_dtype"] = trans_input_dtype(
            params["dtype_input"][v_end_index + 7]
        )

        # >> 处理pageatten场景（以下处理不涉及cpu真值计算，仅为npu生成输入）：
        # 1、生成随机的block_table，并覆写原有bin文件
        # 2、将kv shape 统一转换成bsh
        # 3、生成kv cache
        # 4、将kv cache dump成新的bin文件，供aclnn接口调用
    if ifa_param["pa_flag"] and "output" not in params["action_type"]:
        # 生成blocktable
        blockSize = ifa_param["blocksize"]
        blockTableShape = ifa_param["bt_shape"]
        blockNum = ifa_param["k_cache_shape"][0]
        if 0 in ifa_param["bt_shape"]:
            print("[WARNING]:block_table为空场景，输出空tensor！")
            return torch.zeros(out_shape)
        blockNumPerBlock = []
        block_num_min = 0
        for actual_seq in actualSeqLengths:
            blockNumPerBlock.append(math.ceil(actual_seq / blockSize))
            block_num_min += math.ceil(actual_seq / blockSize)
        if block_num_min > blockNum:
            print(
                f"[ERROR]Wrong input k_cache_shape: get blockNum = {blockNum}, but expect blockNum > {block_num_min}"
            )
            exit(1)
        block_idx_list = np.arange(0, blockNum, 1)
        block_idx_list = np.random.permutation(block_idx_list).astype(np.int32)
        block_idx = 0
        block_table = [-1] * blockTableShape[1]
        block_table = np.tile(block_table, (blockTableShape[0], 1)).astype(np.int32)
        block_table_batch_idx = 0
        for idx in blockNumPerBlock:
            for j in range(idx):
                block_table[block_table_batch_idx][j] = block_idx_list[block_idx]
                block_idx += 1
            block_table_batch_idx += 1
        ifa_param["block_table"] = block_table
        k_np_dtype = tools.get_np_dtype(params["dtype_input"][1])
        v_np_dtype = tools.get_np_dtype(params["dtype_input"][2])
        ifa_param["k_np_dtype"] = k_np_dtype
        ifa_param["v_np_dtype"] = v_np_dtype
        print(
            f"[PageAtten]Input Kdtype:{params['dtype_input'][1]} Vdtype:{params['dtype_input'][2]}"
        )
        k_cache = np.zeros(ifa_param["k_cache_shape"]).astype(k_np_dtype)
        v_cache = np.zeros(ifa_param["v_cache_shape"]).astype(v_np_dtype)
        if not ifa_param["deep_seek_flag"]:
            print("[INFO]:PA + no Deepseek！")
            # gen kv cache
            if len(ifa_param["k_cache_shape"]) == 3:
                # trans kv to bsh(此处使用的tensor, 没有经过n的扩展)
                k_tensor_bsh_raw = trans_bnsd_to_bsh(
                    k_tensor_bnsd_list[0], k_shape_bnsd_list[0]
                )
                v_tensor_bsh_raw = trans_bnsd_to_bsh(
                    v_tensor_bnsd_list[0], v_shape_bnsd_list[0]
                )

                # kv paddIng
                kvh = numKeyValueHeads * q_bnsd_shape[3]
                if ifa_param["k_dtype"] == "int32":
                    kvh = int(kvh / 8)
                k_tensor_bsh = np.zeros(
                    (q_bnsd_shape[0], blockTableShape[1] * blockSize, kvh)
                ).astype(k_np_dtype)
                v_tensor_bsh = np.zeros(
                    (q_bnsd_shape[0], blockTableShape[1] * blockSize, kvh)
                ).astype(v_np_dtype)
                k_tensor_bsh[:, : k_tensor_bsh_raw.shape[1], :] = k_tensor_bsh_raw[
                    :, :, :
                ]
                v_tensor_bsh[:, : v_tensor_bsh_raw.shape[1], :] = v_tensor_bsh_raw[
                    :, :, :
                ]
                for b in range(q_bnsd_shape[0]):
                    for block_i, kv_cache_blk_id in enumerate(block_table[b]):
                        block_offset = block_i * blockSize
                        if kv_cache_blk_id == -1:
                            continue
                        else:
                            k_cache[kv_cache_blk_id, 0:blockSize, :] = k_tensor_bsh[
                                b, block_offset : (block_offset + blockSize), :
                            ]
                            v_cache[kv_cache_blk_id, 0:blockSize, :] = v_tensor_bsh[
                                b, block_offset : (block_offset + blockSize), :
                            ]
            elif len(ifa_param["k_cache_shape"]) == 4:
                kvn = k_tensor_bnsd_list[0].shape[1]
                kvs = k_tensor_bnsd_list[0].shape[2]
                kvd = k_tensor_bnsd_list[0].shape[3]
                k_tensor_bnsd = np.zeros(
                    (q_bnsd_shape[0], kvn, blockTableShape[1] * blockSize, kvd)
                ).astype(k_np_dtype)
                v_tensor_bnsd = np.zeros(
                    (q_bnsd_shape[0], kvn, blockTableShape[1] * blockSize, kvd)
                ).astype(v_np_dtype)
                k_tensor_bnsd[:, :, :kvs, :] = k_tensor_bnsd_list[0][:, :, :, :]
                v_tensor_bnsd[:, :, :kvs, :] = v_tensor_bnsd_list[0][:, :, :, :]

                for b in range(q_bnsd_shape[0]):
                    for block_i, kv_cache_blk_id in enumerate(block_table[b]):
                        block_offset = block_i * blockSize
                        if kv_cache_blk_id == -1:
                            continue
                        else:
                            for n in range(q_bnsd_shape[1]):
                                k_cache[kv_cache_blk_id, n : n + 1, 0:blockSize, :] = (
                                    k_tensor_bnsd[
                                        b,
                                        n : n + 1,
                                        block_offset : (block_offset + blockSize),
                                        :,
                                    ]
                                )
                                v_cache[kv_cache_blk_id, n : n + 1, 0:blockSize, :] = (
                                    v_tensor_bnsd[
                                        b,
                                        n : n + 1,
                                        block_offset : (block_offset + blockSize),
                                        :,
                                    ]
                                )
            elif len(ifa_param["k_cache_shape"]) == 5:  # NZ
                kvn = k_tensor_bnsd_list[0].shape[1]
                kvs = k_tensor_bnsd_list[0].shape[2]
                kvd = k_tensor_bnsd_list[0].shape[3]
                vd = v_tensor_bnsd_list[0].shape[3]
                k_tensor_bnsd = np.zeros(
                    (q_bnsd_shape[0], kvn, blockTableShape[1] * blockSize, kvd)
                ).astype(k_np_dtype)
                v_tensor_bnsd = np.zeros(
                    (q_bnsd_shape[0], kvn, blockTableShape[1] * blockSize, vd)
                ).astype(v_np_dtype)
                k_tensor_bnsd[:, :, :kvs, :] = k_tensor_bnsd_list[0][:, :, :, :]
                v_tensor_bnsd[:, :, :kvs, :] = v_tensor_bnsd_list[0][:, :, :, :]
                k_cache_tensor_bnbd = np.zeros(
                    (k_cache.shape[0], kvn, blockSize, kvd)
                ).astype(k_np_dtype)
                v_cache_tensor_bnbd = np.zeros(
                    (k_cache.shape[0], kvn, blockSize, vd)
                ).astype(k_np_dtype)
                for b in range(q_bnsd_shape[0]):
                    for block_i, kv_cache_blk_id in enumerate(block_table[b]):
                        block_offset = block_i * blockSize
                        if kv_cache_blk_id == -1:
                            continue
                        else:
                            for n in range(q_bnsd_shape[1]):
                                k_cache_tensor_bnbd[
                                    kv_cache_blk_id, n : n + 1, 0:blockSize, :
                                ] = k_tensor_bnsd[
                                    b,
                                    n : n + 1,
                                    block_offset : (block_offset + blockSize),
                                    :,
                                ]
                                v_cache_tensor_bnbd[
                                    kv_cache_blk_id, n : n + 1, 0:blockSize, :
                                ] = v_tensor_bnsd[
                                    b,
                                    n : n + 1,
                                    block_offset : (block_offset + blockSize),
                                    :,
                                ]
                if k_dtype == "int8":
                    base = 32
                else:
                    base = 16
                k_cache = k_cache_tensor_bnbd.reshape(
                    k_cache_tensor_bnbd.shape[0],
                    k_cache_tensor_bnbd.shape[1],
                    k_cache_tensor_bnbd.shape[2],
                    k_cache_tensor_bnbd.shape[3] // base,
                    base,
                ).transpose(0, 1, 3, 2, 4)
                v_cache = v_cache_tensor_bnbd.reshape(
                    k_cache_tensor_bnbd.shape[0],
                    k_cache_tensor_bnbd.shape[1],
                    k_cache_tensor_bnbd.shape[2],
                    v_cache_tensor_bnbd.shape[3] // base,
                    base,
                ).transpose(0, 1, 3, 2, 4)
            else:
                print(
                    f"[ERROR]Wrong input kv_cache_shape: {ifa_param['k_cache_shape']}"
                )
                return torch.zeros(out_shape)
            # 将kv cache 生成新的bin文件

            k_cache_index = 21
            v_cache_index = 22
        else:
            print("[INFO]:PA + Deepseek！")
            if ifa_param["in_quant_flag"]:
                deepseek_inquant_ds_pa_preprocessing(ifa_param, params)
            else:
                deepseek_ds_pa_preprocessing(ifa_param, params)
    # >> kv 伪量化预处理
    if ifa_param["kv_quant_flag"]:
        antiquantscale_shape_raw = ifa_param["antiquantscale_shape_raw"]
        antiquantscale_tensor_raw = ifa_param["antiquantscale_tensor_raw"]
        antiquantoffset_shape_raw = ifa_param["antiquantoffset_shape_raw"]
        antiquantoffset_tensor_raw = ifa_param["antiquantoffset_tensor_raw"]

        # >> [TEMP]pertoken_group：1、将kv量化参数统一转换成2,B,N,S,D/32 2、GQA场景下，扩展kv量化参数
        if ifa_param["kv_quant_ptog_flag"]:
            antiquantscale_tensor = _trans_antiparam_to_2bnsg(
                antiquantscale_shape_raw,
                antiquantscale_tensor_raw,
                inputLayout,
                numKeyValueHeads,
            )
            antiquantoffset_tensor = _trans_antiparam_to_2bnsg(
                antiquantoffset_shape_raw,
                antiquantoffset_tensor_raw,
                inputLayout,
                numKeyValueHeads,
            )
            antiquantscale_tensor = broadcast_kv_dequant_tensor_fp8e8m0(
                antiquantscale_tensor, numKeyValueHeads, numHeads
            )
            antiquantoffset_tensor = broadcast_kv_dequant_tensor_fp8e8m0(
                antiquantoffset_tensor, numKeyValueHeads, numHeads
            )

            ifa_param["antiquantScale"] = antiquantscale_tensor
            ifa_param["antiquantOffset"] = antiquantoffset_tensor
        # pertensor/perchannel ->1、将kv量化参数统一转换成2n1d 2、GQA场景下，扩展kv量化参数
        elif not ifa_param["kv_quant_pto_flag"]:
            # 将kv量化参数转换为2n1d(匹配bnsd)
            antiquantscale_tensor = _trans_antiparam_to_2n1d(
                antiquantscale_shape_raw,
                antiquantscale_tensor_raw,
                inputLayout,
                numKeyValueHeads,
                ifa_param["q_d"],
            )
            antiquantoffset_tensor = _trans_antiparam_to_2n1d(
                antiquantoffset_shape_raw,
                antiquantoffset_tensor_raw,
                inputLayout,
                numKeyValueHeads,
                ifa_param["q_d"],
            )
            # GQA场景，扩展kv反量化参数
            q_dtype_np = get_np_dtype(params["dtype_input"][0])
            antiquantscale_tensor = broadcast_kv_dequant_tensor(
                antiquantscale_tensor, numKeyValueHeads, numHeads, q_dtype_np
            )
            antiquantoffset_tensor = broadcast_kv_dequant_tensor(
                antiquantoffset_tensor, numKeyValueHeads, numHeads, q_dtype_np
            )

            ifa_param["antiquantScale"] = antiquantscale_tensor
            ifa_param["antiquantOffset"] = antiquantoffset_tensor
        else:
            ifa_param["antiquantScale"] = antiquantscale_tensor_raw
            ifa_param["antiquantOffset"] = antiquantoffset_tensor_raw
    # 判断kv分离的伪量化
    if ifa_param["k_antiquant_flag"] and not ifa_param["v_antiquant_flag"]:
        print("[ERROR] k_antiquant exit, but v_antiquant not exit")
        exit(-1)
    elif not ifa_param["k_antiquant_flag"] and ifa_param["v_antiquant_flag"]:
        print("[ERROR] v_antiquant exit, but k_antiquant not exit")
        exit(-1)
    else:
        if ifa_param["k_antiquant_flag"]:
            k_antiquantscale_shape_raw = ifa_param["k_antiquantscale_shape_raw"]
            k_antiquantscale_tensor_raw = ifa_param["k_antiquantscale_tensor_raw"]
            k_antiquantoffset_shape_raw = ifa_param["k_antiquantoffset_shape_raw"]
            k_antiquantoffset_tensor_raw = ifa_param["k_antiquantoffset_tensor_raw"]
            print(f"k_antiquant_pto_flag: {ifa_param['k_antiquant_pto_flag']}")
            print(f"k_antiquant_ptoh_flag: {ifa_param['k_antiquant_ptoh_flag']}")
            print(f"k_antiquant_ptopa_flag: {ifa_param['k_antiquant_ptopa_flag']}")
            print(f"k_antiquant_ptohpa_flag: {ifa_param['k_antiquant_ptohpa_flag']}")
            # pertoken_head
            if ifa_param["k_antiquant_ptoh_flag"]:
                # GQA场景，扩展kv反量化参数
                k_antiquantscale_tensor = broadcast_kv_split_dequant_tensor_ptoh(
                    k_antiquantscale_tensor_raw, numKeyValueHeads, numHeads
                )
                k_antiquantoffset_tensor = broadcast_kv_split_dequant_tensor_ptoh(
                    k_antiquantoffset_tensor_raw, numKeyValueHeads, numHeads
                )
                ifa_param["k_antiquantScale"] = k_antiquantscale_tensor
                ifa_param["k_antiquantOffset"] = k_antiquantoffset_tensor
            # pertensor/perchannel ->1、将kv量化参数统一转换成1n1d 2、GQA场景下，扩展kv量化参数
            elif ifa_param["k_antiquant_pc_flag"] or ifa_param["k_antiquant_pth_flag"]:
                k_antiquantscale_tensor = _trans_antiparam_to_1n1d(
                    k_antiquantscale_shape_raw,
                    k_antiquantscale_tensor_raw,
                    inputLayout,
                    numKeyValueHeads,
                    ifa_param["q_d"],
                    ifa_param["k_antiquant_mode"],
                )
                k_antiquantoffset_tensor = _trans_antiparam_to_1n1d(
                    k_antiquantoffset_shape_raw,
                    k_antiquantoffset_tensor_raw,
                    inputLayout,
                    numKeyValueHeads,
                    ifa_param["q_d"],
                    ifa_param["k_antiquant_mode"],
                )
                # GQA场景，扩展kv反量化参数
                q_dtype_np = get_np_dtype(params["dtype_input"][0])
                k_antiquantscale_tensor = broadcast_kv_split_dequant_tensor(
                    k_antiquantscale_tensor, numKeyValueHeads, numHeads, q_dtype_np
                )
                k_antiquantoffset_tensor = broadcast_kv_split_dequant_tensor(
                    k_antiquantoffset_tensor, numKeyValueHeads, numHeads, q_dtype_np
                )
                ifa_param["k_antiquantScale"] = k_antiquantscale_tensor
                ifa_param["k_antiquantOffset"] = k_antiquantoffset_tensor
            # pertoken_pa -> 1、将bb转成1bs
            elif ifa_param["k_antiquant_ptopa_flag"]:
                if ifa_param["pa_flag"]:
                    ifa_param["k_antiquantScale"] = trans_bb_2_1bs(
                        k_antiquantscale_tensor_raw,
                        ifa_param["block_table"],
                        ifa_param["b"],
                    )
                    ifa_param["k_antiquantOffset"] = trans_bb_2_1bs(
                        k_antiquantoffset_tensor_raw,
                        ifa_param["block_table"],
                        ifa_param["b"],
                    )
                else:
                    print("[ERROR]Got antiquant mode == 4,but not PA!")
                    exit(1)
            # pertoken_head_pa -> 1、将bb转成1bns
            elif ifa_param["k_antiquant_ptohpa_flag"]:
                if ifa_param["pa_flag"]:
                    k_antiquantscale_tensor = trans_bb_2_1bns(
                        k_antiquantscale_tensor_raw,
                        ifa_param["block_table"],
                        ifa_param["b"],
                    )
                    k_antiquantoffset_tensor = trans_bb_2_1bns(
                        k_antiquantoffset_tensor_raw,
                        ifa_param["block_table"],
                        ifa_param["b"],
                    )
                    k_antiquantscale_tensor = broadcast_kv_split_dequant_tensor_ptoh(
                        k_antiquantscale_tensor, numKeyValueHeads, numHeads
                    )
                    k_antiquantoffset_tensor = broadcast_kv_split_dequant_tensor_ptoh(
                        k_antiquantoffset_tensor, numKeyValueHeads, numHeads
                    )
                    ifa_param["k_antiquantScale"] = k_antiquantscale_tensor
                    ifa_param["k_antiquantOffset"] = k_antiquantoffset_tensor
                else:
                    print("[ERROR]Got antiquant mode == 5,but not PA!")
                    exit(1)

            elif ifa_param["k_antiquant_ptog_flag"]:
                k_antiquantscale_tensor = broadcast_kv_split_antiquant_tensor_fp8e8m0(
                    k_antiquantscale_tensor_raw, numKeyValueHeads, numHeads
                )
                k_antiquantoffset_tensor = broadcast_kv_split_antiquant_tensor_fp8e8m0(
                    k_antiquantoffset_tensor_raw, numKeyValueHeads, numHeads
                )

                ifa_param["k_antiquantScale"] = k_antiquantscale_tensor
                ifa_param["k_antiquantOffset"] = k_antiquantoffset_tensor

            # pertoken -> 如果是BS,转成1BS
            else:
                ifa_param["k_antiquantScale"] = _trans_antiparam_to_1bs(
                    k_antiquantscale_tensor_raw
                )
                ifa_param["k_antiquantOffset"] = _trans_antiparam_to_1bs(
                    k_antiquantoffset_tensor_raw
                )

        if ifa_param["v_antiquant_flag"]:
            v_antiquantscale_shape_raw = ifa_param["v_antiquantscale_shape_raw"]
            v_antiquantscale_tensor_raw = ifa_param["v_antiquantscale_tensor_raw"]
            v_antiquantoffset_shape_raw = ifa_param["v_antiquantoffset_shape_raw"]
            v_antiquantoffset_tensor_raw = ifa_param["v_antiquantoffset_tensor_raw"]
            # pertoken_head
            if ifa_param["v_antiquant_ptoh_flag"]:
                # GQA场景，扩展kv反量化参数
                v_antiquantscale_tensor = broadcast_kv_split_dequant_tensor_ptoh(
                    v_antiquantscale_tensor_raw, numKeyValueHeads, numHeads
                )
                v_antiquantoffset_tensor = broadcast_kv_split_dequant_tensor_ptoh(
                    v_antiquantoffset_tensor_raw, numKeyValueHeads, numHeads
                )
                ifa_param["v_antiquantScale"] = v_antiquantscale_tensor
                ifa_param["v_antiquantOffset"] = v_antiquantoffset_tensor
            # pertensor/perchannel ->1、将kv量化参数统一转换成1n1d 2、GQA场景下，扩展kv量化参数
            elif ifa_param["v_antiquant_pc_flag"] or ifa_param["v_antiquant_pth_flag"]:
                v_antiquantscale_tensor = _trans_antiparam_to_1n1d(
                    v_antiquantscale_shape_raw,
                    v_antiquantscale_tensor_raw,
                    inputLayout,
                    numKeyValueHeads,
                    ifa_param["q_d"],
                    ifa_param["v_antiquant_mode"],
                )
                v_antiquantoffset_tensor = _trans_antiparam_to_1n1d(
                    v_antiquantoffset_shape_raw,
                    v_antiquantoffset_tensor_raw,
                    inputLayout,
                    numKeyValueHeads,
                    ifa_param["q_d"],
                    ifa_param["v_antiquant_mode"],
                )
                # GQA场景，扩展kv反量化参数
                q_dtype_np = get_np_dtype(params["dtype_input"][0])
                v_antiquantscale_tensor = broadcast_kv_split_dequant_tensor(
                    v_antiquantscale_tensor, numKeyValueHeads, numHeads, q_dtype_np
                )
                v_antiquantoffset_tensor = broadcast_kv_split_dequant_tensor(
                    v_antiquantoffset_tensor, numKeyValueHeads, numHeads, q_dtype_np
                )
                ifa_param["v_antiquantScale"] = v_antiquantscale_tensor
                ifa_param["v_antiquantOffset"] = v_antiquantoffset_tensor
            # pertoken_pa -> 1、将bb转成1bs
            elif ifa_param["v_antiquant_ptopa_flag"]:
                if ifa_param["pa_flag"]:
                    ifa_param["v_antiquantScale"] = trans_bb_2_1bs(
                        v_antiquantscale_tensor_raw,
                        ifa_param["block_table"],
                        ifa_param["b"],
                    )
                    ifa_param["v_antiquantOffset"] = trans_bb_2_1bs(
                        v_antiquantoffset_tensor_raw,
                        ifa_param["block_table"],
                        ifa_param["b"],
                    )
                else:
                    print("[ERROR]Got antiquant mode == 4,but not PA!")
                    exit(1)

            # per_token_head_pa -> 1、将bnb转成1bns
            elif ifa_param["v_antiquant_ptohpa_flag"]:
                print(v_antiquantscale_tensor_raw.shape)
                if ifa_param["pa_flag"]:
                    v_antiquantscale_tensor = trans_bb_2_1bns(
                        v_antiquantscale_tensor_raw,
                        ifa_param["block_table"],
                        ifa_param["b"],
                    )
                    v_antiquantoffset_tensor = trans_bb_2_1bns(
                        v_antiquantoffset_tensor_raw,
                        ifa_param["block_table"],
                        ifa_param["b"],
                    )
                    v_antiquantscale_tensor = broadcast_kv_split_dequant_tensor_ptoh(
                        v_antiquantscale_tensor, numKeyValueHeads, numHeads
                    )
                    v_antiquantoffset_tensor = broadcast_kv_split_dequant_tensor_ptoh(
                        v_antiquantoffset_tensor, numKeyValueHeads, numHeads
                    )
                    ifa_param["v_antiquantScale"] = v_antiquantscale_tensor
                    ifa_param["v_antiquantOffset"] = v_antiquantoffset_tensor
                else:
                    print("[ERROR]Got antiquant mode == 5,but not PA!")
                    exit(1)

            # per_token_group -> 1、将bb转成1bs
            elif ifa_param["v_antiquant_ptog_flag"]:
                v_antiquantscale_tensor = broadcast_kv_split_antiquant_tensor_fp8e8m0(
                    v_antiquantscale_tensor_raw, numKeyValueHeads, numHeads
                )
                v_antiquantoffset_tensor = broadcast_kv_split_antiquant_tensor_fp8e8m0(
                    v_antiquantoffset_tensor_raw, numKeyValueHeads, numHeads
                )

                ifa_param["v_antiquantScale"] = v_antiquantscale_tensor
                ifa_param["v_antiquantOffset"] = v_antiquantoffset_tensor
            # pertoken -> 如果是BS,转成1BS
            else:
                ifa_param["v_antiquantScale"] = _trans_antiparam_to_1bs(
                    v_antiquantscale_tensor_raw
                )
                ifa_param["v_antiquantOffset"] = _trans_antiparam_to_1bs(
                    v_antiquantoffset_tensor_raw
                )

    # >> 输入量化预处理：获取切分使用的sinner
    ifa_param["sinner"] = 0
    if ifa_param["in_quant_flag"] and not ifa_param["deep_seek_flag"]:
        sinner = get_sinner(ifa_param)
        ifa_param["sinner"] = sinner

    # psefix 预处理：1.转成1nsd 2.GQA场景扩展N; 3.按act_prefix裁剪 4.获取perfix_act_lens
    if ifa_param["prefix_flag"]:
        k_prefix_tensor = ifa_param["k_prefix_tensor"]
        k_prefix_shape = ifa_param["k_prefix_shape"]
        v_prefix_tensor = ifa_param["v_prefix_tensor"]
        v_prefix_shape = ifa_param["v_prefix_shape"]
        k_prefix_tensor, k_prefix_shape = _n_trans_shape_to_bnsd(
            k_prefix_tensor, k_prefix_shape, inputLayout, numKeyValueHeads
        )
        v_prefix_tensor, v_prefix_shape = _n_trans_shape_to_bnsd(
            v_prefix_tensor, v_prefix_shape, inputLayout, numKeyValueHeads
        )

        # kv prefix int32预处理
        if ifa_param["k_prefix_dtype"] == "int32":
            k_prefix_tensor, k_prefix_shape = trans_int32_2_int4_tensor_bnsd(
                k_prefix_tensor, k_prefix_shape
            )
        if ifa_param["v_prefix_dtype"] == "int32":
            v_prefix_tensor, v_prefix_shape = trans_int32_2_int4_tensor_bnsd(
                v_prefix_tensor, v_prefix_shape
            )

        if numKeyValueHeads != numHeads:
            k_prefix_tensor, k_prefix_shape = _t_broadcastKV_sigle(
                numHeads, numKeyValueHeads, k_prefix_tensor, ifa_param["k_prefix_dtype"]
            )
            v_prefix_tensor, v_prefix_shape = _t_broadcastKV_sigle(
                numHeads, numKeyValueHeads, v_prefix_tensor, ifa_param["v_prefix_dtype"]
            )

        if ifa_param["prefix_act_flag"]:
            k_prefix_tensor = k_prefix_tensor[:, :, : ifa_param["prefix_act_lens"], :]
            v_prefix_tensor = v_prefix_tensor[:, :, : ifa_param["prefix_act_lens"], :]
        else:
            ifa_param["prefix_act_lens"] = k_prefix_shape[2]

        ifa_param["k_prefix_tensor"] = k_prefix_tensor
        ifa_param["v_prefix_tensor"] = v_prefix_tensor

        print(f"prefix_act_lens:{str(ifa_param['prefix_act_lens'])}")

    # ===CPU计算入口===
    y_all, lse = _t_increattention_bnsd(ifa_param)
    y_all = torch.from_numpy(y_all)
    y_all = trans_bnsd_to_layout(y_all, out_shape, inputLayout, actualSeqLengths_q)

    print(f"out_shape:{out_shape}")
    if ifa_param["softmax_lse_flag"]:
        print("softmax_lse_flag:", ifa_param["softmax_lse_flag"])
        print("inputLayout:", inputLayout)
        if inputLayout != "TND":
            lse = lse.astype("float64")
            lse = torch.from_numpy(lse)
        elif inputLayout == "TND":
            B = ifa_param["q_shape_bnsd"][0]
            T = ifa_param["q_shape"][0]
            N = ifa_param["q_shape"][1]
            lse_output = np.zeros((T, N, 1), dtype=lse.dtype)
            t_start = 0
            for b_index in range(B):
                act_s = actualSeqLengths_q[b_index]
                t_end = t_start + act_s
                if act_s == 0:
                    continue
                for n_index in range(N):
                    lse_output[t_start:t_end, n_index, :] = lse[
                        b_index, n_index, :act_s, :
                    ]
                t_start += act_s
            lse = lse_output
            lse = torch.from_numpy(lse.astype(np.float32))
        return y_all, lse
    else:
        return y_all


def convert_tensor_to_numpy(obj):
    """
    递归地将字典中的所有 torch.tensor 对象转换为 numpy 数组。
    """
    if isinstance(obj, dict):
        # 如果是字典，则递归处理每个值
        return {k: convert_tensor_to_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        # 如果是列表，则递归处理每个元素
        return [convert_tensor_to_numpy(item) for item in obj]
    elif isinstance(obj, torch.Tensor):
        # 如果是 torch.Tensor，则转换为 numpy 数组
        if obj.dtype == torch.bfloat16:
            obj = obj.to(torch.float32)
        return obj.cpu().numpy()
    else:
        # 其他类型的对象直接返回
        return obj


# PFA
def t_bsh_to_bsnd(tensor, bsh_shape, headnums, actSeqLength, inputLayout="BSH"):
    if inputLayout == "SH":  # SH格式
        print("SH")
        if len(actSeqLength) == 0:
            B = 1
            S = bsh_shape[0]
        else:
            B = len(actSeqLength)
            S = sum(actSeqLength)
        H = bsh_shape[1]
        N = headnums
        D = H // N
        if B == 1:
            return tensor.reshape(B, S, N, D), [B, S, N, D]
        else:
            tmp = torch.zeros((B, S, N, D))
            sums = 0
            for i in range(B):
                acts = actSeqLength[i]
                for j in range(acts):
                    print(sums + j)
                    tmp[i : i + 1, j : j + 1] = tensor[
                        sums + j : sums + j + 1, :
                    ].reshape(N, D)
                sums += acts
            return tmp, [B, S, N, D]
    elif inputLayout == "BSH":
        print("BSH")
        B = bsh_shape[0]
        S = bsh_shape[1]
        H = bsh_shape[2]
        N = headnums
        D = H // N
        print("BSH", tensor.shape)
        return tensor.reshape(B, S, N, D), [B, S, N, D]
    elif inputLayout == "NSD":
        print("NSD")
        B = 1
        S = bsh_shape[1]
        N = bsh_shape[0]
        D = bsh_shape[2]
        H = D * N
        return tensor.reshape(B, N, S, D).permute(0, 2, 1, 3), [B, S, N, D]
    elif inputLayout == "TND":
        N = bsh_shape[1]
        D = bsh_shape[2]
        B = len(actSeqLength)
        S = max(actSeqLength)
        new_tensor = np.zeros((B, S, N, D), dtype=tensor.dtype)
        t_start = 0
        for b_index in range(B):
            act_s = actSeqLength[b_index]
            t_end = t_start + act_s
            if act_s == 0:
                continue
            for s_index in range(act_s):
                new_tensor[b_index, s_index, 0:N, :] = tensor[t_start:t_end, :, :]
            t_start += act_s
        print(f"[TEMP]trans tnd 2 bnsd: {tensor.shape} -> {new_tensor.shape}")
        return new_tensor, [B, S, N, D]
    else:
        print("BNSD", tensor.shape)
        return tensor.permute(0, 2, 1, 3), bsh_shape


def _t_trans_bsh_to_bnsd(tensor, bsh_shape, headnums, actSeqLength, inputLayout="BSH"):
    if inputLayout == "SH":  # SH格式
        print("SH")
        if len(actSeqLength) == 0:
            B = 1
            S = bsh_shape[0]
        else:
            B = len(actSeqLength)
            S = sum(actSeqLength)
        H = bsh_shape[1]
        N = headnums
        D = H // N
        print(B, N, S, D)
        if B == 1:
            return tensor.reshape(B, S, N, D).permute(0, 2, 1, 3), [B, N, S, D]
        else:
            tmp = torch.zeros((B, S, N, D))
            sums = 0
            for i in range(B):
                acts = actSeqLength[i]
                for j in range(acts):
                    tmp[i : i + 1, j : j + 1] = tensor[
                        sums + j : sums + j + 1, :
                    ].reshape(N, D)
                sums += acts
            return tmp.permute(0, 2, 1, 3), [B, N, S, D]
    elif inputLayout == "BSH":
        print("BSH")
        B = bsh_shape[0]
        S = bsh_shape[1]
        H = bsh_shape[2]
        N = headnums
        D = H // N
        print("BSH", tensor.shape)
        return tensor.reshape(B, S, N, D).permute(0, 2, 1, 3), [B, N, S, D]
    elif inputLayout == "NSD":
        print("NSD")
        B = 1
        S = bsh_shape[1]
        N = bsh_shape[0]
        D = bsh_shape[2]
        H = D * N
        return tensor.reshape(B, N, S, D), [B, N, S, D]
    elif inputLayout == "TND":
        B = len(actSeqLength)
        S = max(actSeqLength)
        N = bsh_shape[1]
        D = bsh_shape[2]
        new_tensor = np.zeros((B, N, S, D), dtype=tensor.dtype)
        t_start = 0
        for b_index in range(B):
            act_s = actSeqLength[b_index]
            t_end = t_start + act_s
            if act_s == 0:
                continue
            for n_index in range(N):
                new_tensor[b_index, n_index, 0:act_s, :] = tensor[
                    t_start:t_end, n_index, :
                ]
            t_start += act_s
        print(f"[TEMP]trans tnd 2 bnsd: {tensor.shape} -> {new_tensor.shape}")
        return new_tensor, [B, N, S, D]
    else:
        print("BNSD", tensor.shape)
        return tensor, bsh_shape


def np_bsh_to_bnsd(tensor, bsh_shape, headnums, actSeqLength, inputLayout="BSH"):
    """
    cpu golden  need  bnsd
    """
    if inputLayout == "SH":  # SH格式
        print("SH")
        if len(actSeqLength) == 0:
            B = 1
            S = bsh_shape[0]
        else:
            B = len(actSeqLength)
            S = sum(actSeqLength)
        H = bsh_shape[1]
        N = headnums
        D = H // N
        print(B, N, S, D)
        if B == 1:
            return tensor.reshape(B, S, N, D).transpose(0, 2, 1, 3), [B, N, S, D]
        else:
            tmp = np.zeros((B, S, N, D), dtype=tensor.dtype)
            sums = 0
            for i in range(B):
                acts = actSeqLength[i]
                for j in range(acts):
                    tmp[i : i + 1, j : j + 1] = tensor[
                        sums + j : sums + j + 1, :
                    ].reshape(N, D)
                sums += acts
            return tmp.transpose(0, 2, 1, 3), [B, N, S, D]
    elif inputLayout == "BSH":
        print("BSH")
        B = bsh_shape[0]
        S = bsh_shape[1]
        H = bsh_shape[2]
        N = headnums
        D = H // N
        print("BSH", tensor.shape)
        return tensor.reshape(B, S, N, D).transpose(0, 2, 1, 3), [B, N, S, D]
    elif inputLayout == "NSD":
        print("NSD")
        B = 1
        S = bsh_shape[1]
        N = bsh_shape[0]
        D = bsh_shape[2]
        H = D * N
        return tensor.reshape(B, N, S, D), [B, N, S, D]
    elif inputLayout == "BSND":
        print("BSND")
        B = bsh_shape[0]
        S = bsh_shape[1]
        N = bsh_shape[2]
        D = bsh_shape[3]
        return tensor.transpose(0, 2, 1, 3), [B, N, S, D]
    elif inputLayout == "TND" or inputLayout == "NTD_TND":
        B = len(actSeqLength)
        S = max(actSeqLength)
        N = bsh_shape[1]
        D = bsh_shape[2]
        if inputLayout == "NTD_TND":
            N = bsh_shape[0]
            tensor = tensor.transpose(1, 0, 2)
        new_tensor = np.zeros((B, N, S, D), dtype=tensor.dtype)
        t_start = 0
        for b_index in range(B):
            act_s = actSeqLength[b_index]
            t_end = t_start + act_s
            if act_s == 0:
                continue
            for n_index in range(N):
                new_tensor[b_index, n_index, 0:act_s, :] = tensor[
                    t_start:t_end, n_index, :
                ]
            t_start += act_s
        print(f"[TEMP]trans tnd 2 bnsd: {tensor.shape} -> {new_tensor.shape}")
        return new_tensor, [B, N, S, D]
    else:
        print(inputLayout)
        return tensor, bsh_shape


def alibi_biases(pre_shift_shape):
    # [0,  1, 2]
    # [-1, 0, 1]
    # [-2,-1, 0]
    q_seq = pre_shift_shape[2]
    kv_seq = pre_shift_shape[3]
    alibi_biases = np.zeros([q_seq, kv_seq])
    for x in range(0, kv_seq):
        for y in range(0, q_seq):
            alibi_biases[y, x] = abs(x - y - (kv_seq - q_seq)) * (-1)
    return alibi_biases


def get_all_alibi(
    numHeads, pre_shift_shape, pse_dtype=np.float32
):  # 输入是bnss或者是1nss B, Nq, Sq, Skv, pse_layout, pse_type, pse_dtype,pre_shift_shape
    m = get_slopes(numHeads)  # m  系数
    m = m.numpy()
    alibi_biase = alibi_biases(pre_shift_shape)
    pse_shift = np.zeros(pre_shift_shape, dtype=np.float32)

    for n in range(numHeads):
        pse_shift[:, n : n + 1, :, :] = alibi_biase * m[n]
    return pse_shift.astype(pse_dtype)


def _np_broadcast_pseShift_n(
    pse_shift_tensor, pse_shift_shape, q_batch
):  # pre_shift, pre_shift_shape, q_bnsd_shape[0]
    print(f"broadcast_mask_n:mask shape:{pse_shift_shape} q_batch:{q_batch}")
    # 1nss or bnss
    B_m = pse_shift_shape[0]
    if B_m == q_batch:
        return pse_shift_tensor
    else:
        B = q_batch
        # pse_res = np.zeros([B, pse_shift_shape[1],pse_shift_shape[2], pse_shift_shape[3]])
        pse_res = np.zeros(
            [
                B,
                pse_shift_tensor.shape[1],
                pse_shift_tensor.shape[2],
                pse_shift_tensor.shape[3],
            ],
            dtype=np.float32,
        )
        for i in range(B):
            pse_res[i : i + 1] = pse_shift_tensor[0]
        return pse_res


def _t_broadcast_pseShift_n(
    pse_shift_tensor, pse_shift_shape, q_batch
):  # pre_shift, pre_shift_shape, q_bnsd_shape[0]
    print(f"broadcast_mask_n:mask shape:{pse_shift_shape} q_batch:{q_batch}")
    # 1nss or bnss
    B_m = pse_shift_shape[0]
    if B_m == q_batch:
        return pse_shift_tensor
    else:
        B = q_batch
        pse_res = torch.zeros(
            [B, pse_shift_shape[1], pse_shift_shape[2], pse_shift_shape[3]]
        )
        for i in range(B):
            pse_res[i : i + 1] = pse_shift_tensor[0]
        return pse_res


def _np_broadcast_mask_n(m_tensor, m_shape, cpu_m_shape, numheads, q_batch):
    print(
        f"broadcast_mask_n:mask shape:{m_shape} with numheads:{numheads} q_batch:{q_batch}"
    )
    mask_cur_shape = cpu_m_shape
    if len(m_shape) == 4:
        # b1ss
        B_m = m_shape[0]
        if B_m != 1:
            B = B_m
        else:
            B = q_batch
        m_res = []
        for i in range(B):
            # mask_cur_shape = [m_shape[2], m_shape[3]]
            if B_m == 1:
                mask_cur = m_tensor[:, :, :].reshape(mask_cur_shape)
            else:
                mask_cur = m_tensor[i : i + 1, :, :, :].reshape(mask_cur_shape)
            m_res.append(mask_cur)
        return m_res
    elif len(m_shape) == 3:
        # bss
        B_m = m_shape[0]
        if B_m != 1:
            B = B_m
        else:
            B = q_batch
        m_res = []
        for i in range(B):
            # mask_cur_shape = [m_shape[1], m_shape[2]]
            if B_m == 1:
                mask_cur = m_tensor[:, :, :].reshape(mask_cur_shape)
            else:
                mask_cur = m_tensor[i : i + 1, :, :].reshape(mask_cur_shape)
            m_res.append(mask_cur)
        return m_res
    elif len(m_shape) == 2:
        # ss
        B = q_batch
        m_res = []
        for i in range(B):
            m_res.append(m_tensor)
        return m_res
    else:
        return m_tensor


def npSoftmax(x):
    x_max = x.max(axis=-1, keepdims=True)
    x_sub = x - x_max
    y = np.exp(x_sub)
    x_sum = y.sum(axis=-1, keepdims=True)
    ans = y / x_sum
    return ans, x_max, x_sum


def npSoftmax_new(x, sinks=None):
    x_max = x.max(axis=-1, keepdims=True)
    x_sub = x - x_max
    y = np.exp(x_sub)
    x_sum = y.sum(axis=-1, keepdims=True)
    if sinks is not None:
        sinks_reshaped = sinks.reshape(1, -1, 1, 1)
        sinks_sub = sinks_reshaped - x_max
        sink_exp = np.exp(sinks_sub)
        x_sum += sink_exp.sum(axis=-1, keepdims=True)
    ans = y
    return ans, x_max, x_sum


def softmax_flashv2(x, max_front=None, sum_front=None, update=None, is_fp16=False):
    """
    Compute the softmax function for each channel of the input x.
    """
    if update is None:
        if is_fp16:
            x = x.astype(np.float32)
        x_max = np.max(x, axis=-1, keepdims=True)
        x_sub = x - x_max  # -> x
        x_exp = np.exp(x_sub)  # -> x
        x_sum = np.sum(x_exp, axis=-1, keepdims=True)
        out = x_exp
        exp_max = None
        if is_fp16:
            out = out.astype(np.float16)

        return out, x_max, x_sum, exp_max
    else:
        if is_fp16:
            x = x.astype(np.float32)
            max_front = max_front.astype(np.float32)
            sum_front = sum_front.astype(np.float32)
        x_max_tmp = np.max(x, axis=-1, keepdims=True)  # tmp
        x_sub = x - x_max_tmp  # -> x
        x_exp = np.exp(x_sub)  # -> x
        x_sum = np.sum(x_exp, axis=-1, keepdims=True)  # tmp
        x_max = np.max(
            np.concatenate((max_front, x_max_tmp), axis=-1), axis=-1, keepdims=True
        )  # ->x_max
        x_exp_new = np.exp(x_max_tmp - x_max)  # -> x_max_tmp
        exp_max = np.exp(max_front - x_max)  # -> exp_max
        # update sum
        exp_max = exp_max * sum_front
        reduce_tmp = x_exp_new * x_sum
        x_sum = exp_max + reduce_tmp  # x_sum

        # exp_max = exp_max/x_sum
        exp_max = np.exp(max_front - x_max)
        out = x_exp * x_exp_new
        # out = x_exp*x_exp_new / x_sum

        # ### softmax ratio
        # softma_ratio = x_sum * x_exp_new / x_sum
        # out_new = out * softma_ratio
        # pround("out_new===================", out_new)
        if is_fp16:
            out = out.astype(np.float16)
            # x_max = x_max.astype(np.float16)
            # x_sum = x_sum.astype(np.float16)
            # exp_max = exp_max.astype(np.float16)

        return out, x_max, x_sum, exp_max


def _np_pfaattention_act_int8(
    q_tensor,
    k_tensor,
    v_tensor,
    pse_tensor,
    mask_tensor,
    scalar,
    act_seq,
    act_kv_seq,
    preTokens,
    nextTokens,
    pfa_param,
    dequant_scale1=None,
    dequant_scale2=None,
    quant_scale1=None,
    quant_scale2=None,
    quant_offset2=None,
    out_dtype="<class 'numpy.int8'>",
):
    S = None
    qs_begin = pfa_param["qs_begin"]
    qs_end = pfa_param["qs_end"]
    kvs_begin = pfa_param["kvs_begin"]
    kvs_end = pfa_param["kvs_end"]

    q_tensor = q_tensor[:, :, qs_begin:qs_end, :]
    k_tensor = k_tensor[:, :, kvs_begin:kvs_end, :]
    v_tensor = v_tensor[:, :, kvs_begin:kvs_end, :]

    S = kvs_end - kvs_begin

    if mask_tensor is not None:
        if (
            pfa_param["sparse"] == 2
            or pfa_param["sparse"] == 3
            or pfa_param["sparse"] == 4
        ):
            mask_tensor = mask_tensor[
                :, :, : (qs_end - qs_begin), : (kvs_end - kvs_begin)
            ]
        else:
            mask_tensor = mask_tensor[:, :, qs_begin:qs_end, kvs_begin:kvs_end]
    # 这里需要加paddingmask
    if pse_tensor is not None:
        pse_tensor = pse_tensor[:, :, qs_begin:qs_end, kvs_begin:kvs_end]

    qs = q_tensor.shape[2]
    one_loop_size = 1024
    max_range = (S + one_loop_size - 1) // one_loop_size

    # dequant 算子内部使用float19进行计算
    dequant_scale1.dtype = np.uint32
    dequant_scale1 = np.bitwise_and(dequant_scale1, 0xFFFFE000)
    dequant_scale1.dtype = np.float32

    data_length = one_loop_size
    max_front, sum_front, o_front = None, None, None
    for i in range(max_range):
        if i == S // one_loop_size:
            data_length = S % one_loop_size
        data_start = one_loop_size * i
        data_end = one_loop_size * i + data_length
        ki = k_tensor[:, :, data_start:data_end, :]
        vi = v_tensor[:, :, data_start:data_end, :]  # [2, 2, 128, 128]
        qDtype = q_tensor.dtype

        qk_i = np.matmul(
            q_tensor.astype(np.int32), ki.transpose(0, 1, 3, 2).astype(np.int32)
        )
        qk_i = qk_i.astype(np.float32) * dequant_scale1

        # 转成fp16防止溢出
        qk_i = np.where(qk_i > 65504, 65504, qk_i)
        qk_i = np.where(qk_i < -65504, -65504, qk_i)
        qk_i = qk_i.astype(np.float16)
        qk_i = qk_i * np.float16(scalar)

        # qk_i = qk_i*scalar
        if pse_tensor is not None:
            pse_tensor_i = pse_tensor[:, :, :, data_start:data_end]
            qk_i += pse_tensor_i

        if mask_tensor is not None:
            atten_mask_i = mask_tensor[:, :, :, data_start:data_end]
            # qk_i += atten_mask_i * (-10000.0)
            qk_i[atten_mask_i.astype(np.bool_)] = -65504
        if i == 0:
            py_s_i, max_i, sum_i, exp_max_i = softmax_flashv2(qk_i, is_fp16=True)
            py_s_i = py_s_i * quant_scale1.astype(np.float16)
            py_s_i = np.around(py_s_i)

            py_s_i = np.where(py_s_i > 127, 127, py_s_i)
            py_s_i = np.where(py_s_i < -128, -128, py_s_i)
            py_s_i = py_s_i.astype(np.int8)

            o_i = np.matmul(py_s_i.astype(np.int32), vi.astype(np.int32))
            o_i = o_i.astype(np.float32) * dequant_scale2
            max_front, sum_front, o_front = max_i, sum_i, o_i
        else:
            py_s_i, max_i, sum_i, exp_max_i = softmax_flashv2(
                qk_i, max_front, sum_front, update=True, is_fp16=True
            )
            py_s_i = py_s_i * quant_scale1.astype(np.float16)
            py_s_i = np.around(py_s_i)

            # 转成int8防止溢出
            py_s_i = np.where(py_s_i > 127, 127, py_s_i)
            py_s_i = np.where(py_s_i < -128, -128, py_s_i)
            py_s_i = py_s_i.astype(np.int8)

            o_i = np.matmul(py_s_i.astype(np.int32), vi.astype(np.int32))
            o_i = o_i.astype(np.float32) * dequant_scale2

            # 转成fp16防止溢出
            o_i = np.where(o_i > 65504, 65504, o_i)
            o_i = np.where(o_i < -65504, -65504, o_i)
            o_i = o_i.astype(np.float16)

            o_front_update = np.array(o_front * exp_max_i)
            o_i = o_front_update + np.array(o_i)
            max_front, sum_front, o_front = max_i, sum_i, o_i

    lse = None
    if pfa_param["lseflag"]:
        lse = np.log(sum_front) + max_front

    o_front = o_front / sum_front.astype(np.float16)
    if mask_tensor is not None:
        for i in range(mask_tensor.shape[2]):
            if mask_tensor[:, :, i, :].all() == 1:
                o_front[:, :, i, :] = 0
                if lse is not None:
                    lse[:, :, i, :] = np.inf

    if str(out_dtype) == "<class 'numpy.int8'>":
        o_front = o_front * quant_scale2.astype(np.float16)
        if quant_offset2 is not None:
            o_front += quant_offset2.astype(np.float16)
        o_front = np.around(o_front)
        # 转成int8防止溢出
        o_front = np.where(o_front > 127, 127, o_front)
        o_front = np.where(o_front < -128, -128, o_front)
        o_front = o_front.astype(np.int8)
    else:
        o_front = o_front.astype(np.float16)

    return o_front, lse


def _np_pfaattention_act_fp8(
    q_tensor,
    k_tensor,
    v_tensor,
    pse_tensor,
    mask_tensor,
    scalar,
    act_seq,
    act_kv_seq,
    preTokens,
    nextTokens,
    pfa_param,
    dequant_scale1=None,
    dequant_scale2=None,
    quant_scale1=None,
    quant_scale2=None,
    quant_offset2=None,
    out_dtype="<class 'numpy.int8'>",
):
    print(
        f"q_shape:{q_tensor.shape} k_shape:{k_tensor.shape} v_shape:{v_tensor.shape} scalar:{scalar} act_seq:{act_seq} act_kv_seq:{act_kv_seq}"
    )
    S = None
    qs_begin = pfa_param["qs_begin"]
    qs_end = pfa_param["qs_end"]
    kvs_begin = pfa_param["kvs_begin"]
    kvs_end = pfa_param["kvs_end"]

    q_tensor = q_tensor[:, :, qs_begin:qs_end, :]

    k_tensor = k_tensor[:, :, kvs_begin:kvs_end, :]
    v_tensor = v_tensor[:, :, kvs_begin:kvs_end, :]
    S = kvs_end - kvs_begin

    if mask_tensor is not None:
        print(f"mask_shape:{mask_tensor.shape}")
        if (
            pfa_param["sparse"] == 2
            or pfa_param["sparse"] == 3
            or pfa_param["sparse"] == 4
        ):
            mask_tensor = mask_tensor[
                :, :, : (qs_end - qs_begin), : (kvs_end - kvs_begin)
            ]
        else:
            mask_tensor = mask_tensor[:, :, qs_begin:qs_end, kvs_begin:kvs_end]
    # 这里需要加paddingmask
    if pse_tensor is not None:
        pse_tensor = pse_tensor[:, :, qs_begin:qs_end, kvs_begin:kvs_end]

    qs = q_tensor.shape[2]
    qd = q_tensor.shape[3]
    kd = k_tensor.shape[3]
    vd = v_tensor.shape[3]
    one_loop_size = 128
    if qd > 128 and kd == vd:
        one_loop_size = 64
    max_range = (S + one_loop_size - 1) // one_loop_size

    # dequant 算子内部使用float19进行计算
    dequant_scale1.dtype = np.uint32
    dequant_scale1 = np.bitwise_and(dequant_scale1, 0xFFFFE000)
    dequant_scale1.dtype = np.float32

    dequant_scale2.dtype = np.uint32
    dequant_scale2 = np.bitwise_and(dequant_scale2, 0xFFFFE000)
    dequant_scale2.dtype = np.float32

    data_length = one_loop_size
    max_front, sum_front, o_front = None, None, None
    for i in range(max_range):
        if i == S // one_loop_size:
            data_length = S % one_loop_size
        data_start = one_loop_size * i
        data_end = one_loop_size * i + data_length
        ki = k_tensor[:, :, data_start:data_end, :]
        vi = v_tensor[:, :, data_start:data_end, :]  # [2, 2, 128, 128]
        qDtype = q_tensor.dtype
        print("qDtype:", qDtype)
        print("start matmul1")
        qk_i = np.matmul(
            q_tensor.astype(np.float32), ki.transpose(0, 1, 3, 2).astype(np.float32)
        )
        # qk_i = qk_i.astype(np.float16) * dequant_scale1.astype(np.float16)
        qk_i = qk_i.astype(np.float32) * dequant_scale1
        print("end matmul1")

        qk_i = qk_i * np.float32(scalar)

        # qk_i = qk_i*scalar
        if pse_tensor is not None:
            pse_tensor_i = pse_tensor[:, :, :, data_start:data_end]
            qk_i += pse_tensor_i

        if mask_tensor is not None:
            atten_mask_i = mask_tensor[:, :, :, data_start:data_end]
            # qk_i += atten_mask_i * (-10000.0)
            qk_i[atten_mask_i.astype(np.bool_)] = -65504
        if i == 0:
            py_s_i, max_i, sum_i, exp_max_i = softmax_flashv2(qk_i, is_fp16=False)
            print("start matmul2")
            py_s_i = py_s_i * quant_scale1.astype(np.float32)

            if qDtype == "float8_e5m2":
                py_s_i = py_s_i.astype(float8_e5m2)
            else:
                py_s_i = py_s_i.astype(float8_e4m3fn)

            o_i = np.matmul(py_s_i.astype(np.float32), vi.astype(np.float32))
            print("end matmul2")
            # o_i = o_i.astype(np.float16) * dequant_scale2.astype(np.float16)
            o_i = o_i.astype(np.float32) * dequant_scale2
            max_front, sum_front, o_front = max_i, sum_i, o_i
        else:
            py_s_i, max_i, sum_i, exp_max_i = softmax_flashv2(
                qk_i, max_front, sum_front, update=True, is_fp16=False
            )
            print("start matmul2")
            py_s_i = py_s_i * quant_scale1.astype(np.float32)

            if qDtype == "float8_e5m2":
                py_s_i = py_s_i.astype(float8_e5m2)
            else:
                py_s_i = py_s_i.astype(float8_e4m3fn)

            print(py_s_i.dtype)

            # o_i = np.matmul(py_s_i.astype(np.float32), vi.astype(np.int8).astype(np.float32))
            o_i = np.matmul(py_s_i.astype(np.float32), vi.astype(np.float32))
            o_i = o_i.astype(np.float32) * dequant_scale2
            print("end matmul2")

            o_front_update = np.array(o_front * exp_max_i)
            o_i = o_front_update + np.array(o_i)
            max_front, sum_front, o_front = max_i, sum_i, o_i

    lse = None
    if pfa_param["lseflag"]:
        lse = np.log(sum_front) + max_front

    o_front = o_front / sum_front.astype(np.float32)
    if mask_tensor is not None:
        for i in range(mask_tensor.shape[2]):
            if mask_tensor[:, :, i, :].all() == 1:
                o_front[:, :, i, :] = 0
                if lse is not None:
                    lse[:, :, i, :] = np.inf

    if str(out_dtype) == "<class 'numpy.int8'>":
        o_front = o_front * quant_scale2.astype(np.float16)
        if quant_offset2 is not None:
            o_front += quant_offset2.astype(np.float16)
        o_front = np.around(o_front)
        # 转成int8防止溢出
        o_front = np.where(o_front > 127, 127, o_front)
        o_front = np.where(o_front < -128, -128, o_front)
        o_front = o_front.astype(np.int8)
    else:
        o_front = o_front.astype(np.float16)

    return o_front, lse


def _np_pfaattention_act_hifp8(
    q_tensor_hifp8,
    k_tensor_hifp8,
    v_tensor_hifp8,
    pse_tensor,
    mask_tensor,
    scalar,
    act_seq,
    act_kv_seq,
    preTokens,
    nextTokens,
    pfa_param,
    dequant_scale1=None,
    dequant_scale2=None,
    quant_scale1=None,
    quant_scale2=None,
    quant_offset2=None,
    out_dtype="<class 'numpy.int8'>",
):
    S = None
    q_tensor = trans_np_hifuint8_tensor_to_float32(q_tensor_hifp8).astype(np.float32)
    k_tensor = trans_np_hifuint8_tensor_to_float32(k_tensor_hifp8).astype(np.float32)
    v_tensor = trans_np_hifuint8_tensor_to_float32(v_tensor_hifp8).astype(np.float32)

    print(
        f"q_shape:{q_tensor.shape} k_shape:{k_tensor.shape} v_shape:{v_tensor.shape} scalar:{scalar} act_seq:{act_seq} act_kv_seq:{act_kv_seq}"
    )
    qs_begin = pfa_param["qs_begin"]
    qs_end = pfa_param["qs_end"]
    kvs_begin = pfa_param["kvs_begin"]
    kvs_end = pfa_param["kvs_end"]

    q_tensor = q_tensor[:, :, qs_begin:qs_end, :]

    k_tensor = k_tensor[:, :, kvs_begin:kvs_end, :]
    v_tensor = v_tensor[:, :, kvs_begin:kvs_end, :]
    S = kvs_end - kvs_begin

    if mask_tensor is not None:
        print(f"mask_shape:{mask_tensor.shape}")
        if (
            pfa_param["sparse"] == 2
            or pfa_param["sparse"] == 3
            or pfa_param["sparse"] == 4
        ):
            mask_tensor = mask_tensor[
                :, :, : (qs_end - qs_begin), : (kvs_end - kvs_begin)
            ]
        else:
            mask_tensor = mask_tensor[:, :, qs_begin:qs_end, kvs_begin:kvs_end]
    # 这里需要加paddingmask
    if pse_tensor is not None:
        pse_tensor = pse_tensor[:, :, qs_begin:qs_end, kvs_begin:kvs_end]

    qs = q_tensor.shape[2]
    qd = q_tensor.shape[3]
    kd = k_tensor.shape[3]
    vd = v_tensor.shape[3]
    one_loop_size = 128
    if qd > 128 and kd == vd:
        one_loop_size = 64
    max_range = (S + one_loop_size - 1) // one_loop_size

    # dequant 算子内部使用float19进行计算
    dequant_scale1.dtype = np.uint32
    dequant_scale1 = np.bitwise_and(dequant_scale1, 0xFFFFE000)
    dequant_scale1.dtype = np.float32

    dequant_scale2.dtype = np.uint32
    dequant_scale2 = np.bitwise_and(dequant_scale2, 0xFFFFE000)
    dequant_scale2.dtype = np.float32

    data_length = one_loop_size
    max_front, sum_front, o_front = None, None, None
    for i in range(max_range):
        if i == S // one_loop_size:
            data_length = S % one_loop_size
        data_start = one_loop_size * i
        data_end = one_loop_size * i + data_length
        ki = k_tensor[:, :, data_start:data_end, :]
        vi = v_tensor[:, :, data_start:data_end, :]  # [2, 2, 128, 128]
        qDtype = q_tensor.dtype
        print("start matmul1")
        qk_i = np.matmul(
            q_tensor.astype(np.float32), ki.transpose(0, 1, 3, 2).astype(np.float32)
        )

        # qk_i = qk_i.astype(np.float16) * dequant_scale1.astype(np.float16)
        qk_i = qk_i.astype(np.float32) * dequant_scale1
        print("end matmul1")

        qk_i = qk_i.astype(np.float32)
        qk_i = qk_i * np.float32(scalar)

        # qk_i = qk_i*scalar
        if pse_tensor is not None:
            pse_tensor_i = pse_tensor[:, :, :, data_start:data_end]
            qk_i += pse_tensor_i

        if mask_tensor is not None:
            atten_mask_i = mask_tensor[:, :, :, data_start:data_end]
            # qk_i += atten_mask_i * (-10000.0)
            qk_i[atten_mask_i.astype(np.bool_)] = -65504
        if i == 0:
            py_s_i, max_i, sum_i, exp_max_i = softmax_flashv2(qk_i, is_fp16=False)
            print("start matmul2")
            py_s_i = py_s_i * quant_scale1.astype(np.float32)

            py_s_i = trans_np_float_tensor_to_hifuint8(
                in_tensor=py_s_i, round_mode="hybrid", over_mode=True
            )
            py_s_i = trans_np_hifuint8_tensor_to_float32(py_s_i).astype(np.float32)

            o_i = np.matmul(py_s_i.astype(np.float32), vi.astype(np.float32))
            print("end matmul2")
            # o_i = o_i.astype(np.float16) * dequant_scale2.astype(np.float16)
            o_i = o_i.astype(np.float32) * dequant_scale2
            max_front, sum_front, o_front = max_i, sum_i, o_i
        else:
            py_s_i, max_i, sum_i, exp_max_i = softmax_flashv2(
                qk_i, max_front, sum_front, update=True, is_fp16=False
            )
            print("start matmul2")
            py_s_i = py_s_i * quant_scale1.astype(np.float32)

            py_s_i = trans_np_float_tensor_to_hifuint8(
                in_tensor=py_s_i, round_mode="hybrid", over_mode=True
            )
            py_s_i = trans_np_hifuint8_tensor_to_float32(py_s_i).astype(np.float32)

            # o_i = np.matmul(py_s_i.astype(np.float32), vi.astype(np.int8).astype(np.float32))
            o_i = np.matmul(py_s_i.astype(np.float32), vi.astype(np.float32))
            o_i = o_i.astype(np.float32) * dequant_scale2
            print("end matmul2")

            o_front_update = np.array(o_front * exp_max_i)
            o_i = o_front_update + np.array(o_i)
            max_front, sum_front, o_front = max_i, sum_i, o_i

    lse = None
    if pfa_param["lseflag"]:
        lse = np.log(sum_front) + max_front

    o_front = o_front / sum_front.astype(np.float32)
    if mask_tensor is not None:
        for i in range(mask_tensor.shape[2]):
            if mask_tensor[:, :, i, :].all() == 1:
                o_front[:, :, i, :] = 0
                if lse is not None:
                    lse[:, :, i, :] = np.inf

    if str(out_dtype) == "<class 'numpy.int8'>":
        o_front = o_front * quant_scale2.astype(np.float16)
        if quant_offset2 is not None:
            o_front += quant_offset2.astype(np.float16)
        o_front = np.around(o_front)
        # 转成int8防止溢出
        o_front = np.where(o_front > 127, 127, o_front)
        o_front = np.where(o_front < -128, -128, o_front)
        o_front = o_front.astype(np.int8)
    else:
        o_front = o_front.astype(np.float16)

    return o_front, lse


def _t_pfaattention_act_innerprecise(
    q_tensor,
    k_tensor,
    v_tensor,
    pse_tensor,
    mask_tensor,
    scalar,
    act_seq,
    act_kv_seq,
    preTokens,
    nextTokens,
    dequant_scale1=None,
    dequant_scale2=None,
    quant_scale1=None,
    quant_scale2=None,
    quant_offset2=None,
):
    print(
        f"q_shape:{q_tensor.shape} k_shape:{k_tensor.shape} v_shape:{v_tensor.shape} scalar:{scalar} act_seq:{act_seq} act_kv_seq:{act_kv_seq}"
    )
    if act_seq is not None:
        q_tensor = q_tensor[:, :, :act_seq, :]
    if act_kv_seq is not None:
        k_tensor = k_tensor[:, :, :act_kv_seq, :]
        v_tensor = v_tensor[:, :, :act_kv_seq, :]

    q_t = q_tensor.to(torch.float32)
    k_t = k_tensor.to(torch.float32)
    v_t = v_tensor.to(torch.float32)

    q_tensor = q_t.cuda().requires_grad_(False)
    k_tensor = k_t.cuda().requires_grad_(False)
    v_tensor = v_t.cuda().requires_grad_(False)

    print("start matmul1")

    qkBmmRes = torch.matmul(q_tensor, k_tensor.permute([0, 1, 3, 2]))
    print("end matmul1")
    qkEleRes = qkBmmRes * scalar

    # 这里需要加paddingmask
    if pse_tensor is not None:
        if act_seq is not None:
            pse_tensor = pse_tensor[:, :, :act_seq, :]
        if act_kv_seq is not None:
            pse_tensor = pse_tensor[:, :, :act_seq, :act_kv_seq]
        qkEleRes += pse_tensor

    if mask_tensor is not None:
        print(f"mask_shape:{mask_tensor.shape}")
        if act_seq is not None:
            mask_tensor = mask_tensor[:, :, :act_seq, :]
        if act_kv_seq is not None:
            mask_tensor = mask_tensor[:, :, :act_seq, :act_kv_seq]

        # mask_t = torch.from_numpy(mask_tensor)
        # mask_tensor = mask_tensor.to(bool)
        # mask_tensor = mask_tensor.cuda().requires_grad_(False)
        qkEleRes[mask_tensor.to(bool)] = -1.7 * 10**38
        # qkEleRes += mask_tensor * (-1000000.0)
    softmax_res, x_max, x_sum = _t_softmax(qkEleRes)
    if q_tensor.shape[2] > k_tensor.shape[2] + preTokens:
        ss0 = k_tensor.shape[2] + preTokens
        softmax_res[:, :, ss0:, :] = 0
    if nextTokens < 0:
        ss1 = -nextTokens
        softmax_res[:, :, :ss1, :] = 0
    print("start matmul2")
    bmm2Res = torch.matmul(
        softmax_res.to(torch.float16).to(torch.float32), v_tensor
    ).cpu()
    print("end matmul2")
    print(f"return shape:{bmm2Res.shape}")
    return bmm2Res


def _t_pfaattention_act(
    q_tensor,
    k_tensor,
    v_tensor,
    pse_tensor,
    mask_tensor,
    scalar,
    act_seq,
    act_kv_seq,
    preTokens,
    nextTokens,
    dequant_scale1=None,
    dequant_scale2=None,
    quant_scale1=None,
    quant_scale2=None,
    quant_offset2=None,
):
    print(
        f"q_shape:{q_tensor.shape} k_shape:{k_tensor.shape} v_shape:{v_tensor.shape} scalar:{scalar} act_seq:{act_seq} act_kv_seq:{act_kv_seq}"
    )
    if act_seq is not None:
        q_tensor = q_tensor[:, :, :act_seq, :]
    if act_kv_seq is not None:
        k_tensor = k_tensor[:, :, :act_kv_seq, :]
        v_tensor = v_tensor[:, :, :act_kv_seq, :]

    q_t = q_tensor.to(torch.float32)
    k_t = k_tensor.to(torch.float32)
    v_t = v_tensor.to(torch.float32)

    q_tensor = q_t.cuda().requires_grad_(False)
    k_tensor = k_t.cuda().requires_grad_(False)
    v_tensor = v_t.cuda().requires_grad_(False)

    print("start matmul1")

    qkBmmRes = torch.matmul(q_tensor, k_tensor.permute([0, 1, 3, 2]))
    print("end matmul1")
    qkEleRes = qkBmmRes * scalar

    # 这里需要加paddingmask
    if pse_tensor is not None:
        if act_seq is not None:
            pse_tensor = pse_tensor[:, :, :act_seq, :]
        if act_kv_seq is not None:
            pse_tensor = pse_tensor[:, :, :act_seq, :act_kv_seq]
        qkEleRes += pse_tensor

    if mask_tensor is not None:
        print(f"mask_shape:{mask_tensor.shape}")
        if act_seq is not None:
            mask_tensor = mask_tensor[:, :, :act_seq, :]
        if act_kv_seq is not None:
            mask_tensor = mask_tensor[:, :, :act_seq, :act_kv_seq]

        # mask_t = torch.from_numpy(mask_tensor)
        # mask_tensor = mask_tensor.to(bool)
        # mask_tensor = mask_tensor.cuda().requires_grad_(False)
        qkEleRes[mask_tensor.to(bool)] = -1.7 * 10**38
        # qkEleRes += mask_tensor * (-1000000.0)
    softmax_res, x_max, x_sum = _t_softmax(qkEleRes)
    if q_tensor.shape[2] > k_tensor.shape[2] + preTokens:
        ss0 = k_tensor.shape[2] + preTokens
        softmax_res[:, :, ss0:, :] = 0
    if nextTokens < 0:
        ss1 = -nextTokens
        softmax_res[:, :, :ss1, :] = 0
    print("start matmul2")
    bmm2Res = torch.matmul(
        softmax_res.to(torch.bfloat16).to(torch.float32), v_tensor
    ).cpu()
    print("end matmul2")
    print(f"return shape:{bmm2Res.shape}")
    return bmm2Res


def _np_pfaattention_act(
    q_tensor,
    k_tensor,
    v_tensor,
    pse_tensor,
    mask_tensor,
    scalar,
    act_seq,
    act_kv_seq,
    preTokens,
    nextTokens,
    pfa_param,
    quant_scale2=None,
    quant_offset2=None,
    antiquant_scale_k=None,
    antiquant_scale_v=None,
    antiquant_offset_k=None,
    antiquant_offset_v=None,
    out_dtype="<class 'numpy.int8'>",
    sinks=None,
):
    print(
        f"q_shape:{q_tensor.shape} k_shape:{k_tensor.shape} v_shape:{v_tensor.shape} scalar:{scalar} act_seq:{act_seq} act_kv_seq:{act_kv_seq} preTokens:{preTokens}nextTokens:{nextTokens}"
    )

    # qs_begin = 0
    # qs_end = q_tensor.shape[2]
    # kvs_begin = 0
    # kvs_end = k_tensor.shape[2]
    qs_begin = pfa_param["qs_begin"]
    qs_end = pfa_param["qs_end"]
    kvs_begin = pfa_param["kvs_begin"]
    kvs_end = pfa_param["kvs_end"]

    qdtype = pfa_param["q_dtype"]
    q_tensor = q_tensor[:, :, qs_begin:qs_end, :]

    k_tensor = k_tensor[:, :, kvs_begin:kvs_end, :]
    v_tensor = v_tensor[:, :, kvs_begin:kvs_end, :]
    qDtype = q_tensor.dtype
    # KVcahce反量化  这块逻辑需要修改
    if k_tensor.dtype == np.int8:
        k_tensor = k_tensor.astype(np.float16)
        v_tensor = v_tensor.astype(np.float16)
        if antiquant_offset_k is not None:
            if pfa_param["anti_or_kvanti"] == 2 and pfa_param["k_antiquant_mode"] == 1:
                antiquant_offset_k = np.expand_dims(
                    antiquant_offset_k[:, kvs_begin:kvs_end], axis=-1
                )
                antiquant_offset_v = np.expand_dims(
                    antiquant_offset_v[:, kvs_begin:kvs_end], axis=-1
                )
            k_tensor += antiquant_offset_k
            v_tensor += antiquant_offset_v
        if pfa_param["anti_or_kvanti"] == 2 and pfa_param["k_antiquant_mode"] == 1:
            antiquant_scale_k = np.expand_dims(
                antiquant_scale_k[:, kvs_begin:kvs_end], axis=-1
            )
            antiquant_scale_v = np.expand_dims(
                antiquant_scale_v[:, kvs_begin:kvs_end], axis=-1
            )
        k_tensor *= antiquant_scale_k
        v_tensor *= antiquant_scale_v
    print("start matmul1")
    qkBmmRes = np.matmul(q_tensor, k_tensor.transpose([0, 1, 3, 2]))
    print("end matmul1")
    qkEleRes = qkBmmRes * scalar

    # 这里需要加paddingmask
    if pse_tensor is not None:
        pse_tensor = pse_tensor[:, :, qs_begin:qs_end, kvs_begin:kvs_end]
        print("pse_tensor", pse_tensor.shape)
        qkEleRes += pse_tensor.astype(np.float32)

    if mask_tensor is not None:
        print(f"mask_shape:{mask_tensor.shape}")
        if (
            pfa_param["sparse"] == 2
            or pfa_param["sparse"] == 3
            or pfa_param["sparse"] == 4
        ):
            # actul_prefix = 0
            # if 'actualprefixKV' in pfa_param:
            #     actul_prefix =  pfa_param['actualprefixKV']
            mask_tensor = mask_tensor[
                :, :, : (qs_end - qs_begin), : (kvs_end - kvs_begin)
            ]
        else:
            mask_tensor = mask_tensor[:, :, qs_begin:qs_end, kvs_begin:kvs_end]
        qkEleRes[mask_tensor.astype(np.bool_)] = -1.7 * 10**38
        # qkEleRes += mask_tensor * (-1000000.0)
    softmax_res, x_max, x_sum = npSoftmax_new(qkEleRes, sinks)

    lse = None
    if "lseflag" in pfa_param and pfa_param["lseflag"]:
        lse = np.log(x_sum) + x_max

    print("start matmul2")
    if qdtype == np.float16:
        bmm2Res = np.matmul(softmax_res.astype(np.float16).astype(np.float32), v_tensor)
    elif qdtype == np.float32:
        bmm2Res = np.matmul(softmax_res, v_tensor)
    else:
        bmm2Res = np.matmul(
            softmax_res.astype(tf.bfloat16.as_numpy_dtype).astype(np.float32), v_tensor
        )
    print("end matmul2")

    # /sum
    bmm2Res = bmm2Res / x_sum

    # 20240521新修改
    if mask_tensor is not None:
        for i in range(mask_tensor.shape[2]):
            if mask_tensor[:, :, i, :].all() == 1:
                bmm2Res[:, :, i, :] = 0
                if lse is not None:
                    lse[:, :, i, :] = np.inf

    if str(out_dtype) == "<class 'numpy.int8'>":
        bmm2Res = bmm2Res * quant_scale2.astype(np.float16)
        if quant_offset2 is not None:
            bmm2Res += quant_offset2.astype(np.float16)
        bmm2Res = np.around(bmm2Res)
        # 转成int8防止溢出
        bmm2Res = np.where(bmm2Res > 127, 127, bmm2Res)
        bmm2Res = np.where(bmm2Res < -128, -128, bmm2Res)
        bmm2Res = bmm2Res.astype(np.int8)

    print(f"return shape:{bmm2Res.shape}")
    return bmm2Res, lse


def _t_promtattention_bnsd(
    q_tensor,
    q_shape,
    k_tensor,
    k_shape,
    v_tensor,
    v_shape,
    pse_tensor,
    mask_tensor,
    scale,
    actseqlens,
    actkvseqlens,
    preTokens,
    nextTokens,
    dequant_scale1=None,
    dequant_scale2=None,
    quant_scale1=None,
    quant_scale2=None,
    quant_offset2=None,
    out_dtype="<class 'numpy.int8'>",
    innerprecise=1,
):
    batch_value = q_shape[0]
    numhead_value = q_shape[1]
    actseqlens_size = len(actseqlens)
    print(f"B:{batch_value}  actseqlens:{actseqlens_size}")
    if 0 in q_shape:
        print(f"q shape:{q_shape} is empty,skip")
        return
    y = torch.zeros(q_shape, dtype=torch.float32)
    for b_index in range(batch_value):
        if len(actseqlens) == 0:
            act_seq = None
        else:
            act_seq = int(actseqlens[b_index])

        if len(actkvseqlens) == 0:
            act_kv_seq = None
        else:
            act_kv_seq = int(actkvseqlens[b_index])
        if act_seq == 0 or act_kv_seq == 0 or 0 in k_shape or 0 in v_shape:
            continue
        # bnsd
        for n_index in range(numhead_value):
            if len(q_shape) == 4 and len(k_shape) == 4 and len(v_shape) == 4:
                q_tensor_cur = q_tensor[
                    b_index : (b_index + 1), n_index : (n_index + 1), :, :
                ]
                k_tensor_cur = k_tensor[
                    b_index : (b_index + 1), n_index : (n_index + 1), :, :
                ]
                v_tensor_cur = v_tensor[
                    b_index : (b_index + 1), n_index : (n_index + 1), :, :
                ]
                if mask_tensor is None:
                    mask_cur = None
                else:
                    mask_cur = mask_tensor[
                        b_index : (b_index + 1), n_index : (n_index + 1), :, :
                    ]

                if pse_tensor is None:
                    pse_cur = None
                else:
                    pse_cur = pse_tensor[
                        b_index : (b_index + 1), n_index : (n_index + 1), :, :
                    ]

                if act_seq is None:
                    if innerprecise == 0:
                        y[b_index : (b_index + 1), n_index : (n_index + 1), :, :] = (
                            _t_pfaattention_act_innerprecise(
                                q_tensor_cur,
                                k_tensor_cur,
                                v_tensor_cur,
                                pse_cur,
                                mask_cur,
                                scale,
                                act_seq,
                                act_kv_seq,
                                preTokens,
                                nextTokens,
                                dequant_scale1,
                                dequant_scale2,
                                quant_scale1,
                                quant_scale2,
                                quant_offset2,
                            )
                        )
                    else:
                        y[b_index : (b_index + 1), n_index : (n_index + 1), :, :] = (
                            _t_pfaattention_act(
                                q_tensor_cur,
                                k_tensor_cur,
                                v_tensor_cur,
                                pse_cur,
                                mask_cur,
                                scale,
                                act_seq,
                                act_kv_seq,
                                preTokens,
                                nextTokens,
                                dequant_scale1,
                                dequant_scale2,
                                quant_scale1,
                                quant_scale2,
                                quant_offset2,
                            )
                        )
                else:
                    if innerprecise == 0:
                        y[
                            b_index : (b_index + 1),
                            n_index : (n_index + 1),
                            :act_seq,
                            :,
                        ] = _t_pfaattention_act_innerprecise(
                            q_tensor_cur,
                            k_tensor_cur,
                            v_tensor_cur,
                            pse_cur,
                            mask_cur,
                            scale,
                            act_seq,
                            act_kv_seq,
                            preTokens,
                            nextTokens,
                            dequant_scale1,
                            dequant_scale2,
                            quant_scale1,
                            quant_scale2,
                            quant_offset2,
                        )
                    else:
                        y[
                            b_index : (b_index + 1),
                            n_index : (n_index + 1),
                            :act_seq,
                            :,
                        ] = _t_pfaattention_act(
                            q_tensor_cur,
                            k_tensor_cur,
                            v_tensor_cur,
                            pse_cur,
                            mask_cur,
                            scale,
                            act_seq,
                            act_kv_seq,
                            preTokens,
                            nextTokens,
                            dequant_scale1,
                            dequant_scale2,
                            quant_scale1,
                            quant_scale2,
                            quant_offset2,
                        )

    return y


def _np_promtattention_bnsd(
    q_tensor,
    q_shape,
    k_tensor_list,
    k_shape_list,
    v_tensor_list,
    v_shape_list,
    pse_bnsd_tensor,
    mask_tensor,
    scale,
    actseqlens,
    actkvseqlens,
    preTokens_list,
    nextTokens_list,
    sparse,
    pfa_param,
    dequant_scale1=None,
    dequant_scale2=None,
    quant_scale1=None,
    quant_scale2=None,
    quant_offset2=None,
    antiquant_scale=None,
    antiquant_offset=None,
    out_dtype="<class 'numpy.int8'>",
    q_dtype_origin="origin",
    sinks=None,
):
    batch_value = q_shape[0]
    numhead_value = q_shape[1]
    actseqlens_size = len(actseqlens)
    print(f"B:{batch_value}  actseqlens:{actseqlens_size}")
    # y = np.zeros(q_shape, dtype=np.float32)
    outShape = copy.deepcopy(q_shape)
    outShape[-1] = v_shape_list[0][-1]
    y = np.zeros(outShape, dtype=np.float32)
    lse = np.full(pfa_param["lseshape"], np.inf)
    preTokens = preTokens_list
    nextTokens = nextTokens_list
    tensor_list_flag = len(k_tensor_list)
    if mask_tensor is not None:
        mask_cur = np.zeros([1, 1, q_shape[2], len(mask_tensor[0][0])], dtype="uint8")
    if tensor_list_flag == 1:
        k_tensor = k_tensor_list[0]
        k_shape = k_shape_list[0]
        v_tensor = v_tensor_list[0]
        v_shape = v_shape_list[0]
    for b_index in range(batch_value):
        if tensor_list_flag != 1:
            k_tensor = k_tensor_list[b_index]
            k_shape = k_shape_list[b_index]
            v_tensor = v_tensor_list[b_index]
            v_shape = v_shape_list[b_index]

        if len(actseqlens) == 0:
            act_seq = None
        else:
            act_seq = int(actseqlens[b_index])

        if len(actkvseqlens) == 0:
            act_kv_seq = None
        else:
            act_kv_seq = int(actkvseqlens[b_index])
        if act_seq == 0 or act_kv_seq == 0 or 0 in k_shape or 0 in v_shape:
            continue

        qs_begin = 0
        qs_end = q_tensor.shape[2]
        kvs_begin = 0
        kvs_end = k_tensor.shape[2]
        if "queryPaddingSize" in pfa_param:
            qs_begin = int(
                q_tensor.shape[2] - act_seq - pfa_param["queryPaddingSize"][0]
            )
            qs_end = int(q_tensor.shape[2] - pfa_param["queryPaddingSize"][0])
            print(f"query_left padding--- s_begin:{qs_begin}, s_end:{qs_end}")

        else:
            if act_seq is not None:
                qs_end = act_seq
        if "kvPaddingSize" in pfa_param:
            kvs_begin = int(
                k_tensor.shape[2] - act_kv_seq - pfa_param["kvPaddingSize"][0]
            )
            kvs_end = int(k_tensor.shape[2] - pfa_param["kvPaddingSize"][0])
            print(f"kv_left padding--- s_begin:{kvs_begin}, s_end:{kvs_end}")
        else:
            if act_kv_seq is not None:
                kvs_end = act_kv_seq
        pfa_param["qs_begin"] = qs_begin
        pfa_param["qs_end"] = qs_end
        pfa_param["kvs_begin"] = kvs_begin
        pfa_param["kvs_end"] = kvs_end
        if "actualprefixKV" in pfa_param:
            if pfa_param["actualprefixKV"] == 0:
                pfa_param["kvs_end"] += pfa_param["shared_prefix_k"].shape[2]
            else:
                pfa_param["kvs_end"] += pfa_param["actualprefixKV"]
        pfa_param["sparse"] = sparse

        # bnsd
        if sparse == 4:
            preTokens = (
                preTokens_list[b_index]
                if isinstance(preTokens_list, list)
                else preTokens_list
            )
        if sparse == 3 or sparse == 4:
            nextTokens = (
                nextTokens_list[b_index]
                if isinstance(nextTokens_list, list)
                else nextTokens_list
            )

        for n_index in range(numhead_value):
            if len(q_shape) == 4 and len(k_shape) == 4 and len(v_shape) == 4:
                q_tensor_cur = q_tensor[
                    b_index : (b_index + 1), n_index : (n_index + 1), :, :
                ]
                if tensor_list_flag == 1:
                    k_tensor_cur = k_tensor[
                        b_index : (b_index + 1), n_index : (n_index + 1), :, :
                    ]
                    v_tensor_cur = v_tensor[
                        b_index : (b_index + 1), n_index : (n_index + 1), :, :
                    ]
                else:
                    k_tensor_cur = k_tensor[:, n_index : (n_index + 1), :, :]
                    v_tensor_cur = v_tensor[:, n_index : (n_index + 1), :, :]

                if "shared_prefix_k" in pfa_param:
                    actual_prefix_len = pfa_param["actualprefixKV"]
                    if actual_prefix_len == 0:
                        shared_prefix_k = pfa_param["shared_prefix_k"][
                            :, n_index : (n_index + 1), :, :
                        ]
                        shared_prefix_v = pfa_param["shared_prefix_v"][
                            :, n_index : (n_index + 1), :, :
                        ]
                    else:
                        shared_prefix_k = pfa_param["shared_prefix_k"][
                            :, n_index : (n_index + 1), :actual_prefix_len, :
                        ]
                        shared_prefix_v = pfa_param["shared_prefix_v"][
                            :, n_index : (n_index + 1), :actual_prefix_len, :
                        ]
                    k_tensor_cur = np.append(shared_prefix_k, k_tensor_cur, axis=2)
                    v_tensor_cur = np.append(shared_prefix_v, v_tensor_cur, axis=2)

                if mask_tensor is None:
                    mask_cur = None
                else:
                    mask_cur[0, 0, :, :] = mask_tensor[b_index]
                    # mask_cur = mask_tensor[b_index:(b_index + 1), n_index:(n_index + 1), :, :]

                if pse_bnsd_tensor is None:
                    pse_cur = None
                else:
                    pse_cur = pse_bnsd_tensor[
                        b_index : (b_index + 1), n_index : (n_index + 1), :, :
                    ]

                quant_scale2_cur = None
                quant_offset2_cur = None
                if quant_scale2 is not None:
                    if len(quant_scale2.shape) == 1:
                        quant_scale2_cur = quant_scale2
                    else:
                        quant_scale2_cur = quant_scale2[
                            :, n_index : (n_index + 1), :, :
                        ]
                    if quant_offset2 is not None:
                        if len(quant_offset2.shape) == 1:
                            quant_offset2_cur = quant_offset2
                        else:
                            quant_offset2_cur = quant_offset2[
                                :, n_index : (n_index + 1), :, :
                            ]

                antiquant_scale_k = None
                antiquant_scale_v = None
                antiquant_offset_k = None
                antiquant_offset_v = None
                if k_tensor_cur.dtype == np.int8 and q_tensor_cur.dtype != np.int8:
                    if pfa_param["anti_or_kvanti"] == 1:
                        if len(antiquant_scale.shape) == 1:
                            antiquant_scale_k = antiquant_scale[0]
                            antiquant_scale_v = antiquant_scale[1]
                        else:
                            antiquant_scale_k = antiquant_scale[
                                0, n_index : (n_index + 1), :, :
                            ]
                            antiquant_scale_v = antiquant_scale[
                                1, n_index : (n_index + 1), :, :
                            ]
                        if antiquant_offset is not None:
                            if len(antiquant_offset.shape) == 1:
                                antiquant_offset_k = antiquant_offset[0]
                                antiquant_offset_v = antiquant_offset[1]
                            else:
                                antiquant_offset_k = antiquant_offset[
                                    0, n_index : (n_index + 1), :, :
                                ]
                                antiquant_offset_v = antiquant_offset[
                                    1, n_index : (n_index + 1), :, :
                                ]
                    if pfa_param["anti_or_kvanti"] == 2:  # 分离量化参数
                        if pfa_param["k_antiquant_mode"] == 0:
                            if len(pfa_param["k_antiquant_scale"].shape) == 1:
                                antiquant_scale_k = pfa_param["k_antiquant_scale"]
                                antiquant_scale_v = pfa_param["v_antiquant_scale"]
                            else:
                                antiquant_scale_k = pfa_param["k_antiquant_scale"][
                                    n_index : (n_index + 1), :, :
                                ]
                                antiquant_scale_v = pfa_param["v_antiquant_scale"][
                                    n_index : (n_index + 1), :, :
                                ]
                            if pfa_param["k_antiquant_offset"] is not None:
                                if (
                                    len(pfa_param["k_antiquant_offset"].shape) == 1
                                ):  # petensor
                                    antiquant_offset_k = pfa_param["k_antiquant_offset"]
                                    antiquant_offset_v = pfa_param["v_antiquant_offset"]
                                else:
                                    antiquant_offset_k = pfa_param[
                                        "k_antiquant_offset"
                                    ][n_index : (n_index + 1), :, :]
                                    antiquant_offset_v = pfa_param[
                                        "v_antiquant_offset"
                                    ][n_index : (n_index + 1), :, :]
                        if pfa_param["k_antiquant_mode"] == 1:
                            antiquant_scale_k = pfa_param["k_antiquant_scale"][
                                b_index : (b_index + 1), :
                            ]
                            antiquant_scale_v = pfa_param["v_antiquant_scale"][
                                b_index : (b_index + 1), :
                            ]
                            if pfa_param["k_antiquant_offset"] is not None:
                                antiquant_offset_k = pfa_param["k_antiquant_offset"][
                                    b_index : (b_index + 1), :
                                ]
                                antiquant_offset_v = pfa_param["v_antiquant_offset"][
                                    b_index : (b_index + 1), :
                                ]

                sinks_cur = None
                if sinks is not None:
                    sinks_cur = sinks[n_index : (n_index + 1)]

                if q_tensor.dtype == "int8":
                    (
                        y[
                            b_index : (b_index + 1),
                            n_index : (n_index + 1),
                            qs_begin:qs_end,
                            :,
                        ],
                        lse[
                            b_index : (b_index + 1),
                            n_index : (n_index + 1),
                            qs_begin:qs_end,
                            :,
                        ],
                    ) = _np_pfaattention_act_int8(
                        q_tensor_cur,
                        k_tensor_cur,
                        v_tensor_cur,
                        pse_cur,
                        mask_cur,
                        scale,
                        act_seq,
                        act_kv_seq,
                        preTokens,
                        nextTokens,
                        pfa_param,
                        dequant_scale1,
                        dequant_scale2,
                        quant_scale1,
                        quant_scale2_cur,
                        quant_offset2_cur,
                        out_dtype,
                    )
                elif (
                    q_tensor.dtype == "float8_e5m2" or q_tensor.dtype == "float8_e4m3fn"
                ):
                    (
                        y[
                            b_index : (b_index + 1),
                            n_index : (n_index + 1),
                            qs_begin:qs_end,
                            :,
                        ],
                        lse[
                            b_index : (b_index + 1),
                            n_index : (n_index + 1),
                            qs_begin:qs_end,
                            :,
                        ],
                    ) = _np_pfaattention_act_fp8(
                        q_tensor_cur,
                        k_tensor_cur,
                        v_tensor_cur,
                        pse_cur,
                        mask_cur,
                        scale,
                        act_seq,
                        act_kv_seq,
                        preTokens,
                        nextTokens,
                        pfa_param,
                        dequant_scale1,
                        dequant_scale2,
                        quant_scale1,
                        quant_scale2_cur,
                        quant_offset2_cur,
                        out_dtype,
                    )
                elif q_dtype_origin == "hifloat8":
                    (
                        y[
                            b_index : (b_index + 1),
                            n_index : (n_index + 1),
                            qs_begin:qs_end,
                            :,
                        ],
                        lse[
                            b_index : (b_index + 1),
                            n_index : (n_index + 1),
                            qs_begin:qs_end,
                            :,
                        ],
                    ) = _np_pfaattention_act_hifp8(
                        q_tensor_cur,
                        k_tensor_cur,
                        v_tensor_cur,
                        pse_cur,
                        mask_cur,
                        scale,
                        act_seq,
                        act_kv_seq,
                        preTokens,
                        nextTokens,
                        pfa_param,
                        dequant_scale1,
                        dequant_scale2,
                        quant_scale1,
                        quant_scale2_cur,
                        quant_offset2_cur,
                        out_dtype,
                    )
                else:
                    (
                        y[
                            b_index : (b_index + 1),
                            n_index : (n_index + 1),
                            qs_begin:qs_end,
                            :,
                        ],
                        lse[
                            b_index : (b_index + 1),
                            n_index : (n_index + 1),
                            qs_begin:qs_end,
                            :,
                        ],
                    ) = _np_pfaattention_act(
                        q_tensor_cur,
                        k_tensor_cur,
                        v_tensor_cur,
                        pse_cur,
                        mask_cur,
                        scale,
                        act_seq,
                        act_kv_seq,
                        preTokens,
                        nextTokens,
                        pfa_param,
                        quant_scale2_cur,
                        quant_offset2_cur,
                        antiquant_scale_k,
                        antiquant_scale_v,
                        antiquant_offset_k,
                        antiquant_offset_v,
                        out_dtype,
                        sinks_cur,
                    )

    return y, lse


def conv_float_to_u32(data_f):
    fp = ctypes.pointer(ctypes.c_float(data_f))
    cp = ctypes.cast(fp, ctypes.POINTER(ctypes.c_uint))
    data_hex = cp.contents.value

    result = (data_hex // 8192) * 8192

    return result


def trans_19bit(deqscale):
    res_19bit = np.zeros(deqscale.shape[0], dtype=np.uint64)

    for idx, scale in enumerate(deqscale):
        val = np.bitwise_and(
            ((conv_float_to_u32(scale) >> 13) << 13), 0xFFFFFFFF
        )  # [31:13], M
        val = val.astype(np.uint64)
        res_19bit[idx] = val

    return res_19bit


def qtensor_seqlength(q_shape, inputLayout):
    if inputLayout == "SH":  # SH格式
        return q_shape[0]
    elif inputLayout == "BSH":
        return q_shape[1]
    elif inputLayout == "NSD":
        return q_shape[1]
    elif inputLayout == "BSND":
        return q_shape[1]
    else:  # BNSD
        return q_shape[2]


def kvantiquantmode(flagList, torch_tensor_list, params, v_end_index, numKeyValueHeads):
    antiquant_scale = None
    antiquant_offset = None

    k_antiquant_scale = None
    v_antiquant_scale = None

    k_antiquant_offset = None
    v_antiquant_offset = None

    anti_or_kvanti = 0
    # if len(flagList) > 12:
    # 获取kv量化参数
    flagtensor = 0
    if flagList[12]:
        anti_or_kvanti = 1
        antiquantscale_shape = params["shape_input"][v_end_index + 8]
        antiquantscale_tensor = torch_tensor_list[v_end_index + 8]
        if len(antiquantscale_shape) == 1:  # pertensor
            print("antiquantscale_pertensor")
            antiquant_scale = antiquantscale_tensor
        else:  # perchanel
            # 将kv量化参数转换为2n1d(匹配bnsd)
            antiquantscale_2n1d_tensor = _t_trans_2h_to_2n1d(
                antiquantscale_shape, antiquantscale_tensor, numKeyValueHeads
            )

            antiquant_scale = antiquantscale_2n1d_tensor.permute(0, 2, 1, 3)
    if flagList[13]:
        antiquantoffset_shape = params["shape_input"][v_end_index + 9]
        antiquantoffset_tensor = torch_tensor_list[v_end_index + 9]
        if len(antiquantoffset_shape) == 1:
            print("antiquantoffset_pertensor")
            antiquant_offset = antiquantoffset_tensor
        else:
            antiquantoffset_2n1d_tensor = _t_trans_2h_to_2n1d(
                antiquantoffset_shape, antiquantoffset_tensor, numKeyValueHeads
            )

            antiquant_offset = antiquantoffset_2n1d_tensor.permute(0, 2, 1, 3)
    if flagList[12]:
        return (
            antiquant_scale[:1],
            antiquant_scale[1:2],
            antiquant_offset[:1],
            antiquant_offset[1:2],
        )
    # 判断kv分离的伪量化
    if flagList[17] or flagList[19]:
        anti_or_kvanti = 2
        q_dtype_np = tools.get_np_dtype(params["dtype_input"][0])
        k_antiquantscale_shape = params["shape_input"][v_end_index + 13]
        k_antiquantscale_tensor = torch_tensor_list[v_end_index + 13]
        v_antiquantscale_shape = params["shape_input"][v_end_index + 15]
        v_antiquantscale_tensor = torch_tensor_list[v_end_index + 15]

        if params["k_antiquant_mode"] == 0:
            if (
                len(k_antiquantscale_shape) == 1 and k_antiquantscale_shape[0] == 1
            ):  # pertensor
                print("antiquantscale_pertensor")
                k_antiquant_scale = k_antiquantscale_tensor
                v_antiquant_scale = v_antiquantscale_tensor
            else:  # perchanel
                # 将kv量化参数转换为n1d(匹配bnsd)
                k_antiquantscale_n1d_tensor = _t_trans_h_to_n1d(
                    k_antiquantscale_shape, k_antiquantscale_tensor, numKeyValueHeads
                )
                v_antiquantscale_n1d_tensor = _t_trans_h_to_n1d(
                    v_antiquantscale_shape, v_antiquantscale_tensor, numKeyValueHeads
                )

                k_antiquant_scale = k_antiquantscale_n1d_tensor.permute(0, 2, 1, 3)
                v_antiquant_scale = v_antiquantscale_n1d_tensor.permute(0, 2, 1, 3)
        if params["k_antiquant_mode"] == 1:  # pertoken
            # pertoken -> BS
            k_antiquant_scale = k_antiquantscale_tensor.reshape(
                k_antiquantscale_tensor.shape[0], k_antiquantscale_tensor.shape[1], 1, 1
            )
            v_antiquant_scale = v_antiquantscale_tensor.reshape(
                v_antiquantscale_tensor.shape[0], v_antiquantscale_tensor.shape[1], 1, 1
            )

    if flagList[18] or flagList[20]:
        q_dtype_np = tools.get_np_dtype(params["dtype_input"][0])
        k_antiquantoffset_shape = params["shape_input"][v_end_index + 14]
        k_antiquantoffset_tensor = torch_tensor_list[v_end_index + 14]
        v_antiquantoffset_shape = params["shape_input"][v_end_index + 16]
        v_antiquantoffset_tensor = torch_tensor_list[v_end_index + 16]

        if params["k_antiquant_mode"] == 0:
            if (
                len(k_antiquantoffset_shape) == 1 and k_antiquantoffset_shape[0] == 1
            ):  # pertensor
                print("antiquantscale_pertensor")
                k_antiquant_offset = k_antiquantoffset_tensor
                v_antiquant_offset = v_antiquantoffset_tensor
            else:  # perchanel
                # 将kv量化参数转换为2n1d(匹配bnsd)
                k_antiquantoffset_n1d_tensor = _t_trans_h_to_n1d(
                    k_antiquantoffset_shape, k_antiquantoffset_tensor, numKeyValueHeads
                )
                v_antiquantoffset_n1d_tensor = _t_trans_h_to_n1d(
                    v_antiquantoffset_shape, v_antiquantoffset_tensor, numKeyValueHeads
                )

                k_antiquant_offset = k_antiquantoffset_n1d_tensor.permute(0, 2, 1, 3)
                v_antiquant_offset = v_antiquantoffset_n1d_tensor.permute(0, 2, 1, 3)
        if params["k_antiquant_mode"] == 1:  # pertoken
            # pertoken -> BS
            k_antiquant_offset = k_antiquantoffset_tensor.reshape(
                k_antiquantoffset_tensor.shape[0],
                k_antiquantoffset_tensor.shape[1],
                1,
                1,
            )
            v_antiquant_offset = v_antiquantoffset_tensor.reshape(
                v_antiquantoffset_tensor.shape[0],
                v_antiquantoffset_tensor.shape[1],
                1,
                1,
            )
    if flagList[17]:
        return (
            k_antiquant_scale,
            v_antiquant_scale,
            k_antiquant_offset,
            v_antiquant_offset,
        )
    return k_antiquant_scale, v_antiquant_scale, k_antiquant_offset, v_antiquant_offset


def get_attention_mask_batch_num(npu_m_shape, q_bnsd_shape):
    batch, numhead = None, None
    if len(npu_m_shape) == 2:
        s1 = npu_m_shape[0]
        s2 = npu_m_shape[1]
        return batch, numhead, s1, s2
    if len(npu_m_shape) == 3:
        batch = npu_m_shape[0]
        s1 = npu_m_shape[1]
        s2 = npu_m_shape[2]
        return batch, numhead, s1, s2
    if len(npu_m_shape) == 4:
        batch = npu_m_shape[0]
        numhead = npu_m_shape[1]
        s1 = npu_m_shape[2]
        s2 = npu_m_shape[3]
        return batch, numhead, s1, s2


def _trans_2h_to_2n1d(shape, tensor, numKeyValueHeads):
    if len(shape) == 4:  # 2N1D
        print("2N1D")
        return tensor
    elif len(shape) == 2:  # 2H
        d_num = shape[1] // numKeyValueHeads
        print(
            f"[INFO]_n_trans_2h_to_2n1d : h={shape[1]}, n={numKeyValueHeads} ,d(h/n)={d_num}"
        )
        new_tensor = tensor.reshape(2, 1, numKeyValueHeads, d_num).transpose(0, 2, 1, 3)
        return new_tensor
    elif len(shape) == 3:  # 2ND
        print("2ND")
        N = shape[1]
        D = shape[2]
        new_tensor = tensor.reshape(2, 1, N, D).transpose(0, 2, 1, 3)
        return new_tensor
    else:
        print(f"[ERROR]_n_trans_2h_to_2n1d : len(shape):{len(shape)}")
        return tensor


def _t_trans_2h_to_2n1d(shape, tensor, numKeyValueHeads):
    if len(shape) == 4:  # 2N1D
        print("2N1D")
        return tensor
    elif len(shape) == 2:  # 2H
        d_num = shape[1] // numKeyValueHeads
        print(
            f"[INFO]_n_trans_2h_to_2n1d : h={shape[1]}, n={numKeyValueHeads} ,d(h/n)={d_num}"
        )
        new_tensor = tensor.reshape(2, 1, numKeyValueHeads, d_num).permute(0, 2, 1, 3)
        return new_tensor
    elif len(shape) == 3:  # 2ND
        print("2ND")
        N = shape[1]
        D = shape[2]
        new_tensor = tensor.reshape(2, 1, N, D).permute(0, 2, 1, 3)
        return new_tensor
    else:
        print(f"[ERROR]_n_trans_2h_to_2n1d : len(shape):{len(shape)}")
        return tensor


def _trans_h_to_n1d(shape, tensor, numKeyValueHeads):
    if len(shape) == 3:  # N1D
        print("N1D")
        return tensor
    elif len(shape) == 1:  # H
        d_num = shape[0] // numKeyValueHeads
        print(
            f"[INFO]_n_trans_h_to_n1d : h={shape[1]}, n={numKeyValueHeads} ,d(h/n)={d_num}"
        )
        new_tensor = tensor.reshape(1, numKeyValueHeads, d_num).transpose(1, 0, 2)
        return new_tensor
    elif len(shape) == 2:  # ND
        print("ND")
        N = shape[0]
        D = shape[1]
        new_tensor = tensor.reshape(1, N, D).transpose(1, 0, 2)
        return new_tensor
    else:
        print(f"[ERROR]_n_trans_1h_to_1n1d : len(shape):{len(shape)}")
        return tensor


def _t_trans_h_to_n1d(shape, tensor, numKeyValueHeads):
    if len(shape) == 3:  # N1D
        print("N1D")
        return tensor
    elif len(shape) == 1:  # H
        d_num = shape[0] // numKeyValueHeads
        print(
            f"[INFO]_n_trans_h_to_n1d : h={shape[1]}, n={numKeyValueHeads} ,d(h/n)={d_num}"
        )
        new_tensor = tensor.reshape(1, numKeyValueHeads, d_num).permute(1, 0, 2)
        return new_tensor
    elif len(shape) == 2:  # ND
        print("ND")
        N = shape[0]
        D = shape[1]
        new_tensor = tensor.reshape(1, N, D).permute(1, 0, 2)
        return new_tensor
    else:
        print(f"[ERROR]_n_trans_1h_to_1n1d : len(shape):{len(shape)}")
        return tensor


def _t_trans_1h_to_1n1d(shape, tensor, numHeads, inputLayout):
    if len(shape) == 4:
        # 1n1d->1n1d
        if inputLayout == "BSND":  # 1,1,n,d
            return tensor
        else:  # BNSD 1N1D
            return tensor.transpose(1, 2)
    elif len(shape) == 3:
        if inputLayout == "BSND":  # 1,n,d ->1,n,1,d
            n = shape[1]
            d = shape[2]
            return tensor.reshape(1, 1, n, d)
        elif inputLayout == "BNSD" or inputLayout == "NSD":  # n,1,d -> 1,n,1,d
            n = shape[0]
            d = shape[2]
            return tensor.reshape(1, n, 1, d).transpose(1, 2)
        else:  # bsh 11h->1n1d
            h = shape[2]
            d = h // numHeads
            return tensor.reshape(1, 1, numHeads, d)
    elif len(shape) == 2:
        if inputLayout == "BSH":  # 1,H->1,n,1,d
            d = shape[1] // numHeads
            return tensor.reshape(1, 1, numHeads, d)
        else:  # BSND ND ->1N1D
            n = shape[0]
            d = shape[1]
            return tensor.reshape(1, 1, n, d)
    elif len(shape) == 1:
        # h->1n1d
        h = shape[0]
        d = h // numHeads
        return tensor.reshape(1, 1, numHeads, d)
    else:
        print("[ERROR]trans_to_1n1d: Unknown input shape!")
        exit(1)


def _trans_1h_to_1n1d(shape, tensor, numHeads, inputLayout):
    if len(shape) == 4:
        # 1n1d->1n1d
        if inputLayout == "BSND":  # 1,1,n,d
            return tensor.transpose(0, 2, 1, 3)
        else:  # BNSD 1N1D
            return tensor
    elif len(shape) == 3:
        if inputLayout == "BSND":  # 1,n,d ->1,n,1,d
            n = shape[1]
            d = shape[2]
            return tensor.reshape(1, 1, n, d).transpose(0, 2, 1, 3)
        elif (
            inputLayout == "BNSD" or inputLayout == "NSD" or inputLayout == "BNSD_BSND"
        ):  # n,1,d -> 1,n,1,d
            n = shape[0]
            d = shape[2]
            return tensor.reshape(1, n, 1, d)
        else:  # bsh 11h->1n1d
            h = shape[2]
            d = h // numHeads
            return tensor.reshape(1, 1, numHeads, d).transpose(0, 2, 1, 3)
    elif len(shape) == 2:
        if inputLayout == "BSH":  # 1,H->1,n,1,d
            d = shape[1] // numHeads
            return tensor.reshape(1, 1, numHeads, d).transpose(0, 2, 1, 3)
        else:  # BSND ND ->1N1D
            n = shape[0]
            d = shape[1]
            return tensor.reshape(1, 1, n, d).transpose(0, 2, 1, 3)
    elif len(shape) == 1:
        # h->1n1d
        h = shape[0]
        d = h // numHeads
        return tensor.reshape(1, 1, numHeads, d).transpose(0, 2, 1, 3)
    else:
        print("[ERROR]trans_to_1n1d: Unknown input shape!")
        exit(1)


def gen_outshape(layout, qshape, vshape, numKeyValueHeads, numHeads):
    outshape = copy.deepcopy(qshape)
    if layout == "BSH":
        vd = int(vshape[-1] / numKeyValueHeads)
        outshape[-1] = vd * numHeads
        return outshape
    else:
        outshape[-1] = vshape[-1]
        return outshape


def concat_tensor(tensor1, shape1, tensor2, shape2, n, tnd_flag=False):
    if len(shape1) != len(shape2):
        print(
            f"[ERROR]concat_tensor: 相加的两个tensor 维数不同! shape1 = {shape1}, shape2 =  {shape2}"
        )
        return None, None
    elif len(shape1) == 2:
        # 量化参数1h
        if shape1[0] != shape2[0]:
            print(
                f"[ERROR]concat_tensor: 相加的两个tensor 维度非法! shape1 = {shape1}, shape2 =  {shape2}"
            )
            return None, None
        d1 = int(shape1[1] / n)
        d2 = int(shape2[1] / n)
        tensor1_1nd = tensor1.reshape(shape1[0], n, d1)
        tensor2_1nd = tensor2.reshape(shape2[0], n, d2)
        concatenated_tensor = np.concatenate((tensor1_1nd, tensor2_1nd), axis=2)
        concatenated_shape = [shape1[0], shape1[1] + shape2[1]]
        concatenated_tensor = concatenated_tensor.reshape(concatenated_shape)
    elif len(shape1) == 3:
        if shape1[0] != shape2[0] or shape1[1] != shape2[1]:
            print(
                f"[ERROR]concat_tensor: 相加的两个tensor 维度非法! shape1 = {shape1}, shape2 =  {shape2}"
            )
            return None, None
        if tnd_flag:
            concatenated_tensor = np.concatenate((tensor1, tensor2), axis=2)
            concatenated_shape = [shape1[0], shape1[1], shape1[2] + shape2[2]]
        else:
            d1 = int(shape1[2] / n)
            d2 = int(shape2[2] / n)
            tensor1_bsnd = tensor1.reshape(shape1[0], shape1[1], n, d1)
            tensor2_bsnd = tensor2.reshape(shape2[0], shape2[1], n, d2)
            concatenated_tensor = np.concatenate((tensor1_bsnd, tensor2_bsnd), axis=3)
            concatenated_shape = [shape1[0], shape1[1], shape1[2] + shape2[2]]
            concatenated_tensor = concatenated_tensor.reshape(concatenated_shape)
    elif len(shape1) == 4:
        if shape1[0] != shape2[0] or shape1[1] != shape2[1] or shape1[2] != shape2[2]:
            print(
                f"[ERROR]concat_tensor: 相加的两个tensor 维度非法! shape1 = {shape1}, shape2 =  {shape2}"
            )
            return None, None
        concatenated_tensor = np.concatenate((tensor1, tensor2), axis=3)
        concatenated_shape = [shape1[0], shape1[1], shape1[2], shape1[3] + shape2[3]]
    else:
        print(f"[ERROR]concat_tensor: shape1 维度非法! shape1 = {shape1}")
        return None, None
    return concatenated_tensor, concatenated_shape


def deepseek_ds_pa_preprocessing(pfa_param, params, torch_tensor_list):
    v_end_index = pfa_param["v_end_index"]
    k_cache = torch_tensor_list[v_end_index + 19]
    v_cache = torch_tensor_list[v_end_index + 20]
    k_rope_cache = torch_tensor_list[v_end_index + 23]

    numKeyValueHeads = params["numkeyvalueheads"]
    actualSeqLengthsKV = pfa_param["actualSeqLengthsKV"]
    kv_inputlayout = pfa_param["kv_inputlayout"]
    k_tensor_bnsd_list = []
    k_shape_bnsd_raw_list = []

    v_tensor_bnsd_list = []
    v_shape_bnsd_raw_list = []
    k_shape_num = pfa_param["k_num"]
    for i in range(k_shape_num):
        k_bnsd_tensor, k_bnsd_shape = np_bsh_to_bnsd(
            torch_tensor_list[1],
            pfa_param["k_shape_list_raw"][0],
            numKeyValueHeads,
            actualSeqLengthsKV,
            kv_inputlayout,
        )
        k_tensor_bnsd_list.append(k_bnsd_tensor)  # 转成bnsd原始
        k_shape_bnsd_raw_list.append(k_bnsd_shape)

    for i in range(k_shape_num):
        v_shape = pfa_param["v_shape_list_raw"][i]
        v_bnsd_tensor, v_bnsd_shape = np_bsh_to_bnsd(
            torch_tensor_list[2],
            v_shape,
            numKeyValueHeads,
            actualSeqLengthsKV,
            kv_inputlayout,
        )
        v_tensor_bnsd_list.append(v_bnsd_tensor)  # 转成bnsd原始
        v_shape_bnsd_raw_list.append(v_bnsd_shape)

    k_rope_tensor = torch_tensor_list[v_end_index + 22]
    k_rope_shape = params["shape_input"][v_end_index + 22]
    k_rope_bnsd_tensor, k_rope_bnsd_shape = np_bsh_to_bnsd(
        k_rope_tensor,
        k_rope_shape,
        numKeyValueHeads,
        actualSeqLengthsKV,
        kv_inputlayout,
    )

    blockNumPerBlock = pfa_param["blockNumPerBlock"]
    block_table = pfa_param["block_table"]
    blockSize = params["blocksize"]

    k_tensor_bsh_raw = k_tensor_bnsd_list[0]
    v_tensor_bsh_raw = v_tensor_bnsd_list[0]
    k_rope_bsh_raw = k_rope_bnsd_tensor

    if k_cache.ndim == 3:
        k_tensor_bsh_raw = trans_bnsd_to_bsh(
            k_tensor_bnsd_list[0], k_shape_bnsd_raw_list[0]
        )
        v_tensor_bsh_raw = trans_bnsd_to_bsh(
            v_tensor_bnsd_list[0], v_shape_bnsd_raw_list[0]
        )
        k_rope_bsh_raw = trans_bnsd_to_bsh(k_rope_bnsd_tensor, k_rope_bnsd_shape)
        for b, block_per_b in enumerate(blockNumPerBlock):
            for blockid_index in range(block_per_b):
                blockid = block_table[b][blockid_index]
                block_offset = blockid_index * blockSize
                if blockid_index == block_per_b - 1:
                    blocksize_left = actualSeqLengthsKV[b] - block_offset
                    k_cache[blockid, 0:blocksize_left, :] = k_tensor_bsh_raw[
                        b, block_offset : (block_offset + blocksize_left), :
                    ]
                    v_cache[blockid, 0:blocksize_left, :] = v_tensor_bsh_raw[
                        b, block_offset : (block_offset + blocksize_left), :
                    ]
                    k_rope_cache[blockid, 0:blocksize_left, :] = k_rope_bsh_raw[
                        b, block_offset : (block_offset + blocksize_left), :
                    ]
                else:
                    k_cache[blockid, 0:blockSize, :] = k_tensor_bsh_raw[
                        b, block_offset : (block_offset + blockSize), :
                    ]
                    v_cache[blockid, 0:blockSize, :] = v_tensor_bsh_raw[
                        b, block_offset : (block_offset + blockSize), :
                    ]
                    k_rope_cache[blockid, 0:blockSize, :] = k_rope_bsh_raw[
                        b, block_offset : (block_offset + blockSize), :
                    ]
    if k_cache.ndim == 4:
        # gen kv cache
        for b, block_per_b in enumerate(blockNumPerBlock):
            for blockid_index in range(block_per_b):
                blockid = block_table[b][blockid_index]
                block_offset = blockid_index * blockSize
                if blockid_index == block_per_b - 1:
                    blocksize_left = actualSeqLengthsKV[b] - block_offset
                    k_cache[blockid, :, 0:blocksize_left, :] = k_tensor_bsh_raw[
                        b, :, block_offset : (block_offset + blocksize_left), :
                    ]
                    v_cache[blockid, :, 0:blocksize_left, :] = v_tensor_bsh_raw[
                        b, :, block_offset : (block_offset + blocksize_left), :
                    ]
                    k_rope_cache[blockid, :, 0:blocksize_left, :] = k_rope_bsh_raw[
                        b, :, block_offset : (block_offset + blocksize_left), :
                    ]
                else:
                    k_cache[blockid, :, 0:blockSize, :] = k_tensor_bsh_raw[
                        b, :, block_offset : (block_offset + blockSize), :
                    ]
                    v_cache[blockid, :, 0:blockSize, :] = v_tensor_bsh_raw[
                        b, :, block_offset : (block_offset + blockSize), :
                    ]
                    k_rope_cache[blockid, :, 0:blockSize, :] = k_rope_bsh_raw[
                        b, :, block_offset : (block_offset + blockSize), :
                    ]

    if k_cache.ndim == 5:
        kd = k_tensor_bsh_raw.shape[3]
        vd = v_tensor_bsh_raw.shape[3]
        numheadkv = k_tensor_bsh_raw.shape[1]
        k_cache_bnsd = np.zeros((k_cache.shape[0], numheadkv, blockSize, kd))
        v_cache_bnsd = np.zeros((k_cache.shape[0], numheadkv, blockSize, vd))
        k_rope_cache_bnsd = np.zeros((k_rope_cache.shape[0], numheadkv, blockSize, 64))
        # gen kv cache
        for b, block_per_b in enumerate(blockNumPerBlock):
            for blockid_index in range(block_per_b):
                blockid = block_table[b][blockid_index]
                block_offset = blockid_index * blockSize
                if blockid_index == block_per_b - 1:
                    blocksize_left = actualSeqLengthsKV[b] - block_offset
                    k_cache_bnsd[blockid, :, 0:blocksize_left, :] = k_tensor_bsh_raw[
                        b, :, block_offset : (block_offset + blocksize_left), :
                    ]
                    v_cache_bnsd[blockid, :, 0:blocksize_left, :] = v_tensor_bsh_raw[
                        b, :, block_offset : (block_offset + blocksize_left), :
                    ]
                    k_rope_cache_bnsd[blockid, :, 0:blocksize_left, :] = k_rope_bsh_raw[
                        b, :, block_offset : (block_offset + blocksize_left), :
                    ]
                else:
                    k_cache_bnsd[blockid, :, 0:blockSize, :] = k_tensor_bsh_raw[
                        b, :, block_offset : (block_offset + blockSize), :
                    ]
                    v_cache_bnsd[blockid, :, 0:blockSize, :] = v_tensor_bsh_raw[
                        b, :, block_offset : (block_offset + blockSize), :
                    ]
                    k_rope_cache_bnsd[blockid, :, 0:blockSize, :] = k_rope_bsh_raw[
                        b, :, block_offset : (block_offset + blockSize), :
                    ]

            base = 16
            k_cache = k_cache_bnsd.reshape(
                k_cache_bnsd.shape[0],
                k_cache_bnsd.shape[1],
                k_cache_bnsd.shape[2],
                k_cache_bnsd.shape[3] // base,
                base,
            ).transpose(0, 1, 3, 2, 4)
            v_cache = v_cache_bnsd.reshape(
                v_cache_bnsd.shape[0],
                v_cache_bnsd.shape[1],
                v_cache_bnsd.shape[2],
                v_cache_bnsd.shape[3] // base,
                base,
            ).transpose(0, 1, 3, 2, 4)
            k_rope_cache = k_rope_cache_bnsd.reshape(
                k_rope_cache_bnsd.shape[0],
                k_rope_cache_bnsd.shape[1],
                k_rope_cache_bnsd.shape[2],
                k_rope_cache_bnsd.shape[3] // base,
                base,
            ).transpose(0, 1, 3, 2, 4)

    kv_dtype = tools.get_np_dtype(params["dtype_input"][1])

    if str(kv_dtype) == "<class 'bfloat16'>":
        k_cache = torch.from_numpy(k_cache).to(torch.bfloat16)
        v_cache = torch.from_numpy(v_cache).to(torch.bfloat16)
        k_rope_cache = torch.from_numpy(k_rope_cache).to(torch.bfloat16)
    elif (
        str(kv_dtype) != "<class 'float8_e5m2'>"
        and params["dtype_input"][1] != "hifloat8"
        and params["dtype_input"][1] != "float8_e4m3fn"
    ):
        k_cache = torch.from_numpy(k_cache.astype(kv_dtype))
        v_cache = torch.from_numpy(v_cache.astype(kv_dtype))
        k_rope_cache = torch.from_numpy(k_rope_cache.astype(kv_dtype))


def deepseek_preprocessing(
    params, pfa_param, torch_tensor_list, numHeads, numKeyValueHeads, tnd_flag
):
    v_end_index = pfa_param["v_end_index"]
    # >> q rope info
    q_rope_tensor_shape = params["shape_input"][v_end_index + 21]
    q_rope_dtype = params["dtype_input"][v_end_index + 21]
    q_rope_tensor = torch_tensor_list[v_end_index + 21]

    # >> k rope info
    k_rope_shape = params["shape_input"][v_end_index + 22]
    k_rope_dtype = params["dtype_input"][v_end_index + 22]
    k_rope_tensor = torch_tensor_list[v_end_index + 22]

    # 非全量化场景1.将Q与QROPE拼接2.将K与KROPE拼接3.伪量化场景，k_antiscale与k_rope_antiscale拼接
    # if not ifa_param['in_quant_flag']:
    q_new_tensor, q_new_shape = concat_tensor(
        torch_tensor_list[0],
        params["shape_input"][0],
        q_rope_tensor,
        q_rope_tensor_shape,
        numHeads,
        tnd_flag,
    )
    k_new_tensor, k_new_shape = concat_tensor(
        torch_tensor_list[1],
        params["shape_input"][1],
        k_rope_tensor,
        k_rope_shape,
        numKeyValueHeads,
        tnd_flag,
    )

    pfa_param["q_tensor"] = q_new_tensor
    pfa_param["q_shape"] = q_new_shape
    pfa_param["k_tensor_list"][0] = k_new_tensor
    pfa_param["k_shape_list"][0] = k_new_shape

    return pfa_param


def pfa_get_param_fus(torch_tensor_list, params, qdim):
    pfa_param = {}
    # ===参数获取===
    flag_list = params["flaglist"]
    inputLayout = params["inputlayout"]
    actualSeqLengths = params["actualseqlengths"]
    q_shape = params["shape_input"][0]

    tnd_flag = False
    if inputLayout in ["TND", "TND_NTD", "NTD_TND"]:
        tnd_flag = True
        batch = len(actualSeqLengths)
    else:
        batch = q_shape[0]
    pfa_param["batch"] = batch

    pfa_param["q_tensor"] = torch_tensor_list[0]
    pfa_param["q_shape"] = params["shape_input"][0]

    # >> kv info
    # k和v的位置通过b计算
    k1_shape = params["shape_input"][1]
    kb1 = k1_shape[0]
    # 如果第一个K的B=1,则默认kv列表长度为b;如果第一个K的B!=1,则默认kv列表长度为1
    if kb1 != 1 or inputLayout in ["TND", "NTD_TND"]:
        k_shape_num = v_shape_num = 1
    else:
        k_shape_num = v_shape_num = batch
    pfa_param["v_end_index"] = v_end_index = int(k_shape_num + v_shape_num)
    pfa_param["tnd_flag"] = tnd_flag
    pfa_param["k_num"] = k_shape_num
    pfa_param["k_start_index"] = k_start_index = 1
    pfa_param["k_end_index"] = k_end_index = int(k_shape_num)
    pfa_param["v_start_index"] = v_start_index = int(k_shape_num) + 1
    print(
        f"k_start_index:{k_start_index} k_end_index:{k_end_index} v_start_index:{v_start_index} v_end_index:{v_end_index}"
    )

    pfa_param["k_shape_list"] = pfa_param["k_shape_list_raw"] = params["shape_input"][
        k_start_index : k_end_index + 1
    ]
    pfa_param["k_tensor_list"] = torch_tensor_list[k_start_index : k_end_index + 1]

    if flag_list[26] and qdim == 512:
        pfa_param["v_shape_list"] = params["shape_input"][
            k_start_index : k_end_index + 1
        ]
        pfa_param["v_tensor_list"] = torch_tensor_list[k_start_index : k_end_index + 1]
    else:
        pfa_param["v_shape_list"] = params["shape_input"][
            v_start_index : v_end_index + 1
        ]
        pfa_param["v_tensor_list"] = torch_tensor_list[v_start_index : v_end_index + 1]

    pfa_param["k_shape_list_raw"] = params["shape_input"][
        k_start_index : k_end_index + 1
    ]
    pfa_param["k_tensor_list_raw"] = torch_tensor_list[k_start_index : k_end_index + 1]

    if flag_list[26] and qdim == 512:
        pfa_param["v_shape_list_raw"] = params["shape_input"][
            k_start_index : k_end_index + 1
        ]
        pfa_param["v_tensor_list_raw"] = torch_tensor_list[
            k_start_index : k_end_index + 1
        ]
    else:
        pfa_param["v_shape_list_raw"] = params["shape_input"][
            v_start_index : v_end_index + 1
        ]
        pfa_param["v_tensor_list_raw"] = torch_tensor_list[
            v_start_index : v_end_index + 1
        ]

    return pfa_param


def qtensor_dim(q_shape, inputLayout, numhead):
    if inputLayout == "SH":  # SH格式
        return q_shape[0]
    elif inputLayout == "BSH" or inputLayout == "BSH_NBSD":
        return int(q_shape[2] / numhead)
    else:
        return q_shape[-1]


def aclnnPromptFlashAttention_unification(torch_tensor_list, params):
    # torch_tensor_list ---tensor  params--attr参数

    pfa_param = {}

    action_type = params["action_type"]
    gold = ["bm_output", "bm_output_gold", "bm_gold"]

    flagList = params["flaglist"]
    actualSeqLengths = params["actualseqlengths"]
    actualSeqLengthsKV = params["actualseqlengthskv"]
    numHeads = int(params["numheads"])
    # 设置参数默认值

    if flagList[25] == 0:
        inputLayout = "BSH"
        # 所有的参数的默认值都给写在这里
        numKeyValueHeads = 0
        scaleValue = 1.0
        preTokens = 214748647
        nextTokens = 214748647
        sp_mode = 0
        block_size = 0
    else:
        scaleValue = params["scalevalue"] if "scalevalue" in params else 1
        preTokens = params["pretokens"] if "pretokens" in params else 214748647
        nextTokens = params["nexttokens"] if "nexttokens" in params else 214748647
        inputLayout = params["inputlayout"] if "inputlayout" in params else "BSH"
        numKeyValueHeads = (
            params["numkeyvalueheads"] if "numkeyvalueheads" in params else 0
        )
        sp_mode = params["sparsemode"] if "sparsemode" in params else 0

    if numKeyValueHeads == 0:
        numKeyValueHeads = numHeads
    q_shape = params["shape_input"][0]
    qdim = qtensor_dim(q_shape, inputLayout, numHeads)
    pfa_param = pfa_get_param_fus(torch_tensor_list, params, qdim)
    q_shape = pfa_param["q_shape"]
    q_dtype = get_np_dtype(params["dtype_input"][0])  # numpy类型
    pfa_param["q_dtype"] = q_dtype
    batch = pfa_param["batch"]
    tnd_flag = pfa_param["tnd_flag"]

    kv_inputlayout = inputLayout
    if (inputLayout == "TND" or inputLayout == "NTD_TND") and flagList[14]:
        kv_inputlayout = "BNSD"
    pfa_param["kv_inputlayout"] = kv_inputlayout

    if inputLayout == "SH":
        actualSeqLengthsKV = actualSeqLengths
    else:
        actualSeqLengthsKV = params["actualseqlengthskv"]
    if inputLayout == "TND" or inputLayout == "NTD_TND":
        # 将tnd格式下的act seq转成普通的act seq
        actualSeqLengths = trans_tnd_actseq(actualSeqLengths)
    if (inputLayout == "TND" or inputLayout == "NTD_TND") and flagList[
        14
    ] == 0:  # TND非PA场景需要转成普通actseq
        actualSeqLengthsKV = trans_tnd_actseq(actualSeqLengthsKV)
    # >> actualSeqLengths预处理：actualSeqLengths为单值场景，如果长度为1且b不为1，则将actualSeqLengths扩展为b个单值的列表
    if inputLayout != "TND" and inputLayout != "NTD_TND":
        if actualSeqLengths is not None:
            if len(actualSeqLengths) == 1 and len(actualSeqLengths) != q_shape[0]:
                actualSeqLengths_item = actualSeqLengths[0]
                for b_count in range(batch - 1):
                    actualSeqLengths.append(actualSeqLengths_item)
            # >> actualSeqLengths预处理：actualSeqLengths长度超过
            if len(actualSeqLengths) > batch:
                actualSeqLengths = actualSeqLengths[:batch]
        if actualSeqLengthsKV is not None:
            if len(actualSeqLengthsKV) == 1 and len(actualSeqLengthsKV) != q_shape[0]:
                actualSeqLengthsKV_item = actualSeqLengthsKV[0]
                for b_count in range(batch - 1):
                    actualSeqLengthsKV.append(actualSeqLengthsKV_item)
            # >> actualSeqLengths预处理：actualSeqLengths长度超过
            if len(actualSeqLengthsKV) > batch:
                actualSeqLengthsKV = actualSeqLengthsKV[:batch]
    pfa_param["actualSeqLengthsKV"] = actualSeqLengthsKV
    k_shape_num = pfa_param["k_num"]
    v_shape = params["shape_input"][1 + k_shape_num]
    out_dtype = get_np_dtype(params["dtype_output"][0])  # numpy类型
    out_shape = gen_outshape(inputLayout, q_shape, v_shape, numKeyValueHeads, numHeads)

    if 0 in q_shape or len(q_shape) == 0:
        if params["softmax_lse_flag"]:
            return torch.from_numpy(torch_tensor_list[0]), torch.from_numpy(
                np.array([])
            )
        return torch.from_numpy(torch_tensor_list[0])
    k_shape = pfa_param["k_shape_list"][0]

    pfa_param["q_tensor"] = torch_tensor_list[0]
    if flagList[26]:
        pfa_param = deepseek_preprocessing(
            params, pfa_param, torch_tensor_list, numHeads, numKeyValueHeads, tnd_flag
        )
    q_tensor, q_bnsd_shape = np_bsh_to_bnsd(
        pfa_param["q_tensor"],
        pfa_param["q_shape"],
        numHeads,
        actualSeqLengths,
        inputLayout,
    )
    if 0 in k_shape or len(k_shape) == 0:
        if params["softmax_lse_flag"]:
            return torch.zeros(out_shape), torch.zeros(pfa_param["lseshape"])
        return torch.zeros(out_shape)
    if inputLayout == "TND" or inputLayout == "NTD_TND":
        pfa_param["lseshape"] = [q_shape[0], q_shape[1], 1]
    else:
        pfa_param["lseshape"] = [q_bnsd_shape[0], q_bnsd_shape[1], q_bnsd_shape[2], 1]

    # >> kv info
    k_tensor_list = []
    v_tensor_list = []
    k_shape_bnsd_list = []
    v_shape_bnsd_list = []
    k_tensor_bnsd_list = []
    v_tensor_bnsd_list = []
    k_shape_bnsd_raw_list = []
    v_shape_bnsd_raw_list = []
    k_start_index = pfa_param["k_start_index"]
    k_end_index = pfa_param["k_end_index"]
    v_start_index = pfa_param["v_start_index"]
    v_end_index = pfa_param["v_end_index"]
    kvs_max = 0
    kvs_list = []
    k_dtype = params["dtype_input"][1]
    print(f"k_start_index:{k_start_index}, k_end_index:{k_end_index}")

    for i in range(k_shape_num):
        k_shape = pfa_param["k_shape_list"][i]
        if 0 in k_shape or len(k_shape) == 0:
            if params["softmax_lse_flag"]:
                return torch.zeros(out_shape), torch.zeros(pfa_param["lseshape"])
            return torch.zeros(out_shape)
        k_tensor, k_bnsd_shape = np_bsh_to_bnsd(
            pfa_param["k_tensor_list"][i],
            k_shape,
            numKeyValueHeads,
            actualSeqLengthsKV,
            kv_inputlayout,
        )
        k_tensor_bnsd_list.append(k_tensor)  # 转成bnsd原始
        k_shape_bnsd_raw_list.append(k_bnsd_shape)
        kvs_list.append(k_bnsd_shape[2])
        kvs_max = max(kvs_max, k_bnsd_shape[2])
        if numKeyValueHeads != numHeads:
            print("broadcast k")
            k_tensor, k_bnsd_shape = _np_broadcastKV_sigle(
                numHeads, numKeyValueHeads, k_tensor, k_tensor.dtype
            )
        k_tensor_list.append(k_tensor)
        k_shape_bnsd_list.append(k_bnsd_shape)
    for i in range(k_shape_num):
        v_shape = pfa_param["v_shape_list"][i]
        v_tensor, v_bnsd_shape = np_bsh_to_bnsd(
            pfa_param["v_tensor_list"][i],
            v_shape,
            numKeyValueHeads,
            actualSeqLengthsKV,
            kv_inputlayout,
        )
        v_tensor_bnsd_list.append(v_tensor)
        v_shape_bnsd_raw_list.append(v_bnsd_shape)
        if numKeyValueHeads != numHeads:
            print("broadcast v")
            v_tensor, v_bnsd_shape = _np_broadcastKV_sigle(
                numHeads, numKeyValueHeads, v_tensor, v_tensor.dtype
            )
        v_tensor_list.append(v_tensor)
        v_shape_bnsd_list.append(v_bnsd_shape)
    kvs = kvs_max
    actualprefixKV = 0
    prefix_kvs = 0
    if flagList[21]:
        prefix_k_shape = params["shape_input"][v_end_index + 17]
        prefix_v_shape = params["shape_input"][v_end_index + 18]
        if len(params["prefix_act_lens"]) != 0:
            actualprefixKV = params["prefix_act_lens"][0]

        prefix_k_tensor, prefix_k_bnsd_shape = np_bsh_to_bnsd(
            torch_tensor_list[v_end_index + 17],
            prefix_k_shape,
            numKeyValueHeads,
            actualprefixKV,
            inputLayout,
        )

        prefix_v_tensor, prefix_v_bnsd_shape = np_bsh_to_bnsd(
            torch_tensor_list[v_end_index + 18],
            prefix_v_shape,
            numKeyValueHeads,
            actualprefixKV,
            inputLayout,
        )
        if numKeyValueHeads != numHeads:
            prefix_k_tensor, prefix_k_bnsd_shape = _np_broadcastKV_sigle(
                numHeads, numKeyValueHeads, prefix_k_tensor, prefix_k_tensor.dtype
            )
            prefix_v_tensor, prefix_v_bnsd_shape = _np_broadcastKV_sigle(
                numHeads, numKeyValueHeads, prefix_v_tensor, prefix_v_tensor.dtype
            )
        pfa_param["actualprefixKV"] = actualprefixKV
        pfa_param["shared_prefix_k"] = prefix_k_tensor
        pfa_param["shared_prefix_v"] = prefix_v_tensor
        prefix_kvs = prefix_k_bnsd_shape[2]
        if actualprefixKV == 0:
            actualprefixKV = prefix_k_bnsd_shape[2]
        kvs += prefix_kvs

    pse_bnsd_tensor = None
    pse_shift_shape = params["shape_input"][v_end_index + 1]  ##1nss or bnss
    qs = q_bnsd_shape[2]

    if flagList[3] == 0 or 0 in pse_shift_shape:
        pse_bnsd_tensor = None
    else:
        pse_dtype_torch = get_pt_dtype(params["dtype_input"][v_end_index + 1])
        pse_shift = None
        pse_shift_random_flag = True
        if action_type in gold:
            pse_shift = torch_tensor_list[3]
        else:
            if "prandom" in params:
                if params["prandom"] != 0:
                    pse_shift = torch_tensor_list[v_end_index + 1]
                else:
                    pse_shift_random_flag = False
            else:
                pse_shift_random_flag = False
            if not pse_shift_random_flag:
                print("pse")
                pse_shift = get_all_alibi(numHeads, pse_shift_shape)
                pse_shift[:, :, qs:, :] = 1
                pse_shift[:, :, :, kvs:] = 1
                npu_pse_shift = torch.from_numpy(pse_shift).to(pse_dtype_torch)
                pse_shift = npu_pse_shift.to(torch.float32).numpy().astype(np.float32)
        cpu_pse_shift = pse_shift[:, :, :qs, :kvs]

        pse_bnsd_tensor = _np_broadcast_pseShift_n(
            cpu_pse_shift, pse_shift_shape, q_bnsd_shape[0]
        )  # to bnsd
    m_bnsd_tensor = None
    npu_m_shape = params["shape_input"][v_end_index + 2]
    m_dtype = get_np_dtype(params["dtype_input"][v_end_index + 2])

    randoms = 0
    mrandom_type = "NORMAL"
    if "mrandomtype" in params:
        mrandom_type = params["mrandomtype"]
        if mrandom_type == "ones":
            randoms = int(params["mrandom"])

    if flagList[4] == 0 or 0 in npu_m_shape:
        preTokens = 214748647
        nextTokens = 214748647
    else:
        batch = q_bnsd_shape[0]
        numheads = q_bnsd_shape[1]
        npu_m_shape_s = npu_m_shape
        if sp_mode == 0 or sp_mode == 1:
            batch, numheads, ns1, ns2 = get_attention_mask_batch_num(
                npu_m_shape, q_bnsd_shape
            )  # 获取输入attentionmask的batch 和numhead
            npu_m_shape_s = [ns1, ns2]
        cpu_m_shape = [qs, kvs]  # cpu

        cpu_m_tensor, npu_m_tensor, preTokens, nextTokens = (
            _create_random_mask_by_spars(
                cpu_m_shape,
                npu_m_shape_s,
                m_dtype,
                preTokens,
                nextTokens,
                actualSeqLengths,
                actualSeqLengthsKV,
                actualprefixKV,
                prefix_kvs,
                kvs_list,
                batch,
                numheads,
                sp_mode,
                random_ones=randoms,
            )
        )
        if mrandom_type == "invalid" or mrandom_type == "invaild":
            randoms = int(params["mrandom"])
            cpu_m_tensor[..., :randoms] = 1
            npu_m_tensor[..., :randoms] = 1

        npu_m_tensor = torch.from_numpy(npu_m_tensor)

        if sp_mode == 0 or sp_mode == 1:
            m_bnsd_tensor = _np_broadcast_mask_n(
                cpu_m_tensor, npu_m_shape, cpu_m_shape, numHeads, q_bnsd_shape[0]
            )
        else:
            m_bnsd_tensor = cpu_m_tensor

    dequant_scale1, dequant_scale2, quant_scale1, quant_scale2, quant_offset2 = (
        None,
        None,
        None,
        None,
        None,
    )

    if flagList[7]:
        dequant_scale1 = torch_tensor_list[v_end_index + 3]
        dequant_scale1 = np.frombuffer(dequant_scale1, dtype=np.float32)
        dequant_scale1 = dequant_scale1[:1]

    if flagList[8]:
        quant_scale1 = torch_tensor_list[v_end_index + 4]
    if flagList[9]:
        dequant_scale2 = torch_tensor_list[v_end_index + 5]
        dequant_scale2 = np.frombuffer(dequant_scale2, dtype=np.float32)
        dequant_scale2 = dequant_scale2[:1]
    if flagList[10]:
        quant_scale2 = torch_tensor_list[v_end_index + 6]
        quant_scale2_shape = params["shape_input"][v_end_index + 6]
        per_channel = True
        if len(quant_scale2_shape) == 1:
            if quant_scale2_shape[0] == 1:
                per_channel = False
        if per_channel:
            quant_scale2 = _trans_1h_to_1n1d(
                quant_scale2_shape, quant_scale2, numHeads, inputLayout
            )
    if flagList[11]:
        quant_offset2 = torch_tensor_list[v_end_index + 7]
        quant_offset2_shape = params["shape_input"][v_end_index + 7]
        per_channel = True
        if len(quant_offset2_shape) == 1:
            if quant_offset2_shape[0] == 1:
                per_channel = False
        if per_channel:
            quant_offset2 = _trans_1h_to_1n1d(
                quant_offset2_shape, quant_offset2, numHeads, inputLayout
            )

    antiquant_scale = None
    antiquant_offset = None

    k_antiquant_scale = None
    v_antiquant_scale = None

    k_antiquant_offset = None
    v_antiquant_offset = None

    sinks = None

    anti_or_kvanti = 0
    if len(flagList) > 20:
        # 获取kv量化参数
        if flagList[12]:
            anti_or_kvanti = 1
            antiquantscale_shape = params["shape_input"][v_end_index + 8]
            antiquantscale_tensor = torch_tensor_list[v_end_index + 8]
            if len(antiquantscale_shape) == 1:  # pertensor
                print("antiquantscale_pertensor")
                antiquant_scale = antiquantscale_tensor
            else:  # perchanel
                # 将kv量化参数转换为2n1d(匹配bnsd)
                antiquantscale_2n1d_tensor = _trans_2h_to_2n1d(
                    antiquantscale_shape, antiquantscale_tensor, numKeyValueHeads
                )

                # GQA场景，扩展kv反量化参数
                if numKeyValueHeads != numHeads:
                    print("broadcast kvanti")
                    antiquant_scale = broadcast_kv_dequant_tensor(
                        antiquantscale_2n1d_tensor, numKeyValueHeads, numHeads
                    )
                else:
                    antiquant_scale = antiquantscale_2n1d_tensor
        if flagList[13]:
            antiquantoffset_shape = params["shape_input"][v_end_index + 9]
            antiquantoffset_tensor = torch_tensor_list[v_end_index + 9]
            if len(antiquantoffset_shape) == 1:
                print("antiquantoffset_pertensor")
                antiquant_offset = antiquantoffset_tensor
            else:
                antiquantoffset_2n1d_tensor = _trans_2h_to_2n1d(
                    antiquantoffset_shape, antiquantoffset_tensor, numKeyValueHeads
                )

                # GQA场景，扩展kv反量化参数
                if numKeyValueHeads != numHeads:
                    print("broadcast kvanti")
                    antiquant_offset = broadcast_kv_dequant_tensor(
                        antiquantoffset_2n1d_tensor, numKeyValueHeads, numHeads
                    )
                else:
                    antiquant_offset = antiquantoffset_2n1d_tensor
        # 判断kv分离的伪量化

        if flagList[17] or flagList[19]:
            anti_or_kvanti = 2
            q_dtype_np = tools.get_np_dtype(params["dtype_input"][0])
            k_antiquantscale_shape = params["shape_input"][v_end_index + 13]
            k_antiquantscale_tensor = torch_tensor_list[v_end_index + 13]
            v_antiquantscale_shape = params["shape_input"][v_end_index + 15]
            v_antiquantscale_tensor = torch_tensor_list[v_end_index + 15]

            if params["k_antiquant_mode"] == 0:
                if (
                    len(k_antiquantscale_shape) == 1 and k_antiquantscale_shape[0] == 1
                ):  # pertensor
                    print("antiquantscale_pertensor")
                    k_antiquant_scale = k_antiquantscale_tensor
                    v_antiquant_scale = v_antiquantscale_tensor
                else:  # perchanel
                    # 将kv量化参数转换为n1d(匹配bnsd)
                    k_antiquantscale_n1d_tensor = _trans_h_to_n1d(
                        k_antiquantscale_shape,
                        k_antiquantscale_tensor,
                        numKeyValueHeads,
                    )
                    v_antiquantscale_n1d_tensor = _trans_h_to_n1d(
                        v_antiquantscale_shape,
                        v_antiquantscale_tensor,
                        numKeyValueHeads,
                    )
                    # GQA场景，扩展kv反量化参数
                    if numKeyValueHeads != numHeads:
                        print("broadcast kvanti")
                        k_antiquant_scale = broadcast_kv_split_dequant_tensor(
                            k_antiquantscale_n1d_tensor,
                            numKeyValueHeads,
                            numHeads,
                            q_dtype_np,
                        )
                        v_antiquant_scale = broadcast_kv_split_dequant_tensor(
                            v_antiquantscale_n1d_tensor,
                            numKeyValueHeads,
                            numHeads,
                            q_dtype_np,
                        )
                        print(v_antiquant_scale.shape)
                    else:
                        k_antiquant_scale = k_antiquantscale_n1d_tensor
                        v_antiquant_scale = v_antiquantscale_n1d_tensor
            if params["k_antiquant_mode"] == 1:  # pertoken
                # pertoken -> BS
                k_antiquant_scale = k_antiquantscale_tensor
                v_antiquant_scale = v_antiquantscale_tensor
        pfa_param["k_antiquant_scale"] = k_antiquant_scale
        pfa_param["v_antiquant_scale"] = v_antiquant_scale

        if flagList[18] or flagList[20]:
            q_dtype_np = tools.get_np_dtype(params["dtype_input"][0])
            k_antiquantoffset_shape = params["shape_input"][v_end_index + 14]
            k_antiquantoffset_tensor = torch_tensor_list[v_end_index + 14]
            v_antiquantoffset_shape = params["shape_input"][v_end_index + 16]
            v_antiquantoffset_tensor = torch_tensor_list[v_end_index + 16]

            if params["k_antiquant_mode"] == 0:
                if (
                    len(k_antiquantoffset_shape) == 1
                    and k_antiquantoffset_shape[0] == 1
                ):  # pertensor
                    print("antiquantscale_pertensor")
                    k_antiquant_offset = k_antiquantoffset_tensor
                    v_antiquant_offset = v_antiquantoffset_tensor
                else:  # perchanel
                    # 将kv量化参数转换为2n1d(匹配bnsd)
                    k_antiquantoffset_n1d_tensor = _trans_h_to_n1d(
                        k_antiquantoffset_shape,
                        k_antiquantoffset_tensor,
                        numKeyValueHeads,
                    )
                    v_antiquantoffset_n1d_tensor = _trans_h_to_n1d(
                        v_antiquantoffset_shape,
                        v_antiquantoffset_tensor,
                        numKeyValueHeads,
                    )

                    # GQA场景，扩展kv反量化参数
                    if numKeyValueHeads != numHeads:
                        print("broadcast kvanti")
                        k_antiquant_offset = broadcast_kv_split_dequant_tensor(
                            k_antiquantoffset_n1d_tensor,
                            numKeyValueHeads,
                            numHeads,
                            q_dtype_np,
                        )
                        v_antiquant_offset = broadcast_kv_split_dequant_tensor(
                            v_antiquantoffset_n1d_tensor,
                            numKeyValueHeads,
                            numHeads,
                            q_dtype_np,
                        )
                    else:
                        k_antiquant_offset = k_antiquantoffset_n1d_tensor
                        v_antiquant_offset = v_antiquantoffset_n1d_tensor
            if params["k_antiquant_mode"] == 1:  # pertoken
                # pertoken -> BS
                k_antiquant_offset = k_antiquantoffset_tensor
                v_antiquant_offset = v_antiquantoffset_tensor
        pfa_param["k_antiquant_offset"] = k_antiquant_offset
        pfa_param["v_antiquant_offset"] = v_antiquant_offset

        pfa_param["anti_or_kvanti"] = anti_or_kvanti
        pfa_param["k_antiquant_mode"] = params["k_antiquant_mode"]
        if flagList[14] and "output" not in params["action_type"]:
            # shape_input取tensorshape
            blockSize = params["blocksize"]
            btShape = params["shape_input"][v_end_index + 10]
            kvcacheShape = params["shape_input"][v_end_index + 19]
            if 0 in btShape:
                print("[WARNING]:block_table为空场景，输出空tensor！")
                return torch.zeros(out_shape)
            if 0 in kvcacheShape:
                print("[WARNING]:PA场景下kvcache为空tensor，输出空tensor！")
                return torch.zeros(out_shape)
            block_idx_list = np.arange(kvcacheShape[0])
            block_idx_list = np.random.permutation(block_idx_list)
            block_idx = 0
            block_table = np.full(btShape, -1).astype(np.int32)
            blockNumPerBlock = []
            for actual_seq in actualSeqLengthsKV:
                blockNumPerBlock.append(math.ceil(actual_seq / blockSize))
            for idx, block_per_b in enumerate(blockNumPerBlock):
                for j in range(block_per_b):
                    block_table[idx][j] = block_idx_list[block_idx]
                    block_idx += 1
            pfa_param["blockNumPerBlock"] = blockNumPerBlock
            pfa_param["block_table"] = block_table
            if flagList[26]:
                print("[INFO]:PA + Deepseek！")
                deepseek_ds_pa_preprocessing(pfa_param, params, torch_tensor_list)
            else:
                print("[INFO]:PA + no Deepseek！")
                # trans kv to bsh(此处使用的tensor, 没有经过n的扩展)
                k_cache = torch_tensor_list[v_end_index + 19]
                v_cache = torch_tensor_list[v_end_index + 20]
                print(k_cache.shape)
                print(v_cache.shape)
                if k_cache.ndim == 3:
                    k_tensor_bsh_raw = trans_bnsd_to_bsh(
                        k_tensor_bnsd_list[0], k_shape_bnsd_raw_list[0]
                    )
                    v_tensor_bsh_raw = trans_bnsd_to_bsh(
                        v_tensor_bnsd_list[0], v_shape_bnsd_raw_list[0]
                    )
                    for b, block_per_b in enumerate(blockNumPerBlock):
                        for blockid_index in range(block_per_b):
                            blockid = block_table[b][blockid_index]
                            block_offset = blockid_index * blockSize
                            if blockid_index == block_per_b - 1:
                                blocksize_left = actualSeqLengthsKV[b] - block_offset
                                k_cache[blockid, 0:blocksize_left, :] = (
                                    k_tensor_bsh_raw[
                                        b,
                                        block_offset : (block_offset + blocksize_left),
                                        :,
                                    ]
                                )
                                v_cache[blockid, 0:blocksize_left, :] = (
                                    v_tensor_bsh_raw[
                                        b,
                                        block_offset : (block_offset + blocksize_left),
                                        :,
                                    ]
                                )
                            else:
                                k_cache[blockid, 0:blockSize, :] = k_tensor_bsh_raw[
                                    b, block_offset : (block_offset + blockSize), :
                                ]
                                v_cache[blockid, 0:blockSize, :] = v_tensor_bsh_raw[
                                    b, block_offset : (block_offset + blockSize), :
                                ]
                if k_cache.ndim == 4:
                    k_tensor_bsh_raw = k_tensor_bnsd_list[0]
                    v_tensor_bsh_raw = v_tensor_bnsd_list[0]

                    # gen kv cache
                    for b, block_per_b in enumerate(blockNumPerBlock):
                        for blockid_index in range(block_per_b):
                            blockid = block_table[b][blockid_index]
                            block_offset = blockid_index * blockSize
                            if blockid_index == block_per_b - 1:
                                blocksize_left = actualSeqLengthsKV[b] - block_offset
                                k_cache[blockid, :, 0:blocksize_left, :] = (
                                    k_tensor_bsh_raw[
                                        b,
                                        :,
                                        block_offset : (block_offset + blocksize_left),
                                        :,
                                    ]
                                )
                                v_cache[blockid, :, 0:blocksize_left, :] = (
                                    v_tensor_bsh_raw[
                                        b,
                                        :,
                                        block_offset : (block_offset + blocksize_left),
                                        :,
                                    ]
                                )
                            else:
                                k_cache[blockid, :, 0:blockSize, :] = k_tensor_bsh_raw[
                                    b, :, block_offset : (block_offset + blockSize), :
                                ]
                                v_cache[blockid, :, 0:blockSize, :] = v_tensor_bsh_raw[
                                    b, :, block_offset : (block_offset + blockSize), :
                                ]

                if k_cache.ndim == 5:
                    k_tensor_bsh_raw = k_tensor_bnsd_list[0]
                    v_tensor_bsh_raw = v_tensor_bnsd_list[0]
                    kd = k_tensor_bsh_raw.shape[3]
                    vd = v_tensor_bsh_raw.shape[3]
                    k_cache_bnsd = np.zeros(
                        (k_cache.shape[0], k_cache.shape[1], blockSize, kd)
                    )
                    v_cache_bnsd = np.zeros(
                        (k_cache.shape[0], k_cache.shape[1], blockSize, vd)
                    )
                    # gen kv cache
                    for b, block_per_b in enumerate(blockNumPerBlock):
                        for blockid_index in range(block_per_b):
                            blockid = block_table[b][blockid_index]
                            block_offset = blockid_index * blockSize
                            if blockid_index == block_per_b - 1:
                                blocksize_left = actualSeqLengthsKV[b] - block_offset
                                k_cache_bnsd[blockid, :, 0:blocksize_left, :] = (
                                    k_tensor_bsh_raw[
                                        b,
                                        :,
                                        block_offset : (block_offset + blocksize_left),
                                        :,
                                    ]
                                )
                                v_cache_bnsd[blockid, :, 0:blocksize_left, :] = (
                                    v_tensor_bsh_raw[
                                        b,
                                        :,
                                        block_offset : (block_offset + blocksize_left),
                                        :,
                                    ]
                                )
                            else:
                                k_cache_bnsd[blockid, :, 0:blockSize, :] = (
                                    k_tensor_bsh_raw[
                                        b,
                                        :,
                                        block_offset : (block_offset + blockSize),
                                        :,
                                    ]
                                )
                                v_cache_bnsd[blockid, :, 0:blockSize, :] = (
                                    v_tensor_bsh_raw[
                                        b,
                                        :,
                                        block_offset : (block_offset + blockSize),
                                        :,
                                    ]
                                )

                        if k_dtype == "int8":
                            base = 32
                        else:
                            base = 16
                        k_cache = k_cache_bnsd.reshape(
                            k_cache_bnsd.shape[0],
                            k_cache_bnsd.shape[1],
                            k_cache_bnsd.shape[2],
                            k_cache_bnsd.shape[3] // base,
                            base,
                        ).transpose(0, 1, 3, 2, 4)
                        v_cache = v_cache_bnsd.reshape(
                            v_cache_bnsd.shape[0],
                            v_cache_bnsd.shape[1],
                            v_cache_bnsd.shape[2],
                            v_cache_bnsd.shape[3] // base,
                            base,
                        ).transpose(0, 1, 3, 2, 4)
                        print(f"[INFO]k_cache NZ: {k_cache.shape}")
                        print(f"[INFO]v_cache NZ: {v_cache.shape}")

                kv_dtype = tools.get_np_dtype(params["dtype_input"][1])

                if str(kv_dtype) == "<class 'bfloat16'>":
                    k_cache = torch.from_numpy(k_cache).to(torch.bfloat16)
                    v_cache = torch.from_numpy(v_cache).to(torch.bfloat16)
                elif (
                    str(kv_dtype) != "<class 'float8_e5m2'>"
                    and params["dtype_input"][1] != "hifloat8"
                    and params["dtype_input"][1] != "float8_e4m3fn"
                ):
                    k_cache = torch.from_numpy(k_cache.astype(kv_dtype))
                    v_cache = torch.from_numpy(v_cache.astype(kv_dtype))

    if flagList[15]:
        pfa_param["queryPaddingSize"] = torch_tensor_list[v_end_index + 11]
    if flagList[16]:
        pfa_param["kvPaddingSize"] = torch_tensor_list[v_end_index + 12]
    pfa_param["lseflag"] = params["softmax_lse_flag"]
    pfa_param["lseshape"] = [q_bnsd_shape[0], q_bnsd_shape[1], q_bnsd_shape[2], 1]

    if len(flagList) > 30:
        if flagList[30]:
            sinks = torch_tensor_list[28]

    y_all, lse = _np_promtattention_bnsd(
        q_tensor,
        q_bnsd_shape,
        k_tensor_list,
        k_shape_bnsd_list,
        v_tensor_list,
        v_shape_bnsd_list,
        pse_bnsd_tensor,
        m_bnsd_tensor,
        scaleValue,
        actualSeqLengths,
        actualSeqLengthsKV,
        preTokens,
        nextTokens,
        sp_mode,
        pfa_param,
        dequant_scale1,
        dequant_scale2,
        quant_scale1,
        quant_scale2,
        quant_offset2,
        antiquant_scale,
        antiquant_offset,
        out_dtype,
        params["dtype_input"][0],
        sinks,
    )
    if inputLayout == "BSH":
        y_all = y_all.transpose(0, 2, 1, 3).reshape(out_shape)
    elif inputLayout == "NSD":
        y_all = y_all.reshape(out_shape)
    elif inputLayout == "BSND" or inputLayout == "BNSD_BSND":
        y_all = y_all.transpose(0, 2, 1, 3)
    elif inputLayout == "TND" or inputLayout == "NTD_TND":
        T = sum(actualSeqLengths)
        B = len(actualSeqLengths)
        if inputLayout == "TND":
            N = out_shape[1]
        else:
            N = out_shape[0]
        D = out_shape[2]
        output = np.zeros((T, N, D), dtype=y_all.dtype)
        t_start = 0
        for b_index in range(B):
            act_s = actualSeqLengths[b_index]
            t_end = t_start + act_s
            if act_s == 0:
                continue
            for n_index in range(N):
                output[t_start:t_end, n_index, :] = y_all[b_index, n_index, :act_s, :]
            t_start += act_s
        y_all = output
    elif len(out_shape) == 2:
        if q_bnsd_shape[0] == 1:
            y_all = y_all.transpose(0, 2, 1, 3).reshape(out_shape).astype(np.float32)
        else:
            sums = 0
            B = q_bnsd_shape[0]
            N = q_bnsd_shape[1]
            S = q_bnsd_shape[2]
            D = q_bnsd_shape[3]
            y_all = y_all.transpose(0, 2, 1, 3)
            yNewAll = np.zeros((S, N * D))
            for i in range(B):
                for j in range(actualSeqLengths[i]):
                    yNewAll[sums + j, :] = y_all[i : i + 1, j].reshape(1, N * D)
                sums += actualSeqLengths[i]
            y_all = yNewAll.astype(np.float32)
    else:
        y_all = y_all
    y_all = torch.from_numpy(y_all)
    if params["softmax_lse_flag"] == True:
        if inputLayout == "TND" or inputLayout == "NTD_TND":
            T = sum(actualSeqLengths)
            B = len(actualSeqLengths)
            if inputLayout == "TND":
                N = out_shape[1]
            else:
                N = out_shape[0]
            lse_output = np.zeros((T, N, 1), dtype=lse.dtype)
            t_start = 0
            for b_index in range(B):
                act_s = actualSeqLengths[b_index]
                t_end = t_start + act_s
                if act_s == 0:
                    continue
                for n_index in range(N):
                    lse_output[t_start:t_end, n_index, :] = lse[
                        b_index, n_index, :act_s, :
                    ]
                t_start += act_s
            lse = lse_output
        lse = torch.from_numpy(lse.astype(np.float32))
        return y_all, lse

    # lse = np.zeros([1],dtype=np.float32)
    # # lse = np.zeros(pfa_param['lseshape'], dtype=np.float32)
    # lse = torch.from_numpy(lse.astype(np.float32))
    # return y_all,lse
    return y_all


# FIA
seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

FIA_ENABLE_DEBUG = False
FIA_ENABLE_INFO = True
FIA_USE_TORCH_CPU = True

FIA_ENABLE_DEBUG_DATA = False
FIA_ENABLE_DEBUG_FIA_TENSOR = False
FIA_ENABLE_FUNC_BEGIN_PRINT = False

FIA_ALL_ONE_DEBUG = False

_performance_data = defaultdict(lambda: {"total_time": 0.0, "calls": 0})


def timeit_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        duration = end_time - start_time
        # 更新性能数据
        func_name = func.__qualname__  # 这会包含类名和函数名，例如 MyClass.method
        _performance_data[func_name]["total_time"] += duration
        _performance_data[func_name]["calls"] += 1
        return result

    return wrapper


def print_performance_report(n=10, sort_by="total_time"):
    # 计算平均耗时
    data = []
    for func_name, stats in _performance_data.items():
        total_time = stats["total_time"]
        calls = stats["calls"]
        avg_time = total_time / calls if calls > 0 else 0
        data.append((func_name, total_time, avg_time, calls))
    # 排序
    if sort_by == "total_time":
        sorted_data = sorted(data, key=lambda x: x[1], reverse=True)
    elif sort_by == "avg_time":
        sorted_data = sorted(data, key=lambda x: x[2], reverse=True)
    else:
        raise ValueError("sort_by must be 'total_time' or 'avg_time'")
    # 打印报告
    print(f"{'Function':<40} {'Total Time (s)':<15} {'Avg Time (s)':<15} {'Calls':<10}")
    print("-" * 80)
    for item in sorted_data[:n]:
        func_name, total_time, avg_time, calls = item
        print(f"{func_name:<40} {total_time:15.6f} {avg_time:15.6f} {calls:10}")


def fia_debug_fia_tensor(string):
    if FIA_ENABLE_DEBUG_FIA_TENSOR:
        print(f"[DEBUG] {string}")


def fia_debug_data(string, data):
    if FIA_ENABLE_DEBUG_DATA:
        print(f"[DEBUG] {string} {data.shape} | {data.flatten().round(4)[:10]}")


def fia_debug_func_begin(string):
    if FIA_ENABLE_FUNC_BEGIN_PRINT:
        print(f"[DEBUG] {string}")


def fia_debug(string):
    if FIA_ENABLE_DEBUG:
        print(f"[DEBUG] {string}")


def fia_info(string):
    if FIA_ENABLE_INFO:
        print(f"[INFO] {string}")


def fia_warn(string):
    print(f"[WARNING] {string}")


class StorageMode(Enum):
    CONTIGUOES = 0
    PAGE_ATTENTION = 1
    TENSOR_LIST = 2


class FiaTensor:
    def __init__(
        self,
        data,
        shape,
        dtype,
        layout,
        head_nums=None,
        actual_seq_lens=None,
        name=None,
    ):
        debug_str = f"FiaTensor {name} init, shape: {shape}, layout: {layout}, dtype: {dtype}, head_nums: {head_nums}"
        if actual_seq_lens is not None:
            debug_str += f", actual_seq_lens: {actual_seq_lens}"
        fia_debug_fia_tensor(debug_str)
        if FIA_ALL_ONE_DEBUG:
            self._data = self.init_ones(data, data.shape)
        else:
            self._data = data
        self._shape = shape
        self._layout = layout
        self._bnsd_shape = None
        self._bnsd_data = None
        self._bsnd_shape = None
        self._bsnd_data = None
        self._dtype = self._get_dtype(dtype)
        self._np_dtype = self.get_np_dtype(dtype)
        self._head_nums = head_nums
        self._actual_seq_lens = actual_seq_lens
        self._name = name

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self._data = value
        self._shape = list(value.shape)

    @property
    def shape(self):
        return self._shape

    @property
    def layout(self):
        return self._layout

    @property
    def dtype(self):
        return self._dtype

    @property
    def np_dtype(self):
        return self._np_dtype

    @property
    def bnsd_data(self):
        if self._bnsd_data is None:
            self._bnsd_data = self.to_bnsd()
        return self._bnsd_data

    @property
    def bnsd_shape(self):
        if self._bnsd_shape is None:
            self._bnsd_data = self.to_bnsd()
        return list(self._bnsd_data.shape)

    @property
    def bsnd_data(self):
        if self._bsnd_data is None:
            self._bsnd_data = self.to_bsnd()
        return self._bsnd_data

    @property
    def bsnd_shape(self):
        if self._bsnd_shape is None:
            self._bsnd_data = self.to_bsnd()
        return list(self._bsnd_data.shape)

    def _get_axis(self, axis_name):
        if self.layout != "BNDSD0" and len(self.layout) != len(self.shape):
            raise RuntimeError(
                f"{self._name} the length of layout {self.layout} and shape {self.shape} should be equal"
            )

        if self.layout == "BNDSD0" and len(self.shape) != 5:
            raise RuntimeError(
                f"{self._name} layout is BNDSD0, len(self.shape) = {len(self.shape)} should be 5"
            )

        if axis_name not in self.layout:
            raise RuntimeError(
                f"{self._name} layout {self.layout} do not have axis {axis_name}"
            )

        return self.shape[self.layout.index(axis_name)]

    @property
    def B(self):
        if self.is_tnd_like_layout():
            if not self._actual_seq_lens:
                raise RuntimeError(
                    f"{self._name} layout is {self.layout}, actual_seq_lens should be exists"
                )
            return len(self._actual_seq_lens)
        return self._get_axis("B")

    @property
    def N(self):
        if "N" in self.layout:
            return self._get_axis("N")
        return self._head_nums

    @property
    def S(self):
        if self.is_tnd_like_layout():
            if not self._actual_seq_lens:
                raise RuntimeError(
                    f"{self._name} layout is {self.layout}, actual_seq_lens should be exists"
                )
            return max(self._actual_seq_lens)
        return self._get_axis("S")

    @property
    def D(self):
        if "H" in self.layout:
            if self._head_nums is None:
                raise ValueError(
                    f"{self._name} layout is {self.layout}, head_nums should not be None"
                )
            if self.H % self._head_nums != 0:
                raise RuntimeError(
                    f"H({self.H}) % head_nums({self._head_nums}) should be 0"
                )
            return int(self.H // self._head_nums)
        return self._get_axis("D")

    @property
    def T(self):
        return self._get_axis("T")

    @property
    def H(self):
        return self._get_axis("H")

    def empty(self):
        return 0 in self.shape

    def is_tnd_like_layout(self):
        return self.layout in ["TND", "NTD"]

    @staticmethod
    def _get_dtype(input_dtype):
        if input_dtype == "fp16":
            return "float16"
        elif input_dtype == "int8":
            return "int8"
        elif input_dtype == "uint8":
            return "uint8"
        elif input_dtype == "bf16":
            return "bfloat16"
        elif input_dtype == "bool":
            return "bool"
        elif input_dtype == "int32":
            return "int32"
        elif input_dtype == "fp32":
            return "float32"
        elif input_dtype == "int4":
            return "int4"
        else:
            return input_dtype

    @classmethod
    def transpose(cls, data, dims):
        if isinstance(data, np.ndarray):
            return data.transpose(dims)
        elif isinstance(data, torch.Tensor):
            return data.permute(dims).contiguous()
        else:
            raise RuntimeError(f"Unsupported dtype {type(data)}")

    @classmethod
    def init_zeros(cls, data, shape):
        if isinstance(data, np.ndarray):
            return np.zeros(shape, dtype=data.dtype)
        elif isinstance(data, torch.Tensor):
            return torch.zeros(size=shape, dtype=data.dtype)
        else:
            raise RuntimeError(f"Unsupported dtype {type(data)}")

    @classmethod
    def init_ones(cls, data, shape):
        if isinstance(data, np.ndarray):
            return np.ones(shape, dtype=data.dtype)
        elif isinstance(data, torch.Tensor):
            return torch.ones(size=shape, dtype=data.dtype)
        else:
            raise RuntimeError(f"Unsupported dtype {type(data)}")

    def get_np_dtype(self, type_str):
        type_dict = {
            "fp64": np.float64,
            "fp32": np.float32,
            "fp16": np.float16,
            "int64": np.int64,
            "int32": np.int32,
            "int16": np.int16,
            "int8": np.int8,
            "uint64": np.uint64,
            "uint32": np.uint32,
            "uint16": np.uint16,
            "uint8": np.uint8,
            "bool": np.bool_,
            "complex64": np.complex64,
            "complex128": np.complex128,
            "complex32": np.float16,
            "bf16": tf.bfloat16.as_numpy_dtype,
            "bfloat16": tf.bfloat16.as_numpy_dtype,
            "float4_e2m1": np.uint8,
            "float4_e1m2": np.uint8,
            "float8_e8m0": np.uint8,
            "hifloat8": np.uint8,
            "fp4_e1m2": np.uint8,
            "fp4_e2m1": np.uint8,
            "fp8_e8m0": np.uint8,
            "float32": np.float32,
            "float16": np.float16,
            "qint8": np.int8,
            "qint32": np.int32,
            "quint8": np.uint8,
            "qint16": np.int16,
            "uint1": np.uint8,
            "quint16": np.uint16,
            "fp4_e2m1_as_fp32": np.float32,
        }
        if type_str == "int4":
            from ml_dtypes import int4

            return int4
        elif type_str == "float8_e5m2":
            from ml_dtypes import float8_e5m2

            return float8_e5m2
        elif type_str == "float8_e4m3fn":
            from ml_dtypes import float8_e4m3fn

            return float8_e4m3fn
        else:
            return type_dict[type_str]

    def to_bnsd(self):
        if self.layout == "BNSD":
            return self.data
        elif self.layout == "BSH":
            return self.transpose(
                self.data.reshape(self.B, -1, self._head_nums, self.D), (0, 2, 1, 3)
            )
        elif self.layout == "BSND":
            return self.transpose(self.data, (0, 2, 1, 3))
        elif self.layout == "NBSD":
            return self.transpose(self.data, (1, 0, 2, 3))
        elif self.layout == "TND" or self.layout == "NTD":
            if self._actual_seq_lens is None:
                raise RuntimeError(
                    f"layout is {self.layout}, actual_seq_lens should not be None"
                )
            output_data = self.init_zeros(self.data, (self.B, self.N, self.S, self.D))
            data = (
                self.transpose(self.data, (1, 0, 2))
                if self.layout == "NTD"
                else self.data
            )
            t_start = 0
            for b_idx in range(self.B):
                act_s = self._actual_seq_lens[b_idx]
                t_end = t_start + act_s
                if act_s == 0:
                    continue
                for n_idx in range(self.N):
                    output_data[b_idx, n_idx, 0:act_s, :] = data[
                        t_start:t_end, n_idx, :
                    ]
                t_start += act_s
            return output_data
        else:
            raise ValueError(f"Unsupported layout: {self.layout}")

    def to_bsnd(self):
        return self.transpose(self.bnsd_data, (0, 2, 1, 3))

    def to_layout(self, dst_layout, actual_seq_lens=None):
        if self.layout != "BNSD":
            raise ValueError(f"Unsupported layout {self.layout}")

        B, N, S, D = self.shape
        if dst_layout == "BSH":
            H = N * D
            return self.transpose(self.data, [0, 2, 1, 3]).reshape(B, S, H)
        elif dst_layout == "BSND":
            return self.transpose(self.data, [0, 2, 1, 3])
        elif dst_layout == "NBSD":
            return self.transpose(self.data, [1, 0, 2, 3])
        elif dst_layout == "TND":
            if actual_seq_lens is None:
                raise ValueError("actual_seq_lens must be provided for TND layout.")

            T = sum(actual_seq_lens)
            output_data = self.init_zeros(self.data, (T, N, D))
            t_start = 0

            for b_idx in range(B):
                act_s = actual_seq_lens[b_idx]
                t_end = t_start + act_s
                if act_s == 0:
                    continue
                # 将批次b_idx的数据填充到output[t_start:t_end]
                output_data[t_start:t_end, :, :] = self.transpose(
                    self.data[b_idx, :, :act_s, :], [1, 0, 2]
                )
                t_start += act_s

            return output_data
        elif dst_layout == "NTD":
            if actual_seq_lens is None:
                raise ValueError("actual_seq_lens must be provided for NTD layout.")
            # 先转换为TND，再转置为NTD
            tnd_tensor = self.to_layout("TND", actual_seq_lens)
            return self.transpose(tnd_tensor, [1, 0, 2])
        elif dst_layout == "BNSD":
            return self.data
        else:
            raise RuntimeError(f"Unsupported dst layout: {dst_layout}")

    def trans_to_1n1d(self):
        shape_len = len(self.data.shape)
        if shape_len == 1:
            d = self.data.shape[0] // self._head_nums
            _1n1d = self.data.reshape(((1, self._head_nums, 1, d)))
            self.data = _1n1d
        elif shape_len == 2:
            d = self.data.shape[1]
            _1n1d = self.data.reshape(((1, self._head_nums, 1, d)))
            self.data = _1n1d
        elif shape_len == 3:
            if self.data.shape[0] == 1 and self.data.shape[1] == 1:
                d = self.data.shape[-1] // self._head_nums
                _1n1d = self.data.reshape(((1, self._head_nums, 1, d)))
                self.data = _1n1d
        elif shape_len == 4:
            if self.data.shape[0] == 1 and self.data.shape[1] == 1:
                d = self.data.shape[-1]
                _1n1d = self.data.reshape(((1, self._head_nums, 1, d)))
                self.data = _1n1d
        else:
            print("layout do not support")
        # 根据input将data变成1n1d
        pass


class FiaTensorList(FiaTensor):
    def __init__(
        self,
        data_list,
        shape_list,
        dtype,
        layout,
        head_nums=None,
        actual_seq_lens=None,
        name=None,
    ):
        print()
        debug_str = f"FiaTensorList {name} init, shape_list: {shape_list}, layout: {layout}, dtype: {dtype}, head_nums: {head_nums}"
        if actual_seq_lens is not None:
            debug_str += f", actual_seq_lens: {actual_seq_lens}"
        fia_debug_fia_tensor(debug_str)

        if len(data_list) != len(shape_list):
            raise ValueError(
                f"FiaTensorList {name} len(data_list) = {len(data_list)} should be equal to len(shape_list) = {len(shape_list)}"
            )
        if len(data_list) == 0:
            raise ValueError(f"FiaTensorList {name} len(data_list) should not be 0")

        super().__init__(
            data_list[0], shape_list[0], dtype, layout, head_nums, actual_seq_lens, name
        )
        self._tensor_list = [
            FiaTensor(d, s, dtype, layout, head_nums, actual_seq_lens, name)
            for d, s in zip(data_list, shape_list)
        ]
        self._data_list = data_list
        self._shape_list = shape_list
        self._bnsd_data_list = None
        self._bnsd_shape_list = None
        self._len = len(self._tensor_list)
        self._name = name

    @property
    def S(self):
        S = 0
        for idx in range(self._len):
            S = max(self._tensor_list[idx].S, S)
        return S

    @property
    def kv_s_list(self):
        return [tensor.S for tensor in self._tensor_list]

    def __len__(self):
        return self._len

    def __getitem__(self, index):
        return self._tensor_list[index]

    @property
    def tensor_list(self):
        return self._tensor_list

    @property
    def data_list(self):
        return self._data_list

    @property
    def shape_list(self):
        return self._shape_list

    @property
    def tensor(self):
        return self._tensor_list[0]

    def trans_to_bnsd_list(self):
        data_list = [tensor.bnsd_data for tensor in self._tensor_list]
        shape_list = [tensor.bnsd_shape for tensor in self._tensor_list]
        return data_list, shape_list

    @property
    def bnsd_data_list(self):
        if self._bnsd_data_list is None:
            self._bnsd_data_list, self._bnsd_shape_list = self.trans_to_bnsd_list()
        return self._bnsd_data_list

    @property
    def bnsd_shape_list(self):
        if self._bnsd_shape_list is None:
            self._bnsd_data_list, self._bnsd_shape_list = self.trans_to_bnsd_list()
        return self._bnsd_shape_list


def need_gen_input(action_type):
    if "output" in action_type:
        return False
    return True


def concat_common(tensors, axis):
    if isinstance(tensors[0], np.ndarray):
        return np.concatenate(tensors, axis=axis)
    elif isinstance(tensors[0], torch.Tensor):
        return torch.cat(tensors, dim=axis)
    else:
        raise RuntimeError(f"Unsupported data type {type(tensors[0])}")


def concat_tensor(tensor1, tensor2):
    # Check if the number of dimensions is the same
    if len(tensor1.shape) != len(tensor2.shape):
        raise ValueError(
            f"Number of dimensions mismatch: {len(tensor1.shape)} vs {len(tensor2.shape)}"
        )

    # Check if the layouts are the same
    if tensor1.layout != tensor2.layout:
        raise ValueError(f"Layout mismatch: {tensor1.layout} vs {tensor2.layout}")

    # Check if the dtypes are the same
    if tensor1.dtype != tensor2.dtype:
        raise ValueError(f"Dtype mismatch: {tensor1.dtype} vs {tensor2.dtype}")

    # Check if the head numbers are the same
    if tensor1._head_nums != tensor2._head_nums:
        raise ValueError(
            f"Head numbers mismatch: {tensor1._head_nums} vs {tensor2._head_nums}"
        )

    # Get tensor properties
    layout = tensor1.layout
    dtype = tensor1.dtype
    head_nums = tensor1._head_nums
    actual_seq_lens = tensor1._actual_seq_lens

    # Concatenate based on the layout
    if "D" in layout:
        # Concatenate along the 'D' axis
        d_axis = layout.index("D")
        concat_data = concat_common((tensor1.data, tensor2.data), d_axis)
        concat_shape = list(concat_data.shape)
    elif layout == "BSH":
        # For 'BSH' layout, reshape to 4D and concatenate along the 4th dimension
        concat_shape = [
            tensor1.shape[0],
            tensor1.shape[1],
            tensor1.shape[2] + tensor2.shape[2],
        ]
        tensor1_data_tmp = tensor1.data.reshape(
            tensor1.B, tensor1.S, tensor1.N, tensor1.D
        )
        tensor2_data_tmp = tensor2.data.reshape(
            tensor2.B, tensor2.S, tensor2.N, tensor2.D
        )
        concat_data = concat_common((tensor1_data_tmp, tensor2_data_tmp), 3).reshape(
            concat_shape
        )
    else:
        raise ValueError(f"Unsupported layout: {layout}")

    # Create and return the concatenated tensor
    return FiaTensor(
        concat_data, concat_shape, dtype, layout, head_nums, actual_seq_lens
    )


def concat_tensor_list(tensor_list1, tensor_list2):
    tensor1 = None
    tensor2 = None
    if isinstance(tensor_list1, FiaTensorList):
        if len(tensor_list1) != 1:
            raise ValueError(f"tensor_list1 len = {len(tensor_list1)} must be 1")
        tensor1 = tensor_list1[0]
    else:
        tensor1 = tensor_list1

    if isinstance(tensor_list2, FiaTensorList):
        if len(tensor_list2) != 1:
            raise ValueError(f"tensor_list2 len = {len(tensor_list2)} must be 1")
        tensor2 = tensor_list2[0]
    else:
        tensor2 = tensor_list2

    tensor_concat = concat_tensor(tensor1, tensor2)
    tensor_concat_list = FiaTensorList(
        [tensor_concat.data],
        [tensor_concat.shape],
        tensor_concat.dtype,
        tensor_concat.layout,
        tensor_concat._head_nums,
        tensor_concat._actual_seq_lens,
    )
    return tensor_concat_list


def get_attention_mask_batch_num(npu_m_shape, is_bs):
    batch, numhead = None, None
    if len(npu_m_shape) == 2:
        if is_bs:
            batch = npu_m_shape[0]
            s1 = 1
            s2 = npu_m_shape[1]
        else:
            s1 = npu_m_shape[0]
            s2 = npu_m_shape[1]
        return batch, numhead, s1, s2
    if len(npu_m_shape) == 3:
        batch = npu_m_shape[0]
        s1 = npu_m_shape[1]
        s2 = npu_m_shape[2]
        return batch, numhead, s1, s2
    if len(npu_m_shape) == 4:
        batch = npu_m_shape[0]
        numhead = npu_m_shape[1]
        s1 = npu_m_shape[2]
        s2 = npu_m_shape[3]
        return batch, numhead, s1, s2


def _np_broadcast_mask_n(m_tensor, m_shape, cpu_m_shape, numheads, q_batch):
    print(
        f"broadcast_mask_n:mask shape:{m_shape} with numheads:{numheads} q_batch:{q_batch}"
    )
    mask_cur_shape = cpu_m_shape
    if len(m_shape) == 4:
        # b1ss
        B_m = m_shape[0]
        if B_m != 1:
            B = B_m
        else:
            B = q_batch
        m_res = []
        for i in range(B):
            # mask_cur_shape = [m_shape[2], m_shape[3]]
            if B_m == 1:
                mask_cur = m_tensor[:, :, :].reshape(mask_cur_shape)
            else:
                mask_cur = m_tensor[i : i + 1, :, :, :].reshape(mask_cur_shape)
            m_res.append(mask_cur)
        return m_res
    elif len(m_shape) == 3:
        # bss
        B_m = m_shape[0]
        if B_m != 1:
            B = B_m
        else:
            B = q_batch
        m_res = []
        for i in range(B):
            # mask_cur_shape = [m_shape[1], m_shape[2]]
            if B_m == 1:
                mask_cur = m_tensor[:, :, :].reshape(mask_cur_shape)
            else:
                mask_cur = m_tensor[i : i + 1, :, :].reshape(mask_cur_shape)
            m_res.append(mask_cur)
        return m_res
    elif len(m_shape) == 2:
        # ss
        B = q_batch
        m_res = []
        for i in range(B):
            m_res.append(m_tensor)
        return m_res
    else:
        return m_tensor


def quant(x, qscale, qoffset):
    """
    优化版本：使用矩阵和向量运算替代嵌套循环

    参数说明：
    x: 输入张量，形状为 [N, C, H, W]
    qscale: 量化尺度（标量）
    qoffset: 量化偏移（标量）
    """
    # 将输入转换为half精度
    x_half = np.half(x)

    # 将量化参数转换为half精度
    qscale_half = np.half(qscale)
    qoffset_half = np.half(qoffset)

    # 向量化计算：x * qscale + qoffset
    intermediate = x_half * qscale_half + qoffset_half

    # 应用向量化的饱和函数
    s9_result = s9_saturation_vectorized(intermediate)
    rounded = np.round(s9_result)
    s8_res_cal = s8_saturation_vectorized(rounded)

    return s8_res_cal


def quant_pc(x, qscale, qoffset, n1):
    """
    优化版本：使用矩阵和向量运算替代嵌套循环

    参数说明：
    x: 输入张量，形状为 [N, C, H, W]
    qscale: 量化尺度，形状为 [1, n1, 1, W]
    qoffset: 量化偏移，形状为 [1, n1, 1, W]
    n1: 索引参数
    """
    # 将输入转换为half精度
    x_half = np.half(x)

    # 提取对应的量化参数（广播到与x相同的形状）
    qscale_broadcast = np.half(qscale[0, n1, 0, :])  # 形状: [W]
    qoffset_broadcast = np.half(qoffset[0, n1, 0, :])  # 形状: [W]

    # 重塑量化参数以便进行广播运算
    # 将 [W] 重塑为 [1, 1, 1, W] 以便与 [N, C, H, W] 进行广播
    qscale_reshaped = qscale_broadcast.reshape(1, 1, 1, -1)
    qoffset_reshaped = qoffset_broadcast.reshape(1, 1, 1, -1)

    # 向量化计算：x * qscale + qoffset
    intermediate = x_half * qscale_reshaped + qoffset_reshaped

    # 应用向量化的饱和函数
    s9_result = s9_saturation_vectorized(intermediate)
    rounded = np.round(s9_result)
    s8_res_cal = s8_saturation_vectorized(rounded)

    return s8_res_cal


def s8_saturation_vectorized(inputdata):
    """向量化的s8饱和函数"""
    # 使用np.where替代if-else条件判断
    saturated = np.where(
        inputdata > 127, 127, np.where(inputdata < -128, -128, inputdata)
    )
    return saturated.astype(np.int8)


def s9_saturation_vectorized(inputdata):
    """向量化的s9饱和函数"""
    # 使用np.where替代if-else条件判断
    return np.where(inputdata > 255, 255, np.where(inputdata < -256, -256, inputdata))


class FiaSoftmax:
    @staticmethod
    def softmaxv1(x):
        x = x.astype(np.float32)
        x_max = x.max(axis=-1, keepdims=True)
        x_sub = x - x_max
        y = np.exp(x_sub)
        x_sum = y.sum(axis=-1, keepdims=True)
        ans = y / x_sum
        return ans, x_max, x_sum

    @staticmethod
    def softmax(x, sinks=None):
        if x.dtype != np.float32:
            x = x.astype(np.float32, copy=False)

        x_max = np.max(x, axis=-1, keepdims=True)
        x -= x_max
        np.exp(x, out=x)
        x_sum = np.sum(x, axis=-1, keepdims=True, dtype=np.float32)
        if sinks is not None:
            sinks_reshaped = sinks.reshape(1, -1, 1, 1)
            sinks_sub = sinks_reshaped - x_max
            sink_exp = np.exp(sinks_sub)
            x_sum += sink_exp.sum(axis=-1, keepdims=True)
        return x, x_sum, x_max

    @staticmethod
    def _t_softmax(x):
        x_max = torch.max(x, dim=-1, keepdims=True)[0]
        x_sub = x.sub(x_max)
        y = torch.exp(x_sub)
        x_sum = y.sum(dim=-1, keepdims=True)
        ans = y.div(x_sum)
        return ans, x_max, x_sum


class MaskGenerator:
    @staticmethod
    def _random_fill_tensor(tensor, shape, random_number, value=0):
        for i in range(0, random_number):
            point = []
            for k in range(0, len(shape)):
                point.append(random.randint(1, shape[k]) - 1)
            tensor[point[0], point[1]] = value
        return tensor

    @classmethod
    def _create_mask_right_down(
        cls,
        m_shape,
        pre_tokens,
        next_tokens,
        actualSeqLengths,
        actualSeqLengthsKV,
        actualprefixKV,
        prefix_kvs,
        batch,
        numheads,
        kv_s_list,
        m_dtype,
    ):
        mask_s_q = m_shape[0]
        mask_s_kv = m_shape[1]

        next_tokens_list = []
        re_mask_batch = []
        for i in range(batch):
            if len(actualSeqLengths) == 0:
                S1 = mask_s_q
            else:
                S1 = actualSeqLengths[i]
            if len(actualSeqLengthsKV) != 0:
                S2 = actualSeqLengthsKV[i] + actualprefixKV
            elif len(kv_s_list) > 1:
                S2 = kv_s_list[i] + actualprefixKV
            else:
                S2 = mask_s_kv - prefix_kvs + actualprefixKV
            next_tokens = S2 - S1
            next_tokens_list.append(next_tokens)
            atten_masks = cls._create_mask(m_shape, pre_tokens, next_tokens)
            re_mask_batch.append(atten_masks)
        return re_mask_batch, next_tokens_list

    @staticmethod
    def _create_mask(m_shape, pre_tokens, next_tokens):
        next_masks = np.triu(
            np.ones(m_shape, dtype="uint8"), k=1 + int(next_tokens)
        )  # 生成下三角全是0的矩阵
        pre_mask = np.tril(
            np.ones(m_shape, dtype="uint8"), k=-1 - int(pre_tokens)
        )  # 生成上三角全是0的矩阵
        atten_masks = pre_mask + next_masks

        return atten_masks

    @classmethod
    def _create_mask_no_sparse(
        cls,
        m_shape,
        npu_m_shape,
        pre_tokens,
        next_tokens,
        batch,
        numheads,
        m_dtype,
        random_ones=0,
    ):
        re_mask_batch = []
        re_mask_npu_batch = []

        pad_flag = False
        npu_mask = None
        if m_shape[0] != npu_m_shape[0] or m_shape[1] != npu_m_shape[1]:
            pad_flag = True
            npu_mask = np.ones(npu_m_shape, dtype="uint8")
        cpu_mask = cls._create_mask(m_shape, pre_tokens, next_tokens)
        if pad_flag:
            if batch is None:
                cpu_mask = cls._random_fill_tensor(cpu_mask, m_shape, random_ones, 1)
                npu_mask[: cpu_mask.shape[0], : cpu_mask.shape[1]] = cpu_mask
                return cpu_mask, npu_mask
            for i in range(batch):
                re_mask_num = []
                re_mask_npu_num = []
                re_mask = cls._random_fill_tensor(cpu_mask, m_shape, random_ones, 1)

                npu_mask[: re_mask.shape[0], : re_mask.shape[1]] = re_mask
                if numheads:
                    # cpu
                    re_mask_num.append(re_mask)
                    re_mask_batch.append(re_mask_num)
                    # npu

                    re_mask_npu_num.append(npu_mask)
                    re_mask_npu_batch.append(re_mask_npu_num)
                else:
                    re_mask_batch.append(re_mask)
                    re_mask_npu_batch.append(npu_mask)
            cpu_mask = np.array(re_mask_batch).astype(m_dtype)
            npu_mask = np.array(re_mask_npu_batch).astype(m_dtype)
            return cpu_mask, npu_mask
        else:
            if batch is None:
                cpu_mask = cls._random_fill_tensor(cpu_mask, m_shape, random_ones, 1)
                return cpu_mask, cpu_mask
            for i in range(batch):
                re_mask_num = []
                re_mask = cls._random_fill_tensor(cpu_mask, m_shape, random_ones, 1)
                if numheads:
                    re_mask_num.append(re_mask)
                    re_mask_batch.append(re_mask_num)
                else:
                    re_mask_batch.append(re_mask)

            cpu_mask = np.array(re_mask_batch).astype(m_dtype)
            return cpu_mask, cpu_mask

    @classmethod
    def _create_mask_left_up(
        cls, m_shape, pre_tokens, next_tokens, batch, numheads, m_dtype, random_ones=0
    ):
        re_mask_batch = []
        attentionmask = cls._create_mask(m_shape, pre_tokens, next_tokens)
        for i in range(batch):
            re_mask_batch.append(attentionmask)
        return re_mask_batch

    @classmethod
    def _create_mask_band(
        cls,
        m_shape,
        pre_tokens,
        next_tokens,
        actualSeqLengths,
        actualSeqLengthsKV,
        actualprefixKV,
        prefix_kvs,
        batch,
        numheads,
        kv_s_list,
        m_dtype,
    ):
        mask_s_q = m_shape[0]
        mask_s_kv = m_shape[1]

        pre_tokens_list = []
        next_tokens_list = []
        re_mask_batch = []
        for i in range(batch):
            if len(actualSeqLengths) == 0:
                S1 = mask_s_q
            else:
                S1 = actualSeqLengths[i]
            if len(actualSeqLengthsKV) != 0:
                S2 = actualSeqLengthsKV[i] + actualprefixKV
            elif len(kv_s_list) > 1:
                S2 = kv_s_list[i] + actualprefixKV
            else:
                S2 = mask_s_kv - prefix_kvs + actualprefixKV
            pre_tokens_new = S1 - S2 + pre_tokens
            pre_tokens_list.append(pre_tokens_new)

            next_tokens_new = S2 - S1 + next_tokens
            next_tokens_list.append(next_tokens_new)
            atten_masks = cls._create_mask(m_shape, pre_tokens_new, next_tokens_new)
            re_mask_batch.append(atten_masks)
        return re_mask_batch, pre_tokens_list, next_tokens_list

    @classmethod
    def _create_random_mask_by_spars(
        cls,
        cpu_m_shape,
        npu_m_shape,
        m_dtype,
        pre_tokens,
        next_tokens,
        actualSeqLengths,
        actualSeqLengthsKV,
        actualprefixKV,
        prefix_kvs,
        kv_s_list,
        batch=1,
        numheads=1,
        sp_mode=0,
        random_ones=0,
    ):
        # mask shape [sq,skv]  #mshape  npu  fshape cpu
        print(
            f"[_create_random_mask_by_spars] full_m_shape:{cpu_m_shape} m_shape:{npu_m_shape} datype:{m_dtype} pret:{pre_tokens} nextt:{next_tokens} sp_mode:{sp_mode}"
        )
        if sp_mode == 0:
            cpu_mask, npu_mask = cls._create_mask_no_sparse(
                cpu_m_shape,
                npu_m_shape,
                pre_tokens,
                next_tokens,
                batch,
                numheads,
                m_dtype,
                random_ones,
            )
            return cpu_mask, npu_mask.astype(m_dtype), pre_tokens, next_tokens
        if sp_mode == 1:
            print("[_create_random_mask_by_spars] sp_mode is 1 return all zero mask")
            pre_tokens = 214748647
            next_tokens = 214748647
            cpu_mask, npu_mask = cls._create_mask_no_sparse(
                cpu_m_shape,
                npu_m_shape,
                pre_tokens,
                next_tokens,
                batch,
                numheads,
                m_dtype,
                random_ones,
            )
            return cpu_mask, npu_mask.astype(m_dtype), pre_tokens, next_tokens

        if sp_mode == 2:
            pre_tokens = 214748647
            next_tokens = 0
            print(
                f"[_create_random_mask_by_spars] sp_mode is 2 npu mask shape:{npu_m_shape}"
            )
            npu_mask = np.triu(np.ones(npu_m_shape), k=1)
            cpu_mask = cls._create_mask_left_up(
                cpu_m_shape, pre_tokens, next_tokens, batch, numheads, m_dtype
            )
            return cpu_mask, npu_mask.astype(m_dtype), pre_tokens, next_tokens
        if sp_mode == 3:  # rightdown
            pre_tokens = 214748647
            print(
                f"[_create_random_mask_by_spars] sp_mode is 3 npu mask shape:{npu_m_shape}"
            )
            npu_mask = np.triu(np.ones(npu_m_shape), k=1)
            cpu_mask, next_tokens_new = cls._create_mask_right_down(
                cpu_m_shape,
                pre_tokens,
                next_tokens,
                actualSeqLengths,
                actualSeqLengthsKV,
                actualprefixKV,
                prefix_kvs,
                batch,
                numheads,
                kv_s_list,
                m_dtype,
            )
            return cpu_mask, npu_mask.astype(m_dtype), pre_tokens, next_tokens_new
        if sp_mode == 4:
            npu_mask = np.triu(np.ones(npu_m_shape), k=1)
            cpu_mask, pre_tokens_new, next_tokens_new = cls._create_mask_band(
                cpu_m_shape,
                pre_tokens,
                next_tokens,
                actualSeqLengths,
                actualSeqLengthsKV,
                actualprefixKV,
                prefix_kvs,
                batch,
                numheads,
                kv_s_list,
                m_dtype,
            )
            return (
                np.array(cpu_mask, dtype=m_dtype),
                npu_mask.astype(m_dtype),
                pre_tokens_new,
                next_tokens_new,
            )


class FiaBoradCastTool:
    @classmethod
    def broadcast_kv_n2_to_n1(cls, num_heads, num_kv_heads, kv_tensor, input_dtype):
        factor = num_heads // num_kv_heads
        kv_shape = kv_tensor.shape
        B = kv_shape[0]
        S = kv_shape[2]
        D = kv_shape[3]
        kv_res = np.zeros([B, num_heads, S, D], dtype=input_dtype)
        for i in range(num_heads):
            j = i // factor
            kv_res[:, i : i + 1, :, :] = kv_tensor[:, j : j + 1, :, :]
        return kv_res, kv_res.shape


class FiaLayoutTool:
    @classmethod
    def transpose(cls, data, dims):
        if isinstance(data, np.ndarray):
            return data.transpose(dims)
        elif isinstance(data, torch.Tensor):
            return data.permute(dims).contiguous()
        else:
            raise RuntimeError(f"Unsupported data type {type(data)}")

    @classmethod
    def trans_bnsd_to_target(cls, tensor, shape, target_layout, actual_seq_lens=None):
        B, N, S, D = tensor.shape

        if target_layout == "BSH":
            H = N * D
            return cls.transpose(tensor, [0, 2, 1, 3]).reshape(B, S, H)
        elif target_layout == "BSND":
            return cls.transpose(tensor, [0, 2, 1, 3])
        elif target_layout == "NBSD":
            return cls.transpose(tensor, [1, 0, 2, 3])
        elif target_layout == "TND":
            if actual_seq_lens is None:
                raise ValueError("actual_seq_lens must be provided for TND layout.")

            T = sum(actual_seq_lens)
            output = torch.zeros(size=(T, N, D), dtype=tensor.dtype)
            t_start = 0

            for b_idx in range(B):
                act_s = actual_seq_lens[b_idx]
                t_end = t_start + act_s
                if act_s == 0:
                    continue
                # 将批次b_idx的数据填充到output[t_start:t_end]
                output[t_start:t_end, :, :] = cls.transpose(
                    tensor[b_idx, :, :act_s, :], [1, 0, 2]
                )
                t_start += act_s

            return output
        elif target_layout == "NTD":
            if actual_seq_lens is None:
                raise ValueError("actual_seq_lens must be provided for NTD layout.")
            # 先转换为TND，再转置为NTD
            tnd_tensor = cls.trans_bnsd_to_target(tensor, shape, "TND", actual_seq_lens)
            return cls.transpose(tnd_tensor, [1, 0, 2])
        elif target_layout == "BNSD":
            return tensor
        else:
            raise RuntimeError(
                f"trans_bnsd_to_target does not support target_layout: {target_layout}"
            )

    @classmethod
    def trans_bnsd_to_bsh(cls, tensor, shape):
        return cls.trans_bnsd_to_target(tensor, shape, "BSH")


class FiaOpParam:
    _flag_list_index = {
        "query": 0,
        "key": 1,
        "value": 2,
        "pse_shift": 3,
        "atten_mask": 4,
        "actual_seq_lens_q": 5,
        "actual_seq_lens_kv": 6,
        "dequant_scale1": 7,
        "quant_scale1": 8,
        "dequant_scale2": 9,
        "quant_scale2": 10,
        "quant_offset2": 11,
        "antiquant_scale": 12,
        "antiquant_offset": 13,
        "block_table": 14,
        "q_padding_size": 15,
        "kv_padding_size": 16,
        "k_antiquant_scale": 17,
        "k_antiquant_offset": 18,
        "v_antiquant_scale": 19,
        "v_antiquant_offset": 20,
        "k_shared_prefix": 21,
        "v_shared_prefix": 22,
        "actual_shared_prefix_len": 23,
        "output": 24,
        "input_layout": 25,
        "q_rope": 26,
        "k_rope": 27,
        "key_rope_scale": 28,
        "sinks": 30,
    }

    _param_index = {
        "pse_shift": 1,
        "atten_mask": 2,
        "dequant_scale1": 3,
        "quant_scale1": 4,
        "dequant_scale2": 5,
        "quant_scale2": 6,
        "quant_offset2": 7,
        "antiquant_scale": 8,
        "antiquant_offset": 9,
        "block_table": 10,
        "q_padding_size": 11,
        "kv_padding_size": 12,
        "k_antiquant_scale": 13,
        "k_antiquant_offset": 14,
        "v_antiquant_scale": 15,
        "v_antiquant_offset": 16,
        "k_shared_prefix": 17,
        "v_shared_prefix": 18,
        "k_cache": 19,
        "v_cache": 20,
        "q_rope": 21,
        "k_rope": 22,
        "k_rope_cache": 23,
        "dequant_scale_query": 24,
        "sinks": 26,
    }

    def _get_kv_num(self):
        if self.tnd_flag:
            return 1
        else:
            q_shape = self.data_list[0].shape
            kv_shape = self.data_list[1].shape
            q_b = q_shape[0]
            k_b = kv_shape[0]
            if q_b == k_b:
                return 1
            else:
                return q_b

    @classmethod
    def _get_param_index(cls, data_name, kv_num=1):
        k_start_index = 1
        k_end_index = kv_num
        v_start_index = kv_num + 1
        v_end_index = kv_num + kv_num
        if data_name == "query":
            return 0
        elif data_name == "key":
            return k_start_index, k_end_index + 1
        elif data_name == "value":
            return v_start_index, v_end_index + 1
        else:
            if data_name not in cls._param_index:
                raise ValueError(f"data_name {data_name} is invalid")
            data_index = cls._param_index[data_name]
            return v_end_index + data_index

    @classmethod
    def get_param_index(cls, data_name, kv_num=1):
        index = cls._get_param_index(data_name, kv_num)
        if isinstance(index, tuple):
            return index[0]
        else:
            return index

    def _get_data(self, data_name):
        index = self._get_param_index(data_name, self.kv_num)
        if isinstance(index, tuple):
            return self.data_list[index[0] : index[1]]
        else:
            return self.data_list[index]

    def _get_range(self, range_name):
        index = self._get_param_index(range_name, self.kv_num)
        if isinstance(index, tuple):
            return self.params["range_input"][index[0]]
        else:
            return self.params["range_input"][index]

    def _get_shape(self, shape_name):
        index = self._get_param_index(shape_name, self.kv_num)
        if isinstance(index, tuple):
            return self.params["shape_input"][index[0] : index[1]]
        else:
            return self.params["shape_input"][index]

    def _get_dtype(self, dtype_name):
        index = self._get_param_index(dtype_name, self.kv_num)
        if isinstance(index, tuple):
            return self.params["dtype_input"][index[0]]
        else:
            return self.params["dtype_input"][index]

    def _debug_info(self):
        fia_debug(f"action_type: {self.action_type}")
        fia_debug(f"rope_flag: {self.rope_flag}")
        fia_debug(f"storage_mode: {self.storage_mode}")
        fia_debug(f"kv_num: {self.kv_num}")
        fia_debug(f"pa_flag: {self.pa_flag}")
        fia_debug(f"tnd_flag: {self.tnd_flag}")
        fia_debug(f"num_heads: {self.num_heads}")
        fia_debug(f"num_kv_heads: {self.num_kv_heads}")
        fia_debug(f"input_layout: {self.input_layout}")
        fia_debug(f"q_layout: {self.q_layout}")
        fia_debug(f"out_layout: {self.out_layout}")
        fia_debug(f"kv_layout: {self.kv_layout}")
        fia_debug(f"actual_seq_lens_q_raw: {self.actual_seq_lens_q_raw}")
        fia_debug(f"actual_seq_lens_q: {self.actual_seq_lens_q}")
        fia_debug(f"actual_seq_lens_kv_raw: {self.actual_seq_lens_kv_raw}")
        fia_debug(f"actual_seq_lens_kv: {self.actual_seq_lens_kv}")

        fia_debug(f"is_deepseek_mla: {self.is_deepseek_mla}")
        fia_debug(f"key   shape: {self.key.shape}")
        fia_debug(f"value shape: {self.value.shape}")
        fia_debug(f"key   bnsd shape: {self.key.bnsd_shape}")
        fia_debug(f"value bnsd shape: {self.value.bnsd_shape}")

        fia_debug(f"scale_value: {self.scale_value}")
        fia_debug(f"block_size: {self.block_size}")
        fia_debug(f"inner_precise: {self.inner_precise}")

        fia_debug(f"pre_tokens: {self.pre_tokens}")
        fia_debug(f"next_tokens: {self.next_tokens}")
        fia_debug(f"sparse_mode: {self.sparse_mode}")
        fia_debug(f"softmax_lse_flag: {self.softmax_lse_flag}")
        fia_debug(f"out_quant_flag: {self.out_quant_flag}")
        fia_debug(f"pse_shift_flag: {self.pse_shift_flag}")
        fia_debug(f"atten_mask_flag: {self.atten_mask_flag}")
        fia_debug(f"q_padding_size_flag: {self.q_padding_size_flag}")
        fia_debug(f"kv_padding_size_flag: {self.kv_padding_size_flag}")
        fia_debug(f"shared_prefix_flag: {self.shared_prefix_flag}")
        fia_debug(f"prefix_act_flag: {self.prefix_act_flag}")
        fia_debug(f"sink_flag: {self.sink_flag}")

    def __init__(self, data_list, params):
        self.data_list = data_list
        self.params = params

        self.flag_list = self.str_to_bool_list(self.params["flaglist"])

        self.parse_basic_info()
        self.parse_flag_list()
        self._debug_info()

    def _expand_actual_seq_lens(self, actual_seq_lens):
        if len(actual_seq_lens) == 1 and self.batch > 1:
            return actual_seq_lens * self.batch
        elif len(actual_seq_lens) > self.batch:
            return actual_seq_lens[: self.batch]
        return actual_seq_lens

    @staticmethod
    def _trans_tnd_actseq(actual_seq_lens):
        normal_seq_lens = [actual_seq_lens[0]]
        for i in range(len(actual_seq_lens) - 1):
            seq_len = actual_seq_lens[i + 1] - actual_seq_lens[i]
            if seq_len < 0:
                raise RuntimeError(
                    f"_trans_tnd_actseq: actual_seq_lens[{i}] = {seq_len}, it should >= 0"
                )
            normal_seq_lens.append(seq_len)
        return normal_seq_lens

    def _get_actual_seq_lens_q(self):
        actual_seq_lens_q = self._expand_actual_seq_lens(
            copy.deepcopy(self.params["actualseqlengths"])
        )
        if self.tnd_flag:
            # 将tnd格式下的act seq转成普通的act seq
            fia_debug("_trans_tnd_actseq actual_seq_lens_q")
            actual_seq_lens_q = self._trans_tnd_actseq(actual_seq_lens_q)
        return actual_seq_lens_q

    def _get_actual_seq_lens_kv(self):
        actual_seq_lens_kv = self._expand_actual_seq_lens(
            copy.deepcopy(self.params["actualseqlengthskv"])
        )
        if self.tnd_flag and (not self.pa_flag):
            fia_debug("_trans_tnd_actseq actual_seq_lens_kv")
            # 将tnd格式下的act seq转成普通的act seq
            actual_seq_lens_kv = self._trans_tnd_actseq(actual_seq_lens_kv)
        return actual_seq_lens_kv

    def parse_basic_info(self):
        self.tnd_flag = self._get_tnd_flag()
        self.action_type = self._get_action_type()
        self.input_layout = self._get_input_layout()
        self.kv_num = self._get_kv_num()
        self.num_heads = self._get_num_heads()
        self.num_kv_heads = self._get_num_kv_heads()
        self.rope_flag = self._get_flag("q_rope")
        self.storage_mode = self._get_storage_mode()
        self.pa_flag = self.storage_mode == StorageMode.PAGE_ATTENTION

        self._parse_layout()
        self.actual_seq_lens_q_raw = self._get_actual_seq_lens_q_raw()
        self.actual_seq_lens_kv_raw = self._get_actual_seq_lens_kv_raw()
        if not self.tnd_flag:
            self.batch = FiaTensor(
                self._get_data("query"),
                self._get_shape("query"),
                self._get_dtype("query"),
                self.q_layout,
                name="dummy",
            ).B
        else:
            self.batch = len(self.actual_seq_lens_q_raw)
        self.actual_seq_lens_q = self._get_actual_seq_lens_q()
        self.actual_seq_lens_kv = self._get_actual_seq_lens_kv()

        self.scale_value = self._get_scale_value()
        self.block_size = self._get_block_size()
        self.inner_precise = self._get_inner_precise()

        self.antiquant_mode = self._get_antiquant_mode()
        self.k_antiquant_mode = self._get_k_antiquant_mode()
        self.v_antiquant_mode = self._get_v_antiquant_mode()

        self.pre_tokens = self._get_pretokens()
        self.next_tokens = self._get_nexttokens()
        self.sparse_mode = self._get_sparse_mode()
        self.softmax_lse_flag = self._get_softmax_lse_flag()

        self._parse_input_tensor()
        self._parse_output_tensor()
        self._parse_optional_tensor()
        self._parse_quant_info()

    def _get_flag(self, flag_name):
        if flag_name not in self._flag_list_index:
            raise ValueError(f"param flag_name {flag_name} should in _flag_list_index")
        index = self._flag_list_index[flag_name]
        return self.flag_list[index]

    def _get_action_type(self):
        return self.params["action_type"]

    def _get_input_layout(self):
        return self.params["inputlayout"]

    def _get_actual_seq_lens_q_raw(self):
        return self.params["actualseqlengths"]

    def _get_actual_seq_lens_kv_raw(self):
        return self.params["actualseqlengthskv"]

    def _get_num_heads(self):
        return self.params["numheads"]

    def _get_scale_value(self):
        return self.params["scalevalue"]

    def _get_block_size(self):
        return self.params["blocksize"]

    def _get_inner_precise(self):
        return self.params["innerprecise"]

    def _get_antiquant_mode(self):
        return str(self.params["antiquant_mode"])

    def _get_k_antiquant_mode(self):
        return str(self.params["k_antiquant_mode"])

    def _get_v_antiquant_mode(self):
        return str(self.params["v_antiquant_mode"])

    def _get_pretokens(self):
        return self.params["pretokens"]

    def _get_nexttokens(self):
        return self.params["nexttokens"]

    def _get_sparse_mode(self):
        return self.params["sparsemode"]

    def _get_softmax_lse_flag(self):
        return self.params["softmax_lse_flag"]

    def _get_num_kv_heads(self):
        # 当numKeyValueHeads传入0时，处理为与numHeads相等
        return (
            self.params["numkeyvalueheads"]
            if self.params["numkeyvalueheads"] != 0
            else self.params["numheads"]
        )

    def _get_tnd_flag(self):
        return self.params["inputlayout"] in ["TND", "TND_NTD", "NTD", "NTD_TND"]

    def _get_storage_mode(self):
        if self._get_flag("block_table"):
            return StorageMode.PAGE_ATTENTION

        if self.kv_num > 1:
            return StorageMode.TENSOR_LIST

        return StorageMode.CONTIGUOES

    def _parse_layout(self):
        self.q_layout = self.params["inputlayout"].split("_")[0]
        self.out_layout = self.params["inputlayout"].split("_")[-1]
        self.kv_layout = (
            "BNSD"
            if (self.tnd_flag and (self.storage_mode == StorageMode.PAGE_ATTENTION))
            else self.q_layout
        )
        self.lse_layout = "TND" if self.tnd_flag else "BNSD"

    def _parse_input_tensor(self):
        self.query = FiaTensor(
            self._get_data("query"),
            self._get_shape("query"),
            self._get_dtype("query"),
            self.q_layout,
            self.num_heads,
            self.actual_seq_lens_q,
            name="query",
        )
        self.key = FiaTensorList(
            self._get_data("key"),
            self._get_shape("key"),
            self._get_dtype("key"),
            self.kv_layout,
            self.num_kv_heads,
            self.actual_seq_lens_kv,
            name="key",
        )
        self.value = FiaTensorList(
            self._get_data("value"),
            self._get_shape("value"),
            self._get_dtype("value"),
            self.kv_layout,
            self.num_kv_heads,
            self.actual_seq_lens_kv,
            name="value",
        )
        self.is_deepseek_mla = (
            (self.query.D == 512) and (self.value.D == 512) and self.rope_flag
        )
        if self.is_deepseek_mla:
            self.value = FiaTensorList(
                self._get_data("key"),
                self._get_shape("key"),
                self._get_dtype("key"),
                self.kv_layout,
                self.num_kv_heads,
                self.actual_seq_lens_kv,
                name="value",
            )
        else:
            self.value = FiaTensorList(
                self._get_data("value"),
                self._get_shape("value"),
                self._get_dtype("value"),
                self.kv_layout,
                self.num_kv_heads,
                self.actual_seq_lens_kv,
                name="value",
            )

        self.q_s = self.query.S
        self.kv_s = self.key.S
        self.q_d = self.query.D
        self.v_d = self.value.D
        self.kv_s_list = self.key.kv_s_list
        self.q_t = self.query.T if self.query.is_tnd_like_layout() else 0

        kv_num = self.batch if (self.storage_mode == StorageMode.TENSOR_LIST) else 1
        if kv_num != self.kv_num:
            raise RuntimeError(
                f"kv_num({kv_num}) calculate by batch is not equal to kv_num({self.kv_num}) calculate by data_list num"
            )
        if self.storage_mode == StorageMode.CONTIGUOES:
            if len(self.key) != 1 or len(self.value) != 1:
                raise RuntimeError(
                    f"In CONTIGUOES situation, len(key) = {len(self.key)} \
                    and len(value) = {len(self.value)} should be equal to 1"
                )
        if self.storage_mode == StorageMode.TENSOR_LIST:
            if self.key[0].B != 1 or self.value[0].B != 1:
                raise RuntimeError(
                    f"In TENSOR_LIST situation, key[0].B = {self.key[0].B} \
                    and value[0].B = {self.value[0].B} should be 1"
                )
        self._parse_page_attention_input()

    def _get_kv_cache_layout(self, k_cache_shape):
        kv_cache_layout = "BSH"
        if len(k_cache_shape) == 1:
            fia_info("kv_cache shape is 1")
        elif len(k_cache_shape) == 3:
            kv_cache_layout = "BSH"
        elif len(k_cache_shape) == 4:
            kv_cache_layout = "BNSD"
        elif len(k_cache_shape) == 5:
            kv_cache_layout = "BNDSD0"
        else:
            raise ValueError(
                f"len(k_cache_shape) should be in 3/4/5, got {len(k_cache_shape)}"
            )
        return kv_cache_layout

    def _parse_page_attention_input(self):
        self.block_table = FiaTensor(
            self._get_data("block_table"),
            self._get_shape("block_table"),
            self._get_dtype("block_table"),
            "ND",
            name="block_table",
        )
        k_cache_shape = self._get_shape("k_cache")
        v_cache_shape = self._get_shape("v_cache")
        if len(k_cache_shape) != len(v_cache_shape):
            raise RuntimeError(
                f"len(k_cache_shape) = {len(k_cache_shape)} should be \
                equal to len(v_cahce_shape) = {len(v_cache_shape)}"
            )
        self.kv_cache_layout = self._get_kv_cache_layout(k_cache_shape)
        self.k_cache = FiaTensor(
            self._get_data("k_cache"),
            k_cache_shape,
            self._get_dtype("k_cache"),
            self.kv_cache_layout,
            name="k_cache",
        )
        self.v_cache = FiaTensor(
            self._get_data("v_cache"),
            v_cache_shape,
            self._get_dtype("v_cache"),
            self.kv_cache_layout,
            name="v_cache",
        )
        self.k_rope_cache = FiaTensor(
            self._get_data("k_rope_cache"),
            self._get_shape("k_rope_cache"),
            self._get_dtype("k_rope_cache"),
            self.kv_cache_layout,
            name="k_rope_cache",
        )

    def _get_output_shape(self):
        if self.out_layout == "BSND":
            return [self.batch, self.q_s, self.num_heads, self.v_d]
        elif self.out_layout == "BNSD":
            return [self.batch, self.num_heads, self.q_s, self.v_d]
        elif self.out_layout == "NBSD":
            return [self.num_heads, self.batch, self.q_s, self.v_d]
        elif self.out_layout == "BSH":
            return [self.batch, self.q_s, self.num_heads * self.v_d]
        elif self.out_layout == "TND":
            return [self.q_t, self.num_heads, self.v_d]
        elif self.out_layout == "NTD":
            return [self.num_heads, self.q_t, self.v_d]
        else:
            raise ValueError(f"unsupported out_layout {self.out_layout}")

    def _get_lse_shape(self):
        if self.lse_layout == "TND":
            return [self.q_t, self.num_heads, 1]
        elif self.lse_layout == "BNSD":
            return [self.batch, self.num_heads, self.q_s, 1]
        else:
            raise ValueError(f"unsupported lse_layout {self.lse_layout}")

    def _parse_output_tensor(self):
        output_shape = self._get_output_shape()
        self.output = FiaTensor(
            np.zeros(output_shape),
            output_shape,
            self.params["dtype_output"][0],
            self.out_layout,
            self.num_heads,
            self.actual_seq_lens_q,
            name="output",
        )
        lse_shape = self._get_lse_shape()
        self.lse = FiaTensor(
            np.full(lse_shape, np.inf),
            lse_shape,
            "fp32",
            self.lse_layout,
            self.num_heads,
            self.actual_seq_lens_q,
            name="lse",
        )

    def _parse_optional_tensor(self):
        self.q_rope = FiaTensor(
            self._get_data("q_rope"),
            self._get_shape("q_rope"),
            self._get_dtype("q_rope"),
            self.q_layout,
            self.num_heads,
            self.actual_seq_lens_q,
            name="q_rope",
        )

        self.k_rope = FiaTensor(
            self._get_data("k_rope"),
            self._get_shape("k_rope"),
            self._get_dtype("k_rope"),
            self.kv_layout,
            self.num_kv_heads,
            self.actual_seq_lens_kv,
            name="k_rope",
        )
        self.pse_shift = FiaTensor(
            self._get_data("pse_shift"),
            self._get_shape("pse_shift"),
            self._get_dtype("pse_shift"),
            "ND",
            name="pse_shift",
        )
        self.atten_mask = FiaTensor(
            self._get_data("atten_mask"),
            self._get_shape("atten_mask"),
            self._get_dtype("atten_mask"),
            "ND",
            name="atten_mask",
        )
        self.k_shared_prefix = FiaTensor(
            self._get_data("k_shared_prefix"),
            self._get_shape("k_shared_prefix"),
            self._get_dtype("k_shared_prefix"),
            self.kv_layout,
            self.num_kv_heads,
            name="k_shared_prefix",
        )
        self.v_shared_prefix = FiaTensor(
            self._get_data("v_shared_prefix"),
            self._get_shape("v_shared_prefix"),
            self._get_dtype("v_shared_prefix"),
            self.kv_layout,
            self.num_kv_heads,
            name="v_shared_prefix",
        )
        self.sinks = FiaTensor(
            self._get_data("sinks"),
            self._get_shape("sinks"),
            self._get_dtype("sinks"),
            "ND",
            name="sinks",
        )

    def _get_quant_scale_offset_layout(self, shape):
        dim2layout = {1: "H", 2: "ND"}
        return dim2layout.get(len(shape), "1N1D")

    def _parse_quant_info(self):
        self.quant_scale2 = FiaTensor(
            self._get_data("quant_scale2"),
            self._get_shape("quant_scale2"),
            self._get_dtype("quant_scale2"),
            self._get_quant_scale_offset_layout(self._get_shape("quant_scale2")),
            self.num_heads,
            name="quant_scale2",
        )
        self.quant_offset2 = FiaTensor(
            self._get_data("quant_offset2"),
            self._get_shape("quant_offset2"),
            self._get_dtype("quant_offset2"),
            self._get_quant_scale_offset_layout(self._get_shape("quant_offset2")),
            self.num_heads,
            name="quant_offset2",
        )

        self.out_quant_flag = self._get_flag("quant_scale2")
        self.out_quant_pc_flag = False
        if self.out_quant_flag:
            if not self._get_flag("quant_offset2"):
                # 当不传入quantOffset2时，也需要生成一个和scale一样大小的全0 Offset，用于后续计算
                quant_offset2_shape = self._get_shape("quant_scale2")
                quant_offset2_data = np.zeros(quant_offset2_shape, np.float32)
                self.quant_offset2 = FiaTensor(
                    quant_offset2_data,
                    quant_offset2_shape,
                    "fp32",
                    self._get_quant_scale_offset_layout(quant_offset2_shape),
                    self.num_heads,
                    name="quant_offset2",
                )
            if self._get_shape("quant_scale2") != [1]:
                self.out_quant_pc_flag = True

    def _get_normal_flag(self):
        if (
            not self.flag_list[0]
            or not self.flag_list[1]
            or not self.flag_list[2]
            or not self.flag_list[24]
        ):
            fia_warn("q/k/v/out flag is false, return zero output")
            return False
        if self.query.empty():
            fia_warn("query is empty, return zero output")
            return False
        if self.key.empty():
            fia_warn("key is empty, return zero output")
            return False
        if self.value.empty():
            fia_warn("value is empty, return zero output")
            return False
        if not (
            (self.flag_list[21] and self.flag_list[22])
            or (not self.flag_list[21] and not self.flag_list[22])
        ):
            fia_warn("prefix未成对出现, 返回全0输出")
            return False
        if self.block_table.empty():
            fia_warn("block_table is empty, return zero output")
            return False
        return True

    def _get_prefix_act_lens(self):
        prefix_act_lens = 0
        if self.shared_prefix_flag:
            prefix_act_lens = self.k_shared_prefix.bnsd_shape[2]
            if self.prefix_act_flag:
                prefix_act_lens = self.params["prefix_act_lens"][0]
        return prefix_act_lens

    def _parse_feature_flag(self):
        self.pse_shift_flag = self._get_flag("pse_shift")
        self.atten_mask_flag = self._get_flag("atten_mask")
        self.actual_seq_lens_q_flag = self._get_flag("actual_seq_lens_q")
        self.actual_seq_lens_kv_flag = self._get_flag("actual_seq_lens_kv")
        self.q_padding_size_flag = self._get_flag("q_padding_size")
        if self.q_padding_size_flag:
            self.q_padding_size = max(self._get_range("q_padding_size")[0], 0)
        self.kv_padding_size_flag = self._get_flag("kv_padding_size")
        if self.kv_padding_size_flag:
            self.kv_padding_size = max(self._get_range("kv_padding_size")[0], 0)
        self.shared_prefix_flag = self._get_flag("k_shared_prefix") or self._get_flag(
            "v_shared_prefix"
        )
        self.prefix_act_flag = self._get_flag("actual_shared_prefix_len")
        self.prefix_act_lens = self._get_prefix_act_lens()
        self.prefix_kvs = 0
        self.sink_flag = self._get_flag("sinks")

    def parse_flag_list(self):
        self._parse_feature_flag()
        self.normal_flag = self._get_normal_flag()  # TODO: 处理normal_flag，返回全0

    def str_to_bool_list(self, lst):
        return [True if l else False for l in lst]


class FiaOpPreprocess:
    def __init__(self, data_list, params, op_params):
        self.params = params
        self.data_list = data_list
        self.op_params = op_params
        self.query = op_params.query
        self.key = op_params.key
        self.value = op_params.value
        self.q_rope = op_params.q_rope
        self.k_rope = op_params.k_rope
        self.quant_scale2 = op_params.quant_scale2
        self.quant_offset2 = op_params.quant_offset2

    def preprocess(self):
        if self.op_params.rope_flag:
            self.concat_rope_tensor()

        self.preprocess_kv()
        self.preprocess_shared_prefix()
        self.preprocess_pse_shift()
        # 需要保证输入的mask是正确的
        if self.op_params.sparse_mode in [2, 3, 4] and self.op_params.query.S > 1:
            self.preprocess_atten_mask()
        self.preprocess_block_table()
        self.preprocess_kv_cache()
        self.preprocess_post_quant()

    def preprocess_post_quant(self):
        # 如果是perchannel模式，要将后量化参数统一转换成1n1d格式
        if self.op_params.out_quant_pc_flag:
            fia_debug_func_begin("begin FiaOpPreprocess.preprocess_post_quant")
            self.quant_scale2.trans_to_1n1d()
            self.quant_offset2.trans_to_1n1d()

    def concat_rope_tensor(self):
        # 非全量化场景1.将Q与QROPE拼接2.将K与KROPE拼接3.伪量化场景，k_antiscale与k_rope_antiscale拼接
        self.op_params.query = concat_tensor(self.query, self.q_rope)

        if self.op_params.kv_num == 1:
            self.op_params.key = concat_tensor_list(self.key, self.k_rope)
        else:
            raise ValueError("k tensor 长度不为1, deepseek预处理异常，输出空tensor！")

    def preprocess_kv(self):
        # >> kv预处理：1、将kv list 转换为bnsd 2、GQA场景，将kvn扩展为qn
        return

    def preprocess_block_table(self):
        # block_table: [B, ceil(max_s/block_size)]
        # cache: BnBsH/BnDBsD/NZ
        # 1、生成随机的block_table，并覆写原有bin文件
        if not self.op_params.pa_flag:
            return

        if not need_gen_input(self.op_params.action_type):
            return

        fia_debug_func_begin("begin FiaOpPreprocess.preprocess_block_table")
        # 生成blocktable
        block_num = self.op_params.k_cache.shape[0]
        block_size = self.op_params.block_size
        block_table_shape = self.op_params.block_table.shape

        block_num_each_batch = []
        block_num_expect_min = 0
        for actual_seq in self.op_params.actual_seq_lens_kv:
            block_num_cur_batch = math.ceil(actual_seq / block_size)
            block_num_each_batch.append(block_num_cur_batch)
            block_num_expect_min += block_num_cur_batch

        if block_num_expect_min > block_num:
            raise RuntimeError(
                f"[ERROR]Wrong input k_cache_shape: get block_num = {block_num}, but expect block_num > {block_num_expect_min}"
            )

        block_idx_list = np.arange(0, block_num, 1)
        block_idx_list = np.random.permutation(block_idx_list).astype(np.int32)
        block_table = [-1] * block_table_shape[1]
        block_table = np.tile(block_table, (block_table_shape[0], 1)).astype(np.int32)

        block_idx = 0
        for batch_idx, block_num_cur_batch in enumerate(block_num_each_batch):
            for block_num_cur_batch_idx in range(block_num_cur_batch):
                block_table[batch_idx][block_num_cur_batch_idx] = block_idx_list[
                    block_idx
                ]
                block_idx += 1
        self.op_params.block_table.data = block_table
        if self.params["is_preprocess"]:
            self.params["input_data"].kwargs["block_table"] = block_table

    def _generate_cache(
        self, cache_shape, tensor_bnsd, shape_bnsd, src_dtype, dst_dtype
    ):
        block_table_dim1 = self.op_params.block_table.shape[1]
        block_size = self.op_params.block_size
        block_table = self.op_params.block_table.data
        max_s_batch = block_table_dim1 * block_size
        B, N, S, D = shape_bnsd
        cache_shape = [int(item) for item in cache_shape]
        cache = np.zeros(cache_shape)
        if len(cache_shape) == 3:  # BSH
            # trans kv to bsh(此处使用的tensor, 没有经过n的扩展)
            tensor_bsh_raw = FiaLayoutTool.trans_bnsd_to_bsh(tensor_bnsd, shape_bnsd)
            H = N * D
            if src_dtype == "int32":
                H = int(H / 8)
            tensor_bsh = np.zeros((B, max_s_batch, H))
            tensor_bsh[:, : tensor_bsh_raw.shape[1], :] = tensor_bsh_raw[:, :, :]
            for batch_idx in range(B):
                for block_idx, cache_block_id in enumerate(block_table[batch_idx]):
                    block_offset = block_idx * block_size
                    if cache_block_id == -1:
                        continue
                    else:
                        cache[cache_block_id, 0:block_size, :] = tensor_bsh[
                            batch_idx, block_offset : (block_offset + block_size), :
                        ]
        elif len(cache_shape) == 4:  # BNSD
            k_tensor_bnsd = np.zeros((B, N, max_s_batch, D))
            k_tensor_bnsd[:, :, :S, :] = tensor_bnsd[:, :, :, :]

            for batch_idx in range(B):
                for block_idx, cache_block_id in enumerate(block_table[batch_idx]):
                    block_offset = block_idx * block_size
                    if cache_block_id == -1:
                        continue
                    else:
                        cache[cache_block_id, :, 0:block_size, :] = k_tensor_bnsd[
                            batch_idx, :, block_offset : (block_offset + block_size), :
                        ]
        elif len(cache_shape) == 5:  # NZ
            k_cache_tensor_bnbd = self._generate_cache(
                [cache_shape[0], N, block_size, D],
                tensor_bnsd,
                shape_bnsd,
                src_dtype,
                dst_dtype,
            )
            D0 = 32 if src_dtype == "int8" else 16
            cache = k_cache_tensor_bnbd.reshape(
                k_cache_tensor_bnbd.shape[0],
                k_cache_tensor_bnbd.shape[1],
                k_cache_tensor_bnbd.shape[2],
                k_cache_tensor_bnbd.shape[3] // D0,
                D0,
            ).transpose(0, 1, 3, 2, 4)
        else:
            raise ValueError(
                f"cache shape dim should be 3/4/5, but got {len(cache_shape)}"
            )
        return cache.astype(dst_dtype)

    def _preprocess_kv_cache_rope(self):
        fia_debug_func_begin("begin FiaOpPreprocess._preprocess_kv_cache_rope")
        k_cache = self._generate_cache(
            self.op_params.k_cache.shape,
            self.key.bnsd_data,
            self.key.bnsd_shape,
            self.key.dtype,
            self.key.np_dtype,
        )

        v_cache = None
        if not self.op_params.is_deepseek_mla:
            v_cache = self._generate_cache(
                self.op_params.v_cache.shape,
                self.value.bnsd_data,
                self.value.bnsd_shape,
                self.value.dtype,
                self.value.np_dtype,
            )
        else:
            v_cache = k_cache
        k_rope_cache = self._generate_cache(
            self.op_params.k_rope_cache.shape,
            self.k_rope.bnsd_data,
            self.k_rope.bnsd_shape,
            self.key.dtype,
            self.key.np_dtype,
        )
        # return torch.zeros(out_shape)
        # 将kv cache 生成新的bin文件
        k_cache_index = FiaOpParam.get_param_index("k_cache")
        v_cache_index = FiaOpParam.get_param_index("v_cache")
        k_rope_cache_index = FiaOpParam.get_param_index("k_rope_cache")
        if self.params["is_preprocess"]:
            self.params["input_data"].kwargs["k_cache"] = k_cache
            self.params["input_data"].kwargs["v_cache"] = v_cache
            self.params["input_data"].kwargs["k_rope_cache"] = k_rope_cache

    def _preprocess_kv_cache_no_rope(self):
        k_cache = self._generate_cache(
            self.op_params.k_cache.shape,
            self.key.bnsd_data_list[0],
            self.key.bnsd_shape_list[0],
            self.key.dtype,
            self.key.np_dtype,
        )
        v_cache = self._generate_cache(
            self.op_params.v_cache.shape,
            self.value.bnsd_data_list[0],
            self.value.bnsd_shape_list[0],
            self.value.dtype,
            self.value.np_dtype,
        )

        # 将kv cache 生成新的bin文件
        k_cache_index = FiaOpParam.get_param_index("k_cache")
        v_cache_index = FiaOpParam.get_param_index("v_cache")
        if self.params["is_preprocess"]:
            self.params["input_data"].kwargs["k_cache"] = k_cache
            self.params["input_data"].kwargs["v_cache"] = v_cache

    def preprocess_kv_cache(self):
        # 2、将kv shape 统一转换成bsh
        # 3、生成kv cache
        # 4、将kv cache dump成新的bin文件，供aclnn接口调用
        if not self.op_params.pa_flag:
            return

        if not need_gen_input(self.op_params.action_type):
            return

        fia_info(
            f"[PageAtten]Input Kdtype:{self.params['dtype_input'][1]} Vdtype:{self.params['dtype_input'][2]}"
        )
        if not self.op_params.rope_flag:
            self._preprocess_kv_cache_no_rope()
        else:
            self._preprocess_kv_cache_rope()

    def preprocess_pse_shift(self):
        # >> pse 处理
        if not self.op_params.pse_shift_flag:
            return

        if self.op_params.action_type in ["bm_output_gold", "bm_output"]:
            return

    def preprocess_atten_mask(self):
        # >> m预处理：1、将m扩展为BN1S  2、padding场景下，偏移部分设置为1 3、针对FP16格式，将tensor转成0/1
        sparse_mode = self.op_params.sparse_mode
        q_shape_bnsd = self.op_params.query.bnsd_shape
        randoms = 0
        mrandom_type = "NORMAL"
        if "mrandomtype" in self.params:
            mrandom_type = self.params["mrandomtype"]
            if mrandom_type == "ones":
                randoms = int(self.params["mrandom"])
        if (not self.op_params.atten_mask_flag) or self.op_params.atten_mask.empty():
            self.op_params.pre_tokens = 214748647
            self.op_params.next_tokens = 214748647
        else:
            fia_debug_func_begin("begin FiaOpPreprocess.preprocess_atten_mask")
            batch = q_shape_bnsd[0]
            num_heads = q_shape_bnsd[1]
            npu_m_shape_s = self.op_params.atten_mask.shape
            self.op_params.is_mask_bs = False
            if sparse_mode == 0 or sparse_mode == 1:
                self.op_params.is_mask_bs = (
                    self.op_params.atten_mask.shape[0] == self.op_params.batch
                ) and (len(self.op_params.atten_mask.shape) == 2)
                batch, num_heads, ns1, ns2 = get_attention_mask_batch_num(
                    self.op_params.atten_mask.shape, self.op_params.is_mask_bs
                )  # 获取输入attentionmask的batch 和numhead
                npu_m_shape_s = [ns1, ns2]

            q_s = self.op_params.query.bnsd_shape[2]
            kvs = self.op_params.kv_s  # TODO: 考虑prefix
            if self.op_params.shared_prefix_flag:
                kvs += self.op_params.prefix_act_lens
            cpu_m_shape = [q_s, kvs]  # cpu
            (
                cpu_m_tensor,
                npu_m_tensor,
                self.op_params.pre_tokens,
                self.op_params.next_tokens,
            ) = MaskGenerator._create_random_mask_by_spars(
                cpu_m_shape,
                npu_m_shape_s,
                self.op_params.atten_mask.dtype,
                self.op_params.pre_tokens,
                self.op_params.next_tokens,
                self.op_params.actual_seq_lens_q,
                self.op_params.actual_seq_lens_kv,
                self.op_params.prefix_act_lens,
                self.op_params.prefix_act_lens,
                self.op_params.kv_s_list,
                batch,
                num_heads,
                sparse_mode,
                random_ones=randoms,
            )
            if mrandom_type == "invalid" or mrandom_type == "invaild":
                randoms = int(self.params["mrandom"])
                cpu_m_tensor[..., :randoms] = 1
                npu_m_tensor[..., :randoms] = 1
            if self.op_params.is_mask_bs:
                npu_m_tensor = npu_m_tensor.reshape(self.op_params.atten_mask.shape)
            atten_mask_index = FiaOpParam.get_param_index("atten_mask")

            if (sparse_mode == 0 or sparse_mode == 1) and (
                not self.op_params.is_mask_bs
            ):
                m_tensor = _np_broadcast_mask_n(
                    cpu_m_tensor,
                    self.op_params.atten_mask.shape,
                    cpu_m_shape,
                    self.op_params.num_heads,
                    q_shape_bnsd[0],
                )
            else:
                m_tensor = cpu_m_tensor
            self.op_params.atten_mask.data = np.array(m_tensor)

    def preprocess_shared_prefix(self):
        # psefix 预处理：1.转成1nsd 2.GQA场景扩展N; 3.按act_prefix裁剪 4.获取perfix_act_lens
        if not self.op_params.shared_prefix_flag:
            return

        if self.op_params.prefix_act_flag:
            self.op_params.k_shared_prefix.data = (
                self.op_params.k_shared_prefix.bnsd_data[
                    :, :, : self.op_params.prefix_act_lens, :
                ]
            )
            self.op_params.v_shared_prefix.data = (
                self.op_params.v_shared_prefix.bnsd_data[
                    :, :, : self.op_params.prefix_act_lens, :
                ]
            )
        else:
            self.op_params.k_shared_prefix.data = (
                self.op_params.k_shared_prefix.bnsd_data
            )
            self.op_params.v_shared_prefix.data = (
                self.op_params.v_shared_prefix.bnsd_data
            )
        fia_info(f"prefix_act_lens:{str(self.op_params.prefix_act_lens)}")


class FiaOpPreprocessTorch(FiaOpPreprocess):
    def __init__(self, data_list, params, op_params, device=None):
        super().__init__(data_list, params, op_params)
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

    def preprocess(self):
        if self.op_params.rope_flag:
            self.concat_rope_tensor()
        else:
            raise ValueError("G only supported rope")
        self.preprocess_block_table()

    def preprocess_block_table(self):
        import triton
        import torch.nn.functional as F

        key = self.op_params.key
        fia_debug("key_shape: key.shape")
        d = key.D
        kv_head_nums = key.N
        # pa场景生成blocktable
        block_size = 64
        batch = self.op_params.batch
        self.op_params.cache_seqlens = torch.tensor(
            self.op_params.actual_seq_lens_kv_raw, dtype=torch.int32
        ).to(self.device)
        max_seqlen = self.op_params.cache_seqlens.max().item()
        max_seqlen_pad = triton.cdiv(max_seqlen, 256) * 256
        padding_length = max_seqlen_pad - max_seqlen
        # 裁剪ks
        k_new_tensor = key.bsnd_data[:, : key.S, :, :]
        print(f"k_cache_before_padding:{k_new_tensor.shape}")
        k_new_tensor = F.pad(
            k_new_tensor, (0, 0, 0, 0, 0, padding_length), "constant", 0
        )
        print(f"k_cache_after_padding:{k_new_tensor.shape}")
        block_table = torch.arange(
            batch * max_seqlen_pad // block_size, dtype=torch.int32
        ).view(batch, max_seqlen_pad // block_size)
        fia_debug("block_table shape: block_table.shape")
        self.op_params.block_table = block_table.to(self.device)
        k_new_tensor = k_new_tensor.transpose(1, 2)
        blocked_k = k_new_tensor.reshape(
            block_table.numel(), block_size, kv_head_nums, d
        )
        fia_debug("k_cache shape: blocked_k.shape")
        for i in range(batch):
            if blocked_k.dtype == torch.int8:
                blocked_k.view(batch, max_seqlen_pad, kv_head_nums, d)[
                    i, self.op_params.cache_seqlens[i].item() :
                ] = 0
            else:
                blocked_k.view(batch, max_seqlen_pad, kv_head_nums, d)[
                    i, self.op_params.cache_seqlens[i].item() :
                ] = float("nan")
        self.op_params.blocked_k = blocked_k.to(self.device)


class FiaOpPreprocessNumpy(FiaOpPreprocess):
    pass


class FiaOpForward:
    def __init__(self, data_list, params, mode="numpy", device=None):
        self.params = params
        self.data_list = data_list
        self.op_params = FiaOpParam(data_list, params)
        self.mode = mode
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        if mode == "numpy":
            self.fia_op_preprocess = FiaOpPreprocessNumpy(
                data_list, params, self.op_params
            )
        else:
            self.fia_op_preprocess = FiaOpPreprocessTorch(
                data_list, params, self.op_params, device
            )
        self.act_seq_kv = 0
        self.act_seq_q = 0
        self.lse_default_value = np.inf
        self.n_factor = self.op_params.num_heads // self.op_params.num_kv_heads

    def _post_quant(self, y, n1):
        if self.op_params.out_quant_flag:
            if self.op_params.out_quant_pc_flag:
                fia_debug_func_begin("begin FiaOpForward._post_quant quant_pc")
                y = quant_pc(
                    y,
                    self.op_params.quant_scale2.data,
                    self.op_params.quant_offset2.data,
                    n1,
                )
            else:
                fia_debug_func_begin("begin FiaOpForward._post_quant quant")
                y = quant(
                    y,
                    self.op_params.quant_scale2.data[0],
                    self.op_params.quant_offset2.data[0],
                )
        return y

    def _calculate_q_s_begin_end(self):
        q_s_begin = 0
        q_s_end = self.query.S
        if self.op_params.q_padding_size_flag:
            q_s_begin = int(
                self.query.S - self.act_seq_q - self.op_params.q_padding_size
            )
            q_s_end = int(self.query.S - self.op_params.q_padding_size)
            fia_debug(f"query_left padding--- s_begin:{q_s_begin}, s_end:{q_s_end}")
        else:
            if self.act_seq_q is not None:
                q_s_end = self.act_seq_q
        self.q_s_begin = q_s_begin
        self.q_s_end = q_s_end

    def _calculate_kv_s_begin_end(self, bidx):
        kv_s_begin = 0

        if self.op_params.storage_mode == StorageMode.TENSOR_LIST:
            kv_s_end = self.key._bnsd_shape_list[bidx][2]
        else:
            kv_s_end = self.key.S
        if self.op_params.kv_padding_size_flag:
            kv_s_begin = int(
                kv_s_end - self.act_seq_kv - self.op_params.kv_padding_size
            )
            kv_s_end = int(kv_s_end - self.op_params.kv_padding_size)
            fia_debug(f"kv_left padding--- s_begin:{kv_s_begin}, s_end:{kv_s_end}")
        else:
            if self.op_params.actual_seq_lens_kv_flag:
                if self.act_seq_kv is not None:
                    kv_s_end = self.act_seq_kv

        self.kv_s_begin = kv_s_begin
        self.kv_s_end = kv_s_end

    def _calculate_s_begin_end(self, bidx):
        self._calculate_q_s_begin_end()
        self._calculate_kv_s_begin_end(bidx)

    def _get_matmul_dtype(self):
        matmul_dtype = np.float32
        return matmul_dtype

    @timeit_decorator
    def _calculte_bmm1(self, q, k, matmul_dtype):
        qkBmmRes = np.matmul(q, k.transpose(0, 1, 3, 2), dtype=matmul_dtype)
        fia_debug_data("mm1 output", qkBmmRes)
        return qkBmmRes

    @timeit_decorator
    def _calculte_scale(self, qkBmmRes):
        scale_value = self.op_params.scale_value
        qkEleRes = qkBmmRes * scale_value
        fia_debug_data("mm1*scale output", qkEleRes)
        return qkEleRes

    @timeit_decorator
    def _calculate_pse(self, qkEleRes, b_idx, n_idx):
        if not self.op_params.pse_shift_flag:
            return qkEleRes

        pse_cur = self.pse_cur[
            :, :, self.q_s_begin : self.q_s_end, self.kv_s_begin : self.kv_s_end
        ]
        np.add(qkEleRes, pse_cur, out=qkEleRes)

        fia_debug_data("calculate pse output", qkEleRes)
        return qkEleRes

    @timeit_decorator
    def _calculate_atten_mask(self, qkEleRes):
        if not self.op_params.atten_mask_flag:
            return qkEleRes, None

        fia_debug_func_begin("begin FiaOpForward._calculate_atten_mask")
        current_mask = self.mask_cur
        fia_debug_data("_calculate_atten_mask mask", current_mask)

        # Adjust mask dimensions based on sparse mode
        if self.op_params.sparse_mode in [2, 3, 4]:
            current_mask = current_mask[
                :,
                :,
                : (self.q_s_end - self.q_s_begin),
                : (self.kv_s_end - self.kv_s_begin),
            ]
        else:
            current_mask = current_mask[
                :, :, self.q_s_begin : self.q_s_end, self.kv_s_begin : self.kv_s_end
            ]
        # Apply mask to attention scores
        if self.op_params.atten_mask.dtype == "float16":
            qkEleRes = np.where(current_mask, qkEleRes - 10000, qkEleRes)
        else:
            qkEleRes[current_mask.astype(np.bool_)] = -1.7e38

        fia_debug_data("_calculate_atten_mask output", qkEleRes)
        return qkEleRes, current_mask

    @timeit_decorator
    def _calculate_softmax(self, qkEleRes, sinks=None):
        return FiaSoftmax.softmax(qkEleRes, sinks)

    @timeit_decorator
    def _calculate_bmm2(self, softmax_res, softmax_sum, v_cur, matmul_dtype):
        assert isinstance(softmax_res, np.ndarray), "softmax_res must be a numpy array"
        assert isinstance(v_cur, np.ndarray), "v_cur must be a numpy array"
        assert isinstance(softmax_sum, (np.ndarray, float)), (
            "softmax_sum must be a numpy array or float"
        )

        if self.query.dtype == "float16":
            softmax_res = softmax_res.astype(np.float16)
        elif self.query.dtype == "bfloat16":
            softmax_res = softmax_res.astype(tf.bfloat16.as_numpy_dtype)

        bmm2Res = np.matmul(softmax_res, v_cur, dtype=matmul_dtype)

        if isinstance(softmax_sum, np.ndarray):
            bmm2Res /= softmax_sum
        else:
            bmm2Res = bmm2Res / softmax_sum
        fia_debug_data("bmm2 output", bmm2Res)

        return bmm2Res

    @timeit_decorator
    def _calculate_bmm2_mask(self, bmm2Res, mask_cur, n1):
        if self.op_params.out_quant_flag:
            bmm2Res = self._post_quant(bmm2Res, n1)
        if mask_cur is not None:
            reshaped_bmm2 = bmm2Res.reshape(-1, mask_cur.shape[2], bmm2Res.shape[3])
            mask_to_zero = mask_cur.all(axis=(0, 1, 3))
            reshaped_bmm2[:, mask_to_zero, :] = 0
            fia_debug_data("bmm2 mask output", bmm2Res)
        return bmm2Res

    @timeit_decorator
    def _calculate_lse(self, softmax_sum, softmax_max, mask_cur):
        lse_flag = self.op_params.softmax_lse_flag
        if not lse_flag:
            return None

        fia_debug_func_begin("begin FiaOpForward._calculate_lse")
        lse = np.log(softmax_sum) + softmax_max
        if mask_cur is not None:
            mask_to_default = mask_cur.all(axis=(0, 1, 3))
            reshaped_lse = lse.reshape(-1, mask_cur.shape[2], lse.shape[3])
            reshaped_lse[:, mask_to_default, :] = self.lse_default_value
        fia_debug_data("lse output", lse)

        return lse

    @staticmethod
    def _get_torch_dtype(dtype):
        if dtype == "bfloat16":
            return torch.bfloat16
        else:
            return torch.float32

    def compute_once_bnsd(self, q, k, v, b_idx, n_idx, sinks=None):
        matmul_dtype = self._get_matmul_dtype()
        fia_debug_data("q", q)
        fia_debug_data("k", k)
        fia_debug_data("v", v)

        qkBmmRes = self._calculte_bmm1(q, k, matmul_dtype)
        qkEleRes = self._calculte_scale(qkBmmRes)
        qkEleRes = self._calculate_pse(qkEleRes, b_idx, n_idx)
        qkEleRes, mask_cur = self._calculate_atten_mask(qkEleRes)
        if self.op_params.sink_flag:
            fia_debug_data("_t_ifaattention_act sink input", sinks)
        softmax_res, softmax_sum, softmax_max = self._calculate_softmax(qkEleRes, sinks)
        bmm2Res = self._calculate_bmm2(softmax_res, softmax_sum, v, matmul_dtype)
        bmm2Res = self._calculate_bmm2_mask(bmm2Res, mask_cur, n_idx)
        lse = self._calculate_lse(softmax_sum, softmax_max, mask_cur)
        return bmm2Res, lse

    @timeit_decorator
    def _get_atc_seq_qkv(self, b_idx, q_s):
        debug_parts = [f"b_idx:{b_idx}"]

        if self.op_params.actual_seq_lens_q_flag:
            self.act_seq_q = self.op_params.actual_seq_lens_q[b_idx]
            debug_parts.append(f"act_seq_q:{self.act_seq_q}")
        else:
            self.act_seq_q = q_s
            debug_parts.append(f"q_s:{self.act_seq_q}")

        prefix_len = (
            self.op_params.prefix_act_lens if self.op_params.shared_prefix_flag else 0
        )

        if self.op_params.actual_seq_lens_kv_flag:
            self.act_seq_kv = self.op_params.actual_seq_lens_kv[b_idx]
            debug_parts.append(f"act_seq_kv:{self.act_seq_kv}")
        else:
            self.act_seq_kv = self.op_params.kv_s
            debug_parts.append(f"k_s:{self.act_seq_kv}")

        fia_debug(" ".join(debug_parts))

        return self.act_seq_kv, self.act_seq_q

    @timeit_decorator
    def _process_shared_prefix(self, k, v, bidx, nidx):
        if nidx is not None:
            nidx = nidx // self.n_factor
        if self.op_params.shared_prefix_flag:
            k_shared_prefix = self.op_params.k_shared_prefix.data
            v_shared_prefix = self.op_params.v_shared_prefix.data
            k_shared_prefix = k_shared_prefix[0:1, nidx : nidx + 1, :, :]
            v_shared_prefix = v_shared_prefix[0:1, nidx : nidx + 1, :, :]
            if k_shared_prefix is None or v_shared_prefix is None:
                raise ValueError("Shared prefix data is not available.")
            k = np.concatenate((k_shared_prefix, k), axis=2)
            v = np.concatenate((v_shared_prefix, v), axis=2)
        return k, v

    def _get_atten_mask_cur(self, b_idx):
        # 判断attenmask是否为空
        mask_cur = None
        if self.op_params.atten_mask_flag:
            if self.op_params.prefix_act_flag or self.op_params.prefix_act_lens > 0:
                mask_cur = np.zeros(
                    [
                        1,
                        1,
                        self.op_params.q_s,
                        self.op_params.kv_s + self.op_params.prefix_act_lens,
                    ],
                    dtype="uint8",
                )
            else:
                mask_cur = np.zeros(
                    [1, 1, self.op_params.q_s, self.op_params.kv_s], dtype="uint8"
                )
            mask_cur[0, 0, :, :] = self.op_params.atten_mask.data[b_idx]
        return mask_cur

    def _get_pse_cur(self, b_idx, n_idx=None):
        # 判断pse是否为空,如果非空,检查pse第一维是否为1：如果格式为1n1s,则直接传入下层计算;如果格式为bn1s,则按B拆分后进入下层。
        if not self.op_params.pse_shift_flag:
            pse_cur = None
        elif self.op_params.pse_shift.shape[0] == 1:
            if n_idx is None:
                pse_cur = self.op_params.pse_shift.data[:, :, :, :]
            else:
                pse_cur = self.op_params.pse_shift.data[:, n_idx : (n_idx + 1), :, :]
        else:
            if n_idx is None:
                pse_cur = self.op_params.pse_shift.data[b_idx : (b_idx + 1), :, :, :]
            else:
                pse_cur = self.op_params.pse_shift.data[
                    b_idx : (b_idx + 1), n_idx : (n_idx + 1), :, :
                ]
        return pse_cur

    def _get_sink_cur(self, n_idx):
        # 判断sink是否为空
        sink_cur = None
        if self.op_params.sink_flag:
            sink_cur = self.op_params.sinks.data[n_idx : (n_idx + 1)]
        return sink_cur

    def _get_k_shape_by_idx(self, b_idx):
        if self.op_params.storage_mode != StorageMode.TENSOR_LIST:
            return self.key.bnsd_shape
        return self.key.bnsd_shape_list[b_idx]

    def _get_v_shape_by_idx(self, b_idx):
        if self.op_params.storage_mode != StorageMode.TENSOR_LIST:
            return self.key.bnsd_shape
        return self.key.bnsd_shape_list[b_idx]

    def _get_q_by_idx(self, b_idx, q_s_begin, q_s_end, n_idx=None):
        if n_idx is None:
            return self.op_params.query.bnsd_data[
                b_idx : (b_idx + 1), :, q_s_begin:q_s_end, :
            ]
        else:
            return self.op_params.query.bnsd_data[
                b_idx : (b_idx + 1), n_idx : (n_idx + 1), q_s_begin:q_s_end, :
            ]

    def _get_from_list(self, data_list, b_idx, kv_s_begin, kv_s_end, n_idx=None):
        if n_idx is None:
            return data_list[b_idx][:, :, kv_s_begin:kv_s_end, :]
        else:
            return data_list[b_idx][:, n_idx : (n_idx + 1), kv_s_begin:kv_s_end, :]

    def _get_from_tensor(self, tensor, b_idx, kv_s_begin, kv_s_end, n_idx=None):
        if n_idx is None:
            return tensor[b_idx : (b_idx + 1), :, kv_s_begin:kv_s_end, :]
        else:
            return tensor[
                b_idx : (b_idx + 1), n_idx : (n_idx + 1), kv_s_begin:kv_s_end, :
            ]

    def _get_kv_by_idx(self, fia_tensor, b_idx, kv_s_begin, kv_s_end, n_idx=None):
        storage_mode = self.op_params.storage_mode
        if n_idx is not None:
            n_idx = n_idx // self.n_factor

        if storage_mode != StorageMode.TENSOR_LIST:
            return self._get_from_tensor(
                fia_tensor.bnsd_data, b_idx, kv_s_begin, kv_s_end, n_idx
            )
        else:
            return self._get_from_list(
                fia_tensor.bnsd_data_list, b_idx, kv_s_begin, kv_s_end, n_idx
            )

    def _get_k_by_idx(self, b_idx, kv_s_begin, kv_s_end, n_idx=None):
        return self._get_kv_by_idx(self.key, b_idx, kv_s_begin, kv_s_end, n_idx)

    def _get_v_by_idx(self, b_idx, kv_s_begin, kv_s_end, n_idx=None):
        return self._get_kv_by_idx(self.value, b_idx, kv_s_begin, kv_s_end, n_idx)

    def compute_bnsd(self):
        y = self.attention_out_bnsd.data
        lse = self.lse_bnsd.data
        if (
            self.op_params.q_padding_size_flag or self.op_params.kv_padding_size_flag
        ) and self.op_params.sparse_mode == 0:
            self.op_params.pre_tokens = 2147483647
            self.op_params.next_tokens = 2147483647
        for b_idx in range(self.op_params.batch):
            act_seq_kv, act_seq_q = self._get_atc_seq_qkv(b_idx, self.op_params.q_s)
            if act_seq_kv == 0 and self.op_params.q_s == 1:
                for n_idx in range(self.op_params.num_heads):
                    if self.op_params.out_quant_flag:
                        y[b_idx : (b_idx + 1), n_idx : (n_idx + 1), :, :] = (
                            self._post_quant(
                                y[b_idx : (b_idx + 1), n_idx : (n_idx + 1), :, :], n_idx
                            )
                        )
                continue
            if (
                act_seq_kv == 0
                or act_seq_q == 0
                or 0 in self._get_k_shape_by_idx(b_idx)
                or 0 in self._get_v_shape_by_idx(b_idx)
            ):
                fia_debug("skip calc for actual seq 0 or kv shape has 0")
                continue
            for n_idx in range(self.op_params.num_heads):
                self._calculate_s_begin_end(b_idx)
                q = self._get_q_by_idx(b_idx, self.q_s_begin, self.q_s_end, n_idx)
                k = self._get_k_by_idx(b_idx, self.kv_s_begin, self.kv_s_end, n_idx)
                v = self._get_v_by_idx(b_idx, self.kv_s_begin, self.kv_s_end, n_idx)
                k, v = self._process_shared_prefix(k, v, b_idx, n_idx)
                if self.op_params.shared_prefix_flag:
                    # if self.op_params.prefix_act_lens == 0:
                    #     self.kv_s_end += self.op_params.k_shared_prefix.shape.S
                    # else:
                    self.kv_s_end += self.op_params.prefix_act_lens
                self.mask_cur = self._get_atten_mask_cur(b_idx)
                self.pse_cur = self._get_pse_cur(b_idx, n_idx)
                sinks = self._get_sink_cur(n_idx)
                (
                    y[
                        b_idx : (b_idx + 1),
                        n_idx : (n_idx + 1),
                        self.q_s_begin : self.q_s_end,
                        :,
                    ],
                    lse[
                        b_idx : (b_idx + 1),
                        n_idx : (n_idx + 1),
                        self.q_s_begin : self.q_s_end,
                        :,
                    ],
                ) = self.compute_once_bnsd(q, k, v, b_idx, n_idx, sinks)
        self.attention_out_bnsd.data = y
        return y, lse

    def padding_size_overflow(self):
        if self.op_params.kv_padding_size_flag:
            max_act_seq = max(self.op_params.actual_seq_lens_kv_raw)
            kv_s = self.op_params.kv_s
            if kv_s - self.op_params.kv_padding_size - max_act_seq < 0:
                fia_warn("paddingsize 溢出，输出空tensor！")
                return True
        return False

    def padding_size_overflow(self):
        if not self.op_params.kv_padding_size_flag:
            return False

        max_act_seq = max(self.op_params.actual_seq_lens_kv_raw)
        kv_s = self.op_params.kv_s
        kv_padding_size = self.op_params.kv_padding_size

        if kv_s - kv_padding_size - max_act_seq < 0:
            fia_warn(
                f"kv_padding_size overflow！kv_s={kv_s}，kv_padding_size={kv_padding_size}，max_act_seq={max_act_seq}"
            )
            return True

        return False

    def route_to_old(self):
        if (
            (self.op_params.input_layout in ["SH", "NSD"])
            or (self.op_params.query.dtype not in ["float16", "bfloat16", "float32"])
            or (self.op_params.key.dtype not in ["float16", "bfloat16", "float32"])
            or (self.op_params.query.dtype != self.op_params.key.dtype)
        ):
            return True
        # if self.op_params.query.D not in [64, 128, 192, 512]:
        #     return True
        fia_debug("*********************************ROUTE TO FIA success")
        return False

    def route_to_old_ifa(self):
        if self.op_params.sparse_mode == 4:
            self.op_params.sparse_mode = 0

        if (
            (self.op_params.query.S == 1)
            or (self.op_params.rope_flag and self.op_params.query.D == 512)
            or (self.op_params.query.dtype != self.op_params.key.dtype)
        ):
            fia_debug("*********************************ROUTE TO IFA")
            return True
        fia_debug("*********************************ROUTE TO PFA")
        return False

    def forward_numpy_old(self):
        if self.route_to_old_ifa():
            return aclnn_op_func_ifa_cpu(self.data_list, self.params)
        else:
            return aclnnPromptFlashAttention_unification(self.data_list, self.params)

    def forward_numpy(self):
        if self.route_to_old():
            return self.forward_numpy_old()

        if (not self.op_params.normal_flag) or self.padding_size_overflow():
            return torch.zeros(self.op_params.output.shape)

        self.fia_op_preprocess.preprocess()

        self.query = self.op_params.query
        self.key = self.op_params.key
        self.value = self.op_params.value

        self.attention_out_bnsd = FiaTensor(
            np.zeros(self.op_params.output.bnsd_shape, dtype=np.float32),
            self.op_params.output.bnsd_shape,
            "fp32",
            "BNSD",
            name="attention_out_bnsd",
        )
        self.lse_bnsd = FiaTensor(
            np.full(self.op_params.lse.bnsd_shape, self.lse_default_value),
            self.op_params.lse.bnsd_shape,
            "fp32",
            "BNSD",
            name="lse_bnsd",
        )

        self.compute_bnsd()

        y_all = self.attention_out_bnsd.to_layout(
            self.op_params.out_layout, self.op_params.actual_seq_lens_q
        )
        fia_debug_data("final output", y_all)

        if self.op_params.softmax_lse_flag:
            lse = self.lse_bnsd.to_layout(
                self.op_params.lse_layout, self.op_params.actual_seq_lens_q
            )
            fia_debug_data("final lse output", lse)
            return torch.from_numpy(y_all), torch.from_numpy(lse)
        else:
            return torch.from_numpy(y_all)

    def forward_torch(self):
        print("flash_mla_with_kvcache...")
        from flash_mla import flash_mla_with_kvcache, get_mla_metadata

        self.query = self.op_params.query
        self.key = self.op_params.key
        self.value = self.op_params.value

        self.fia_op_preprocess.preprocess()

        causal = True if self.op_params.sparse_mode == 3 else False

        q_s = self.query.S
        q_head_nums = self.query.N
        head_dim_v = self.value.D
        kv_head_nums = self.key.N

        q = self.op_params.query.bsnd_data
        tile_scheduler_metadata, num_splits = get_mla_metadata(
            self.op_params.cache_seqlens,
            q_s * q_head_nums // kv_head_nums,
            kv_head_nums,
        )

        def flash_mla():
            fia_debug(
                "*************************************start flash_mla_with_kvcache*************************"
            )
            return flash_mla_with_kvcache(
                q,
                self.op_params.blocked_k,
                self.op_params.block_table,
                self.op_params.cache_seqlens,
                head_dim_v,
                tile_scheduler_metadata,
                num_splits,
                causal=causal,
            )

        out_flash, lse_flash = flash_mla()
        return out_flash.cpu()

    def forward(self):
        if self.mode == "numpy":
            return self.forward_numpy()
        elif self.mode == "torch":
            return self.forward_torch()
        else:
            raise ValueError(f"Unsupported mode {self.mode}")


# SplitFuse

IS_INF_FLAG = False
SPARSE_MODE_INT_MAX = 2147483647

dtypeMap = {
    torch.float16: np.float16,
    torch.bfloat16: bfloat16,
    torch.float32: np.float32,
}

maskTypeMap = {
    ## sparseMode : golden maskType
    0: 0,
    3: 1,
    4: 2,
}


class TestFIAV4SplitFuse:
    @dataclass
    class AuxAttrs:
        preTokens: int
        nextTokens: int
        num_heads: int
        kv_heads: int
        head_size: int
        num_blocks: int
        block_size: int
        mask_type: int
        dtype: any
        kv_dtype: int
        layout_dtype: int
        max_q_seqlen: int
        max_kv_seqlen: int
        inner_prec: int
        scale: float
        sparseMode: int
        kv_storage_mode: str

    @dataclass
    class AttentionInputs:
        query: any
        key_cache: any
        value_cache: any
        block_tables: any
        q_seqlen_list: any
        k_seqlen_list: any
        global_mask: any
        learnable_sink: any
        auxAttrs: any

    @classmethod
    def group_matmul(
        cls, head, kv_head, left, right, high_prec=1, is_benchmark_task=True
    ):
        group_num = head // kv_head
        score = None
        dtype = np.float32
        if is_benchmark_task:
            dtype = np.float64
        for i in range(kv_head):
            group_score = np.matmul(
                left[i * group_num : (i + 1) * group_num, :, :].astype(dtype),
                right[i : (i + 1), :, :].astype(dtype),
            )
            if score is None:
                score = group_score
            else:
                score = np.concatenate((score, group_score), 0)
        return score

    @classmethod
    def softmax_numpy(cls, sim, sink_matrix, batch_i):
        row_max = np.max(sim, axis=-1, keepdims=True)
        valid_row_mask = ~np.isneginf(row_max)
        # add sink rowmax
        if sink_matrix is not None:
            assert sink_matrix.shape == row_max.shape, (
                f"sink_matrix 形状 {sink_matrix.shape} 与 row_max 形状 {row_max.shape} 不一致！"
            )
            row_max[valid_row_mask] = np.maximum(
                row_max[valid_row_mask], sink_matrix[valid_row_mask]
            )

        sim_sub = sim - row_max
        sim_sub_high = sim.astype(np.float64) - row_max.astype(np.float64)

        sim_sub = np.exp(sim_sub)
        sim_sub_high = np.exp(sim_sub_high)
        row_sum = np.sum(sim_sub, axis=-1, keepdims=True)
        row_sum_high = np.sum(sim_sub_high, axis=-1, keepdims=True)

        if sink_matrix is not None:
            sink_exp = np.exp(sink_matrix - row_max)
            sink_exp_high = np.exp(
                sink_matrix.astype(np.float64) - row_max.astype(np.float64)
            )
            row_sum = row_sum + sink_exp
            row_sum_high = row_sum_high + sink_exp_high

        soft_res = sim_sub / row_sum
        lse = np.squeeze((np.log(row_sum_high) + row_max.astype(np.float64)), axis=-1)

        return soft_res, lse, row_max

    def softmax1(
        self,
        qk_result,
        is_first,
        gm,
        is_kvs_last_loop,
        sink_matrix,
        interm_dtype=np.float16,
    ):
        sim = qk_result.astype(interm_dtype)
        lm = np.max(sim, axis=-1, keepdims=True)
        if is_first:
            hm = lm
            dm = 0

        else:
            hm = np.maximum(gm, lm)
            dm = gm - hm

        valid_hm_mask = ~np.isneginf(hm)
        if sink_matrix is not None and is_kvs_last_loop:
            assert sink_matrix.shape == hm.shape, (
                f"sink_matrix 形状 {sink_matrix.shape} 与 hm 形状 {hm.shape} 不一致！"
            )
            hm[valid_hm_mask] = np.maximum(
                hm[valid_hm_mask], sink_matrix[valid_hm_mask]
            )
            dm = gm - hm if not is_first else 0

        gm = hm
        sim_sub = sim - hm
        sim_sub = np.exp(sim_sub.astype(interm_dtype))

        row_sum = np.sum(sim_sub, axis=-1, keepdims=True)

        sink_exp = None
        if sink_matrix is not None and is_kvs_last_loop:
            sink_exp = np.exp((sink_matrix - hm).astype(interm_dtype)).astype(
                interm_dtype
            )

        return sim_sub, row_sum, dm, gm, sink_exp

    def qkMM1(self, query, key, is_benchmark_task):
        result = None
        qk_k = key.shape[1]
        qk_k_split = 128
        qk_k_loop = (qk_k + 127) // 128
        for qk_k_loop_idx in range(qk_k_loop):
            sub_k = (
                128
                if qk_k_loop_idx != (qk_k_loop - 1)
                else (qk_k - qk_k_loop_idx * 128)
            )
            partial_Query = query[
                :, :, qk_k_loop_idx * 128 : qk_k_loop_idx * 128 + sub_k
            ]
            partial_Key = key[:, qk_k_loop_idx * 128 : qk_k_loop_idx * 128 + sub_k, :]
            result_split = self.group_matmul(
                partial_Query.shape[0],
                partial_Key.shape[0],
                partial_Query,
                partial_Key,
                0,
                is_benchmark_task,
            )
            if result is None:
                result = result_split
            else:
                result = result + result_split
        return result

    def pvMM2(self, p, value, is_benchmark_task):
        result = None
        pv_k = value.shape[1]
        pv_k_split = 128
        pv_k_loop = (pv_k + 127) // 128
        for pv_k_loop_idx in range(pv_k_loop):
            sub_k = (
                128
                if pv_k_loop_idx != (pv_k_loop - 1)
                else (pv_k - pv_k_loop_idx * 128)
            )
            partial_P = p[:, :, pv_k_loop_idx * 128 : pv_k_loop_idx * 128 + sub_k]
            partial_Value = value[
                :, pv_k_loop_idx * 128 : pv_k_loop_idx * 128 + sub_k, :
            ]
            result_split = self.group_matmul(
                partial_P.shape[0],
                partial_Value.shape[0],
                partial_P,
                partial_Value,
                0,
                is_benchmark_task,
            )
            if result is None:
                result = result_split
            else:
                result = result + result_split
        return result

    # ===================== 核心SWA计算函数（你的参数版）=====================
    def calc_swa_kvsLoop(
        self,
        preToken,
        nextToken,
        qSeqlen,
        kvSeqlen,
        qSBlockIdx,  # 传入：q_id（Q分块索引）
        qSBlockSize,  # 传入：q_sub_len（Q子块长度）
        curQSBlockTile,  # 传入：q_chunk_size（Q分块大小）
        MAX_KV_STACK_LEN,  # 传入：kv_chunk_size（KV分块大小）
    ):
        # 初始化变量（和C++完全一致）
        leftPointPreToken = kvSeqlen
        leftPointNextToken = 0
        startIdx = 0
        kvSLoopNumTotal = 0
        preTokenEndLen = 0
        preTokenStartLen = 0
        nextTokenStartLen = 0

        # ===================== 1. 计算左窗口 startIdx =====================
        if preToken < 0 and (-preToken) >= qSeqlen:
            startIdx = (kvSeqlen // MAX_KV_STACK_LEN) + 1
        elif preToken != SPARSE_MODE_INT_MAX:
            leftPointPreToken = kvSeqlen - qSeqlen - preToken
            preTokenStartLen = qSBlockIdx * curQSBlockTile + leftPointPreToken
            preTokenEndLen = (
                qSBlockIdx * curQSBlockTile + qSBlockSize + leftPointPreToken
            )
            startIdx = max(0, preTokenStartLen) // MAX_KV_STACK_LEN
        else:
            startIdx = 0

        # ===================== 2. 计算右窗口 kvSLoopNumTotal =====================
        if nextToken < 0 and (-nextToken) >= kvSeqlen:
            kvSLoopNumTotal = 0
        elif nextToken != SPARSE_MODE_INT_MAX:
            leftPointNextToken = kvSeqlen - qSeqlen + nextToken
            nextTokenStartLen = qSBlockIdx * curQSBlockTile + leftPointNextToken
            nextTokenEndLen = (
                qSBlockIdx * curQSBlockTile + qSBlockSize + leftPointNextToken
            )

            noSkipKvS = min(
                kvSeqlen,
                (nextTokenEndLen + MAX_KV_STACK_LEN - 1)
                // MAX_KV_STACK_LEN
                * MAX_KV_STACK_LEN,
            )
            noSkipKvS = kvSeqlen if noSkipKvS <= 0 else noSkipKvS
            kvSLoopNumTotal = (noSkipKvS + MAX_KV_STACK_LEN - 1) // MAX_KV_STACK_LEN
        else:
            noSkipKvS = kvSeqlen
            (noSkipKvS + MAX_KV_STACK_LEN - 1) // MAX_KV_STACK_LEN

        return startIdx, kvSLoopNumTotal

    def ref_flash_attention(
        self,
        query,
        key,
        value,
        scale,
        mask,
        attention_inputs: AttentionInputs,
        sink_matrix,
        batch_i,
        swa_auxAttrs: AuxAttrs,
        is_benchmark_task,
    ):
        data_type = attention_inputs.auxAttrs.dtype
        inner_prec = attention_inputs.auxAttrs.inner_prec
        interm_dtype = np.float16 if inner_prec == 1 else np.float32
        query = np.transpose(query, (1, 0, 2))
        print(f"lwg_query.shape:{query.shape}")
        key = np.transpose(key, (1, 2, 0))
        print(f"lwg_key.shape:{key.shape}")
        value = np.transpose(value, (1, 0, 2))
        scale = np.float16(scale) if inner_prec == 1 else np.float32(scale)
        sparseMode = swa_auxAttrs.sparseMode
        preToken = swa_auxAttrs.preTokens
        nextToken = swa_auxAttrs.nextTokens

        kv_seqlen = key.shape[2]  # KV总长度
        q_seqlen = query.shape[1]  # Q总长度
        kv_chunk_size = 512  # KV分块大小
        q_chunk_size = 128  # Q分块大小：128
        num_heads = query.shape[0]
        head_size_vo = value.shape[2]

        # 存储最终结果
        out_shape = (num_heads, q_seqlen, head_size_vo)
        final_output = np.zeros(out_shape, dtype=np.float32)
        # ✅ 修复1：初始化完整的LSE存储数组（和Q长度对齐）
        final_lse = np.zeros((num_heads, q_seqlen), dtype=np.float32)
        if is_benchmark_task:
            final_output = np.transpose(final_output, (1, 0, 2))
            return final_output, final_lse

        # ============== 外层循环：Q 按 128 切块 ==============
        for q_id, q_start in enumerate(range(0, q_seqlen, q_chunk_size)):
            # 1. 切分当前 Q 小块
            q_sub_len = min(q_chunk_size, q_seqlen - q_start)
            sub_query = query[:, q_start : q_start + q_sub_len, :]

            # 每个 Q 小块独立初始化中间状态
            gm_chunk = None
            gl_chunk = None
            go_chunk = None

            kvSLoopNumTotal = 0
            MASK_TYPE = 0  # no mask
            startIdx = 0
            if mask is not None:
                if sparseMode == 3:  # causal
                    MASK_TYPE = 3
                    diffS = kv_seqlen - q_seqlen
                    diffS = 0 if diffS < 0 else diffS
                    noSkipKvS = (q_id + 1) * q_chunk_size + diffS
                    noSkipKvS = min(noSkipKvS, kv_seqlen)
                    kvSLoopNumTotal = (noSkipKvS + kv_chunk_size - 1) // kv_chunk_size
                elif sparseMode == 4:  # swa mask
                    MASK_TYPE = 4
                    startIdx, kvSLoopNumTotal = self.calc_swa_kvsLoop(
                        preToken,
                        nextToken,
                        q_seqlen,
                        kv_seqlen,
                        qSBlockIdx=q_id,
                        qSBlockSize=q_sub_len,
                        curQSBlockTile=q_chunk_size,
                        MAX_KV_STACK_LEN=kv_chunk_size,
                    )
                    if startIdx >= kvSLoopNumTotal or kvSLoopNumTotal <= 0:
                        continue
            sink_matrix_sub = None
            if sink_matrix is not None:
                sink_matrix_sub = sink_matrix[:, q_start : q_start + q_sub_len, :]

            is_kvs_last_loop = False

            # ============== 内层循环：KV 按 512 切块 ==============
            for kv_id, kv_start in enumerate(range(0, kv_seqlen, kv_chunk_size)):
                sub_kv_len = min(kv_chunk_size, kv_seqlen - kv_start)
                if MASK_TYPE == 0:  # no mask
                    is_kvs_last_loop = kv_start + kv_chunk_size >= kv_seqlen
                elif MASK_TYPE == 3 or MASK_TYPE == 4:  # causal & swa
                    is_kvs_last_loop = (kv_id + 1) >= kvSLoopNumTotal
                else:
                    exit(-99)
                sub_key = key[:, :, kv_start : kv_start + sub_kv_len]
                sub_value = value[:, kv_start : kv_start + sub_kv_len, :]
                sub_mask = None
                if mask is not None:
                    sub_mask = mask[
                        q_start : q_start + q_sub_len, kv_start : kv_start + sub_kv_len
                    ].astype(interm_dtype)

                # QK 计算
                qk_result = self.qkMM1(sub_query, sub_key, is_benchmark_task).astype(
                    interm_dtype
                )
                qk_result = qk_result * scale
                if mask is not None:
                    qk_result += sub_mask

                # 分块 Softmax
                p_result, row_sum, dm, gm_chunk, sink_exp = self.softmax1(
                    qk_result,
                    kv_start == 0,
                    gm_chunk,
                    is_kvs_last_loop,
                    sink_matrix_sub,
                    interm_dtype,
                )
                p_result = p_result.astype(data_type)

                # PV 计算
                lo = self.pvMM2(p_result, sub_value, is_benchmark_task).astype(
                    interm_dtype
                )

                # 累积中间结果
                if kv_start == 0:
                    gl_chunk = row_sum
                    go_chunk = lo
                else:
                    dm = np.exp(dm)
                    gl_chunk = gl_chunk * dm
                    gl_chunk = gl_chunk + row_sum
                    go_chunk = go_chunk * dm
                    go_chunk = go_chunk + lo

                # sink 处理
                if is_kvs_last_loop and sink_exp is not None:
                    gl_chunk = gl_chunk + sink_exp

                if is_kvs_last_loop:
                    break

            # ============== 当前 Q 小块计算完成 ==============
            # 计算当前块的输出O
            go_chunk = go_chunk / gl_chunk
            final_output[:, q_start : q_start + q_sub_len, :] = go_chunk

            # ✅ 修复2：计算当前Q块的LSE，并存入对应位置（核心！）
            lse_chunk = np.squeeze((np.log(gl_chunk) + gm_chunk), axis=-1).astype(
                np.float32
            )
            final_lse[:, q_start : q_start + q_sub_len] = lse_chunk
        final_output = np.transpose(final_output, (1, 0, 2))
        # ✅ 修复3：返回拼接完成的完整LSE
        return final_output.astype(data_type), final_lse

    def ref_masked_attention(
        self,
        query,  # (q_seqlen, num_heads, head_size)
        key,  # (k_seqlen, kv_heads, head_size)
        value,
        scale: float,
        mask,  # (q_seqlen, k_seqlen)
        sink_matrix,
        batch_i,
        is_benchmark_task,
    ):
        if not is_benchmark_task:
            q_seqlen = query.shape[0]  # Q总长度
            num_heads = query.shape[1]
            head_size_vo = value.shape[2]

            out_shape = (q_seqlen, num_heads, head_size_vo)
            final_output = np.zeros(out_shape, dtype=np.float32)
            # ✅ 修复1：初始化完整的LSE存储数组（和Q长度对齐）
            final_lse = np.zeros((num_heads, q_seqlen), dtype=np.float32)
            return final_output, final_lse

        query = np.transpose(query, (1, 0, 2))
        key = np.transpose(key, (1, 2, 0))
        sim_high = self.group_matmul(
            query.shape[0], key.shape[0], query, key, 1, is_benchmark_task
        )  # (head_num, q_seqlen, k_seqlen)
        sim_high = sim_high * scale
        if mask is not None:
            sim_high = sim_high + (
                mask[: sim_high.shape[-2], : sim_high.shape[-1]]
            ).astype(np.float32)
        p_high, lse_high, gm = self.softmax_numpy(sim_high, sink_matrix, batch_i)
        lse_high = lse_high.astype(np.float64)
        p = p_high.astype(query.dtype)
        p_high = p_high.astype(np.float32)
        value = np.transpose(value, (1, 0, 2))

        out_high = self.group_matmul(
            query.shape[0], key.shape[0], p_high, value, 1, is_benchmark_task
        )
        out_high = np.transpose(out_high, (1, 0, 2))
        return out_high, lse_high

    def ref_single_query_cached_kv_attention(
        self,
        attention_inputs: AttentionInputs,
        output,
        golden_gpu_output,
        golden_lse_output,
        golden_gpu_lse_output,
        is_benchmark_task,
    ) -> None:
        num_heads = attention_inputs.auxAttrs.num_heads
        kv_heads = attention_inputs.auxAttrs.kv_heads
        head_size_qk = attention_inputs.auxAttrs.head_size
        head_size_vo = attention_inputs.auxAttrs.head_size
        block_size = attention_inputs.auxAttrs.block_size
        max_q_seqlen = attention_inputs.auxAttrs.max_q_seqlen
        inner_prec = attention_inputs.auxAttrs.inner_prec
        scale = attention_inputs.auxAttrs.scale
        learnable_sink = attention_inputs.learnable_sink
        sparseMode = attention_inputs.auxAttrs.sparseMode
        if learnable_sink is not None:
            learnable_sink = learnable_sink.astype(np.float32)

        swa_auxAttrs = attention_inputs.auxAttrs

        batch = len(attention_inputs.q_seqlen_list)
        cu_seqlen = 0
        kv_seqlen_now = 0
        for i in range(batch):
            q_seqlen = int(attention_inputs.q_seqlen_list[i])
            k_seqlen = int(attention_inputs.k_seqlen_list[i])
            # qs=0场景（TND累加和数组中actualSeqQ[i]==actualSeqQ[i-1]或首轴为0）：
            # 该batch无query token，输出区域为空切片无需填充，跳过计算避免
            # np.max对zero-size数组归约报错。
            if q_seqlen == 0:
                cu_seqlen += q_seqlen
                kv_seqlen_now += k_seqlen
                continue
            # kvs=0场景（PA下actualSeqKV[i]==0）：有query但无KV，输出为0、lse为-inf。
            # 跳过ref计算避免np.stack对空list报错，直接填充golden结果。
            if k_seqlen == 0:
                output[cu_seqlen : cu_seqlen + q_seqlen, :, :] = 0
                golden_gpu_output[cu_seqlen : cu_seqlen + q_seqlen, :, :] = 0
                golden_lse_output[:, cu_seqlen : cu_seqlen + q_seqlen] = -np.inf
                golden_gpu_lse_output[:, cu_seqlen : cu_seqlen + q_seqlen] = -np.inf
                cu_seqlen += q_seqlen
                kv_seqlen_now += k_seqlen
                continue

            q = None
            if attention_inputs.auxAttrs.layout_dtype == 1:
                q = attention_inputs.query[cu_seqlen : (cu_seqlen + q_seqlen), :, :]
            else:
                q = attention_inputs.query[
                    i * max_q_seqlen : (i * max_q_seqlen + q_seqlen), :, :
                ]
            keys = None
            values = None
            if attention_inputs.auxAttrs.kv_dtype == 1:
                keys = []
                values = []
                block_table = attention_inputs.block_tables[i]
                for j in range(k_seqlen):
                    if block_size == 0:
                        print(f"block_size: {block_size} \n")
                    else:
                        block_number = int(block_table[j // block_size])
                        block_offset = j % block_size

                    if attention_inputs.auxAttrs.kv_storage_mode == "bnbd":
                        # BNBD layout: (numBlocks, kvHeads, blockSize, headSize)
                        k = attention_inputs.key_cache[block_number, :, block_offset, :]
                        v = attention_inputs.value_cache[
                            block_number, :, block_offset, :
                        ]
                    else:
                        # BBH layout: (numBlocks, blockSize, kvHeads, headSize)
                        k = attention_inputs.key_cache[block_number, block_offset, :, :]
                        v = attention_inputs.value_cache[
                            block_number, block_offset, :, :
                        ]
                    k = k.reshape(kv_heads, head_size_qk)
                    keys.append(k)

                    v = v.reshape(kv_heads, head_size_vo)
                    values.append(v)
                keys = np.stack(keys, axis=0)
                values = np.stack(values, axis=0)
            elif attention_inputs.auxAttrs.kv_dtype == 0:
                if attention_inputs.auxAttrs.layout_dtype == 1:
                    keys = attention_inputs.key_cache[
                        kv_seqlen_now : kv_seqlen_now + k_seqlen, :, :
                    ]
                    values = attention_inputs.value_cache[
                        kv_seqlen_now : kv_seqlen_now + k_seqlen, :, :
                    ]
                else:
                    keys = attention_inputs.key_cache[i, :, :, :]
                    values = attention_inputs.value_cache[i, :, :, :]

            if attention_inputs.auxAttrs.mask_type == 1:
                mask = attention_inputs.global_mask[
                    cu_seqlen : (cu_seqlen + q_seqlen), :
                ]
            elif attention_inputs.auxAttrs.mask_type == 2:
                mask = attention_inputs.global_mask[
                    cu_seqlen : (cu_seqlen + q_seqlen), :
                ]
            elif attention_inputs.auxAttrs.mask_type == 0:
                mask = None

            sink_matrix = None
            if learnable_sink is not None:
                # [num_heads → [num_heads, 1, 1]
                sink_expanded = np.expand_dims(learnable_sink, axis=1)
                sink_expanded = np.expand_dims(sink_expanded, axis=2)
                # [num_heads, 1, 1] → [num_heads, q_seqlen, 1]
                sink_matrix = np.broadcast_to(
                    sink_expanded, shape=(learnable_sink.shape[0], q_seqlen, 1)
                )

            preTokens = attention_inputs.auxAttrs.preTokens
            nextTokens = attention_inputs.auxAttrs.nextTokens

            preTokensChange = preTokens - k_seqlen + q_seqlen
            nextTokensChange = nextTokens + k_seqlen - q_seqlen
            nextTokensError = -nextTokensChange if nextTokensChange < 0 else 0
            preTokensError = (
                (q_seqlen - k_seqlen - preTokensChange)
                if q_seqlen > k_seqlen + preTokensChange
                else 0
            )
            actualSeq = q_seqlen

            actualSeq -= nextTokensError
            actualSeq -= preTokensError
            print(
                f"ljl-2 {i},:{preTokens},{nextTokens},{preTokensChange},{nextTokensChange},{preTokensError},{nextTokensError},{actualSeq}"
            )
            if actualSeq != q_seqlen and sparseMode == 4:
                if nextTokensError != 0:
                    # 前n行置0
                    actualSeq = q_seqlen - actualSeq
                elif preTokensError != 0:
                    # 后n行置0
                    actualSeq = actualSeq

            out_normal, lse = self.ref_masked_attention(
                q, keys, values, scale, mask, sink_matrix, i, is_benchmark_task
            )
            out_gpu, lse_gpu = self.ref_flash_attention(
                q,
                keys,
                values,
                scale,
                mask,
                attention_inputs,
                sink_matrix,
                i,
                swa_auxAttrs,
                is_benchmark_task,
            )
            out_gpu_test = torch.from_numpy(out_gpu.astype(np.float32))
            nan_out_gpu = torch.isnan(out_gpu_test)
            nan_count = nan_out_gpu.sum().item()

            out = out_normal.reshape(-1, num_heads, head_size_vo)
            out = out.reshape(-1, num_heads, head_size_vo)
            out_gpu = out_gpu.reshape(-1, num_heads, head_size_vo)

            if attention_inputs.auxAttrs.layout_dtype == 1:
                output[cu_seqlen : cu_seqlen + q_seqlen, :, :] = out
                golden_gpu_output[cu_seqlen : cu_seqlen + q_seqlen, :, :] = out_gpu

                golden_lse_output[:, cu_seqlen : cu_seqlen + q_seqlen] = lse
                golden_gpu_lse_output[:, cu_seqlen : cu_seqlen + q_seqlen] = lse_gpu

                if actualSeq != q_seqlen and sparseMode == 4:
                    if nextTokensError != 0:
                        output[cu_seqlen : cu_seqlen + actualSeq, :, :] = 0  # 前n行置0
                        golden_gpu_output[cu_seqlen : cu_seqlen + actualSeq, :, :] = 0
                        golden_lse_output[:, cu_seqlen : cu_seqlen + actualSeq] = np.inf
                        golden_gpu_lse_output[:, cu_seqlen : cu_seqlen + actualSeq] = (
                            np.inf
                        )
                    elif preTokensError != 0:
                        output[cu_seqlen + actualSeq : cu_seqlen + q_seqlen, :, :] = (
                            0  # 后n行置0
                        )
                        golden_gpu_output[
                            cu_seqlen + actualSeq : cu_seqlen + q_seqlen, :, :
                        ] = 0
                        golden_lse_output[
                            :, cu_seqlen + actualSeq : cu_seqlen + q_seqlen
                        ] = np.inf
                        golden_gpu_lse_output[
                            :, cu_seqlen + actualSeq : cu_seqlen + q_seqlen
                        ] = np.inf
            else:
                output[i * max_q_seqlen : i * max_q_seqlen + q_seqlen, :, :] = out
                golden_gpu_output[
                    i * max_q_seqlen : i * max_q_seqlen + q_seqlen, :, :
                ] = out_gpu

                golden_lse_output[:, i * max_q_seqlen : i * max_q_seqlen + q_seqlen] = (
                    lse
                )
                golden_gpu_lse_output[
                    :, i * max_q_seqlen : i * max_q_seqlen + q_seqlen
                ] = lse_gpu
                if actualSeq != q_seqlen and sparseMode == 4:
                    if nextTokensError != 0:
                        output[
                            i * max_q_seqlen : i * max_q_seqlen + actualSeq, :, :
                        ] = 0
                        golden_gpu_output[
                            i * max_q_seqlen : i * max_q_seqlen + actualSeq, :, :
                        ] = 0

                        golden_lse_output[
                            :, i * max_q_seqlen : i * max_q_seqlen + actualSeq
                        ] = np.inf
                        golden_gpu_lse_output[
                            :, i * max_q_seqlen : i * max_q_seqlen + actualSeq
                        ] = np.inf
                    elif preTokensError != 0:
                        output[
                            i * max_q_seqlen + actualSeq : i * max_q_seqlen + q_seqlen,
                            :,
                            :,
                        ] = 0
                        golden_gpu_output[
                            i * max_q_seqlen + actualSeq : i * max_q_seqlen + q_seqlen,
                            :,
                            :,
                        ] = 0

                        golden_lse_output[
                            :,
                            i * max_q_seqlen + actualSeq : i * max_q_seqlen + q_seqlen,
                        ] = np.inf
                        golden_gpu_lse_output[
                            :,
                            i * max_q_seqlen + actualSeq : i * max_q_seqlen + q_seqlen,
                        ] = np.inf

            cu_seqlen += q_seqlen
            kv_seqlen_now += k_seqlen

    def calc_data(self, attention_inputs: AttentionInputs, is_benchmark_task):
        num_tokens = attention_inputs.query.shape[0]
        shape_out = (
            num_tokens,
            attention_inputs.auxAttrs.num_heads,
            attention_inputs.auxAttrs.head_size,
        )
        golden_output = np.zeros(shape_out, dtype=np.float32)
        golden_gpu_output = np.zeros(shape_out, dtype=np.float32)

        lse_shape_out = (attention_inputs.auxAttrs.num_heads, num_tokens)
        golden_lse_output = np.zeros(lse_shape_out, dtype=np.float32)
        golden_gpu_lse_output = np.zeros(lse_shape_out, dtype=np.float32)

        self.ref_single_query_cached_kv_attention(
            attention_inputs,
            golden_output,
            golden_gpu_output,
            golden_lse_output,
            golden_gpu_lse_output,
            is_benchmark_task,
        )

        golden_lse_output = np.transpose(golden_lse_output, (1, 0))
        golden_lse_output = np.expand_dims(golden_lse_output, axis=2)
        golden_gpu_lse_output = np.transpose(golden_gpu_lse_output, (1, 0))
        golden_gpu_lse_output = np.expand_dims(golden_gpu_lse_output, axis=2)

        return (
            golden_output,
            golden_gpu_output,
            golden_lse_output,
            golden_gpu_lse_output,
        )


def construct_all_inf_rows(Q, K, seed=42):
    """
    构造函数：修改 Q 的部分行为 -inf，并调整 K 以确保 S 对应行为全 -inf。

    参数:
        Q: torch.Tensor, 形状 (T, N, D), 类型建议为 float16 或 float32
        K: torch.Tensor, 形状 (T, N, D)
        seed: 随机种子，用于选择失效的行
    返回:
        Q_mod: 修改后的 Q
        K_mod: 修改后的 K (全正数)
        target_rows: 被设置为无效的行索引列表
    """
    T, N, D = Q.shape
    # 局部生成器，保证函数内部行为只受 seed 影响
    gen = torch.Generator(device="cpu").manual_seed(seed)
    Q_mod = Q.clone()
    # 修改 K：确保全正且无零，防止 NaN
    K_mod = torch.abs(K) + 1e-3
    # 随机选择要“作废”的行数 (1 到 T/2 行)
    num_rows = torch.randint(1, max(2, T // 2 + 1), (1,), generator=gen).item()
    target_rows = torch.randperm(T, generator=gen)[:num_rows].tolist()
    # 注入 -inf
    for row in target_rows:
        Q_mod[row, :, :] = float("-inf")
    return Q_mod, K_mod, target_rows


def gen_list_from_cumSum(seqlenArray):
    seqlenList = []
    preSeqSum = 0
    for i in range(len(seqlenArray)):
        seqlenList.append(seqlenArray[i] - preSeqSum)
        preSeqSum = seqlenArray[i]
    return seqlenList


def gen_actual_seqlen_list_golden(
    actualseqlengths, actualseqlengthskv, inputLayout, pagedAttentionFlag
):
    qSeqlenList = []
    kvSeqlenList = []
    if inputLayout == "TND":
        qSeqlenList = gen_list_from_cumSum(actualseqlengths)
        if pagedAttentionFlag:
            kvSeqlenList = list(actualseqlengthskv)
        else:
            kvSeqlenList = gen_list_from_cumSum(actualseqlengthskv)
    else:
        qSeqlenList = list(actualseqlengths)
        kvSeqlenList = list(actualseqlengthskv)
    return qSeqlenList, kvSeqlenList


def create_binary_matrix(qSeqlen, kvSeqlen, preToken, nextToken):
    preToken = kvSeqlen - qSeqlen - preToken
    nextToken = kvSeqlen - qSeqlen + nextToken
    matrix = [[0 for _ in range(kvSeqlen)] for _ in range(qSeqlen)]
    for i in range(qSeqlen):
        for j in range(kvSeqlen):
            is_below_pretoken_line = (-i + j) < preToken
            is_above_nexttoken_line = (-i + j) > nextToken
            if is_below_pretoken_line or is_above_nexttoken_line:
                matrix[i][j] = 1

    return np.array(matrix)


def aclnn_op_func_fia_split_fuse_golden(input_data: InputDataset, is_benchmark_task):
    input_data_dtype = input_data.kwargs["query"].dtype

    if input_data_dtype == torch.float16:
        query = input_data.kwargs["query"].numpy()
    elif input_data_dtype == torch.bfloat16:
        query = input_data.kwargs["query"].to(torch.float32).numpy().astype(bfloat16)
    else:
        query = input_data.kwargs["query"].numpy()

    if input_data_dtype == torch.float16:
        key = input_data.kwargs["key"][0].numpy()
    elif input_data_dtype == torch.bfloat16:
        key = input_data.kwargs["key"][0].to(torch.float32).numpy().astype(bfloat16)
    else:
        key = input_data.kwargs["key"][0].numpy()

    if input_data_dtype == torch.float16:
        value = input_data.kwargs["value"][0].numpy()
    elif input_data_dtype == torch.bfloat16:
        value = input_data.kwargs["value"][0].to(torch.float32).numpy().astype(bfloat16)
    else:
        value = input_data.kwargs["value"][0].numpy()

    blockTable = None
    pagedAttentionFlag = False
    if input_data.kwargs["blockTableOptional"] != None:
        blockTable = input_data.kwargs["blockTableOptional"].numpy()
        pagedAttentionFlag = True
    ## gen actual seqlen
    inputLayout = input_data.kwargs["inputLayout"]
    batch = len(input_data.kwargs["actualSeqLengthsOptional"])
    actualseqlengths = [0] * batch
    actualseqlengthsKv = [0] * batch
    for i in range(batch):
        actualseqlengths[i] = input_data.kwargs["actualSeqLengthsOptional"][i]
        actualseqlengthsKv[i] = input_data.kwargs["actualSeqLengthsKvOptional"][i]
    qSeqlenList, kvSeqlenList = gen_actual_seqlen_list_golden(
        actualseqlengths, actualseqlengthsKv, inputLayout, pagedAttentionFlag
    )
    maxKvSeqlen = max(kvSeqlenList)
    maxQSeqlen = max(qSeqlenList)
    totalQTokens = sum(qSeqlenList)
    preTokens = input_data.kwargs["preTokens"]
    nextTokens = input_data.kwargs["nextTokens"]
    ## gen mask
    fullMask = None
    pre_mask_factor = (
        -3e38
        if input_data_dtype == torch.bfloat16 or input_data_dtype == torch.float32
        else -6e4
    )
    if (
        input_data.kwargs["attenMaskOptional"] != None
        and input_data.kwargs["sparseMode"] == 3
    ):
        maskDtype = dtypeMap[input_data_dtype]
        fullMask = np.zeros(shape=(totalQTokens, maxKvSeqlen)).astype(maskDtype)
        prevQseqlen = 0
        for i in range(len(qSeqlenList)):
            qSeqlen = qSeqlenList[i]
            kSeqlen = kvSeqlenList[i]
            tri = np.ones((qSeqlen, qSeqlen))
            tri = np.triu(tri, 1)
            tri *= pre_mask_factor
            fullMask[
                prevQseqlen : (prevQseqlen + qSeqlen), kSeqlen - qSeqlen : kSeqlen
            ] = tri
            prevQseqlen += qSeqlen

    if input_data.kwargs["sparseMode"] == 4:
        pre_mask_factor = -3e38
        maskDtype = bfloat16
        print(
            f"ljl-totalQTokens:{totalQTokens},maxKvSeqlen:{maxKvSeqlen},{kvSeqlenList}"
        )
        fullMask = np.zeros(shape=(totalQTokens, maxKvSeqlen)).astype(maskDtype)
        prevQseqlen = 0
        for i in range(len(qSeqlenList)):
            qSeqlen = qSeqlenList[i]
            kSeqlen = kvSeqlenList[i]
            tri = create_binary_matrix(qSeqlen, kSeqlen, preTokens, nextTokens)
            tri = tri.astype(maskDtype)
            tri *= pre_mask_factor
            fullMask[prevQseqlen : (prevQseqlen + qSeqlen), :kSeqlen] = tri
            prevQseqlen += qSeqlen

    learnable_sink = None
    if input_data.kwargs.get("learnableSinkOptional") is not None:
        sink_torch_dtype = input_data.kwargs["learnableSinkOptional"].dtype
        if sink_torch_dtype == torch.float16:
            learnable_sink = input_data.kwargs["learnableSinkOptional"].numpy()
            learnable_sink = learnable_sink.astype(np.float32)
        elif sink_torch_dtype == torch.bfloat16:
            learnable_sink = (
                input_data.kwargs["learnableSinkOptional"].to(torch.float32).numpy()
            )
            learnable_sink_bf16 = np.array(learnable_sink, dtype=bfloat16)
            # 2. 强制第三头的sink值为BF16近似值（关键：和NPU侧完全一致）
            learnable_sink = learnable_sink_bf16.astype(np.float32)
        else:
            learnable_sink = input_data.kwargs["learnableSinkOptional"].numpy()
    else:
        print("golden, sink为空 \n")

    numHeads = input_data.kwargs["numHeads"]
    # if numHeads == 0:
    # exit(-100)
    kvHeads = input_data.kwargs["numKeyValueHeads"]
    headSize = query.shape[2] if inputLayout == "TND" else 0
    numBlocks = key.shape[0] if pagedAttentionFlag == True else 0
    blockSize = input_data.kwargs["blockSize"]
    maskType = maskTypeMap[input_data.kwargs["sparseMode"]]
    sparseMode = input_data.kwargs["sparseMode"]
    dtype = dtypeMap[input_data_dtype]
    kvOrgMode = 1 if pagedAttentionFlag == True else 0
    layoutMode = 1 if inputLayout == "TND" else 0
    goldenGpuPrecision = input_data.kwargs["innerPrecise"]
    softmaxLseFlag = input_data.kwargs["softmaxLseFlag"]
    scale = float(input_data.kwargs["scaleValue"])

    kvStorageMode = "bbh"
    if pagedAttentionFlag == True:
        if len(key.shape) == 3:  # BBH: (numBlocks, blockSize, H)
            key = key.reshape(key.shape[:-1] + (kvHeads, headSize))
            value = value.reshape(value.shape[:-1] + (kvHeads, headSize))
        else:  # BNBD: already (numBlocks, kvHeads, blockSize, headSize)
            kvStorageMode = "bnbd"
    # print(key.shape)
    testObj = TestFIAV4SplitFuse()
    auxAttrs = testObj.AuxAttrs(
        preTokens,
        nextTokens,
        numHeads,
        kvHeads,
        headSize,
        numBlocks,
        blockSize,
        maskType,
        dtype,
        kvOrgMode,
        layoutMode,
        maxQSeqlen,
        maxKvSeqlen,
        goldenGpuPrecision,
        scale,
        sparseMode,
        kvStorageMode,
    )
    attentionInputs = testObj.AttentionInputs(
        query,
        key,
        value,
        blockTable,
        qSeqlenList,
        kvSeqlenList,
        fullMask,
        learnable_sink,
        auxAttrs,
    )
    golden_output, golden_gpu_output, golden_lse_output, golden_gpu_lse_output = (
        testObj.calc_data(attentionInputs, is_benchmark_task)
    )
    if inputLayout == "TND" and totalQTokens < query.shape[0]:
        golden_output[totalQTokens:, :, :] = 0
        golden_gpu_output[totalQTokens:, :, :] = 0
    if golden_output.dtype == "bfloat16":
        print("=================================走入bf16分支", golden_output.dtype)
        golden_output = torch.from_numpy(golden_output.astype(np.float32))
        print(f"golden_output_dtype:{golden_output.dtype} \n")
        golden_gpu_output = torch.from_numpy(golden_gpu_output.astype(np.float32))
        golden_lse_output = torch.from_numpy(golden_lse_output.astype(np.float32))
        golden_gpu_lse_output = torch.from_numpy(
            golden_gpu_lse_output.astype(np.float32)
        )
    else:
        golden_output = torch.from_numpy(golden_output)
        golden_gpu_output = torch.from_numpy(golden_gpu_output)
        golden_lse_output = torch.from_numpy(golden_lse_output)
        golden_gpu_lse_output = torch.from_numpy(golden_gpu_lse_output)
    if not softmaxLseFlag:
        golden_lse_output = torch.tensor([])
        golden_gpu_lse_output = torch.tensor([])

    if not is_benchmark_task:
        # 标杆返回这个
        return golden_gpu_output, golden_gpu_lse_output
    else:
        # 真值返回下面的
        return golden_output, golden_lse_output


# ATK 处理逻辑
arr_tuple_none = -9223372036854775808
dtype_map = {
    torch.float16: "fp16",
    torch.bfloat16: "bf16",
    torch.float32: "fp32",
    torch.int8: "int8",
    torch.bool: "bool",
    torch.int32: "int32",
    torch.int64: "int32",
    torch.uint8: "uin8",
    torch.int4: "int4",
    torch.float64: "fp64",
}


def overwrite_structured_mask(input_data):
    """
    根据 sparseMode 强制修改 attenMask 的数值结构：
    Mode 2/3: 下三角保留 (Causal)，上三角遮蔽。
    Mode 4:   Band 结构。
    """
    mode = input_data.kwargs.get("sparseMode", 0)
    mask_tensor = input_data.kwargs.get("attenMaskOptional", None)

    query = input_data.kwargs["query"].cpu().to(dtype=torch.float32).numpy()
    inputLaout = input_data.kwargs["inputLayout"]
    pfaFlag = False
    if inputLaout == "BNSD":
        pfaFlag = query.shape[2] > 1
    if inputLaout in ["BSH", "BSND"]:
        pfaFlag = query.shape[1] > 1

    # 如果没有 mask 或者 mode 是 0/1 (Default/All)，通常保持随机或全零即可，不做强制修改
    # (或者根据需求，Mode 0 也可以重写，这里主要处理 2,3,4)
    if mask_tensor is None or mode not in [2, 3, 4] or pfaFlag == False:
        return input_data

    # 获取 Mask 的 Shape (预期是 2048x2048 或 [B, 1, 2048, 2048])
    shape = mask_tensor.shape
    # 获取最后两个维度
    S = shape[-2]
    KVS = shape[-1]

    # 构造标准的结构化 Mask (numpy)
    # PFA 定义: 1 代表遮蔽(Masked), 0 代表保留(Keep)
    new_mask = np.zeros((S, KVS), dtype=np.int8)

    if mode == 2 or mode == 3:
        # Mode 2 (LeftUpCausal) / Mode 3 (RightDownCausal)
        # 构造上三角 Mask (k=1 表示对角线往上一格开始全是 1)
        new_mask = np.triu(np.ones((S, KVS), dtype=np.int8), k=1)

    elif mode == 4:
        # Mode 4 (Band)
        # 需要读取 preTokens 和 nextTokens
        pre_t = input_data.kwargs.get("preTokens", 2147483647)
        next_t = input_data.kwargs.get("nextTokens", 2147483647)

        # 如果是 Tensor/List 取第一个值简化处理 (因为 Mode 4 Mask 是固定的 2048x2048)
        if hasattr(pre_t, "item"):
            pre_t = pre_t.item()
        if isinstance(pre_t, (list, tuple)):
            pre_t = pre_t[0]
        if hasattr(next_t, "item"):
            next_t = next_t.item()
        if isinstance(next_t, (list, tuple)):
            next_t = next_t[0]

        # 利用广播机制生成 Band
        rows = np.arange(S)[:, None]
        cols = np.arange(KVS)[None, :]
        # 遮蔽条件: j > i + next  OR  j < i - pre
        mask_condition = (cols > rows + next_t) | (cols < rows - pre_t)
        new_mask[mask_condition] = 1

    # --- 将构造好的 2D Mask 广播回原始 Shape ---

    # 1. 转回 Tensor
    # 保持和原 Mask 相同的 dtype (通常是 bool 或 int8)
    orig_dtype = mask_tensor.dtype
    new_mask_tensor = torch.from_numpy(new_mask).to(orig_dtype)

    # 2. 恢复维度 (Broadcast)
    # 如果原 Mask 是 [B, 1, 2048, 2048]，需要把 2D 扩展回去
    if len(shape) == 4:
        # [2048, 2048] -> [1, 1, 2048, 2048] -> [B, 1, 2048, 2048]
        new_mask_tensor = new_mask_tensor.unsqueeze(0).unsqueeze(0)
        new_mask_tensor = new_mask_tensor.expand(
            shape[0], shape[1], -1, -1
        ).contiguous()
    elif len(shape) == 3:
        new_mask_tensor = new_mask_tensor.unsqueeze(0)
        new_mask_tensor = new_mask_tensor.expand(shape[0], -1, -1).contiguous()
    # 3. 覆盖回 input_data
    input_data.kwargs["attenMaskOptional"] = new_mask_tensor
    return input_data


def get_split_fuse_flag(input_data):
    layout = input_data.kwargs["inputLayout"]
    return layout == "TND"


def load_kv_cache(input_data: InputDataset):
    key_shape = list(input_data.kwargs["key"][0].shape)
    if len(key_shape) == 3:
        H = key_shape[2]
    else:
        H = input_data.kwargs["numHeads"] * key_shape[3]

    blocktable_shape = list(input_data.kwargs["blockTableOptional"].shape)

    total_blocks = blocktable_shape[0] * blocktable_shape[1]
    cache_shape = [total_blocks, input_data.kwargs["blockSize"], H]

    block_table = input_data.kwargs.pop("block_table")
    k_cache = input_data.kwargs.pop("k_cache")
    v_cache = input_data.kwargs.pop("v_cache")

    k_cache_tensor = (
        torch.tensor(k_cache, dtype=torch.float32)
        .to(dtype=input_data.kwargs["key"][0].dtype)
        .reshape(cache_shape)
        .npu()
    )
    v_cache_tensor = (
        torch.tensor(v_cache, dtype=torch.float32)
        .to(dtype=input_data.kwargs["value"][0].dtype)
        .reshape(cache_shape)
        .npu()
    )

    input_data.kwargs["key"][0] = k_cache_tensor
    input_data.kwargs["value"][0] = v_cache_tensor

    block_table_tensor = torch.tensor(block_table, dtype=torch.int32)
    input_data.kwargs["blockTableOptional"] = (
        torch.tensor(block_table_tensor, dtype=torch.int32)
        .reshape(blocktable_shape)
        .npu()
    )

    if input_data.kwargs["keyRopeOptional"] is not None:
        k_rope_cache = input_data.kwargs.pop("k_rope_cache")
        k_rope_cache = (
            torch.tensor(k_rope_cache, dtype=torch.float32)
            .to(dtype=input_data.kwargs["keyRopeOptional"].dtype)
            .reshape(k_rope_cache.shape)
            .npu()
        )
        input_data.kwargs["keyRopeOptional"] = k_rope_cache

    return


def trans_input_to_params(
    input_data: InputDataset, is_benchmark_task, is_preprocess=False
):
    tensor_list = [None] * 29
    shape_input = [[1]] * 29
    range_input = [["null", "null"]] * 29
    dtype_input = ["fp16"] * 29
    format_input = ["ND"] * 29
    type_input = ["tensor"] * 29

    params = {
        "dtype_output": ["fp16"],
        "attr_1": "actualseqlengths",
        "actualseqlengths": [],
        "required_actualseqlengths": 1,
        "attr_2": "actualseqlengthskv",
        "actualseqlengthskv": [],
        "required_actualseqlengthskv": 1,
        "attr_3": "prefix_act_lens",
        "prefix_act_lens": [],
        "required_prefix_act_lens": 1,
        "attr_4": "numheads",
        "numheads": 8,
        "required_numheads": 1,
        "attr_5": "scalevalue",
        "scalevalue": 0.08838834764831843,
        "required_scalevalue": 1,
        "attr_6": "pretokens",
        "pretokens": 2147483647,
        "required_pretokens": 1,
        "attr_7": "nexttokens",
        "nexttokens": 2147483647,
        "required_nexttokens": 1,
        "attr_8": "inputlayout",
        "inputlayout": "BNSD",
        "required_inputlayout": 1,
        "attr_9": "numkeyvalueheads",
        "numkeyvalueheads": 8,
        "required_numkeyvalueheads": 1,
        "attr_10": "sparsemode",
        "sparsemode": 0,
        "required_sparsemode": 1,
        "attr_11": "innerprecise",
        "innerprecise": 0,
        "required_innerprecise": 1,
        "attr_12": "blocksize",
        "blocksize": 0,
        "required_blocksize": 1,
        "attr_13": "antiquant_mode",
        "antiquant_mode": 0,
        "required_antiquant_mode": 1,
        "attr_14": "softmax_lse_flag",
        "softmax_lse_flag": False,
        "required_softmax_lse_flag": 1,
        "attr_15": "k_antiquant_mode",
        "k_antiquant_mode": 0,
        "required_k_antiquant_mode": 1,
        "attr_16": "v_antiquant_mode",
        "v_antiquant_mode": 0,
        "required_v_antiquant_mode": 1,
        "attr_17": "query_quant_mode",
        "query_quant_mode": 0,
        "required_query_quant_mode": 1,
        "attr_18": "fused_flag",
        "fused_flag": "yes",
        "required_fused_flag": 1,
        "attr_19": "mrandomtype",
        "mrandomtype": "Normal",
        "required_mrandomtype": 1,
        "attr_20": "mrandom",
        "mrandom": 0,
        "required_mrandom": 1,
        "attr_21": "prandom",
        "prandom": 0,
        "required_prandom": 1,
        "attr_22": "enablegpu",
        "enablegpu": "True",
        "required_enablegpu": 1,
        "attr_23": "flaglist",
        "flaglist": [
            1,
            1,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            1,
            0,
            0,
            0,
            0,
            0,
        ],
        "required_flaglist": 1,
    }

    params["is_benchmark_task"] = is_benchmark_task
    params["is_preprocess"] = is_preprocess
    params["input_data"] = input_data

    if input_data.kwargs["query"].dtype == torch.int8:
        tensor_list[0] = query = (
            input_data.kwargs["query"].to(dtype=torch.int8).cpu().numpy()
        )
        tensor_list[1] = key = (
            input_data.kwargs["key"][0].to(dtype=torch.int8).cpu().numpy()
        )
        tensor_list[2] = value = (
            input_data.kwargs["value"][0].to(dtype=torch.int8).cpu().numpy()
        )
    elif input_data.kwargs["antiquantScaleOptional"] is not None:
        tensor_list[0] = query = (
            input_data.kwargs["query"].to(dtype=torch.float32).cpu().numpy()
        )
        tensor_list[1] = key = (
            input_data.kwargs["key"][0].to(dtype=torch.int8).cpu().numpy()
        )
        tensor_list[2] = value = (
            input_data.kwargs["value"][0].to(dtype=torch.int8).cpu().numpy()
        )
    else:
        tensor_list[0] = query = (
            input_data.kwargs["query"].to(dtype=torch.float16).cpu().numpy()
        )
        tensor_list[1] = key = (
            input_data.kwargs["key"][0].to(dtype=torch.float16).cpu().numpy()
        )
        tensor_list[2] = value = (
            input_data.kwargs["value"][0].to(dtype=torch.float16).cpu().numpy()
        )

    tensor_list[3] = pse = (
        input_data.kwargs["pseShiftOptional"].to(dtype=torch.float32).cpu().numpy()
        if input_data.kwargs["pseShiftOptional"] is not None
        else np.array([], dtype=np.float32)
    )
    tensor_list[4] = attenmask = (
        input_data.kwargs["attenMaskOptional"].to(dtype=torch.bool).cpu().numpy()
        if input_data.kwargs["attenMaskOptional"] is not None
        else np.array([], dtype=np.float32)
    )

    tensor_list[5] = dequantscale1 = (
        input_data.kwargs["deqScale1Optional"].to(dtype=torch.float32).cpu().numpy()
        if input_data.kwargs["deqScale1Optional"] is not None
        else np.array([], dtype=np.float32)
    )
    tensor_list[6] = quantscale1 = (
        input_data.kwargs["quantScale1Optional"].to(dtype=torch.float32).cpu().numpy()
        if input_data.kwargs["quantScale1Optional"] is not None
        else np.array([], dtype=np.float32)
    )
    tensor_list[7] = dequantscale2 = (
        input_data.kwargs["deqScale2Optional"].to(dtype=torch.float32).cpu().numpy()
        if input_data.kwargs["deqScale2Optional"] is not None
        else np.array([], dtype=np.float32)
    )

    tensor_list[8] = quantscale2 = (
        input_data.kwargs["quantScale2Optional"].to(dtype=torch.float32).cpu().numpy()
        if input_data.kwargs["quantScale2Optional"] is not None
        else np.array([], dtype=np.float32)
    )
    tensor_list[9] = quantoffset2 = (
        input_data.kwargs["quantOffset2Optional"].to(dtype=torch.float32).cpu().numpy()
        if input_data.kwargs["quantOffset2Optional"] is not None
        else np.array([], dtype=np.float32)
    )
    tensor_list[10] = antiquantscale = (
        input_data.kwargs["antiquantScaleOptional"]
        .to(dtype=torch.float32)
        .cpu()
        .numpy()
        if input_data.kwargs["antiquantScaleOptional"] is not None
        else np.array([], dtype=np.float32)
    )
    tensor_list[11] = antiquantoffset = (
        input_data.kwargs["antiquantOffsetOptional"]
        .to(dtype=torch.float32)
        .cpu()
        .numpy()
        if input_data.kwargs["antiquantOffsetOptional"] is not None
        else np.array([], dtype=np.float32)
    )
    tensor_list[12] = blocktable = (
        input_data.kwargs["blockTableOptional"].cpu().numpy()
        if input_data.kwargs["blockTableOptional"] is not None
        else np.array([], dtype=np.int32)
    )

    tensor_list[13] = q_padding_size = (
        input_data.kwargs["queryPaddingSizeOptional"]
        .to(dtype=torch.int32)
        .cpu()
        .numpy()
        if input_data.kwargs["queryPaddingSizeOptional"] is not None
        else np.array([], dtype=np.uint64)
    )
    tensor_list[14] = padding_size = (
        input_data.kwargs["kvPaddingSizeOptional"].to(dtype=torch.int32).cpu().numpy()
        if input_data.kwargs["kvPaddingSizeOptional"] is not None
        else np.array([], dtype=np.uint64)
    )
    tensor_list[15] = k_antiquantscale = (
        input_data.kwargs["keyAntiquantScaleOptional"].cpu()
        if input_data.kwargs["keyAntiquantScaleOptional"] is not None
        else np.array([], dtype=np.float32)
    )
    tensor_list[16] = k_antiquantoffset = (
        input_data.kwargs["keyAntiquantOffsetOptional"].cpu()
        if input_data.kwargs["keyAntiquantOffsetOptional"] is not None
        else np.array([], dtype=np.float32)
    )
    tensor_list[17] = v_antiquantscale = (
        input_data.kwargs["valueAntiquantScaleOptional"].cpu()
        if input_data.kwargs["valueAntiquantScaleOptional"] is not None
        else np.array([], dtype=np.float32)
    )

    tensor_list[18] = v_antiquantoffset = (
        input_data.kwargs["valueAntiquantOffsetOptional"].cpu()
        if input_data.kwargs["valueAntiquantOffsetOptional"] is not None
        else np.array([], dtype=np.float32)
    )
    tensor_list[19] = k_prefix = (
        input_data.kwargs["keySharedPrefixOptional"]
        .to(dtype=torch.float32)
        .cpu()
        .numpy()
        if input_data.kwargs["keySharedPrefixOptional"] is not None
        else np.array([], dtype=np.int8)
    )
    tensor_list[20] = v_prefix = (
        input_data.kwargs["valueSharedPrefixOptional"]
        .to(dtype=torch.float32)
        .cpu()
        .numpy()
        if input_data.kwargs["valueSharedPrefixOptional"] is not None
        else np.array([], dtype=np.int8)
    )

    tensor_list[23] = q_rope = (
        input_data.kwargs["queryRopeOptional"].cpu().numpy()
        if input_data.kwargs["queryRopeOptional"] is not None
        else np.array([], dtype=np.float32)
    )
    tensor_list[24] = k_rope = (
        input_data.kwargs["keyRopeOptional"].cpu().numpy()
        if input_data.kwargs["keyRopeOptional"] is not None
        else np.array([], dtype=np.float32)
    )

    tensor_list[26] = k_rope_antiquantScale = (
        input_data.kwargs["keyRopeAntiquantScaleOptional"].cpu().numpy()
        if input_data.kwargs["keyRopeAntiquantScaleOptional"] is not None
        else np.array([], dtype=np.float32)
    )
    tensor_list[27] = dequantScale_query = (
        input_data.kwargs["dequantScaleQueryOptional"].cpu().numpy()
        if input_data.kwargs["dequantScaleQueryOptional"] is not None
        else np.array([], dtype=np.float32)
    )
    tensor_list[28] = sinks = (
        input_data.kwargs["learnableSinkOptional"].cpu().numpy()
        if input_data.kwargs["learnableSinkOptional"] is not None
        else np.array([], dtype=np.float32)
    )

    shape_input[0] = list(input_data.kwargs["query"].shape)
    dtype_input[0] = dtype_map[input_data.kwargs["query"].dtype]
    params["dtype_output"][0] = dtype_map[input_data.kwargs["query"].dtype]

    shape_input[1] = list(input_data.kwargs["key"][0].shape)
    dtype_input[1] = dtype_map[input_data.kwargs["key"][0].dtype]
    shape_input[2] = list(input_data.kwargs["value"][0].shape)
    dtype_input[2] = dtype_map[input_data.kwargs["value"][0].dtype]

    if input_data.kwargs["pseShiftOptional"] is not None:
        params["flaglist"][3] = 1
        dtype_input[3] = dtype_map[input_data.kwargs["pseShiftOptional"].dtype]
        shape_input[3] = list(input_data.kwargs["pseShiftOptional"].shape)

    if input_data.kwargs["attenMaskOptional"] is not None:
        params["flaglist"][4] = 1
        range_input[4][0] = [0, 1]
        dtype_input[4] = dtype_map[input_data.kwargs["attenMaskOptional"].dtype]
        shape_input[4] = list(input_data.kwargs["attenMaskOptional"].shape)

    if input_data.kwargs["actualSeqLengthsOptional"][0] != arr_tuple_none:
        params["actualseqlengths"] = list(input_data.kwargs["actualSeqLengthsOptional"])
        params["flaglist"][5] = 1
    else:
        Q_S = (
            shape_input[0][1]
            if input_data.kwargs["inputLayout"] in ["BSND", "BSH"]
            else shape_input[0][2]
        )
        params["actualseqlengths"] = [Q_S]

    if input_data.kwargs["actualSeqLengthsKvOptional"][0] != arr_tuple_none:
        params["actualseqlengthskv"] = list(
            input_data.kwargs["actualSeqLengthsKvOptional"]
        )
        params["flaglist"][6] = 1
    else:
        KV_S = (
            shape_input[1][1]
            if input_data.kwargs["inputLayout"] in ["BSND", "BSH"]
            else shape_input[1][2]
        )
        params["actualseqlengthskv"] = [KV_S]

    if input_data.kwargs["deqScale1Optional"] is not None:
        params["flaglist"][7] = 1
        dtype_input[5] = dtype_map[input_data.kwargs["deqScale1Optional"].dtype]
        shape_input[5] = list(input_data.kwargs["deqScale1Optional"].shape)

    if input_data.kwargs["quantScale1Optional"] is not None:
        params["flaglist"][8] = 1
        dtype_input[6] = dtype_map[input_data.kwargs["quantScale1Optional"].dtype]
        shape_input[6] = list(input_data.kwargs["quantScale1Optional"].shape)

    if input_data.kwargs["deqScale2Optional"] is not None:
        params["flaglist"][9] = 1
        dtype_input[7] = dtype_map[input_data.kwargs["deqScale2Optional"].dtype]
        shape_input[7] = list(input_data.kwargs["deqScale2Optional"].shape)

    if input_data.kwargs["quantScale2Optional"] is not None:
        params["flaglist"][10] = 1
        dtype_input[8] = dtype_map[input_data.kwargs["quantScale2Optional"].dtype]
        shape_input[8] = list(input_data.kwargs["quantScale2Optional"].shape)

    if input_data.kwargs["quantOffset2Optional"] is not None:
        params["flaglist"][11] = 1
        dtype_input[9] = dtype_map[input_data.kwargs["quantOffset2Optional"].dtype]
        shape_input[9] = list(input_data.kwargs["quantOffset2Optional"].shape)

    if input_data.kwargs["antiquantScaleOptional"] is not None:
        params["flaglist"][12] = 1
        dtype_input[10] = dtype_map[input_data.kwargs["antiquantScaleOptional"].dtype]
        shape_input[10] = list(input_data.kwargs["antiquantScaleOptional"].shape)

    if input_data.kwargs["antiquantOffsetOptional"] is not None:
        params["flaglist"][13] = 1
        dtype_input[11] = dtype_map[input_data.kwargs["antiquantOffsetOptional"].dtype]
        shape_input[11] = list(input_data.kwargs["antiquantOffsetOptional"].shape)

    if input_data.kwargs["blockTableOptional"] is not None:
        params["flaglist"][14] = 1
        dtype_input[12] = dtype_map[input_data.kwargs["blockTableOptional"].dtype]
        shape_input[12] = list(input_data.kwargs["blockTableOptional"].shape)
        actual_seq = (
            key.shape[1]
            if input_data.kwargs["inputLayout"] in ["BSH", "BSND"]
            else key.shape[2]
        )
        headdim = (
            key.shape[3]
            if input_data.kwargs["inputLayout"] in ["BSND", "BNSD"]
            else key.shape[2] / input_data.kwargs["numKeyValueHeads"]
        )
        block_num = math.ceil(actual_seq / input_data.kwargs["blockSize"])
        cache_shape = [
            block_num,
            input_data.kwargs["blockSize"],
            int(input_data.kwargs["numKeyValueHeads"] * headdim),
        ]
        shape_input[21] = cache_shape
        shape_input[22] = cache_shape

    if input_data.kwargs["queryPaddingSizeOptional"] is not None:
        params["flaglist"][15] = 1
        range_input[15][0] = q_padding_size
        dtype_input[13] = dtype_map[input_data.kwargs["queryPaddingSizeOptional"].dtype]
        shape_input[13] = list(input_data.kwargs["queryPaddingSizeOptional"].shape)

    if input_data.kwargs["kvPaddingSizeOptional"] is not None:
        params["flaglist"][16] = 1
        range_input[16][0] = padding_size
        dtype_input[14] = dtype_map[input_data.kwargs["kvPaddingSizeOptional"].dtype]
        shape_input[14] = list(input_data.kwargs["kvPaddingSizeOptional"].shape)

    if input_data.kwargs["keyAntiquantScaleOptional"] is not None:
        params["flaglist"][17] = 1
        dtype_input[15] = dtype_map[
            input_data.kwargs["keyAntiquantScaleOptional"].dtype
        ]
        shape_input[15] = list(input_data.kwargs["keyAntiquantScaleOptional"].shape)

    if input_data.kwargs["keyAntiquantOffsetOptional"] is not None:
        params["flaglist"][18] = 1
        dtype_input[16] = dtype_map[
            input_data.kwargs["keyAntiquantOffsetOptional"].dtype
        ]
        shape_input[16] = list(input_data.kwargs["keyAntiquantOffsetOptional"].shape)

    if input_data.kwargs["valueAntiquantScaleOptional"] is not None:
        params["flaglist"][19] = 1
        dtype_input[17] = dtype_map[
            input_data.kwargs["valueAntiquantScaleOptional"].dtype
        ]
        shape_input[17] = list(input_data.kwargs["valueAntiquantScaleOptional"].shape)

    if input_data.kwargs["valueAntiquantOffsetOptional"] is not None:
        params["flaglist"][20] = 1
        dtype_input[18] = dtype_map[
            input_data.kwargs["valueAntiquantOffsetOptional"].dtype
        ]
        shape_input[18] = list(input_data.kwargs["valueAntiquantOffsetOptional"].shape)

    if input_data.kwargs["keySharedPrefixOptional"] is not None:
        params["flaglist"][21] = 1
        dtype_input[19] = dtype_map[input_data.kwargs["keySharedPrefixOptional"].dtype]
        shape_input[19] = list(input_data.kwargs["keySharedPrefixOptional"].shape)

    if input_data.kwargs["valueSharedPrefixOptional"] is not None:
        params["flaglist"][22] = 1
        dtype_input[20] = dtype_map[
            input_data.kwargs["valueSharedPrefixOptional"].dtype
        ]
        shape_input[20] = list(input_data.kwargs["valueSharedPrefixOptional"].shape)

    if input_data.kwargs["queryRopeOptional"] is not None:
        params["flaglist"][26] = 1
        dtype_input[23] = dtype_map[input_data.kwargs["queryRopeOptional"].dtype]
        shape_input[23] = list(input_data.kwargs["queryRopeOptional"].shape)

    if input_data.kwargs["keyRopeOptional"] is not None:
        params["flaglist"][27] = 1
        dtype_input[24] = dtype_map[input_data.kwargs["keyRopeOptional"].dtype]
        shape_input[24] = list(input_data.kwargs["keyRopeOptional"].shape)
        actual_seq = (
            key.shape[1]
            if input_data.kwargs["inputLayout"] in ["BSH", "BSND"]
            else key.shape[2]
        )
        block_num = math.ceil(actual_seq / input_data.kwargs["blockSize"])
        cache_shape = [block_num, input_data.kwargs["blockSize"], 64]
        dtype_input[25] = dtype_map[input_data.kwargs["keyRopeOptional"].dtype]
        shape_input[25] = cache_shape
        tensor_list[25] = torch.zeros(
            cache_shape, dtype=input_data.kwargs["keyRopeOptional"].dtype
        )

    params["prefix_act_lens"] = list(input_data.kwargs["actualSharedPrefixLenOptional"])
    if input_data.kwargs["actualSharedPrefixLenOptional"][0] != arr_tuple_none:
        params["flaglist"][23] = 1
        params["prefix_act_lens"] = list(
            input_data.kwargs["actualSharedPrefixLenOptional"]
        )

    params["numheads"] = input_data.kwargs["numHeads"]
    params["scalevalue"] = input_data.kwargs["scaleValue"]
    params["pretokens"] = input_data.kwargs["preTokens"]
    params["nexttokens"] = input_data.kwargs["nextTokens"]
    params["inputlayout"] = input_data.kwargs["inputLayout"]
    params["numkeyvalueheads"] = input_data.kwargs["numKeyValueHeads"]
    params["sparsemode"] = input_data.kwargs["sparseMode"]
    # params["innerprecise"] = input_data.kwargs["innerPrecise"]
    params["innerprecise"] = 0
    params["blocksize"] = input_data.kwargs["blockSize"]
    params["antiquant_mode"] = input_data.kwargs["antiquantMode"]
    params["softmax_lse_flag"] = input_data.kwargs["softmaxLseFlag"]
    params["k_antiquant_mode"] = input_data.kwargs["keyAntiquantMode"]
    params["v_antiquant_mode"] = input_data.kwargs["valueAntiquantMode"]
    params["query_quant_mode"] = input_data.kwargs["queryQuantMode"]
    params["softmax_lse_flag"] = input_data.kwargs["softmaxLseFlag"]
    params["shape_input"] = shape_input
    params["dtype_input"] = dtype_input
    params["range_input"] = range_input
    params["format_input"] = format_input
    params["type_input"] = type_input
    params["action_type"] = "bm"

    return tensor_list, params


def init_kv_cache(input_data: InputDataset, is_benchmark_task, is_preprocess=True):
    tensor_list, params = trans_input_to_params(
        input_data, is_benchmark_task, is_preprocess
    )
    if is_preprocess:
        FiaOpForward(tensor_list, params).forward()
        return


def fill_useless_out(input_data: InputDataset):
    input_data_dtype = input_data.kwargs["query"].cpu().dtype
    if input_data_dtype == torch.float16:
        query = input_data.kwargs["query"].cpu().numpy()
    elif input_data_dtype == torch.bfloat16:
        query = (
            input_data.kwargs["query"].cpu().to(torch.float32).numpy().astype(bfloat16)
        )
    else:
        query = input_data.kwargs["query"].cpu().numpy()
    if input_data.kwargs["blockTableOptional"] != None:
        pagedAttentionFlag = True
    else:
        pagedAttentionFlag = False
    ## gen actual seqlen
    inputLayout = input_data.kwargs["inputLayout"]
    batch = len(input_data.kwargs["actualSeqLengthsOptional"])
    actualseqlengths = [0] * batch
    actualseqlengthsKv = [0] * batch
    for i in range(batch):
        actualseqlengths[i] = input_data.kwargs["actualSeqLengthsOptional"][i]
        actualseqlengthsKv[i] = input_data.kwargs["actualSeqLengthsKvOptional"][i]
    qSeqlenList, kvSeqlenList = gen_actual_seqlen_list_golden(
        actualseqlengths, actualseqlengthsKv, inputLayout, pagedAttentionFlag
    )
    maxQSeqlen = max(qSeqlenList)
    totalQTokens = sum(qSeqlenList)

    if totalQTokens < query.shape[0]:
        attentionOut = input_data.kwargs["attentionOut"]
        attentionOut[totalQTokens:, :, :] = 0


def aclnn_op_func_fia_cpu(
    input_data: InputDataset, is_benchmark_task, is_preprocess=False
):
    tensor_list, params = trans_input_to_params(
        input_data, is_benchmark_task, is_preprocess
    )
    if input_data.kwargs["softmaxLseFlag"]:
        output, output_lse = FiaOpForward(tensor_list, params).forward()
        output_lse = output_lse.to(dtype=torch.float32)
    else:
        output = FiaOpForward(tensor_list, params).forward()

    if input_data.kwargs["quantScale2Optional"] is not None:
        output = output.to(dtype=torch.int8)
    else:
        output = output.to(dtype=input_data.kwargs["query"].dtype)

    if input_data.kwargs["softmaxLseFlag"]:
        return output, output_lse
    else:
        return output


@register("executor_fused_infer_attention_score_v4")
class fusedInferAttentionScoreApi(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(fusedInferAttentionScoreApi, self).__init__(task_result)

    def init_by_input_data(self, input_data: InputDataset):
        # 将cache生成逻辑数据直接存放input_data里
        input_data = overwrite_structured_mask(input_data)
        self.split_fuse_flag = get_split_fuse_flag(input_data)
        if not self.split_fuse_flag and input_data.kwargs["blockTableOptional"] != None:
            init_kv_cache(input_data, is_benchmark_task=True, is_preprocess=True)
        if self.split_fuse_flag:
            fill_useless_out(input_data)

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        if self.split_fuse_flag:
            output, output_lse = aclnn_op_func_fia_split_fuse_golden(
                input_data, self.task_result.is_benchmark_task
            )
            if input_data.kwargs["softmaxLseFlag"] == True:
                return output, output_lse
            else:
                return output
        else:
            is_benchmark_task = True  # 单标杆
            # is_benchmark_task = self.task_result.is_benchmark_task # 双标杆
            if input_data.kwargs["softmaxLseFlag"]:
                output, output_lse = aclnn_op_func_fia_cpu(
                    input_data, is_benchmark_task, is_preprocess=False
                )
                return output, output_lse
            else:
                output = aclnn_op_func_fia_cpu(
                    input_data, is_benchmark_task, is_preprocess=False
                )
                return output


@register("executor_aclnn_fused_infer_attention_score_v4")
class aclnnFusedInferAttentionScoreApi(AclnnBaseApi):
    def __init__(self, task_result: TaskResult, backend):
        super(aclnnFusedInferAttentionScoreApi, self).__init__(task_result, backend)

    def gen_compressed_triU_mask(self, dim_num, mask_dtype):
        mask_shape_four_dims = (1, 1, 2048, 2048)
        mask_four_dims = torch.zeros(mask_shape_four_dims, dtype=mask_dtype)
        mask_triU = torch.triu(torch.ones(2048, 2048), diagonal=1)
        mask_four_dims[:] = mask_triU
        if dim_num == 2:
            return mask_four_dims[0][0]
        elif dim_num == 3:
            return mask_four_dims[0]
        elif dim_num == 4:
            return mask_four_dims
        else:
            print("invalid dim num, will provide a four dim mask anyway")
            return mask_four_dims
        return mask_four_dims[0][0]

    def init_by_input_data(self, input_data: InputDataset):
        torch.npu.synchronize()
        self.split_fuse_flag = get_split_fuse_flag(input_data)
        if self.split_fuse_flag:
            np.random.seed(self.task_result.case_config.id)
            if input_data.kwargs["attenMaskOptional"] != None:
                mask_dim_num = len(input_data.kwargs["attenMaskOptional"].shape)
                mask_dtype = input_data.kwargs["attenMaskOptional"].dtype
                input_data.kwargs["attenMaskOptional"] = self.gen_compressed_triU_mask(
                    mask_dim_num, mask_dtype
                ).npu()

        input_args = []  # 算子的入参列表
        if (
            not self.split_fuse_flag
            and input_data.kwargs["blockTableOptional"] is not None
        ):
            load_kv_cache(input_data)

        # 处理actual输入None
        input_args, output_packages = super().init_by_input_data(input_data)

        AclIntArrayPtr = ctypes.POINTER(AclIntArray)
        if (
            input_data.kwargs["actualSeqLengthsOptional"] is None
            or input_data.kwargs["actualSeqLengthsOptional"][0] == arr_tuple_none
            or input_data.kwargs["actualSeqLengthsOptional"][0] < 0
        ):
            input_args[5] = ctypes.cast(None, AclIntArrayPtr)

        if (
            input_data.kwargs["actualSeqLengthsKvOptional"] is None
            or input_data.kwargs["actualSeqLengthsKvOptional"][0] == arr_tuple_none
            or input_data.kwargs["actualSeqLengthsKvOptional"][0] < 0
        ):
            input_args[6] = ctypes.cast(None, AclIntArrayPtr)

        if (
            input_data.kwargs["actualSharedPrefixLenOptional"] is None
            or input_data.kwargs["actualSharedPrefixLenOptional"][0] == arr_tuple_none
            or input_data.kwargs["actualSharedPrefixLenOptional"][0] < 0
        ):
            input_args[23] = ctypes.cast(None, AclIntArrayPtr)

        output_packages = []  # 算子的出参数据包列表
        input_args.pop()
        if input_data.kwargs["softmaxLseFlag"] == True:
            input_args.pop()
            output_packages.append(input_args[-2])
            output_packages.append(input_args[-1])
        else:
            output_packages.append(input_args[-2])

        # 将所有type是tensor values是None的输入 改为AclTensor类型的空指针
        for i, (name, kwarg) in enumerate(input_data.kwargs.items()):
            if (
                kwarg is None
                and self.task_result.case_config.inputs[i].type == "tensor"
            ):
                from atk.tasks.backends.lib_interface.acl_wrapper import TensorPtr

                input_args[i] = TensorPtr()
            elif (
                name == "actualSeqLengthsOptional"
                and self.task_result.case_config.inputs[i][0].dtype == "bool"
            ):
                input_args[i] = pointer(AclIntArray)()
        return input_args, output_packages

    def __call__(self):
        self.backend.aclnn_x_get_workspace_size()
        self.backend.aclnn_x()

    def after_call(self, output_packages):
        output = []
        for output_pack in output_packages:
            temp_output_pack = self.acl_tensor_to_torch(output_pack)
            if self.split_fuse_flag:
                temp_output_pack = temp_output_pack.to(dtype=torch.float32)
            output.append(temp_output_pack)
        return output
