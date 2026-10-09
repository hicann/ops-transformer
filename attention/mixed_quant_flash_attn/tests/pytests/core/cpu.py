#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

import os
import sys
import torch
import torch_npu
import math
import numpy as np
import ctypes

_ACL_FLOAT = 0
_ACL_FLOAT16 = 1
_ACL_BF16 = 27
_ACL_FORMAT_ND = 2
_ACL_MEM_MALLOC_HUGE_FIRST = 0
_ACL_MEMCPY_HOST_TO_DEVICE = 1
_ACL_MEMCPY_DEVICE_TO_HOST = 2

_c_int64_p = ctypes.POINTER(ctypes.c_int64)
_c_void_p_p = ctypes.POINTER(ctypes.c_void_p)

_fmm_libascendcl = None
_fmm_libnnopbase = None
_fmm_libopapi = None
_fmm_acl_stream = None
_fmm_acl_inited = False

_fmm_dt_map = {
    torch.float16: _ACL_FLOAT16,
    torch.bfloat16: _ACL_BF16,
    torch.float32: _ACL_FLOAT,
}

_FP4_E2M1_LUT = torch.tensor(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=torch.bfloat16,
)

_FP4_E1M2_LUT = torch.tensor(
    [
        0.0,
        0.25,
        0.5,
        0.75,
        1.0,
        1.25,
        1.5,
        1.75,
        -0.0,
        -0.25,
        -0.5,
        -0.75,
        -1.0,
        -1.25,
        -1.5,
        -1.75,
    ],
    dtype=torch.bfloat16,
)


def get_axis_from_tensor(
    q, k, cu_seqlens_q, seqused_q, seqused_kv, max_seqlen_kv, layout_q, layout_kv
):
    b = -1
    n1 = -1
    n2 = -1
    s1 = -1
    s2 = -1
    d = -1
    block_num = -1
    block_size = -1
    if layout_q == "BNSD":
        b = q.shape[0]
        n1 = q.shape[1]
        s1 = q.shape[2]
        d = q.shape[3]
    elif layout_q == "BSND":
        b = q.shape[0]
        s1 = q.shape[1]
        n1 = q.shape[2]
        d = q.shape[3]
    elif layout_q == "TND":
        if cu_seqlens_q is None:
            raise ValueError("cu_seqlens_q cannot be None when layout_q is TND")
        b = len(cu_seqlens_q) - 1
        s1 = max(
            max(seqused_q),
            max(
                [
                    cu_seqlens_q[i + 1] - cu_seqlens_q[i]
                    for i in range(len(cu_seqlens_q) - 1)
                ]
            ),
        )
        n1 = q.shape[1]
        d = q.shape[2]
    else:
        raise ValueError("layout_q must be BNSD, BSND or TND")
    if "PA" in layout_kv:
        if seqused_kv is None and max_seqlen_kv == -1:
            raise ValueError(
                f"seqused_kv or max_seqlen_kv must be valid when layout_kv is {layout_kv}"
            )
        s2 = max(seqused_kv) if seqused_kv is not None else max_seqlen_kv
        if max_seqlen_kv > -1:
            s2 = max(max_seqlen_kv, s2)
    if layout_kv == "PA_NZ":
        block_num = k.shape[0]
        n2 = k.shape[1]
        block_size = k.shape[3]
    elif layout_kv == "PA_BNBD":
        block_num = k.shape[0]
        n2 = k.shape[1]
        block_size = k.shape[2]
        print(f"PA_BNBD shape{k.shape} block_size:{block_size}")
    elif layout_kv == "PA_BBND":
        block_num = k.shape[0]
        block_size = k.shape[1]
        n2 = k.shape[2]
    else:
        raise ValueError("layout_kv must be PA_BNBD, PA_BBND or PA_NZ")
    return b, n1, n2, s1, s2, d, block_num, block_size


def unsplit_float32_pa(pa_tensor):
    """
    Inverse of golden `split_float32_to_uint16` PA packing for hifp4 descale.

    Input:  PA-packed float32 tensor with last dim 16, where each group of 16
            PA float32 == view-as-float32 of [low16(orig 16 f32), high16(orig 16 f32)].
    Output: logical float32 tensor with last dim 16 (the original 16 e6m2e1e16 values).
    """
    shape = pa_tensor.shape
    if pa_tensor.numel() == 0:
        return torch.zeros(shape, dtype=torch.float32)
    pa_u16 = pa_tensor.contiguous().view(torch.uint16).reshape(shape[:-1] + (32,))
    low = pa_u16[..., :16].to(torch.int32)
    high = pa_u16[..., 16:32].to(torch.int32)
    combined = (high << 16) | (low & 0xFFFF)
    return combined.view(torch.float32)


def trans_pa_to_no_pa(
    batch,
    n2,
    s2,
    d,
    layout_kv,
    quant_data,
    descale,
    block_table,
    block_num,
    block_size,
    quant_compute_mode,
    gs_flag,
    is_key,
):
    if quant_compute_mode == 1:
        group_size = 32
    elif quant_compute_mode == 2:
        group_size = 64
    else:
        raise NotImplementedError(
            f"unsupported quant_compute_mode: {quant_compute_mode}"
        )
    if gs_flag:
        data = torch.zeros((batch, n2, s2, d), dtype=quant_data.dtype)
        if quant_compute_mode == 1:
            if is_key:
                scale = torch.zeros(
                    (1, batch, n2, s2, math.ceil(d / group_size / 2) * 2),
                    dtype=descale.dtype,
                )
            else:
                scale = torch.zeros(
                    (1, batch, n2, d, math.ceil(s2 / group_size / 2) * 2),
                    dtype=descale.dtype,
                )
        elif quant_compute_mode == 2:
            if is_key:
                scale = torch.zeros(
                    (1, batch, n2, s2, math.ceil(d / group_size)), dtype=descale.dtype
                )
            else:
                scale = torch.zeros(
                    (1, batch, n2, d, math.ceil(s2 / group_size)), dtype=descale.dtype
                )
    else:
        data = torch.zeros((batch, s2, n2, d), dtype=quant_data.dtype)
        if quant_compute_mode == 1:
            if is_key:
                scale = torch.zeros(
                    (1, batch, s2, n2, math.ceil(d / group_size / 2) * 2),
                    dtype=descale.dtype,
                )
            else:
                scale = torch.zeros(
                    (1, batch, n2, d, math.ceil(s2 / group_size / 2) * 2),
                    dtype=descale.dtype,
                )
        else:
            if is_key:
                scale = torch.zeros(
                    (1, batch, s2, n2, math.ceil(d / group_size)), dtype=descale.dtype
                )
            else:
                scale = torch.zeros(
                    (1, batch, n2, d, math.ceil(s2 / group_size)), dtype=descale.dtype
                )
    for b_idx in range(batch):
        block_ids = block_table[b_idx]
        for block_idx, block_id in enumerate(block_ids):
            if block_id > -1:
                block_data = quant_data[block_id]
                block_scale = descale[block_id]
                if layout_kv == "PA_NZ":
                    if gs_flag:
                        block_data_ = block_data.permute(0, 2, 1, 3).reshape(
                            n2, block_size, d
                        )
                    else:
                        block_data_ = block_data.permute(2, 0, 1, 3).reshape(
                            block_size, n2, d
                        )
                    if quant_compute_mode == 1:
                        if is_key:
                            if gs_flag:
                                block_scale_ = block_scale.permute(
                                    0, 1, 3, 2, 4
                                ).reshape(
                                    n2, block_size, math.ceil(d / group_size / 2) * 2
                                )
                            else:
                                block_scale_ = block_scale.permute(
                                    1, 3, 0, 2, 4
                                ).reshape(
                                    block_size, n2, math.ceil(d / group_size / 2) * 2
                                )
                        else:
                            block_scale_ = block_scale.permute(0, 1, 3, 2, 4).reshape(
                                n2, d, math.ceil(block_size / group_size / 2) * 2
                            )
                    elif quant_compute_mode == 2:
                        block_scale = unsplit_float32_pa(block_scale)
                        if is_key:
                            if gs_flag:
                                block_scale_ = block_scale.permute(0, 1, 3, 2).reshape(
                                    n2, block_size, math.ceil(d / group_size)
                                )
                            else:
                                block_scale_ = block_scale.permute(1, 3, 0, 2).reshape(
                                    block_size, n2, math.ceil(d / group_size)
                                )
                        else:
                            block_scale_ = block_scale.permute(0, 1, 3, 2).reshape(
                                n2, d, math.ceil(block_size / group_size)
                            )
                    else:
                        raise NotImplementedError(
                            f"unsupported quant_compute_mode: {quant_compute_mode}"
                        )
                elif layout_kv == "PA_BBND":
                    # PA_BBND: (blockNum, blockSize, N, D) for KV
                    # K-scale: (blockNum, blockSize, N, D/groupSize)
                    # V-scale: (blockNum, N, D, blockSize/groupSize)
                    if gs_flag:
                        block_data_ = block_data.permute(1, 0, 2)
                    else:
                        block_data_ = block_data.reshape(block_size, n2, d)
                    if is_key:
                        if gs_flag:
                            block_scale_ = block_scale.permute(1, 0, 2)
                        else:
                            block_scale_ = block_scale.reshape(
                                block_size, n2, math.ceil(d / group_size)
                            )
                    else:
                        block_scale_ = block_scale.reshape(
                            n2, d, math.ceil(block_size / group_size)
                        )

                elif layout_kv == "PA_BNBD":
                    # PA_BNBD: (blockNum, N, blockSize, D) for KV
                    # K-scale: (blockNum, N, blockSize, D/groupSize)
                    # V-scale: (blockNum, N, D, blockSize/groupSize)
                    if gs_flag:
                        block_data_ = block_data.reshape(n2, block_size, d)
                    else:
                        block_data_ = block_data.reshape(n2, block_size, d).permute(
                            1, 0, 2
                        )
                    if is_key:
                        if gs_flag:
                            block_scale_ = block_scale.reshape(
                                n2, block_size, math.ceil(d / group_size)
                            )
                        else:
                            block_scale_ = block_scale.reshape(
                                n2, block_size, math.ceil(d / group_size)
                            ).permute(1, 0, 2)
                    else:
                        if gs_flag:
                            block_scale_ = block_scale.reshape(
                                n2, d, math.ceil(block_size / group_size)
                            )
                        else:
                            block_scale_ = block_scale.reshape(
                                n2, d, math.ceil(block_size / group_size)
                            )

                else:
                    raise NotImplementedError(f"unsupported layout_kv: {layout_kv}")
                if gs_flag:
                    if (block_idx + 1) * block_size > s2:
                        data[b_idx, :, block_idx * block_size : s2] = block_data_[
                            :, : (s2 - block_idx * block_size)
                        ]
                    else:
                        data[
                            b_idx,
                            :,
                            block_idx * block_size : (block_idx + 1) * block_size,
                        ] = block_data_
                else:
                    if (block_idx + 1) * block_size > s2:
                        data[b_idx, block_idx * block_size : s2] = block_data_[
                            : (s2 - block_idx * block_size)
                        ]
                    else:
                        data[
                            b_idx, block_idx * block_size : (block_idx + 1) * block_size
                        ] = block_data_
                if is_key:
                    if gs_flag:
                        if (block_idx + 1) * block_size > s2:
                            scale[0, b_idx, :, block_idx * block_size : s2] = (
                                block_scale_[:, : (s2 - block_idx * block_size)]
                            )
                        else:
                            scale[
                                0,
                                b_idx,
                                :,
                                block_idx * block_size : (block_idx + 1) * block_size,
                            ] = block_scale_
                    else:
                        if (block_idx + 1) * block_size > s2:
                            scale[0, b_idx, block_idx * block_size : s2] = block_scale_[
                                : (s2 - block_idx * block_size)
                            ]
                        else:
                            scale[
                                0,
                                b_idx,
                                block_idx * block_size : (block_idx + 1) * block_size,
                            ] = block_scale_
                else:
                    if quant_compute_mode == 1:
                        if (block_idx + 1) * block_size > s2:
                            scale[
                                0,
                                b_idx,
                                :,
                                :,
                                math.ceil(block_idx * block_size / group_size / 2)
                                * 2 : math.ceil(s2 / group_size / 2) * 2,
                            ] = block_scale_[
                                :,
                                :,
                                : math.ceil(
                                    (s2 - block_idx * block_size) / group_size / 2
                                )
                                * 2,
                            ]
                        else:
                            scale[
                                0,
                                b_idx,
                                :,
                                :,
                                math.ceil(block_idx * block_size / group_size / 2)
                                * 2 : math.ceil(
                                    (block_idx + 1) * block_size / group_size / 2
                                )
                                * 2,
                            ] = block_scale_
                    else:
                        if (block_idx + 1) * block_size > s2:
                            scale[
                                0,
                                b_idx,
                                :,
                                :,
                                math.ceil(
                                    block_idx * block_size / group_size
                                ) : math.ceil(s2 / group_size),
                            ] = block_scale_[
                                :,
                                :,
                                : math.ceil((s2 - block_idx * block_size) / group_size),
                            ]
                        else:
                            scale[
                                0,
                                b_idx,
                                :,
                                :,
                                math.ceil(
                                    block_idx * block_size / group_size
                                ) : math.ceil(
                                    (block_idx + 1) * block_size / group_size
                                ),
                            ] = block_scale_
            else:
                break
    return data, scale


def softmax_flash(x, block_sum, block_max, block_exp, is_first):
    # this func is only used by quant_dequant
    x = x.to(torch.float32)
    x_max = x.max(axis=-1, keepdims=True).values
    pre_max = block_max
    block_max = torch.max(
        torch.hstack([block_max, x_max]), axis=1, keepdims=True
    ).values
    if is_first:
        block_exp = block_exp
    else:
        block_exp = torch.exp(pre_max - block_max)
    x_sub = x - block_max
    mask = torch.isposinf(x) & torch.isposinf(block_max)
    x_sub = torch.where(mask, 0, x_sub)
    y = torch.exp(x_sub)
    x_sum = y.sum(axis=-1, keepdims=True)
    if is_first:
        block_sum = x_sum
    else:
        block_sum *= block_exp
        block_sum += x_sum
    return y, block_sum, block_max, block_exp


def unpack_int4(src):
    unpack_shape = list(src.shape)
    unpack_shape[-1] = unpack_shape[-1] << 1
    unpack_shape = tuple(unpack_shape)
    shift = torch.tensor([0, 4], dtype=torch.uint8)
    return (
        torch.bitwise_and(src.to(torch.uint8).reshape([-1, 1]) >> shift, 0b00001111)
        .reshape([-1])
        .reshape(unpack_shape)
    )


def cvt_fp4_e2m1_to_bfloat16(x):
    Fp4e2m1ToBf16 = {
        "0": 0.0,
        "1": 0.5,
        "2": 1.0,
        "3": 1.5,
        "4": 2.0,
        "5": 3.0,
        "6": 4.0,
        "7": 6.0,
        "8": -0.0,
        "9": -0.5,
        "10": -1.0,
        "11": -1.5,
        "12": -2.0,
        "13": -3.0,
        "14": -4.0,
        "15": -6.0,
    }
    x = int(x)
    first_fp4val = x & 0x0F
    first_fp4str = str(first_fp4val)
    return Fp4e2m1ToBf16[first_fp4str]


def new_trans_np_fp4_e2m1_tensor_to_bfloat16(in_tensor):
    shape = in_tensor.shape
    idx = in_tensor.reshape(-1).to(torch.long) & 0x0F
    return _FP4_E2M1_LUT[idx].reshape(shape)


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


def new_trans_np_fp4_e1m2_tensor_to_bfloat16(in_tensor):
    shape = in_tensor.shape
    idx = in_tensor.reshape(-1).to(torch.long) & 0x0F
    return _FP4_E1M2_LUT[idx].reshape(shape)


def trans_tensor_e6m2e1e16_to_bf16(tensor):
    # 每4个共用一个
    new_tensor = torch.zeros(
        [
            tensor.shape[0],
            tensor.shape[1],
            tensor.shape[2],
            tensor.shape[3],
            tensor.shape[4] * 16,
        ],
        dtype=torch.bfloat16,
    )
    from ml_dtypes import bfloat16

    # 使用nditer遍历tensor中的每个元素
    tensor_u32 = tensor.view(torch.uint32).numpy()
    with np.nditer(tensor_u32, flags=["multi_index"], op_flags=["readwrite"]) as it:
        for x in it:
            E6M2_1 = x & 0xFF

            e6m2_e = E6M2_1 >> 2
            e6m2_m0 = E6M2_1 & 0x1
            e6m2_m1 = (E6M2_1 & 0x3) >> 1

            e6m2_val = 2.0 ** (e6m2_e - 48) * (1 + e6m2_m1 * 0.5 + e6m2_m0 * 0.25)

            E1_8 = np.zeros(8).astype(np.int32)
            for k in range(8):
                E1_8[k] = (x >> (8 + k)) & 0x1  # 8~15bit

            E1_8x2 = np.zeros(16).astype(np.int32)
            for k in range(8):
                E1_8x2[k * 2 : k * 2 + 2] = E1_8[k]

            E1_16 = np.zeros(16).astype(np.int32)
            for k in range(16):
                E1_16[k] = (x >> (16 + k)) & 0x1  # 16~31bit
            E16G = E1_16 + E1_8x2  # Fused 16 exp vals

            S16G = np.zeros(16).astype(bfloat16)
            for k in range(16):
                S16G[k] = (e6m2_val * 2 ** (E16G[k])).astype(bfloat16)

            new_tensor[it.multi_index[0]][it.multi_index[1]][it.multi_index[2]][
                it.multi_index[3]
            ][it.multi_index[4] * 16 : it.multi_index[4] * 16 + 16] = torch.from_numpy(
                S16G.view(np.uint16)
            ).view(torch.bfloat16)

    return new_tensor


def antiquant_data(
    data, output_dtype, quant_compute_mode, antiquant_scale, gs_flag=True, is_key=True
):
    if quant_compute_mode == 1:
        out_antiquant_scale = antiquant_scale.to(torch.bfloat16)
        out_data = new_trans_np_fp4_e2m1_tensor_to_bfloat16(data)
        if output_dtype == torch.float16:
            out_antiquant_scale = out_antiquant_scale.to(torch.float16)
            out_data = out_data.to(torch.float16)
    elif quant_compute_mode == 2:
        out_antiquant_scale = trans_tensor_e6m2e1e16_to_bf16(antiquant_scale)
        out_data = new_trans_np_fp4_e1m2_tensor_to_bfloat16(data)
        if output_dtype == torch.float16:
            out_antiquant_scale = out_antiquant_scale.to(torch.float16)
            out_data = out_data.to(torch.float16)
    else:
        raise NotImplementedError(
            f"unsupported quant_compute_mode: {quant_compute_mode}"
        )
    if quant_compute_mode == 3:
        return out_data * out_antiquant_scale
    elif quant_compute_mode in [1, 2]:
        g_size = 32
        if quant_compute_mode == 2:
            g_size = 4
        if is_key:
            for i in range(out_antiquant_scale.shape[-1]):
                if gs_flag:
                    out_data[:, :, :, i * g_size : (i + 1) * g_size] = (
                        out_data[:, :, :, i * g_size : (i + 1) * g_size]
                        * out_antiquant_scale[0, :, :, :, i : i + 1]
                    )
                else:
                    out_data[:, :, :, i * g_size : (i + 1) * g_size] = (
                        out_data[:, :, :, i * g_size : (i + 1) * g_size]
                        * out_antiquant_scale[0, :, :, :, i : i + 1]
                    )
            return out_data
        else:
            for i in range(out_antiquant_scale.shape[-1]):
                if gs_flag:
                    out_data[:, :, i * g_size : (i + 1) * g_size, :] = out_data[
                        :, :, i * g_size : (i + 1) * g_size, :
                    ] * out_antiquant_scale[0, :, :, :, i : i + 1].permute(0, 1, 3, 2)
                else:
                    out_data[:, i * g_size : (i + 1) * g_size, :, :] = out_data[
                        :, i * g_size : (i + 1) * g_size, :, :
                    ] * out_antiquant_scale[0, :, :, :, i : i + 1].permute(0, 3, 1, 2)
            return out_data
    else:
        raise NotImplementedError(
            f"unsupported quant_compute_mode: {quant_compute_mode}"
        )


def _fmm_init():
    global \
        _fmm_libascendcl, \
        _fmm_libnnopbase, \
        _fmm_libopapi, \
        _fmm_acl_stream, \
        _fmm_acl_inited
    if _fmm_acl_inited:
        return
    cann_home = os.environ.get(
        "ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest"
    )
    lib64 = f"{cann_home}/lib64"
    _fmm_libascendcl = ctypes.CDLL(f"{lib64}/libascendcl.so")
    _fmm_libnnopbase = ctypes.CDLL(f"{lib64}/libnnopbase.so")
    _fmm_libopapi = ctypes.CDLL(f"{lib64}/libopapi.so")
    _fmm_libascendcl.aclInit.restype = ctypes.c_int32
    _fmm_libascendcl.aclInit.argtypes = [ctypes.c_char_p]
    _fmm_libascendcl.aclrtSetDevice.restype = ctypes.c_int32
    _fmm_libascendcl.aclrtSetDevice.argtypes = [ctypes.c_int32]
    _fmm_libascendcl.aclrtCreateStream.restype = ctypes.c_int32
    _fmm_libascendcl.aclrtCreateStream.argtypes = [_c_void_p_p]
    _fmm_libascendcl.aclrtSynchronizeStream.restype = ctypes.c_int32
    _fmm_libascendcl.aclrtSynchronizeStream.argtypes = [ctypes.c_void_p]
    _fmm_libascendcl.aclrtMalloc.restype = ctypes.c_int32
    _fmm_libascendcl.aclrtMalloc.argtypes = [
        _c_void_p_p,
        ctypes.c_size_t,
        ctypes.c_int32,
    ]
    _fmm_libascendcl.aclrtFree.restype = ctypes.c_int32
    _fmm_libascendcl.aclrtFree.argtypes = [ctypes.c_void_p]
    _fmm_libascendcl.aclrtMemcpy.restype = ctypes.c_int32
    _fmm_libascendcl.aclrtMemcpy.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int32,
    ]

    _fmm_libnnopbase.aclCreateTensor.restype = ctypes.c_void_p
    _fmm_libnnopbase.aclCreateTensor.argtypes = [
        _c_int64_p,
        ctypes.c_uint64,
        ctypes.c_int32,
        _c_int64_p,
        ctypes.c_int64,
        ctypes.c_int32,
        _c_int64_p,
        ctypes.c_uint64,
        ctypes.c_void_p,
    ]
    _fmm_libnnopbase.aclDestroyTensor.restype = ctypes.c_int32
    _fmm_libnnopbase.aclDestroyTensor.argtypes = [ctypes.c_void_p]
    _fmm_libopapi.aclnnBatchMatMulGetWorkspaceSize.restype = ctypes.c_int32
    _fmm_libopapi.aclnnBatchMatMulGetWorkspaceSize.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int8,
        ctypes.POINTER(ctypes.c_uint64),
        _c_void_p_p,
    ]
    _fmm_libopapi.aclnnBatchMatMul.restype = ctypes.c_int32
    _fmm_libopapi.aclnnBatchMatMul.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint64,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    ret = _fmm_libascendcl.aclInit(None)
    if ret not in (0, 100002, 500000):
        raise AssertionError(f"aclInit failed ret={ret}")
    ret = _fmm_libascendcl.aclrtSetDevice(0)
    if ret not in (0, 100002, 500000, 107001):
        raise AssertionError(f"aclrtSetDevice failed ret={ret}")
    _fmm_acl_stream = ctypes.c_void_p()
    _fmm_libascendcl.aclrtCreateStream(ctypes.byref(_fmm_acl_stream))
    _fmm_acl_inited = True


def _fmm_make_strides(shape):
    strides = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        strides[i] = shape[i + 1] * strides[i + 1]
    return strides


def _fmm_create_acl_tensor(
    shape, acl_dtype, device_addr, fmt=_ACL_FORMAT_ND, strides=None, storage_shape=None
):
    if strides is None:
        strides = _fmm_make_strides(shape)
    if storage_shape is None:
        storage_shape = shape
    shape_arr = (ctypes.c_int64 * len(shape))(*shape)
    strides_arr = (ctypes.c_int64 * len(strides))(*strides)
    storage_arr = (ctypes.c_int64 * len(storage_shape))(*storage_shape)
    return _fmm_libnnopbase.aclCreateTensor(
        ctypes.cast(shape_arr, _c_int64_p),
        len(shape),
        acl_dtype,
        ctypes.cast(strides_arr, _c_int64_p),
        0,
        fmt,
        ctypes.cast(storage_arr, _c_int64_p),
        len(storage_shape),
        device_addr,
    )


def _fmm_malloc_copy(host_bytes):
    n = len(host_bytes)
    dev = ctypes.c_void_p()
    assert (
        _fmm_libascendcl.aclrtMalloc(ctypes.byref(dev), n, _ACL_MEM_MALLOC_HUGE_FIRST)
        == 0
    ), "aclrtMalloc failed"
    buf = (ctypes.c_ubyte * n).from_buffer_copy(host_bytes)
    assert (
        _fmm_libascendcl.aclrtMemcpy(dev, n, buf, n, _ACL_MEMCPY_HOST_TO_DEVICE) == 0
    ), "aclrtMemcpy H2D failed"
    return dev


def _fmm_memcpy_d2h(dev, nbytes):
    buf = (ctypes.c_ubyte * nbytes)()
    assert (
        _fmm_libascendcl.aclrtMemcpy(
            buf, nbytes, dev, nbytes, _ACL_MEMCPY_DEVICE_TO_HOST
        )
        == 0
    ), "aclrtMemcpy D2H failed"
    return bytes(buf)


def _fmm_free(tensor, dev):
    if tensor is not None:
        _fmm_libnnopbase.aclDestroyTensor(tensor)
    if dev is not None:
        _fmm_libascendcl.aclrtFree(dev)


def npu_batch_matmul_fp32(x, x2):
    """直调 aclnnBatchMatMul, 输入 bf16/fp16, 输出 fp32。
    x: (M, K) bf16/fp16
    x2: (K, N) bf16/fp16
    y: (M, N) fp32 CPU tensor
    """
    _fmm_init()
    M, K = x.shape
    K2, N = x2.shape
    assert K == K2, f"K mismatch: x K={K} vs x2 K={K2}"
    assert x.dtype == x2.dtype, f"dtype mismatch: {x.dtype} vs {x2.dtype}"

    x_acl_dtype = _fmm_dt_map[x.dtype]
    y_acl_dtype = _ACL_FLOAT

    x_contig = x.contiguous()
    x2_contig = x2.contiguous()

    x_np = (
        x_contig.view(torch.uint16).numpy()
        if x.dtype in (torch.bfloat16, torch.float16)
        else x_contig.numpy()
    )
    x2_np = (
        x2_contig.view(torch.uint16).numpy()
        if x2.dtype in (torch.bfloat16, torch.float16)
        else x2_contig.numpy()
    )

    x_dev = _fmm_malloc_copy(x_np.tobytes())
    x2_dev = _fmm_malloc_copy(x2_np.tobytes())
    # 3D shape [1, M, K] / [1, K, N]
    x_t = _fmm_create_acl_tensor([1, M, K], x_acl_dtype, x_dev)
    x2_t = _fmm_create_acl_tensor([1, K, N], x_acl_dtype, x2_dev)

    y_np = np.zeros((1, M, N), dtype=np.float32)
    y_dev = _fmm_malloc_copy(y_np.tobytes())
    y_t = _fmm_create_acl_tensor([1, M, N], y_acl_dtype, y_dev)

    ws_size = ctypes.c_uint64(0)
    executor = ctypes.c_void_p()
    ret = _fmm_libopapi.aclnnBatchMatMulGetWorkspaceSize(
        x_t, x2_t, y_t, ctypes.c_int8(0), ctypes.byref(ws_size), ctypes.byref(executor)
    )
    if ret != 0:
        _fmm_free(x_t, x_dev)
        _fmm_free(x2_t, x2_dev)
        _fmm_free(y_t, y_dev)
        raise RuntimeError(f"aclnnBatchMatMul GetWorkspaceSize failed ret={ret}")
    ws = ctypes.c_void_p()
    if ws_size.value > 0:
        assert (
            _fmm_libascendcl.aclrtMalloc(
                ctypes.byref(ws), ws_size.value, _ACL_MEM_MALLOC_HUGE_FIRST
            )
            == 0
        ), "ws malloc failed"
    ret = _fmm_libopapi.aclnnBatchMatMul(ws, ws_size.value, executor, _fmm_acl_stream)
    if ret != 0:
        if ws.value:
            _fmm_libascendcl.aclrtFree(ws)
            _fmm_free(x_t, x_dev)
            _fmm_free(x2_t, x2_dev)
            _fmm_free(y_t, y_dev)
        raise RuntimeError(f"aclnnBatchMatMul exec failed ret={ret}")
    _fmm_libascendcl.aclrtSynchronizeStream(_fmm_acl_stream)
    if ws.value:
        _fmm_libascendcl.aclrtFree(ws)

    y_bytes = _fmm_memcpy_d2h(y_dev, M * N * 4)
    y_out = torch.frombuffer(y_bytes, dtype=torch.float32).reshape(M, N)

    _fmm_free(x_t, x_dev)
    _fmm_free(x2_t, x2_dev)
    _fmm_free(y_t, y_dev)
    return y_out


def tforward(sinner=-1, souter=-1, use_npu=False, **kwargs):
    q = kwargs["q"]
    k = kwargs["k"]
    v = kwargs["v"]
    k_descale = kwargs["k_descale"]
    v_descale = kwargs["v_descale"]
    block_table = kwargs["block_table"]
    cu_seqlens_q = kwargs["cu_seqlens_q"]
    seqused_q = kwargs["seqused_q"]
    seqused_kv = kwargs["seqused_kv"]
    sinks = kwargs["sinks"]
    attn_mask = kwargs["attn_mask"]
    quant_compute_mode = kwargs["quant_compute_mode"]
    softmax_scale = kwargs["softmax_scale"]
    mask_mode = kwargs["mask_mode"]
    win_left = kwargs["win_left"]
    win_right = kwargs["win_right"]
    max_seqlen_q = kwargs["max_seqlen_q"]
    max_seqlen_kv = kwargs["max_seqlen_kv"]
    layout_q = kwargs["layout_q"]
    layout_kv = kwargs["layout_kv"]
    layout_attn_out = kwargs["layout_attn_out"]
    return_softmax_lse = kwargs["return_softmax_lse"]

    print(f"k shape: {k.shape}, dtype: {k.dtype}")
    print(f"v shape: {v.shape}, dtype: {v.dtype}")
    if quant_compute_mode in (1, 2):
        k = unpack_int4(k)
        v = unpack_int4(v)
    print("after unpack")
    print(f"q shape: {q.shape}, dtype: {q.dtype}")
    print(f"k shape: {k.shape}, dtype: {k.dtype}")
    print(f"v shape: {v.shape}, dtype: {v.dtype}")
    print(f"k_descale shape: {k_descale.shape}, dtype: {k_descale.dtype}")
    print(f"v_descale shape: {v_descale.shape}, dtype: {v_descale.dtype}")

    batch, n1, n2, s1, s2, d, block_num, block_size = get_axis_from_tensor(
        q, k, cu_seqlens_q, seqused_q, seqused_kv, max_seqlen_kv, layout_q, layout_kv
    )

    if layout_q == layout_attn_out:
        attn_out = torch.zeros_like(q)
    else:
        raise NotImplementedError(
            f"unsupported layout_attn_out({layout_attn_out}) != layout_q({layout_q})"
        )
    print(f"attn_out shape: {attn_out.shape}, dtype: {attn_out.dtype}")
    attn_out_shape = attn_out.shape

    g = n1 // n2
    gs_flag = True if layout_q == "BNSD" else False
    if gs_flag:
        q = q.reshape(batch, n2, g, s1, d)
        attn_out = attn_out.reshape(batch, n2, g, s1, d)
    else:
        if layout_q == "BSND":
            q = q.reshape(batch, s1, n2, g, d)
            attn_out = attn_out.reshape(batch, s1, n2, g, d)
        else:
            q = q.reshape(-1, n2, g, d)
            attn_out = attn_out.reshape(-1, n2, g, d)

    if layout_q in ["BSND", "BNSD"]:
        x_max = torch.zeros((batch, n1, s1))
        x_sum = torch.zeros((batch, n1, s1))
    elif layout_q == "TND":
        x_max = torch.zeros((q.shape[0], n1))
        x_sum = torch.zeros((q.shape[0], n1))
    else:
        raise NotImplementedError(f"unsupported layout_q({layout_q})")

    if block_table is not None:
        k, k_descale_no_pa = trans_pa_to_no_pa(
            batch,
            n2,
            s2,
            d,
            layout_kv,
            k,
            k_descale,
            block_table,
            block_num,
            block_size,
            quant_compute_mode,
            gs_flag,
            True,
        )
        v, v_descale_no_pa = trans_pa_to_no_pa(
            batch,
            n2,
            s2,
            d,
            layout_kv,
            v,
            v_descale,
            block_table,
            block_num,
            block_size,
            quant_compute_mode,
            gs_flag,
            False,
        )
    else:
        k_descale_no_pa = k_descale
        v_descale_no_pa = v_descale

    v0_block = None
    k = antiquant_data(k, q.dtype, quant_compute_mode, k_descale_no_pa, gs_flag, True)
    v = antiquant_data(v, q.dtype, quant_compute_mode, v_descale_no_pa, gs_flag, False)
    print(f"batch: {batch}, n2: {n2}, g: {g}")
    for b_idx in range(batch):
        kvs_cur_batch = s2
        if seqused_kv is not None:
            kvs_cur_batch = seqused_kv[b_idx]
        qs_cur_batch = s1
        if cu_seqlens_q is not None:
            qs_cur_batch = cu_seqlens_q[b_idx + 1] - cu_seqlens_q[b_idx]
        elif seqused_q is not None:
            qs_cur_batch = seqused_q[b_idx]
        valid_qs_cur_batch = qs_cur_batch
        if layout_q == "TND" and seqused_q is not None:
            valid_qs_cur_batch = seqused_q[b_idx]
        if kvs_cur_batch == 0 or valid_qs_cur_batch == 0:
            continue
        print(
            f"kvs_cur_batch: {kvs_cur_batch}, valid_qs_cur_batch: {valid_qs_cur_batch}"
        )
        for n2_idx in range(n2):
            isinvalid = (kvs_cur_batch < valid_qs_cur_batch) and mask_mode != 0
            souter_loop_times = (
                math.ceil(g * valid_qs_cur_batch / souter) if souter > 0 else 1
            )
            for souter_idx in range(souter_loop_times):
                if souter_idx == souter_loop_times - 1:
                    souter_actual = g * valid_qs_cur_batch - souter * (
                        souter_loop_times - 1
                    )
                else:
                    souter_actual = souter
                block_sum = torch.zeros((souter_actual, 1), dtype=torch.float32)
                block_max = torch.ones((souter_actual, 1), dtype=torch.float32) * (
                    -float("inf")
                )
                block_exp = torch.ones((souter_actual, 1), dtype=torch.float32)
                if gs_flag:
                    q_block = q[b_idx, n2_idx][:, :valid_qs_cur_batch].reshape(-1, d)[
                        souter_idx * souter : souter_idx * souter + souter_actual
                    ]
                else:
                    if layout_q == "TND":
                        q_block = q[:valid_qs_cur_batch, n2_idx].reshape(-1, d)[
                            souter_idx * souter : souter_idx * souter + souter_actual
                        ]
                    else:
                        q_block = q[b_idx, :valid_qs_cur_batch, n2_idx].reshape(-1, d)[
                            souter_idx * souter : souter_idx * souter + souter_actual
                        ]
                if quant_compute_mode == 3:
                    v0_block = q_block * k_descale_no_pa[n2_idx].reshape(-1, d)
                out_block = torch.zeros_like(q_block)
                sinner_loop_times = (
                    math.ceil(kvs_cur_batch / sinner) if sinner > 0 else 1
                )
                for sinner_idx in range(sinner_loop_times):
                    if sinner_idx == sinner_loop_times - 1:
                        sinner_actual = kvs_cur_batch - sinner * (sinner_loop_times - 1)
                    else:
                        sinner_actual = sinner
                    if gs_flag:
                        k_block = k[b_idx, n2_idx][
                            sinner_idx * sinner : sinner_idx * sinner + sinner_actual
                        ]
                        v_block = v[b_idx, n2_idx][
                            sinner_idx * sinner : sinner_idx * sinner + sinner_actual
                        ]
                    else:
                        k_block = k[b_idx, :, n2_idx][
                            sinner_idx * sinner : sinner_idx * sinner + sinner_actual
                        ]
                        v_block = v[b_idx, :, n2_idx][
                            sinner_idx * sinner : sinner_idx * sinner + sinner_actual
                        ]
                    mask_block = None
                    if attn_mask is not None:
                        if mask_mode == 3:
                            mask_block = torch.zeros(
                                (q_block.shape[0], k_block.shape[0]), dtype=torch.uint8
                            )
                            for m_idx in range(0, souter_actual):
                                s1_idx = (
                                    (souter_idx * souter + m_idx) % valid_qs_cur_batch
                                    if gs_flag
                                    else (souter_idx * souter + m_idx) // g
                                )
                                s2idx_start = sinner_idx * sinner
                                s2idx_end = sinner_idx * sinner + sinner_actual
                                if (
                                    (kvs_cur_batch - valid_qs_cur_batch + 1 + s1_idx)
                                    > 0
                                ) and s2idx_end <= (
                                    kvs_cur_batch - valid_qs_cur_batch + 1 + s1_idx
                                ):
                                    continue
                                mask_num = s2idx_end - (
                                    kvs_cur_batch - valid_qs_cur_batch + 1 + s1_idx
                                )
                                mask_block[m_idx, -mask_num:] = 1
                        else:
                            raise NotImplementedError(
                                f"unimplemented sparse_mode: {mask_mode}"
                            )
                    if use_npu:
                        mm1 = npu_batch_matmul_fp32(
                            q_block, k_block.transpose(1, 0).contiguous()
                        )
                    else:
                        mm1 = torch.matmul(
                            q_block.to(torch.float32),
                            k_block.to(torch.float32).transpose(1, 0),
                        )
                    if mask_block is not None:
                        mm1 = mm1 + torch.where(
                            mask_block.to(torch.bool), -torch.inf, 0.0
                        ).to(torch.float32)
                    p_block, block_sum, block_max, block_exp = softmax_flash(
                        mm1 * softmax_scale,
                        block_sum,
                        block_max,
                        block_exp,
                        sinner_idx == 0,
                    )
                    if use_npu:
                        mm2 = npu_batch_matmul_fp32(
                            p_block.to(q.dtype), v_block.contiguous()
                        )
                    else:
                        mm2 = torch.matmul(p_block, v_block.to(torch.float32))
                    if sinner_idx == 0:
                        out_block = mm2
                    else:
                        out_block = out_block * block_exp + mm2
                    if sinner_idx == sinner_loop_times - 1:
                        out_block /= block_sum
                if isinvalid:
                    for m_idx in range(0, souter_actual):
                        s1_idx = (
                            (souter_idx * souter + m_idx) % valid_qs_cur_batch
                            if gs_flag
                            else (souter_idx * souter + m_idx) // g
                        )
                        if s1_idx < (valid_qs_cur_batch - kvs_cur_batch):
                            out_block[m_idx] = 0
                if layout_q == "TND":
                    x_max[cu_seqlens_q[b_idx] : cu_seqlens_q[b_idx + 1], n2_idx][
                        souter_idx * souter : souter_idx * souter + souter_actual
                    ] = block_max
                    x_sum[cu_seqlens_q[b_idx] : cu_seqlens_q[b_idx + 1], n2_idx][
                        souter_idx * souter : souter_idx * souter + souter_actual
                    ] = block_sum
                else:
                    if layout_q == "BNSD":
                        for m_idx in range(
                            souter_idx * souter, souter_idx * souter + souter_actual
                        ):
                            g_idx = m_idx // valid_qs_cur_batch
                            s1_idx = m_idx % valid_qs_cur_batch
                            x_max[b_idx, n2_idx * g + g_idx, s1_idx] = block_max[
                                m_idx - souter_idx * souter
                            ]
                            x_sum[b_idx, n2_idx * g + g_idx, s1_idx] = block_sum[
                                m_idx - souter_idx * souter
                            ]
                    else:
                        for m_idx in range(
                            souter_idx * souter, souter_idx * souter + souter_actual
                        ):
                            s1_idx = m_idx // g
                            g_idx = m_idx % g
                            n1_idx = n2_idx * g + g_idx
                            x_max[b_idx, n1_idx, s1_idx] = block_max[
                                m_idx - souter_idx * souter
                            ]
                            x_sum[b_idx, n1_idx, s1_idx] = block_sum[
                                m_idx - souter_idx * souter
                            ]
                if gs_flag:
                    for m_idx in range(
                        souter_idx * souter, souter_idx * souter + souter_actual
                    ):
                        g_idx = m_idx // valid_qs_cur_batch
                        s1_idx = m_idx % valid_qs_cur_batch
                        attn_out[b_idx, n2_idx, g_idx, s1_idx] = out_block[
                            m_idx - souter_idx * souter
                        ]
                else:
                    if layout_attn_out == "TND":
                        for m_idx in range(
                            souter_idx * souter, souter_idx * souter + souter_actual
                        ):
                            t_idx = cu_seqlens_q[b_idx] + m_idx
                            attn_out[t_idx, n2_idx, g_idx] = out_block[
                                m_idx - souter_idx * souter
                            ]
                    else:
                        for m_idx in range(
                            souter_idx * souter, souter_idx * souter + souter_actual
                        ):
                            s1_idx = m_idx // g
                            g_idx = m_idx % g
                            attn_out[b_idx, s1_idx, n2_idx, g_idx] = out_block[
                                m_idx - souter_idx * souter
                            ]
    attn_out = attn_out.reshape(attn_out_shape)
    return attn_out, x_max, x_sum


def run_cpu_golden(input_data):
    sinner = input_data.get("sinner", 512)
    souter = input_data.get("souter", 32)
    use_npu = input_data.get("use_npu", False)
    input_data = {
        k: v for k, v in input_data.items() if k not in ("sinner", "souter", "use_npu")
    }
    out_cpu, x_max, x_sum = tforward(sinner, souter, use_npu=use_npu, **input_data)
    lse = torch.log(x_sum) + x_max
    seqused_q = input_data.get("seqused_q")
    seqused_kv = input_data.get("seqused_kv")
    layout_q = input_data.get("layout_q")
    if seqused_q is not None or seqused_kv is not None:
        batch = lse.shape[0]
        for b in range(batch):
            if (
                seqused_kv is not None
                and int(seqused_kv[b]) == 0
                and layout_q in ("BNSD", "BSND")
            ):
                lse[b, :, :] = float("inf")
                continue
            if seqused_q is not None:
                s_used = (
                    seqused_q[b]
                    if isinstance(seqused_q, torch.Tensor)
                    else seqused_q[b]
                )
                lse[b, :, s_used:] = float("inf")
    return {"attn_out": out_cpu, "softmax_lse": lse}


def main():
    import argparse

    parser = argparse.ArgumentParser(description="CPU golden computation")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to testcase directory containing *_input.pt",
    )
    args = parser.parse_args()

    input_path = os.path.join(
        args.input_dir, os.path.basename(args.input_dir) + "_input.pt"
    )
    if not os.path.exists(input_path):
        files = os.listdir(args.input_dir)
        input_files = [f for f in files if f.endswith("_input.pt")]
        if not input_files:
            print(f"No _input.pt found in {args.input_dir}")
            sys.exit(1)
        input_path = os.path.join(args.input_dir, input_files[0])

    data = torch.load(input_path, map_location="cpu", weights_only=False)
    input_data = {}
    params = {}
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            input_data[k] = v
        else:
            params[k] = v
    input_data["params"] = params

    result = run_cpu_golden(input_data)
    case_name = os.path.basename(args.input_dir)
    output_path = os.path.join(args.input_dir, case_name + "_cpu.pt")
    torch.save(result, output_path)
    print(f"CPU golden saved to {output_path}")


if __name__ == "__main__":
    main()
