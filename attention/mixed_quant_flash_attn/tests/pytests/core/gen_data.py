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

import math
import random
import ctypes
import numpy as np
import torch
import torch.nn.functional as F


fp4_e1m2_values = [
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
]

fp4_e2m1_values = [
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
]

value_to_code_e1m2 = {val: idx for idx, val in enumerate(fp4_e1m2_values)}
value_to_code_e2m1 = {val: idx for idx, val in enumerate(fp4_e2m1_values)}
code_to_value_e1m2 = {idx: val for idx, val in enumerate(fp4_e1m2_values)}
code_to_value_e2m1 = {idx: val for idx, val in enumerate(fp4_e2m1_values)}


def get_torch_dtype(type_str):
    type_dict = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp4_e2m1": torch.uint8,
        "fp8_e8m0": torch.float8_e8m0fnu,
        "fp8_e4m3": torch.float8_e4m3fn,
        "hifp4": torch.uint8,
        "hifp4_scale": torch.float32,
        "fp32_e6m2e1e16": torch.float32,
    }
    return type_dict.get(type_str)


def generate_fp4_tensor(shape, dtype):
    num_elements = 1
    for s in shape:
        num_elements *= s
    if dtype == "fp4_e1m2":
        sampled_values = random.choices(fp4_e1m2_values, k=num_elements)
        encoded_codes = [value_to_code_e1m2[val] for val in sampled_values]
    elif dtype == "fp4_e2m1":
        sampled_values = random.choices(fp4_e2m1_values, k=num_elements)
        encoded_codes = [value_to_code_e2m1[val] for val in sampled_values]
    else:
        raise ValueError(f"Unsupported fp4 dtype: {dtype}")
    fp4_tensor_encoded = torch.tensor(encoded_codes, dtype=torch.uint8).reshape(shape)
    return fp4_tensor_encoded


def pack_int4(src):
    pack_size = 2
    shift = torch.tensor([0, 4], dtype=torch.uint8)
    tensor = src
    if src.numel() % pack_size != 0:
        tensor = F.pad(
            src.flatten(), (0, pack_size - src.numel() % pack_size), mode="constant"
        )
    reshaped = tensor.reshape([-1, 2])
    out = torch.sum(
        torch.bitwise_and(reshaped.view(torch.uint8), 0b00001111) << shift, dim=1
    ).to(torch.uint8)
    return out


def trans_np_fp4_e2m1_to_bfloat16(in_tensor):
    shape = in_tensor.shape
    flat = in_tensor.reshape(-1).view(np.uint8)
    out = np.zeros(flat.shape, dtype=np.float64)
    for i in range(flat.size):
        code = int(flat[i]) & 0x0F
        out[i] = code_to_value_e2m1[code]
    return out.reshape(shape)


def trans_np_fp4_e1m2_to_bfloat16(in_tensor):
    shape = in_tensor.shape
    flat = in_tensor.reshape(-1).view(np.uint8)
    out = np.zeros(flat.shape, dtype=np.float64)
    for i in range(flat.size):
        code = int(flat[i]) & 0x0F
        out[i] = code_to_value_e1m2[code]
    return out.reshape(shape)


def ieee_754_conversion(sign, exponent_raw, mantissa, exp_len=8, mant_len=7):
    sign_mult = -1 if sign == 1 else 1
    exponent = exponent_raw - (2 ** (exp_len - 1) - 1)
    mant_mult = 1
    for b in range(mant_len - 1, -1, -1):
        if mantissa & (2**b):
            mant_mult += 1 / (2 ** (mant_len - b))
    return sign_mult * (2**exponent) * mant_mult


def trans_tensor_fp8_e8m0_to_bf16(tensor):
    from ml_dtypes import bfloat16

    new_tensor = np.zeros_like(tensor).astype(bfloat16)
    with np.nditer(tensor, flags=["multi_index"], op_flags=["readwrite"]) as it:
        for x in it:
            new_value = ieee_754_conversion(0, int(x), 0)
            new_tensor[it.multi_index] = new_value
    return new_tensor


def float32_to_float8_e6m2(x):
    if x < 2 ** (-48):
        x = 2 ** (-48)
    if x > 1.5 * 2 ** (15):
        x = 1.5 * 2 ** (15)
    x_ptr = ctypes.pointer(ctypes.c_float(x))
    x_u32 = ctypes.c_uint32
    uint32_ptr = ctypes.cast(x_ptr, ctypes.POINTER(x_u32))
    x_u32 = uint32_ptr.contents.value
    f32_e = x_u32 >> 23
    f8_e = f32_e - 127 + 48
    m = (x_u32 >> 21) & 0x3
    return np.uint8(f8_e << 2 | m)


def gen_tensor_hifp4_scale(shape_input, data_range):
    input_min, input_max = data_range
    tensor = np.random.uniform(low=input_min, high=input_max, size=shape_input).astype(
        np.float32
    )
    new_tensor = np.zeros(shape_input).astype(np.uint32)
    with np.nditer(tensor, flags=["multi_index"], op_flags=["readwrite"]) as it:
        for x in it:
            E6M2_1 = float32_to_float8_e6m2(x)
            E1_8 = np.uint8(np.random.randint(0, 255))
            E1_16 = np.uint16(np.random.randint(0, 65535))
            max_pos_e8 = np.random.randint(0, 8)
            E1_8 = E1_8 | (1 << max_pos_e8)
            for k in range(8):
                if (E1_8 & (1 << k)) != 0:
                    pos = np.random.randint(0, 2)
                    E1_16 = E1_16 | (1 << (2 * k + pos))
            EG = np.uint32(E6M2_1) | np.uint32(E1_8) << 8 | np.uint32(E1_16) << 16
            new_tensor[it.multi_index] = EG
    return new_tensor.view(np.float32)


def trans_tensor_e6m2e1e16_to_bf16(tensor):
    from ml_dtypes import bfloat16

    new_tensor = np.zeros(
        [
            tensor.shape[0],
            tensor.shape[1],
            tensor.shape[2],
            tensor.shape[3],
            tensor.shape[4] * 16,
        ]
    ).astype(bfloat16)
    tensor_u32 = tensor.view(np.uint32)
    with np.nditer(tensor_u32, flags=["multi_index"], op_flags=["readwrite"]) as it:
        for x in it:
            E6M2_1 = x & 0xFF
            e6m2_e = E6M2_1 >> 2
            e6m2_m1 = (E6M2_1 & 0x3) >> 1
            e6m2_m0 = E6M2_1 & 0x1
            e6m2_val = 2.0 ** (e6m2_e - 48) * (1 + e6m2_m1 * 0.5 + e6m2_m0 * 0.25)
            E1_8 = np.zeros(8).astype(np.int32)
            for k in range(8):
                E1_8[k] = (x >> (8 + k)) & 0x1
            E1_8x2 = np.zeros(16).astype(np.int32)
            for k in range(8):
                E1_8x2[k * 2 : k * 2 + 2] = E1_8[k]
            E1_16 = np.zeros(16).astype(np.int32)
            for k in range(16):
                E1_16[k] = (x >> (16 + k)) & 0x1
            E16G = E1_16 + E1_8x2
            S16G = np.zeros(16).astype(bfloat16)
            for k in range(16):
                S16G[k] = (e6m2_val * 2 ** (E16G[k])).astype(bfloat16)
            idx = it.multi_index
            new_tensor[
                idx[0], idx[1], idx[2], idx[3], idx[4] * 16 : idx[4] * 16 + 16
            ] = S16G
    return new_tensor


def generate_input_shape(
    b, n, s, d, layout, cu_seqlen=None, quant_compute_mode=None, dtype=None, isKey=None
):
    """
    :param b: batch size or block num
    :param n: head num
    :param s: sequence length or block size
    :param d: head dim
    :param layout:
    :param cu_seqlen: when layout=TND, must exist
    :param quant_compute_mode:
    :param dtype:
    :param isKey:
    :return: shape

    scale shape conventions (PA_NZ):
      quant_compute_mode=1 (mxfp4, fp8_e8m0 scale, group_size=32):
        kv   (fp4_e2m1):     (b, n, d//64, s, 64)
        k_scale (fp8_e8m0):  (b, n, s//16, d//64, 16, 2)   # 2 sub-groups per 64 D
        v_scale (fp8_e8m0):  (b, n, d//16, s//64, 16, 2)   # 2 sub-groups per 64 S
      quant_compute_mode=2 (hifp4, fp32_e6m2e1e16 scale, group_size=4, 1 u32 -> 16 scales):
        kv   (fp4_e1m2):     (b, n, d//64, s, 64)
        k_scale (e6m2e1e16): (b, n, s//16, d//64, 16)      # 1 u32 per 64 D-elements
        v_scale (e6m2e1e16): (b, n, d//16, s//64, 16)      # 1 u32 per 64 S-elements
    """
    if layout == "BNSD":
        if quant_compute_mode == 1:
            if isKey:
                return (1, b, n, s, math.ceil(d / 64) * 2)
            else:
                return (1, b, n, d, math.ceil(s / 64) * 2)
        if quant_compute_mode == 2:
            if isKey:
                return (1, b, n, s, math.ceil(d / 64))
            else:
                return (1, b, n, d, math.ceil(s / 64))
        return (b, n, s, d)
    elif layout == "BSND":
        if quant_compute_mode == 1:
            if isKey:
                return (1, b, s, n, math.ceil(d / 64) * 2)
            else:
                return (1, b, n, d, math.ceil(s / 64) * 2)
        if quant_compute_mode == 2:
            if isKey:
                return (1, b, s, n, math.ceil(d / 64))
            else:
                return (1, b, n, d, math.ceil(s / 64))
        return (b, s, n, d)
    elif layout == "TND":
        if cu_seqlen is None:
            raise ValueError("cu_seqlen cannot be None when layout is TND")
        if len(cu_seqlen) != (b + 1):
            raise ValueError("the length of cu_seqlen must equal to (b+1)")
        return (cu_seqlen[-1], n, d)
    elif layout == "PA_NZ":
        if quant_compute_mode is None:
            raise ValueError("quant_compute_mode cannot be None when layout is PA_NZ")
        if dtype is None:
            raise ValueError("dtype cannot be None when layout is PA_NZ")
        if quant_compute_mode == 1:
            if dtype == "fp4_e2m1":
                return (b, n, d // 64, s, 64)
            elif dtype == "fp8_e8m0":
                if isKey is None:
                    raise ValueError(
                        "isKey cannot be None when dtype is fp8_e8m0 and layout is PA_NZ"
                    )
                if isKey:
                    return (b, n, s // 16, d // 64, 16, 2)
                else:
                    return (b, n, d // 16, s // 64, 16, 2)
            else:
                raise NotImplementedError(f"unsupported dtype {dtype}")
        elif quant_compute_mode == 2:
            if dtype == "fp4_e1m2":
                return (b, n, d // 64, s, 64)
            elif dtype in ("fp32_e6m2e1e16", "hifp4_scale"):
                if isKey is None:
                    raise ValueError(
                        "isKey cannot be None when dtype is fp32_e6m2e1e16 and layout is PA_NZ"
                    )
                if isKey:
                    return (b, n, s // 16, d // 64, 16)
                else:
                    return (b, n, d // 16, s // 64, 16)
            else:
                raise NotImplementedError(
                    f"unsupported dtype {dtype} for quant_compute_mode=2"
                )
        else:
            raise NotImplementedError(
                f"unsupported quant_compute_mode {quant_compute_mode} for PA_NZ"
            )
    elif layout == "PA_BBND":
        if quant_compute_mode is None:
            raise ValueError("quant_compute_mode cannot be None when layout is PA_BBND")
        if dtype is None:
            raise ValueError("dtype cannot be None when layout is PA_BBND")
        if quant_compute_mode == 1:
            group_size = 32
            if dtype == "fp4_e2m1":
                return (b, s, n, d)
            elif dtype == "fp8_e8m0":
                if isKey is None:
                    raise ValueError(
                        "isKey cannot be None when dtype is fp8_e8m0 and layout is PA_BBND"
                    )
                if isKey:
                    return (b, s, n, d // group_size)
                else:
                    return (b, n, d, s // group_size)
            else:
                raise NotImplementedError(f"unsupported dtype {dtype}")
        elif quant_compute_mode == 2:
            group_size = 64
            if dtype == "fp4_e1m2":
                return (b, s, n, d)
            elif dtype in ("fp32_e6m2e1e16", "hifp4_scale"):
                if isKey is None:
                    raise ValueError(
                        "isKey cannot be None when dtype is fp32_e6m2e1e16 and layout is PA_BBND"
                    )
                if isKey:
                    return (b, s, n, d // group_size)
                else:
                    return (b, n, d, s // group_size)
            else:
                raise NotImplementedError(
                    f"unsupported dtype {dtype} for quant_compute_mode=2"
                )
        else:
            raise NotImplementedError(
                f"unsupported quant_compute_mode {quant_compute_mode} for PA_BBND"
            )
    elif layout == "PA_BNBD":
        if quant_compute_mode is None:
            raise ValueError("quant_compute_mode cannot be None when layout is PA_BNBD")
        if dtype is None:
            raise ValueError("dtype cannot be None when layout is PA_BNBD")
        if quant_compute_mode == 1:
            group_size = 32
            if dtype == "fp4_e2m1":
                return (b, n, s, d)
            elif dtype == "fp8_e8m0":
                if isKey is None:
                    raise ValueError(
                        "isKey cannot be None when dtype is fp8_e8m0 and layout is PA_BNBD"
                    )
                if isKey:
                    return (b, n, s, d // group_size)
                else:
                    return (b, n, d, s // group_size)
            else:
                raise NotImplementedError(f"unsupported dtype {dtype}")
        elif quant_compute_mode == 2:
            group_size = 64
            if dtype == "fp4_e1m2":
                return (b, n, s, d)
            elif dtype in ("fp32_e6m2e1e16", "hifp4_scale"):
                if isKey is None:
                    raise ValueError(
                        "isKey cannot be None when dtype is fp32_e6m2e1e16 and layout is PA_BNBD"
                    )
                if isKey:
                    return (b, n, s, d // group_size)
                else:
                    return (b, n, d, s // group_size)
            else:
                raise NotImplementedError(
                    f"unsupported dtype {dtype} for quant_compute_mode=2"
                )
        else:
            raise NotImplementedError(
                f"unsupported quant_compute_mode {quant_compute_mode} for PA_BNBD"
            )
    else:
        raise NotImplementedError(f"unsupported layout: {layout}")


def split_float32_to_uint16(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32, copy=False)
    uint32_view = arr.view(np.uint32)
    high = (uint32_view >> 16).astype(np.uint16)
    low = (uint32_view & 0xFFFF).astype(np.uint16)
    return np.concatenate([low, high], axis=-1)


def pack_descale_pa_bbnd(
    descale_logical, layout, block_table, block_num, block_size, b, n, s, d, isKey
):
    """
    Pack a logical descale tensor into PA_BBND physical layout.
    PA_BBND K-scale: (blockNum, blockSize, N, D/groupSize)
    PA_BBND V-scale: (blockNum, blockSize/groupSize, N, D)
    No split_float32_to_uint16 — simple reshape/transpose from logical BNSD/BSND.
    """
    src = descale_logical[0]
    if isKey:
        d_groups = src.shape[-1]
        cache = np.zeros(
            (block_num, block_size, n, d_groups), dtype=descale_logical.dtype
        )
        for bIdx in range(b):
            num_blocks = (
                int((block_table[bIdx] >= 0).sum().item())
                if isinstance(block_table, torch.Tensor)
                else int(np.sum(block_table[bIdx] >= 0))
            )
            for blockIdx in range(num_blocks):
                blockId = int(block_table[bIdx][blockIdx])
                if blockId < 0:
                    continue
                s_start = blockIdx * block_size
                s_end = min((blockIdx + 1) * block_size, s)
                if layout in ("BSND", "BSH"):
                    cur = src[bIdx, s_start:s_end].reshape(-1, n, d_groups)
                else:
                    cur = src[bIdx, :, s_start:s_end].transpose(1, 0, 2)
                cache[blockId, : s_end - s_start] = cur
    else:
        s_groups = src.shape[-1]
        group_size = s // s_groups
        sg_per_block = block_size // group_size
        cache = np.zeros((block_num, n, d, sg_per_block), dtype=descale_logical.dtype)
        for bIdx in range(b):
            num_blocks = (
                int((block_table[bIdx] >= 0).sum().item())
                if isinstance(block_table, torch.Tensor)
                else int(np.sum(block_table[bIdx] >= 0))
            )
            for blockIdx in range(num_blocks):
                blockId = int(block_table[bIdx][blockIdx])
                if blockId < 0:
                    continue
                sg_start = blockIdx * sg_per_block
                sg_end = min((blockIdx + 1) * sg_per_block, s_groups)
                cur = src[bIdx, :, :, sg_start:sg_end]
                cache[blockId, :, :, : sg_end - sg_start] = cur
    return cache


def pack_descale_pa_bnbd(
    descale_logical, layout, block_table, block_num, block_size, b, n, s, d, isKey
):
    """
    Pack a logical descale tensor into PA_BNBD physical layout.
    PA_BNBD K-scale: (blockNum, N, blockSize, D/groupSize)
    PA_BNBD V-scale: (blockNum, N, D, blockSize/groupSize)
    No split_float32_to_uint16 — simple reshape/transpose from logical BNSD/BSND.
    """
    src = descale_logical[0]
    if isKey:
        d_groups = src.shape[-1]
        cache = np.zeros(
            (block_num, n, block_size, d_groups), dtype=descale_logical.dtype
        )
        for bIdx in range(b):
            num_blocks = (
                int((block_table[bIdx] >= 0).sum().item())
                if isinstance(block_table, torch.Tensor)
                else int(np.sum(block_table[bIdx] >= 0))
            )
            for blockIdx in range(num_blocks):
                blockId = int(block_table[bIdx][blockIdx])
                if blockId < 0:
                    continue
                s_start = blockIdx * block_size
                s_end = min((blockIdx + 1) * block_size, s)
                if layout in ("BSND", "BSH"):
                    cur = (
                        src[bIdx, s_start:s_end]
                        .reshape(-1, n, d_groups)
                        .transpose(1, 0, 2)
                    )
                else:
                    cur = src[bIdx, :, s_start:s_end]
                cache[blockId, :, : s_end - s_start] = cur
    else:
        s_groups = src.shape[-1]
        sg_per_block = block_size * s_groups // s
        cache = np.zeros((block_num, n, d, sg_per_block), dtype=descale_logical.dtype)
        for bIdx in range(b):
            num_blocks = (
                int((block_table[bIdx] >= 0).sum().item())
                if isinstance(block_table, torch.Tensor)
                else int(np.sum(block_table[bIdx] >= 0))
            )
            for blockIdx in range(num_blocks):
                blockId = int(block_table[bIdx][blockIdx])
                if blockId < 0:
                    continue
                sg_start = blockIdx * sg_per_block
                sg_end = min((blockIdx + 1) * sg_per_block, s_groups)
                cur = src[bIdx, :, :, sg_start:sg_end]
                cache[blockId, :, :, : sg_end - sg_start] = cur
    return cache


def pack_descale_pa_nz(
    descale_logical, layout, block_table, block_num, block_size, b, n, s, d, isKey
):
    """
    Pack a logical hifp4 descale tensor into PA_NZ physical layout, mirroring the
    golden demo's `generate_pa_data` for `fp32_e6m2e1e16`.

    Logical shape (BNSD):
        K: (1, b, n, s, d//64)
        V: (1, b, n, d, s//64)
    Physical PA_NZ shape:
        K: (block_num, n, block_size//16, d//64, 16)
        V: (block_num, n, d//16, block_size//64, 16)

    For each batch and each block, the corresponding slice of the logical descale
    is reshaped to (n, block_size//16, 16, last_dim), transposed to
    (n, block_size//16, last_dim, 16), then `split_float32_to_uint16(...).view(float32)`
    is written into the cache. The split/transpose is required by the NPU decoder.
    """
    if layout not in ("BNSD", "BSND", "BSH"):
        raise NotImplementedError(
            f"unsupported layout for descale PA packing: {layout}"
        )
    if isKey:
        last_dim = d // 64
        cache = np.zeros(
            (block_num, n, block_size // 16, last_dim, 16), dtype=descale_logical.dtype
        )
        G = 1
    else:
        last_dim = s // 64
        cache = np.zeros(
            (block_num, n, d // 16, block_size // 64, 16), dtype=descale_logical.dtype
        )
        G = 64

    src = descale_logical[0]
    for bIdx in range(b):
        num_blocks = (
            int((block_table[bIdx] >= 0).sum().item())
            if isinstance(block_table, torch.Tensor)
            else int(np.sum(block_table[bIdx] >= 0))
        )
        for blockIdx in range(num_blocks):
            blockId = int(block_table[bIdx][blockIdx])
            if blockId < 0:
                continue
            if isKey:
                if layout in ("BSND", "BSH"):
                    cur = (
                        src[bIdx, blockIdx * block_size : (blockIdx + 1) * block_size]
                        .reshape(-1, n, last_dim)
                        .transpose(1, 0, 2)
                    )
                else:
                    cur = src[
                        bIdx, :, blockIdx * block_size : (blockIdx + 1) * block_size
                    ].reshape(n, -1, last_dim)
                target_s = math.ceil(cur.shape[1] / 16) * 16
                if cur.shape[1] < target_s:
                    padded = np.ones((n, target_s, last_dim), dtype=cur.dtype)
                    padded[:, : cur.shape[1], :] = cur
                    cur = padded
                tmp = split_float32_to_uint16(
                    cur.reshape(n, target_s // 16, 16, last_dim)
                    .transpose(0, 1, 3, 2)
                    .copy()
                ).view(np.float32)
                cache[blockId, :, : tmp.shape[1]] = tmp
            else:
                cur = src[
                    bIdx,
                    :,
                    :,
                    blockIdx * block_size // G : (blockIdx + 1) * block_size // G,
                ].reshape(n, d, -1)
                tmp = split_float32_to_uint16(
                    cur.reshape(n, d // 16, 16, cur.shape[2])
                    .transpose(0, 1, 3, 2)
                    .copy()
                ).view(np.float32)
                cache[blockId, :, :, : tmp.shape[2]] = tmp
    return cache


def generate_torch_data(data_range, shape, dtype):
    torch_dtype = get_torch_dtype(dtype)
    if dtype in ("fp4_e2m1", "fp4_e1m2"):
        encoded = generate_fp4_tensor(shape, dtype)
        pack_shape = list(shape)
        pack_shape[-1] = pack_shape[-1] >> 1
        pack_shape = tuple(pack_shape)
        pack_data = pack_int4(encoded).reshape(pack_shape)
        return pack_data
    elif dtype in ("fp32_e6m2e1e16", "hifp4_scale"):
        return torch.from_numpy(gen_tensor_hifp4_scale(shape, data_range)).to(
            torch_dtype
        )
    else:
        return torch.empty(shape).uniform_(data_range[0], data_range[1]).to(torch_dtype)


def resolve_kv_scale_dtype(quant_compute_mode, q_dtype):
    if quant_compute_mode == 1:
        return "fp8_e8m0"
    elif quant_compute_mode == 2:
        return "fp32_e6m2e1e16"
    else:
        raise NotImplementedError(
            f"unimplemented quant_compute_mode: {quant_compute_mode}"
        )


def parse_quant_compute_mode(quant_compute_mode):
    if quant_compute_mode == 1:
        kv_dtype_str = "fp4_e2m1"
        kv_descale_dtype_str = "fp8_e8m0"
        kv_descale_range = (0.01, 10)
    elif quant_compute_mode == 2:
        kv_dtype_str = "fp4_e1m2"
        kv_descale_dtype_str = "fp32_e6m2e1e16"
        kv_descale_range = (125, 129)
    else:
        raise NotImplementedError(
            f"unsupported quant_compute_mode: {quant_compute_mode}"
        )
    return kv_dtype_str, kv_descale_dtype_str, kv_descale_range


def generate_inputs(params, seed=42):
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    # 轴信息
    b = params.get("B", 1)
    n1 = params["N1"]
    n2 = params.get("N2", n1)
    s1 = params.get("S1", 1)
    s2 = params.get("S2", s1)
    d = params["D"]

    # seq信息
    cu_seqlens_q = params.get("cu_seqlens_q", None)
    seqused_q = params.get("seqused_q", None)
    seqused_kv = params.get("seqused_kv", None)
    max_seqlen_q = params.get("max_seqlen_q", -1)
    max_seqlen_kv = params.get("max_seqlen_kv", -1)

    # layout信息
    layout_q = params.get("layout_q", "BNSD")
    layout_kv = params.get("layout_kv", layout_q)
    layout_attn_out = params.get("layout_attn_out", layout_q)

    # PA信息
    block_size = params.get("block_size", 128)
    block_num = params.get("block_num", 0)
    block_table = params.get("block_table", None)

    # mask信息
    mask_mode = params.get("mask_mode", 0)
    attn_mask = params.get("attn_mask", None)
    win_left = params.get("win_left", -1)
    win_right = params.get("win_right", -1)

    # lse
    return_softmax_lse = params.get("return_softmax_lse", False)

    # sinks
    sinks = params.get("sinks", None)

    softmax_scale = params.get("softmax_scale", 1.0 / (d**0.5))

    # dtype信息
    q_dtype_str = params.get("q_dtype", "bf16")
    quant_compute_mode = params.get("quant_compute_mode", 1)
    kv_dtype_str, kv_descale_dtype_str, kv_descale_range_default = (
        parse_quant_compute_mode(quant_compute_mode)
    )

    # data range信息
    q_range = params.get("q_range", (-10.0, 10.0))
    kv_range = params.get("kv_range", (-10.0, 10.0))
    kv_descale_range = params.get("kv_descale_range", kv_descale_range_default)

    q_shape = generate_input_shape(b, n1, s1, d, layout_q)

    tensors = {
        "q": None,
        "k": None,
        "v": None,
        "k_descale": None,
        "v_descale": None,
        "block_table": None,
        "cu_seqlens_q": None,
        "seqused_q": None,
        "seqused_kv": None,
        "sinks": None,
        "attn_mask": None,
        "metadata": None,
        "quant_compute_mode": None,
        "softmax_scale": None,
        "mask_mode": None,
        "win_left": None,
        "win_right": None,
        "max_seqlen_q": None,
        "max_seqlen_kv": None,
        "layout_q": None,
        "layout_kv": None,
        "layout_attn_out": None,
        "return_softmax_lse": None,
        "batch_size": b,
        "num_heads_q": n1,
        "num_heads_kv": n2,
        "head_dim": d,
    }
    if "PA" in layout_kv:
        if block_num <= 0:
            block_num = sum([math.ceil(x / block_size) for x in seqused_kv])
        if block_table is None:
            max_block_num_per_batch = math.ceil(max(seqused_kv) / block_size)
            block_table = torch.ones(b, max_block_num_per_batch, dtype=torch.int32) * -1
            global_block_offset = 0
            for bIdx in range(b):
                kvs_cur_batch = seqused_kv[bIdx]
                num_blocks = math.ceil(kvs_cur_batch / block_size)
                start_id = global_block_offset
                end_id = global_block_offset + num_blocks
                block_id_list = list(range(start_id, end_id))
                random.shuffle(block_id_list)
                # block_num 偏小时循环复用，保证 id < block_num 不越界
                block_ids_tensor = (
                    torch.tensor(block_id_list, dtype=torch.int32) % block_num
                )
                block_table[bIdx][:num_blocks] = torch.tensor(block_ids_tensor)
                global_block_offset += num_blocks
        kv_shape = generate_input_shape(
            block_num,
            n2,
            block_size,
            d,
            layout_kv,
            quant_compute_mode=quant_compute_mode,
            dtype=kv_dtype_str,
        )
        k_descale_shape = generate_input_shape(
            block_num,
            n2,
            block_size,
            d,
            layout_kv,
            quant_compute_mode=quant_compute_mode,
            dtype=kv_descale_dtype_str,
            isKey=True,
        )
        v_descale_shape = generate_input_shape(
            block_num,
            n2,
            block_size,
            d,
            layout_kv,
            quant_compute_mode=quant_compute_mode,
            dtype=kv_descale_dtype_str,
            isKey=False,
        )
        tensors["block_table"] = block_table
    else:
        kv_shape = generate_input_shape(b, n2, s2, d, layout_kv)
        k_descale_shape = generate_input_shape(
            b,
            n2,
            s2,
            d,
            layout_kv,
            quant_compute_mode=quant_compute_mode,
            dtype=kv_descale_dtype_str,
            isKey=True,
        )
        v_descale_shape = generate_input_shape(
            b,
            n2,
            s2,
            d,
            layout_kv,
            quant_compute_mode=quant_compute_mode,
            dtype=kv_descale_dtype_str,
            isKey=False,
        )
    if mask_mode in [3, 4]:
        attn_mask = torch.triu(torch.ones((2048, 2048), dtype=torch.int8), diagonal=1)

    tensors["q"] = generate_torch_data(q_range, q_shape, q_dtype_str)
    tensors["k"] = generate_torch_data(kv_range, kv_shape, kv_dtype_str)
    tensors["v"] = generate_torch_data(kv_range, kv_shape, kv_dtype_str)

    if "PA" in layout_kv and quant_compute_mode == 2:
        logical_layout = "BNSD" if layout_q == "BNSD" else "BSND"
        k_descale_logical_shape = generate_input_shape(
            b,
            n2,
            s2,
            d,
            logical_layout,
            quant_compute_mode=quant_compute_mode,
            dtype=kv_descale_dtype_str,
            isKey=True,
        )
        v_descale_logical_shape = generate_input_shape(
            b,
            n2,
            s2,
            d,
            logical_layout,
            quant_compute_mode=quant_compute_mode,
            dtype=kv_descale_dtype_str,
            isKey=False,
        )
        k_descale_logical = generate_torch_data(
            kv_descale_range, k_descale_logical_shape, kv_descale_dtype_str
        )
        v_descale_logical = generate_torch_data(
            kv_descale_range, v_descale_logical_shape, kv_descale_dtype_str
        )
        if layout_kv == "PA_NZ":
            k_descale_pa = pack_descale_pa_nz(
                k_descale_logical.numpy(),
                logical_layout,
                block_table,
                block_num,
                block_size,
                b,
                n2,
                s2,
                d,
                isKey=True,
            )
            v_descale_pa = pack_descale_pa_nz(
                v_descale_logical.numpy(),
                logical_layout,
                block_table,
                block_num,
                block_size,
                b,
                n2,
                s2,
                d,
                isKey=False,
            )
        elif layout_kv == "PA_BBND":
            k_descale_pa = pack_descale_pa_bbnd(
                k_descale_logical.numpy(),
                logical_layout,
                block_table,
                block_num,
                block_size,
                b,
                n2,
                s2,
                d,
                isKey=True,
            )
            v_descale_pa = pack_descale_pa_bbnd(
                v_descale_logical.numpy(),
                logical_layout,
                block_table,
                block_num,
                block_size,
                b,
                n2,
                s2,
                d,
                isKey=False,
            )
        elif layout_kv == "PA_BNBD":
            k_descale_pa = pack_descale_pa_bnbd(
                k_descale_logical.numpy(),
                logical_layout,
                block_table,
                block_num,
                block_size,
                b,
                n2,
                s2,
                d,
                isKey=True,
            )
            v_descale_pa = pack_descale_pa_bnbd(
                v_descale_logical.numpy(),
                logical_layout,
                block_table,
                block_num,
                block_size,
                b,
                n2,
                s2,
                d,
                isKey=False,
            )
        else:
            raise NotImplementedError(
                f"unsupported layout_kv for descale packing: {layout_kv}"
            )
        tensors["k_descale"] = torch.from_numpy(k_descale_pa).to(torch.float32)
        tensors["v_descale"] = torch.from_numpy(v_descale_pa).to(torch.float32)
    else:
        tensors["k_descale"] = generate_torch_data(
            kv_descale_range, k_descale_shape, kv_descale_dtype_str
        )
        tensors["v_descale"] = generate_torch_data(
            kv_descale_range, v_descale_shape, kv_descale_dtype_str
        )

    tensors["cu_seqlens_q"] = (
        torch.tensor(cu_seqlens_q, dtype=torch.int32)
        if isinstance(cu_seqlens_q, list)
        else cu_seqlens_q
    )
    tensors["seqused_q"] = (
        torch.tensor(seqused_q, dtype=torch.int32)
        if isinstance(seqused_q, list)
        else seqused_q
    )
    tensors["seqused_kv"] = (
        torch.tensor(seqused_kv, dtype=torch.int32)
        if isinstance(seqused_kv, list)
        else seqused_kv
    )
    tensors["sinks"] = sinks
    tensors["attn_mask"] = attn_mask
    tensors["metadata"] = None
    tensors["quant_compute_mode"] = quant_compute_mode
    tensors["softmax_scale"] = softmax_scale
    tensors["mask_mode"] = mask_mode
    tensors["win_left"] = win_left
    tensors["win_right"] = win_right
    tensors["max_seqlen_q"] = max_seqlen_q
    tensors["max_seqlen_kv"] = max_seqlen_kv
    tensors["layout_q"] = layout_q
    tensors["layout_kv"] = layout_kv
    tensors["layout_attn_out"] = layout_attn_out
    tensors["return_softmax_lse"] = return_softmax_lse

    return tensors
