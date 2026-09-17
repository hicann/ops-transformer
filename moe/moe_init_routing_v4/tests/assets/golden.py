#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

__spec__ = {
    "moe_init_routing_v4": "MoeInitRoutingV4KernelSpec",
    "aclnnMoeInitRoutingV4": "AclnnMoeInitRoutingV4Spec",
}

import numpy


def To_BF16(x):
    x = numpy.float64(x)
    tmp = numpy.abs(x)
    E = numpy.floor(numpy.log2(tmp + 2 ** (-1000)))
    E = -126 if E < -126 else E
    res = numpy.round(x * 2 ** (-E + 7)) * 2 ** (E - 7)
    return res


def To_BF16_Array(x):
    x = numpy.asarray(x, dtype=numpy.float64)
    tmp = numpy.abs(x)
    E = numpy.floor(numpy.log2(tmp + 2 ** (-1000)))
    E = numpy.maximum(E, -126)
    res = numpy.round(x * 2 ** (-E + 7)) * 2 ** (E - 7)
    return res


def _float32_to_bf16_trunc_bits(x):
    x_f32 = numpy.asarray(x, dtype=numpy.float32)
    return (x_f32.view(numpy.uint32) >> 16).astype(numpy.uint16)


def _bf16_bits_to_float32(x):
    bits = numpy.asarray(x, dtype=numpy.uint16)
    return (bits.astype(numpy.uint32) << 16).view(numpy.float32)


def _mxfp8_roundscale_quant_numpy(x, dst_type_str, clamp_amax=False, block_size=32):
    from ttk.utilities.dtypes import (
        numpy_float8_e4m3fn,
        numpy_float8_e5m2,
        numpy_float8_e8m0,
    )

    x_array = numpy.asarray(x)
    x_f32 = x_array.astype(numpy.float32)
    rows, cols = x_f32.shape
    valid_scale_cols = (cols + block_size - 1) // block_size
    scale_cols = ((valid_scale_cols + 1) // 2) * 2
    fp8_emax = 15 if dst_type_str == "float8_e5m2" else 8

    scale_code = numpy.zeros((rows, scale_cols), dtype=numpy.uint8)
    inv_scale = numpy.zeros((rows, valid_scale_cols), dtype=numpy.float32)
    is_fp16_input = x_array.dtype == numpy.float16

    for row in range(rows):
        for block_idx in range(valid_scale_cols):
            start = block_idx * block_size
            end = min(start + block_size, cols)
            block = x_f32[row, start:end]
            amax_bits_array = _float32_to_bf16_trunc_bits(block) & numpy.uint16(0x7FFF)
            if is_fp16_input:
                block_fp16 = x_array[row, start:end]
                special_mask = numpy.isinf(block_fp16) | numpy.isnan(block_fp16)
                amax_bits_array[special_mask] = numpy.uint16(0x7F80)

            amax_bits = (
                int(numpy.max(amax_bits_array)) if amax_bits_array.size > 0 else 0
            )
            if clamp_amax and amax_bits < 0x38D2:
                amax_bits = 0x38D2

            if amax_bits >= 0x7F80:
                scale_code[row, block_idx] = numpy.uint8(0xFF)
                inv_scale[row, block_idx] = _bf16_bits_to_float32(
                    numpy.array([0x7F81], dtype=numpy.uint16)
                )[0]
                continue
            if amax_bits == 0:
                scale_code[row, block_idx] = numpy.uint8(0)
                inv_scale[row, block_idx] = 0.0
                continue

            exp_bits = amax_bits >> 7
            mant_bits = amax_bits & 0x007F
            mant_add = 1 if mant_bits > 0x0060 else 0
            rounded_scale = (
                1 if exp_bits <= fp8_emax else exp_bits - fp8_emax + mant_add
            )
            scale_code[row, block_idx] = numpy.uint8(rounded_scale & 0xFF)

            inv_exp = 0x00FE - rounded_scale
            inv_bits = numpy.uint16((inv_exp << 7) & 0xFFFF)
            if rounded_scale == 0x00FE:
                inv_bits = numpy.uint16(0x0040)
            inv_scale[row, block_idx] = _bf16_bits_to_float32(
                numpy.array([inv_bits], dtype=numpy.uint16)
            )[0]

    quant_f32 = x_f32.copy()
    for block_idx in range(valid_scale_cols):
        start = block_idx * block_size
        end = min(start + block_size, cols)
        quant_f32[:, start:end] *= inv_scale[:, block_idx : block_idx + 1]

    quant_f32 = numpy.nan_to_num(quant_f32, nan=0.0, copy=False)
    if dst_type_str == "float8_e5m2":
        quant_x = quant_f32.astype(numpy_float8_e5m2(), copy=False)
    else:
        quant_x = quant_f32.astype(numpy_float8_e4m3fn(), copy=False)
    quant_scale = scale_code.view(numpy_float8_e8m0())
    return quant_scale, quant_x


def E6M2_REC(x):
    if numpy.isnan(x):
        from ml_dtypes import bfloat16

        return bfloat16(numpy.nan)
    else:
        E6 = numpy.floor(numpy.log2(x + 2 ** (-1000)))
        M2 = x * 2 ** (-E6 + 2) - 4

        if M2 == 0:
            M7 = 0.0
        elif M2 == 1:
            M7 = 77
        elif M2 == 2:
            M7 = 43
        elif M2 == 3:
            M7 = 18
        else:
            print("Unexpected Input")
            exit()

        if M2 == 0:
            E8 = -E6
        else:
            E8 = -E6 - 1

        res = 2 ** (E8) * (1 + M7 * 2 ** (-7))
        return res


def BF16_to_E6M2(x):
    from ml_dtypes import bfloat16

    if (
        numpy.isinf(x)
        or numpy.isnan(x)
        or (x >= 1.625 * (2**15))
        or (x == 0)
        or (x < 2 ** (-48))
    ):
        return bfloat16(numpy.nan)
    else:
        x = 2 ** (-48) if (x < 2 ** (-48)) else x
        E = numpy.floor(numpy.log2(x))
        E6M2 = numpy.round(x * 2 ** (-E + 2)) * 2 ** (-2 + E)
    return E6M2


def float_to_e6m2_int(x):
    if numpy.isnan(x):
        return 0xFF
    else:
        E6 = numpy.floor(numpy.log2(x + 2 ** (-1000)))
        M2 = x * (2 ** (-E6 + 2)) - 4.0
        E6 = E6 + 48
        B8_E6M2 = numpy.uint8(int(E6) * 4) | numpy.uint8(int(M2))
        return B8_E6M2


def dynamic_quant_hifp4(x):
    from ml_dtypes import bfloat16
    from en_dtypes import float4_e1m2

    G = 64
    shape = x.shape
    last_dim = shape[-1]
    Ncnt = numpy.ceil(last_dim / G).astype(int)
    x_2d = x.reshape(-1, last_dim)
    Mi = x_2d.shape[0]
    Ni = x_2d.shape[1]
    res = numpy.zeros((Mi, Ni))
    scale = numpy.zeros((Mi, Ncnt)).astype(numpy.float32)
    scale_uint8 = scale.view(numpy.uint8).reshape(Mi * Ncnt * 4)
    ksi = 0

    for i in range(Mi):
        for j in range(Ncnt):
            ori = x_2d[i, j * G : j * G + G]
            S = numpy.ones(G)
            S[ori < 0] = -1
            S = S.T
            tmpG = numpy.abs(ori)

            V16 = numpy.zeros(16)
            for k in range(16):
                V16[k] = numpy.max(tmpG[k * 4 : k * 4 + 4])

            V8 = numpy.zeros(8)
            for k in range(8):
                V8[k] = numpy.max(V16[k * 2 : k * 2 + 2])

            Vmax = numpy.max(V8).astype(bfloat16)

            Const_rec = numpy.uint16(0x3E12).view(bfloat16)
            SF = Vmax * Const_rec

            E6M2 = BF16_to_E6M2(SF)
            E6M2_code = float_to_e6m2_int(E6M2)

            REC_E6M2 = E6M2_REC(E6M2)

            E1_8 = To_BF16_Array((V8 * REC_E6M2)) >= 4

            E1_8x2 = numpy.zeros(16)
            for k in range(8):
                E1_8x2[k * 2 : k * 2 + 2] = E1_8[k]
            E1_16 = (
                To_BF16_Array((V16 * REC_E6M2.astype(bfloat16) * 2 ** (-E1_8x2))) >= 2
            )

            e6m2_uint8 = numpy.uint8(E6M2_code)

            scale_uint8[ksi] = e6m2_uint8
            ksi += 1

            E1_8_int = E1_8.astype(numpy.int32)
            E1_8_bit = numpy.array(
                [
                    numpy.packbits(E1_8_int[:8][::-1])[0],
                ],
                dtype=numpy.uint8,
            )
            scale_uint8[ksi] = E1_8_bit[0]
            ksi += 1

            E1_16_int = E1_16.astype(numpy.int32)
            E1_16_bit = numpy.array(
                [
                    numpy.packbits(E1_16_int[:8][::-1])[0],
                    numpy.packbits(E1_16_int[8:][::-1])[0],
                ],
                dtype=numpy.uint8,
            )
            scale_uint8[ksi] = E1_16_bit[0]
            ksi += 1
            scale_uint8[ksi] = E1_16_bit[1]
            ksi += 1

            DE16 = E1_16 + E1_8x2
            DE64 = numpy.zeros(G)
            for k in range(16):
                DE64[k * 4 : k * 4 + 4] = DE16[k]
            in_grp = bfloat16(tmpG) * bfloat16(REC_E6M2) * 2 ** (-DE64)

            res[i, j * G : j * G + G] = S * in_grp
    res = res.astype(float4_e1m2)

    scale_shape = list(shape[:-1]) + [Ncnt]
    scale = scale.reshape(scale_shape)
    output_data = [res.reshape(shape), scale]
    return output_data


def _moe_init_routing_v4_numpy(
    x,
    expert_idx,
    scale,
    offset,
    topk_weight,
    active_num,
    expert_capacity,
    expert_num,
    drop_pad_mode,
    expert_tokens_num_type,
    expert_tokens_num_flag,
    quant_mode,
    active_expert_range,
    row_idx_type,
):
    expert_start = active_expert_range[0]
    expert_end = active_expert_range[1]
    num_rows = x.shape[0]
    h = x.shape[1]
    k = expert_idx.shape[-1]
    total_length = num_rows * k
    if active_num == 0 or active_num == -1:
        x_out_num = total_length
    else:
        x_out_num = min(active_num, total_length)

    if drop_pad_mode == 1 and quant_mode == -1:
        expert_idx_flat = expert_idx.copy().reshape(-1)
        sorted_expert_indices = numpy.argsort(expert_idx_flat, axis=-1, kind="stable")
        sorted_expert_idx = expert_idx_flat[sorted_expert_indices]

        valid_mask = (sorted_expert_idx >= expert_start) & (
            sorted_expert_idx < expert_end
        )
        valid_sorted_row_idx = sorted_expert_indices[valid_mask]
        valid_sorted_expert_idx = sorted_expert_idx[valid_mask]

        sort_row_tmp = numpy.full((expert_num * expert_capacity), -1, dtype=numpy.int64)
        expert_offset = numpy.zeros(expert_num, dtype=numpy.int64)
        for row_idx, expert_id in zip(valid_sorted_row_idx, valid_sorted_expert_idx):
            expert_id = int(expert_id)
            if expert_id < 0 or expert_id >= expert_num:
                continue
            offset_idx = expert_offset[expert_id]
            expert_offset[expert_id] += 1
            if offset_idx >= expert_capacity:
                continue
            sort_row_tmp[expert_id * expert_capacity + offset_idx] = row_idx

        expanded_row_idx = numpy.full((num_rows * k,), -1, dtype=numpy.int32)
        valid_capacity_mask = sort_row_tmp != -1
        expanded_row_idx[sort_row_tmp[valid_capacity_mask]] = numpy.arange(
            expert_num * expert_capacity, dtype=numpy.int32
        )[valid_capacity_mask]

        expanded_x = numpy.zeros((expert_num * expert_capacity, h), dtype=x.dtype)
        expanded_x[valid_capacity_mask] = x[sort_row_tmp[valid_capacity_mask] // k, :]
        expanded_x = expanded_x.reshape((expert_num, expert_capacity, h))

        expanded_scale = None
        if scale is not None:
            scale_shape = (expert_num * expert_capacity,) + scale.shape[1:]
            expanded_scale = numpy.zeros(scale_shape, dtype=scale.dtype)
            if scale.shape[0] == 1:
                expanded_scale[valid_capacity_mask] = scale[0]
            else:
                expanded_scale[valid_capacity_mask] = scale[
                    sort_row_tmp[valid_capacity_mask] // k
                ]
            if expanded_scale.ndim == 2 and expanded_scale.shape[-1] == 1:
                expanded_scale = expanded_scale.reshape(-1)

        if expert_tokens_num_flag:
            expert_tokens_count = numpy.bincount(
                valid_sorted_expert_idx - expert_start,
                minlength=expert_end - expert_start,
            )
            expert_tokens_count = expert_tokens_count[
                : expert_end - expert_start
            ].astype(numpy.int64)
        else:
            expert_tokens_count = numpy.array([], dtype=numpy.int64)

        # DropPad uses the same source-to-destination map as expanded_x.
        expanded_topk_weight = None
        if topk_weight is not None:
            topk_weight_flat = numpy.asarray(topk_weight).reshape(-1)
            expanded_topk_weight = numpy.zeros(
                (expert_num * expert_capacity, 1), dtype=numpy.float32
            )
            expanded_topk_weight[valid_capacity_mask, 0] = topk_weight_flat[
                sort_row_tmp[valid_capacity_mask]
            ]
        return (
            expanded_x,
            expanded_row_idx,
            expert_tokens_count,
            expanded_scale,
            expanded_topk_weight,
        )

    expert_idx_in = expert_idx.copy().reshape(-1)
    actual_expert_total_num = numpy.sum(
        (expert_idx_in >= expert_start) & (expert_idx_in < expert_end)
    )

    expert_idx_in[(expert_idx_in < expert_start)] = numpy.int32(
        numpy.iinfo(numpy.int32).max
    )
    sorted_expert_indices = numpy.argsort(expert_idx_in, axis=-1, kind="stable")
    sorted_expert_idx = expert_idx_in[sorted_expert_indices]
    if row_idx_type == 1:
        expanded_row_idx = sorted_expert_indices
    else:
        expanded_row_idx = numpy.ones(num_rows * k).astype(numpy.int32) * -1
        tmp_indices = numpy.arange(actual_expert_total_num)
        expanded_row_idx[sorted_expert_indices[:actual_expert_total_num]] = tmp_indices

    if quant_mode == -1:
        if scale is None:
            expanded_scale = None
        else:
            from ttk.utilities.dtypes import (
                numpy_float8_e4m3fn,
                numpy_float8_e5m2,
                numpy_float8_e8m0,
                numpy_float4_e2m1,
            )

            if (
                x.dtype == numpy_float8_e4m3fn()
                or x.dtype == numpy_float8_e5m2()
                or x.dtype == numpy_float4_e2m1()
            ):
                scale = scale.astype(numpy_float8_e8m0())
                expanded_scale = scale[
                    sorted_expert_indices[:actual_expert_total_num] // k
                ]
            else:
                expanded_scale = scale[
                    sorted_expert_indices[:actual_expert_total_num] // k
                ].flatten()
        expanded_x = x[sorted_expert_indices[:actual_expert_total_num] // k, :]

    elif quant_mode == 0:
        expanded_scale = None
        expanded_x = x[sorted_expert_indices[:actual_expert_total_num] // k, :]
        x_fp16 = expanded_x.astype(numpy.float16)
        scale_fp16 = scale.astype(numpy.float16)
        expanded_x = x_fp16 * scale_fp16[0]
        offset_fp16 = offset.astype(numpy.float16)
        expanded_x = expanded_x + offset_fp16[0]
        expanded_x = numpy.rint(expanded_x)
        expanded_x = numpy.clip(expanded_x, -128, 127)
        expanded_x = expanded_x.astype(numpy.int8)

    elif quant_mode == 1:
        expanded_x = x[sorted_expert_indices // k, :]
        expanded_x = expanded_x.astype(numpy.float32)
        if scale is None:
            expanded_x = expanded_x[:actual_expert_total_num, :]
            x_abs = numpy.abs(expanded_x)
            x_max = numpy.max(x_abs, axis=-1, keepdims=True)
            expanded_scale = x_max / 127
            expanded_x = expanded_x / expanded_scale
            expanded_x = numpy.round(expanded_x).astype(numpy.int8)
        else:
            expended_scale = scale[
                sorted_expert_idx[:actual_expert_total_num] - expert_start, :
            ]
            expanded_x = expanded_x[:actual_expert_total_num, :]
            expanded_x = expanded_x * expended_scale
            x_abs = numpy.abs(expanded_x)
            x_max = numpy.max(x_abs, axis=-1, keepdims=True)
            expanded_scale = x_max / 127
            expanded_x = expanded_x / expanded_scale
            expanded_x = numpy.round(expanded_x).astype(numpy.int8)

    elif quant_mode == 2 or quant_mode == 3 or quant_mode == 16 or quant_mode == 17:
        expanded_x = x[sorted_expert_indices // k, :]
        expanded_x = expanded_x[:actual_expert_total_num, :]
        quant_mode_dtype_str_map = {
            2: "float8_e5m2",
            3: "float8_e4m3fn",
            16: "float8_e5m2",
            17: "float8_e4m3fn",
        }
        expanded_scale, expanded_x = _mxfp8_roundscale_quant_numpy(
            expanded_x,
            quant_mode_dtype_str_map[quant_mode],
            clamp_amax=(quant_mode == 16 or quant_mode == 17),
        )

    elif quant_mode == 6:
        expanded_scale = None
        expanded_x = x[sorted_expert_indices // k, :]
        expanded_x = expanded_x[:actual_expert_total_num, :]

        if expanded_x.dtype == "bfloat16":
            expanded_x = expanded_x.astype(numpy.float32)

        from ttk.utilities import numpy_hifloat8

        expanded_x = expanded_x.astype(numpy_hifloat8(), copy=False)

    elif quant_mode == 7:
        expanded_scale = None
        HIFLOAT8_MIN = -32768.0
        HIFLOAT8_MAX = 32768.0
        expanded_x = x[sorted_expert_indices // k, :]
        expanded_x = expanded_x.astype(numpy.float32)
        expanded_x = expanded_x[:actual_expert_total_num, :]

        expanded_x = expanded_x * scale
        expanded_x = numpy.clip(expanded_x, HIFLOAT8_MIN, HIFLOAT8_MAX)

        from ttk.utilities import numpy_hifloat8

        expanded_x = expanded_x.astype(numpy_hifloat8(), copy=False)

    elif quant_mode == 8:
        HIFLOAT8_MIN = -32768.0
        HIFLOAT8_MAX = 32768.0
        expanded_x = x[sorted_expert_indices // k, :]
        expanded_x = expanded_x.astype(numpy.float32)
        expanded_x = expanded_x[:actual_expert_total_num, :]

        x_abs = numpy.abs(expanded_x)
        x_max_per_token = numpy.max(x_abs, axis=-1, keepdims=True)
        expanded_scale = x_max_per_token / 32768.0

        expanded_scale_safe = numpy.where(expanded_scale == 0, 1.0, expanded_scale)
        expanded_x = expanded_x / expanded_scale_safe
        expanded_x = numpy.clip(expanded_x, HIFLOAT8_MIN, HIFLOAT8_MAX)

        from ttk.utilities import numpy_hifloat8

        expanded_x = expanded_x.astype(numpy_hifloat8(), copy=False)

    elif quant_mode == 9:
        expanded_x = x[sorted_expert_indices // k, :]
        expanded_x = expanded_x[:actual_expert_total_num, :].astype(numpy.float32)
        from ttk.utilities.dtypes import mx_quantize

        expanded_scale, expanded_x = mx_quantize(
            expanded_x,
            mx_ele_dtype="float4_e2m1",
            axis=-1,
            block_size=32,
            round_mode="rint",
            scale_alg=0,
        )

    elif quant_mode == 10:
        expanded_x = x[sorted_expert_indices // k, :]
        expanded_x = expanded_x[:actual_expert_total_num, :]
        res = dynamic_quant_hifp4(expanded_x)
        expanded_x = res[0]
        expanded_scale = res[1]

    elif quant_mode == 11 or quant_mode == 12:
        expanded_x = x[sorted_expert_indices // k, :]
        expanded_x = expanded_x[:actual_expert_total_num, :]
        quant_mode_dtype_str_map = {11: "float8_e5m2", 12: "float8_e4m3fn"}
        from ttk.utilities import get_dtype_range
        from ttk.utilities import numpy_float8_e5m2, numpy_float8_e4m3fn

        def block_max_with_padding(expanded_x, row_block_size, col_block_size):
            batch, rows, cols = expanded_x.shape
            pad_rows = (row_block_size - rows % row_block_size) % row_block_size
            pad_cols = (col_block_size - cols % col_block_size) % col_block_size

            x_padded = numpy.pad(
                expanded_x,
                ((0, 0), (0, pad_rows), (0, pad_cols)),
                mode="constant",
                constant_values=0,
            )

            padded_batch, padded_rows, padded_cols = x_padded.shape
            row_blocks = padded_rows // row_block_size
            col_blocks = padded_cols // col_block_size

            result = numpy.zeros((padded_batch, row_blocks, col_blocks))

            for k in range(padded_batch):
                for i in range(row_blocks):
                    for j in range(col_blocks):
                        block = x_padded[
                            k,
                            i * row_block_size : (i + 1) * row_block_size,
                            j * col_block_size : (j + 1) * col_block_size,
                        ]
                        result[k, i, j] = numpy.max(block)
            return result

        def pad_to_even_zero(tensor, axis):
            if tensor.shape[axis] % 2 == 0:
                return tensor
            pad_width = [(0, 0)] * tensor.ndim
            pad_width[axis] = (0, 1)
            return numpy.pad(tensor, pad_width, mode="constant", constant_values=0)

        row_block_size = 1
        col_block_size = 128
        expand_flag = False
        if expanded_x.ndim == 2:
            expand_flag = True
            expanded_x = numpy.expand_dims(expanded_x, axis=0)
        dst_type_str = quant_mode_dtype_str_map[quant_mode]
        max_value = 0

        if dst_type_str == "float8_e5m2":
            max_value = (2 - pow(2, -2)) * pow(2, 15)
        elif dst_type_str == "float8_e4m3fn":
            max_value = (2 - pow(2, -2)) * pow(2, 8)

        x_abs = numpy.abs(expanded_x)
        block_max = block_max_with_padding(x_abs, row_block_size, col_block_size)

        block_max_f32 = block_max.astype(numpy.float32)
        scale = block_max_f32 / max_value

        block_max_uint32 = block_max_f32.view(numpy.uint32)
        inf_nan_mask = block_max_uint32 >= numpy.uint32(0x7F800000)
        fp32_nan = numpy.array(numpy.int32(0x7FC00000).view(numpy.float32))
        scale = numpy.where(inf_nan_mask, fp32_nan, scale)

        min_normal_f32 = numpy.finfo(numpy.float32).tiny
        scale = numpy.where(scale < min_normal_f32, 0, scale)
        scale = pad_to_even_zero(scale, axis=2)

        scale_expanded = numpy.zeros_like(expanded_x).astype(numpy.float32)
        for k in range(scale.shape[0]):
            for i in range(scale.shape[1]):
                for j in range(scale.shape[2]):
                    scale_expanded[
                        k,
                        i * row_block_size : (i + 1) * row_block_size,
                        j * col_block_size : (j + 1) * col_block_size,
                    ] = scale[k, i, j]

        x_f32 = expanded_x.astype(numpy.float32)
        out_f32 = x_f32 / scale_expanded

        max_norm = get_dtype_range(dst_type_str)[1]
        numpy.clip(out_f32, a_min=-max_norm, a_max=max_norm, out=out_f32)

        output_scale = scale.astype("float32")
        round_data = numpy.round(out_f32, 8)
        round_data = numpy.nan_to_num(round_data, nan=0.0, copy=False)

        if dst_type_str == "float8_e5m2":
            round_data = round_data.astype(numpy_float8_e5m2(), copy=False)
        elif dst_type_str == "float8_e4m3fn":
            round_data = round_data.astype(numpy_float8_e4m3fn(), copy=False)

        if (
            expand_flag == True
            and round_data.shape[0] == 1
            and output_scale.shape[0] == 1
        ):
            round_data = numpy.squeeze(round_data, axis=0)
            output_scale = numpy.squeeze(output_scale, axis=0)

        expanded_x = round_data
        expanded_scale = output_scale

    elif quant_mode in (4, 5, 14, 15):
        expanded_x = x[sorted_expert_indices[:actual_expert_total_num] // k, :]
        expanded_x_f32 = expanded_x.astype(numpy.float32)
        num_rows_q, h_q = expanded_x_f32.shape
        group_size = 128
        num_groups = (h_q + group_size - 1) // group_size
        padded_h = num_groups * group_size

        expanded_x_padded = numpy.pad(
            expanded_x_f32,
            ((0, 0), (0, padded_h - h_q)),
            mode="constant",
            constant_values=0,
        )
        grouped = expanded_x_padded.reshape(num_rows_q, num_groups, group_size)

        amax = numpy.max(numpy.abs(grouped), axis=-1)

        if quant_mode in (14, 15):
            amax = numpy.maximum(amax, numpy.float32(0.0001))

        amax_uint32 = amax.view(numpy.uint32)
        inf_nan_mask = amax_uint32 >= numpy.uint32(0x7F800000)

        fp8_max = {
            4: numpy.float32(57344.0),
            5: numpy.float32(448.0),
            14: numpy.float32(57344.0),
            15: numpy.float32(448.0),
        }[quant_mode]

        raw_scale = numpy.array(amax / fp8_max, dtype=numpy.float32)

        raw_scale_int = raw_scale.view(numpy.int32)
        exp_bits = (raw_scale_int >> numpy.int32(23)) & numpy.int32(0xFF)
        mant_bits = raw_scale_int & numpy.int32(0x007FFFFF)
        mant_add = numpy.where(
            mant_bits != numpy.int32(0), numpy.int32(1), numpy.int32(0)
        ).astype(numpy.int32)
        rounded_exp = exp_bits - numpy.int32(127) + mant_add
        biased_exp = rounded_exp + numpy.int32(127)
        round_scale_int = (biased_exp << numpy.int32(23)).astype(numpy.int32)
        round_scale = round_scale_int.view(numpy.float32).copy()
        nonzero_mask = raw_scale != numpy.float32(0.0)
        round_scale = numpy.where(nonzero_mask, round_scale, numpy.float32(0.0))

        fp32_nan = numpy.array(numpy.int32(0x7FC00000).view(numpy.float32))
        round_scale = numpy.where(inf_nan_mask, fp32_nan, round_scale)

        divisor = numpy.where(
            round_scale != numpy.float32(0.0), round_scale, numpy.float32(1.0)
        )
        divisor_expanded = divisor[:, :, numpy.newaxis]
        quantized_padded = grouped / divisor_expanded
        quantized_padded = numpy.where(
            inf_nan_mask[:, :, numpy.newaxis], numpy.float32(0.0), quantized_padded
        )

        from ml_dtypes import float8_e5m2, float8_e4m3fn

        fp8_dtype = {
            4: float8_e5m2,
            5: float8_e4m3fn,
            14: float8_e5m2,
            15: float8_e4m3fn,
        }[quant_mode]
        expanded_x = quantized_padded.reshape(num_rows_q, padded_h)[:, :h_q].astype(
            fp8_dtype
        )
        expanded_scale = round_scale

    elif quant_mode == 13:
        from ttk.utilities.dtypes import numpy_int4

        expanded_x = x[sorted_expert_indices // k, :].astype(numpy.float32)

        if scale is None:
            expanded_x = expanded_x[:actual_expert_total_num, :]
        else:
            scale_fp32 = scale.astype(numpy.float32)
            smooth_scale = (
                scale_fp32.reshape(1, -1) if scale_fp32.ndim == 1 else scale_fp32
            )
            expanded_x = expanded_x[:actual_expert_total_num, :] * smooth_scale

        x_abs = numpy.abs(expanded_x)
        x_max = numpy.max(x_abs, axis=-1, keepdims=True)
        expanded_scale = numpy.where(
            x_max == 0,
            numpy.float32(0.0),
            (x_max * numpy.float32(1.0 / 7.0)).astype(numpy.float32),
        )
        mul = numpy.where(
            expanded_scale == 0,
            numpy.float32(0.0),
            (numpy.float32(1.0) / expanded_scale).astype(numpy.float32),
        )
        expanded_x = numpy.rint((expanded_x * mul).astype(numpy.float32))
        expanded_x = numpy.clip(expanded_x, -8, 7).astype(numpy_int4())

        if actual_expert_total_num < x_out_num:
            expanded_x = numpy.concatenate(
                [
                    expanded_x,
                    numpy.ones(
                        (x_out_num - actual_expert_total_num, h), dtype=expanded_x.dtype
                    ),
                ],
                axis=0,
            )
            expanded_scale = numpy.concatenate(
                [
                    expanded_scale.reshape(-1),
                    numpy.ones(
                        (x_out_num - actual_expert_total_num,),
                        dtype=expanded_scale.dtype,
                    ),
                ]
            )
        else:
            expanded_x = expanded_x[:x_out_num, :]
            expanded_scale = expanded_scale.reshape(-1)[:x_out_num]

    if expert_tokens_num_type == 0:
        counts = numpy.bincount(
            sorted_expert_idx[:actual_expert_total_num] - expert_start,
            minlength=expert_end - expert_start,
        )
        expert_tokens_count = numpy.cumsum(counts).astype(numpy.int64)
    elif expert_tokens_num_type == 1:
        expert_tokens_count = numpy.bincount(
            sorted_expert_idx[:actual_expert_total_num] - expert_start
        )
        expert_tokens_count = numpy.concatenate(
            [
                expert_tokens_count,
                numpy.zeros(
                    (expert_end - expert_start) - len(expert_tokens_count)
                ).astype(numpy.int64),
            ]
        )
    elif expert_tokens_num_type == 2:
        expert_id, counts = numpy.unique(
            sorted_expert_idx[:actual_expert_total_num], return_counts=True
        )
        expert_tokens_count = numpy.column_stack((expert_id, counts))
        if expert_tokens_count.shape[0] < expert_num:
            expert_tokens_count = numpy.concatenate(
                (expert_tokens_count, [[0, 0]]), axis=0
            )

    # Dropless output follows the valid sorted source indices used by expanded_x.
    expanded_topk_weight = None
    if topk_weight is not None:
        topk_weight_flat = numpy.asarray(topk_weight).reshape(-1)
        effective_count = int(actual_expert_total_num)
        if int(active_num) > 0:
            effective_count = min(int(active_num), effective_count)
        expanded_topk_weight = topk_weight_flat[
            sorted_expert_indices[:effective_count]
        ].reshape(-1, 1)
    return (
        expanded_x,
        expanded_row_idx.astype(numpy.int32),
        expert_tokens_count.astype(numpy.int64),
        expanded_scale,
        expanded_topk_weight,
    )


def MoeInitRoutingV4(*input_arrays, **kwargs):
    x = input_arrays[0]
    expert_idx = input_arrays[1]
    scale = input_arrays[2] if len(input_arrays) > 2 else None
    offset = input_arrays[3] if len(input_arrays) > 3 else None
    active_num = input_arrays[4] if len(input_arrays) > 4 else None
    topk_weight = input_arrays[5] if len(input_arrays) > 5 else None

    if scale is not None and numpy.asarray(scale).ndim == 0:
        scale = None
    if offset is not None and numpy.asarray(offset).ndim == 0:
        offset = None
    expert_num = int(kwargs.get("expert_num", 256))
    active_expert_range = kwargs.get("active_expert_range", [0, expert_num])
    # V4 currently accepts active_num but processes the complete B*S*K length.
    # Keep the input in the interface and normalize it until dynamic semantics
    # are implemented by the operator.
    active_num = _normalize_v4_active_num(
        active_num,
        int(numpy.asarray(x).shape[0]) * int(numpy.asarray(expert_idx).shape[-1]),
    )
    return _moe_init_routing_v4_numpy(
        numpy.asarray(x),
        numpy.asarray(expert_idx),
        None if scale is None else numpy.asarray(scale),
        None if offset is None else numpy.asarray(offset),
        None if topk_weight is None else numpy.asarray(topk_weight),
        active_num,
        int(kwargs.get("expert_capacity", -1)),
        expert_num,
        int(kwargs.get("drop_pad_mode", 0)),
        int(kwargs.get("expert_tokens_num_type", 0)),
        bool(kwargs.get("expert_tokens_num_flag", False)),
        int(kwargs.get("quant_mode", -1)),
        active_expert_range,
        int(kwargs.get("row_idx_type", 0)),
    )


def _drop_rowidx_tail_len(npu_arr, golden_arr, golden0_arr, min_len):
    """SCATTER 模式下 expanded_row_idx 的 golden 返回完整 n*k 置换（含尾部无效区），
    而 NPU 只写出前 actualExpertIdxNum 个有效条目。此函数计算 golden 的有效前缀长度，
    使 compare 只比对该前缀，避免尾部无效数据导致误报。"""
    if npu_arr is None or golden_arr is None or min_len <= 0:
        return min_len
    golden_flat = numpy.asarray(golden_arr).reshape(-1)
    npu_flat = numpy.asarray(npu_arr).reshape(-1)
    golden_shape = getattr(golden_arr, "shape", ())
    golden_dtype = getattr(golden_arr, "dtype", None)
    if len(golden_shape) != 1 or getattr(golden_dtype, "kind", "?") not in ("i", "u"):
        return min_len
    if golden_flat.size != npu_flat.size or golden_flat.size <= 0:
        return min_len
    if int(golden_flat.min()) < 0 or int(golden_flat.max()) != golden_flat.size - 1:
        return min_len
    if numpy.unique(golden_flat).size != golden_flat.size:
        return min_len
    if golden0_arr is None:
        return min_len
    golden0 = numpy.asarray(_torch_to_numpy(golden0_arr))
    if golden0.ndim < 1:
        return min_len
    prefix_len = int(golden0.shape[0])
    if 0 <= prefix_len < golden_flat.size:
        return prefix_len
    return min_len


def _torch_to_numpy(tensor):
    if tensor is None:
        return None
    if hasattr(tensor, "detach"):
        tensor = tensor.detach().cpu()
    if hasattr(tensor, "numpy"):
        try:
            return tensor.numpy()
        except (TypeError, RuntimeError):
            import torch

            return tensor.to(torch.float32).numpy()
    return numpy.asarray(tensor)


def _unpack_int4(raw_bytes):
    """将 packed uint8（每字节存 2 个 int4 nibble）解包为有符号 int8 数组。
    低 nibble 在前，高 nibble 在后；>=8 的 nibble 减 16 转为负数。"""
    lo = (raw_bytes & 0x0F).astype(numpy.int8)
    hi = ((raw_bytes >> 4) & 0x0F).astype(numpy.int8)
    nibbles = numpy.empty(raw_bytes.size * 2, dtype=numpy.int8)
    nibbles[0::2] = lo
    nibbles[1::2] = hi
    nibbles[nibbles >= 8] -= 16
    return nibbles


def _quant_ulp_result(npu_vals, golden_vals):
    """量化输出 ULP 判据：绝对误差 <= 1 视为通过，错误占比超过 ptol(0.1%) 才判 FAIL。
    用于 int8/int4 量化输出，消除 float16 计算精度导致的 off-by-1 误报。"""
    min_len = min(npu_vals.size, golden_vals.size)
    npu_vals = npu_vals[:min_len]
    golden_vals = golden_vals[:min_len]
    diff = numpy.abs(numpy.subtract(npu_vals, golden_vals))
    total = diff.size
    bad_count = int(numpy.sum(diff > 1))
    precision = float(total - bad_count) / total * 100.0 if total > 0 else 100.0
    return {
        "pass": (1 - precision / 100.0) <= 0.001,
        "precision": precision,
        "error_info": None
        if bad_count == 0
        else f"quant ulp mismatch {bad_count}/{total}",
    }


def _to_list(val):
    if val is None:
        return None
    if hasattr(val, "tolist"):
        return val.tolist()
    if hasattr(val, "detach"):
        return val.detach().cpu().tolist()
    return list(val)


def _normalize_v4_active_num(active_num, total_length):
    """Normalize V4's optional active_num to the currently supported value.

    V4 accepts the optional input for interface compatibility, but the current
    tiling/kernel contract always processes the complete B*S*K routing length.
    Keep reading the input so the golden mirrors the full API signature, then
    normalize it to that length until dynamic active_num semantics are added.
    """
    if active_num is not None:
        value = _torch_to_numpy(active_num)
        if value is not None and numpy.asarray(value).size > 0:
            _ = numpy.asarray(value).reshape(-1)[0].item()
    return int(total_length)


def _numpy_to_torch(arr, template):
    if arr is None or template is None:
        return None
    import torch

    arr_np = numpy.asarray(arr)
    if hasattr(template, "device"):
        device = template.device
    else:
        device = None
    if hasattr(template, "dtype"):
        target_torch_dtype = template.dtype
    else:
        target_torch_dtype = None

    arr_dtype_name = str(arr_np.dtype).lower()
    target_dtype_name = (
        str(target_torch_dtype).lower() if target_torch_dtype is not None else ""
    )
    custom_keywords = [
        "e2m1",
        "e1m2",
        "float4",
        "hifloat8",
        "e4m3",
        "e5m2",
        "e8m0",
        "hif8",
        "float8",
    ]
    is_custom_dtype = arr_np.dtype.kind not in ("f", "i", "u", "b") or any(
        x in arr_dtype_name or x in target_dtype_name for x in custom_keywords
    )
    if is_custom_dtype:
        # golden 返回 float32 但目标 dtype 为 float8/float4/hifloat8 时，需先 astype 到
        # 目标 numpy dtype（itemsize=1），再 view(uint8)。若直接对 float32（itemsize=4）
        # 做 view(uint8)，元素数会放大 4 倍，导致 reshape 失败。
        target_np_dtype_name = target_dtype_name.replace("torch.", "")
        if arr_dtype_name not in target_np_dtype_name and arr_np.dtype.kind == "f":
            from ttk.utilities.dtypes import (
                numpy_float8_e4m3fn,
                numpy_float8_e5m2,
                numpy_float8_e8m0,
                numpy_float4_e2m1,
                numpy_float4_e1m2,
                numpy_hifloat8,
            )

            dtype_map = {
                "float8_e4m3fn": numpy_float8_e4m3fn,
                "float8_e5m2": numpy_float8_e5m2,
                "float8_e8m0": numpy_float8_e8m0,
                "float8_e8m0fnu": numpy_float8_e8m0,
                "float4_e2m1": numpy_float4_e2m1,
                "float4_e1m2": numpy_float4_e1m2,
                "hifloat8": numpy_hifloat8,
                "hif8": numpy_hifloat8,
            }
            for key, dtype_fn in dtype_map.items():
                if key in target_np_dtype_name:
                    try:
                        arr_np = arr_np.astype(dtype_fn())
                    except (TypeError, ValueError):
                        pass
                    break
        try:
            raw_uint8 = numpy.asarray(arr_np.view(numpy.uint8).copy()).reshape(
                arr_np.shape
            )
        except (ValueError, TypeError):
            raw_uint8 = numpy.frombuffer(arr_np.tobytes(), dtype=numpy.uint8).reshape(
                arr_np.shape
            )
        result = torch.from_numpy(raw_uint8)
        if device is not None:
            result = result.to(device=device)
        if target_torch_dtype is not None:
            try:
                result = result.view(dtype=target_torch_dtype)
            except (TypeError, RuntimeError, Exception):
                pass
        return result

    try:
        result = torch.from_numpy(arr_np.copy())
    except (TypeError, RuntimeError):
        result = torch.as_tensor(numpy.asarray(arr_np, dtype=numpy.float32))
    if device is not None:
        result = result.to(device=device)
    if target_torch_dtype is not None:
        try:
            result = result.to(dtype=target_torch_dtype)
        except (TypeError, RuntimeError):
            pass
    return result


class MoeInitRoutingV4KernelSpec:
    """Kernel/GEIR spec — compare truncates NPU output to golden's size on shape mismatch.

    When all outputs have matching shapes (kernel mode), compare compares normally.
    When any output has a shape mismatch (GEIR mode with padded outputs), compare
    truncates both sides to min_len before comparing.
    """

    tolerance = {
        "float16": {"standard": "stat_rel_err"},
        "bfloat16": {"standard": "stat_rel_err"},
        "float32": {"standard": "stat_rel_err"},
    }

    @staticmethod
    def compare(*outputs, **kwargs):
        results = []
        half = len(outputs) // 2
        for i in range(half):
            npu_out = outputs[i]
            golden_out = outputs[half + i]
            if golden_out is None:
                results.append({"pass": True, "precision": 100.0})
                continue
            if npu_out is None:
                results.append(
                    {
                        "pass": False,
                        "precision": 0.0,
                        "error_info": f"output[{i}] npu is None",
                    }
                )
                continue
            npu_arr = _torch_to_numpy(npu_out)
            golden_arr = _torch_to_numpy(golden_out)
            npu_flat = numpy.asarray(npu_arr).reshape(-1)
            golden_flat = numpy.asarray(golden_arr).reshape(-1)
            min_len = min(npu_flat.size, golden_flat.size)
            # SCATTER 的 expanded_row_idx：golden 为完整置换，只比头部有效区
            prefix_len = _drop_rowidx_tail_len(
                npu_arr, golden_arr, outputs[half], min_len
            )
            if prefix_len < min_len:
                npu_flat = npu_flat[:prefix_len]
                golden_flat = golden_flat[:prefix_len]
                min_len = prefix_len
            npu_cmp = npu_flat[:min_len]
            golden_cmp = golden_flat[:min_len]
            dtype_name = str(golden_arr.dtype).lower()
            all_custom_keywords = [
                "e2m1",
                "e1m2",
                "float4",
                "hifloat8",
                "e4m3",
                "e5m2",
                "e8m0",
                "hif8",
                "int4",
                "uint1",
                "int2",
                "uint2",
            ]
            is_custom_dtype = any(
                x in dtype_name or x in str(npu_arr.dtype).lower()
                for x in all_custom_keywords
            )
            if is_custom_dtype:
                # 自定义 dtype（float4/float8/hifloat8/int4 等）走 raw uint8 字节级比对

                def _to_raw_uint8(arr):
                    if arr.dtype.kind in ("i", "u"):
                        vals = arr.reshape(-1).astype(numpy.uint8)
                        if vals.size > 0 and numpy.max(vals) <= 15:
                            n = vals.size - vals.size % 2
                            lo = vals[:n:2]
                            hi = vals[1:n:2]
                            return ((hi << 4) | lo).astype(numpy.uint8)
                        return vals
                    try:
                        raw = arr.view(numpy.uint8).reshape(-1).astype(numpy.uint8)
                    except (ValueError, TypeError):
                        raw = numpy.frombuffer(arr.tobytes(), dtype=numpy.uint8)
                    if raw.size > 0 and numpy.max(raw) <= 15:
                        n = raw.size - raw.size % 2
                        lo = raw[:n:2]
                        hi = raw[1:n:2]
                        return ((hi << 4) | lo).astype(numpy.uint8)
                    return raw

                npu_raw = _to_raw_uint8(numpy.asarray(npu_arr))
                golden_raw = _to_raw_uint8(numpy.asarray(golden_arr))
                min_len_raw = min(npu_raw.size, golden_raw.size)
                is_hifloat8 = (
                    "hifloat8" in dtype_name
                    or "hif8" in dtype_name
                    or "hifloat8" in str(npu_arr.dtype).lower()
                    or "hif8" in str(npu_arr.dtype).lower()
                )
                if is_hifloat8:
                    npu_i8 = numpy.asarray(npu_raw[:min_len_raw]).astype(numpy.int8)
                    golden_i8 = numpy.asarray(golden_raw[:min_len_raw]).astype(
                        numpy.int8
                    )
                    diff = numpy.abs(numpy.subtract(npu_i8, golden_i8))
                    npu_nan = numpy.isnan(npu_flat[:min_len].astype(numpy.float32))
                    golden_nan = numpy.isnan(
                        golden_flat[:min_len].astype(numpy.float32)
                    )
                    both_nan = numpy.logical_and(npu_nan, golden_nan)
                    diff[both_nan[: diff.size]] = 0
                    diff_count = int(numpy.sum(diff > 1))
                    precision = (
                        float(min_len_raw - diff_count) / min_len_raw * 100.0
                        if min_len_raw > 0
                        else 100.0
                    )
                    results.append(
                        {
                            "pass": (1 - precision / 100.0) <= 0.001,
                            "precision": precision,
                            "error_info": None
                            if diff_count == 0
                            else f"ulp mismatch {diff_count}/{min_len_raw}",
                        }
                    )
                else:
                    is_int4_quant = (
                        "int4" in dtype_name or "int4" in str(npu_arr.dtype).lower()
                    )
                    if is_int4_quant:
                        # int4: 先解包 packed nibble，再用 ULP 判据（差 > 1 计为错误）
                        npu_nib = _unpack_int4(npu_raw[:min_len_raw])
                        golden_nib = _unpack_int4(golden_raw[:min_len_raw])
                        results.append(_quant_ulp_result(npu_nib, golden_nib))
                    else:
                        diff_count = int(
                            numpy.sum(npu_raw[:min_len_raw] != golden_raw[:min_len_raw])
                        )
                        precision = (
                            float(min_len_raw - diff_count) / min_len_raw * 100.0
                            if min_len_raw > 0
                            else 100.0
                        )
                        results.append(
                            {
                                "pass": diff_count == 0,
                                "precision": precision,
                                "error_info": None
                                if diff_count == 0
                                else f"byte mismatch {diff_count}/{min_len_raw}",
                            }
                        )
                continue
            is_int = golden_arr.dtype.kind in ("i", "u")
            is_float = golden_arr.dtype.kind == "f" or any(
                x in dtype_name
                for x in [
                    "float",
                    "bfloat",
                    "hifloat",
                    "e4m3",
                    "e5m2",
                    "e8m0",
                    "e2m1",
                    "e1m2",
                ]
            )
            if is_int and not is_float:
                # int8 量化输出用 ULP 判据（差 > 1 计为错误）；int32/int64 等精确匹配
                is_quant_output = "int8" in dtype_name
                if is_quant_output:
                    npu_i = npu_cmp.astype(numpy.int32)
                    golden_i = golden_cmp.astype(numpy.int32)
                    results.append(_quant_ulp_result(npu_i, golden_i))
                else:
                    diff_count = int(numpy.sum(npu_cmp != golden_cmp))
                    precision = (
                        float(min_len - diff_count) / min_len * 100.0
                        if min_len > 0
                        else 100.0
                    )
                    results.append(
                        {
                            "pass": diff_count == 0,
                            "precision": precision,
                            "error_info": None
                            if diff_count == 0
                            else f"mismatch {diff_count}/{min_len}",
                        }
                    )
            else:
                if "bfloat16" in dtype_name or "bf16" in dtype_name:
                    rtol, atol = 0.004, 0.004
                elif "float32" in dtype_name:
                    rtol, atol = 0.0001, 0.0001
                elif "float16" in dtype_name:
                    rtol, atol = 0.001, 0.001
                else:
                    rtol = kwargs.get("rtol", 0.001)
                    atol = kwargs.get("atol", 0.001)
                npu_cmp_f = npu_cmp.astype(numpy.float32)
                golden_cmp_f = golden_cmp.astype(numpy.float32)
                close = numpy.isclose(
                    npu_cmp_f, golden_cmp_f, rtol=rtol, atol=atol, equal_nan=True
                )
                fulfill = (
                    float(numpy.sum(close)) / min_len * 100.0 if min_len > 0 else 100.0
                )
                results.append(
                    {
                        "pass": fulfill >= 99.5,
                        "precision": fulfill,
                        "error_info": None
                        if fulfill >= 99.5
                        else f"fulfill={fulfill:.2f}%",
                    }
                )
        return results[0] if len(results) == 1 else results


class E2eMoeInitRoutingBaseSpec:
    """E2E spec for torch_npu.npu_moe_init_routing_v2 — same as ACLNN but with snake_case params."""

    tolerance = {
        "float16": {"standard": "stat_rel_err"},
        "bfloat16": {"standard": "stat_rel_err"},
        "float32": {"standard": "stat_rel_err"},
    }

    @staticmethod
    def golden(
        x,
        expert_idx,
        scale=None,
        offset=None,
        active_num=-1,
        expert_capacity=-1,
        expert_num=-1,
        drop_pad_mode=0,
        expert_tokens_num_type=0,
        expert_tokens_num_flag=False,
        quant_mode=-1,
        active_expert_range=None,
        row_idx_type=0,
        expanded_x_out=None,
        expanded_row_idx_out=None,
        expert_tokens_count_or_cumsum_out=None,
        expanded_scale_out=None,
        **kwargs,
    ):
        x_np = _torch_to_numpy(x)
        expert_idx_np = _torch_to_numpy(expert_idx)
        scale_np = _torch_to_numpy(scale)
        offset_np = _torch_to_numpy(offset)
        if active_expert_range is not None:
            active_expert_range = _to_list(active_expert_range)
        else:
            active_expert_range = [0, int(expert_num)]

        results = _moe_init_routing_v4_numpy(
            x_np,
            expert_idx_np,
            scale_np,
            offset_np,
            None,
            _normalize_v4_active_num(
                active_num, int(x_np.shape[0]) * int(expert_idx_np.shape[-1])
            ),
            int(expert_capacity),
            int(expert_num),
            int(drop_pad_mode),
            int(expert_tokens_num_type),
            bool(expert_tokens_num_flag),
            int(quant_mode),
            active_expert_range,
            int(row_idx_type),
        )

        return list(results)

    @staticmethod
    def compare(*outputs, **kwargs):
        results = []
        half = len(outputs) // 2
        for i in range(half):
            npu_out = outputs[i]
            golden_out = outputs[half + i]
            if golden_out is None:
                results.append({"pass": True, "precision": 100.0})
                continue
            if npu_out is None:
                results.append(
                    {
                        "pass": False,
                        "precision": 0.0,
                        "error_info": f"output[{i}] npu is None",
                    }
                )
                continue
            npu_arr = _torch_to_numpy(npu_out)
            golden_arr = _torch_to_numpy(golden_out)
            npu_flat = numpy.asarray(npu_arr).reshape(-1)
            golden_flat = numpy.asarray(golden_arr).reshape(-1)
            min_len = min(npu_flat.size, golden_flat.size)
            # SCATTER 的 expanded_row_idx：golden 为完整置换，只比头部有效区
            prefix_len = _drop_rowidx_tail_len(
                npu_arr, golden_arr, outputs[half], min_len
            )
            if prefix_len < min_len:
                npu_flat = npu_flat[:prefix_len]
                golden_flat = golden_flat[:prefix_len]
                min_len = prefix_len
            npu_cmp = npu_flat[:min_len]
            golden_cmp = golden_flat[:min_len]
            dtype_name = str(golden_arr.dtype).lower()
            all_custom_keywords = [
                "e2m1",
                "e1m2",
                "float4",
                "hifloat8",
                "e4m3",
                "e5m2",
                "e8m0",
                "hif8",
                "int4",
                "uint1",
                "int2",
                "uint2",
            ]
            is_custom_dtype = any(
                x in dtype_name or x in str(npu_arr.dtype).lower()
                for x in all_custom_keywords
            )
            if is_custom_dtype:
                # 自定义 dtype（float4/float8/hifloat8/int4 等）走 raw uint8 字节级比对

                def _to_raw_uint8(arr):
                    if arr.dtype.kind in ("i", "u"):
                        vals = arr.reshape(-1).astype(numpy.uint8)
                        if vals.size > 0 and numpy.max(vals) <= 15:
                            n = vals.size - vals.size % 2
                            lo = vals[:n:2]
                            hi = vals[1:n:2]
                            return ((hi << 4) | lo).astype(numpy.uint8)
                        return vals
                    try:
                        raw = arr.view(numpy.uint8).reshape(-1).astype(numpy.uint8)
                    except (ValueError, TypeError):
                        raw = numpy.frombuffer(arr.tobytes(), dtype=numpy.uint8)
                    if raw.size > 0 and numpy.max(raw) <= 15:
                        n = raw.size - raw.size % 2
                        lo = raw[:n:2]
                        hi = raw[1:n:2]
                        return ((hi << 4) | lo).astype(numpy.uint8)
                    return raw

                npu_raw = _to_raw_uint8(numpy.asarray(npu_arr))
                golden_raw = _to_raw_uint8(numpy.asarray(golden_arr))
                min_len_raw = min(npu_raw.size, golden_raw.size)
                is_hifloat8 = (
                    "hifloat8" in dtype_name
                    or "hif8" in dtype_name
                    or "hifloat8" in str(npu_arr.dtype).lower()
                    or "hif8" in str(npu_arr.dtype).lower()
                )
                if is_hifloat8:
                    # hifloat8: 按 int8 解释，ULP 差 > 1 计为错误，NaN 位置双方一致视为相等
                    npu_i8 = numpy.asarray(npu_raw[:min_len_raw]).astype(numpy.int8)
                    golden_i8 = numpy.asarray(golden_raw[:min_len_raw]).astype(
                        numpy.int8
                    )
                    diff = numpy.abs(numpy.subtract(npu_i8, golden_i8))
                    npu_nan = numpy.isnan(npu_flat[:min_len].astype(numpy.float32))
                    golden_nan = numpy.isnan(
                        golden_flat[:min_len].astype(numpy.float32)
                    )
                    both_nan = numpy.logical_and(npu_nan, golden_nan)
                    diff[both_nan[: diff.size]] = 0
                    diff_count = int(numpy.sum(diff > 1))
                    precision = (
                        float(min_len_raw - diff_count) / min_len_raw * 100.0
                        if min_len_raw > 0
                        else 100.0
                    )
                    results.append(
                        {
                            "pass": (1 - precision / 100.0) <= 0.001,
                            "precision": precision,
                            "error_info": None
                            if diff_count == 0
                            else f"ulp mismatch {diff_count}/{min_len_raw}",
                        }
                    )
                else:
                    is_int4_quant = (
                        "int4" in dtype_name or "int4" in str(npu_arr.dtype).lower()
                    )
                    if is_int4_quant:
                        # int4: 先解包 packed nibble，再用 ULP 判据（差 > 1 计为错误）
                        npu_nib = _unpack_int4(npu_raw[:min_len_raw])
                        golden_nib = _unpack_int4(golden_raw[:min_len_raw])
                        results.append(_quant_ulp_result(npu_nib, golden_nib))
                    else:
                        diff_count = int(
                            numpy.sum(npu_raw[:min_len_raw] != golden_raw[:min_len_raw])
                        )
                        precision = (
                            float(min_len_raw - diff_count) / min_len_raw * 100.0
                            if min_len_raw > 0
                            else 100.0
                        )
                        results.append(
                            {
                                "pass": diff_count == 0,
                                "precision": precision,
                                "error_info": None
                                if diff_count == 0
                                else f"byte mismatch {diff_count}/{min_len_raw}",
                            }
                        )
                continue
            is_int = golden_arr.dtype.kind in ("i", "u")
            is_float = golden_arr.dtype.kind == "f" or any(
                x in dtype_name
                for x in [
                    "float",
                    "bfloat",
                    "hifloat",
                    "e4m3",
                    "e5m2",
                    "e8m0",
                    "e2m1",
                    "e1m2",
                ]
            )
            if is_int and not is_float:
                # int8 量化输出用 ULP 判据（差 > 1 计为错误）；int32/int64 等精确匹配
                is_quant_output = "int8" in dtype_name
                if is_quant_output:
                    npu_i = npu_cmp.astype(numpy.int32)
                    golden_i = golden_cmp.astype(numpy.int32)
                    results.append(_quant_ulp_result(npu_i, golden_i))
                else:
                    diff_count = int(numpy.sum(npu_cmp != golden_cmp))
                    precision = (
                        float(min_len - diff_count) / min_len * 100.0
                        if min_len > 0
                        else 100.0
                    )
                    results.append(
                        {
                            "pass": diff_count == 0,
                            "precision": precision,
                            "error_info": None
                            if diff_count == 0
                            else f"mismatch {diff_count}/{min_len}",
                        }
                    )
            else:
                if "bfloat16" in dtype_name or "bf16" in dtype_name:
                    rtol, atol = 0.004, 0.004
                elif "float32" in dtype_name:
                    rtol, atol = 0.0001, 0.0001
                elif "float16" in dtype_name:
                    rtol, atol = 0.001, 0.001
                else:
                    rtol = kwargs.get("rtol", 0.001)
                    atol = kwargs.get("atol", 0.001)
                npu_cmp_f = npu_cmp.astype(numpy.float32)
                golden_cmp_f = golden_cmp.astype(numpy.float32)
                close = numpy.isclose(
                    npu_cmp_f, golden_cmp_f, rtol=rtol, atol=atol, equal_nan=True
                )
                fulfill = (
                    float(numpy.sum(close)) / min_len * 100.0 if min_len > 0 else 100.0
                )
                results.append(
                    {
                        "pass": fulfill >= 99.5,
                        "precision": fulfill,
                        "error_info": None
                        if fulfill >= 99.5
                        else f"fulfill={fulfill:.2f}%",
                    }
                )
        return results[0] if len(results) == 1 else results


class AclnnMoeInitRoutingBaseSpec:
    """ACLNN spec — golden returns actual shape, compare truncates NPU output to golden's size."""

    tolerance = {
        "float16": {"standard": "stat_rel_err"},
        "bfloat16": {"standard": "stat_rel_err"},
        "float32": {"standard": "stat_rel_err"},
    }

    @staticmethod
    def golden(
        x,
        expertIdx,
        scaleOptional,
        offsetOptional,
        activeNum,
        expertCapacity,
        expertNum,
        dropPadMode,
        expertTokensNumType,
        expertTokensNumFlag,
        quantMode,
        activeExpertRangeOptional,
        rowIdxType,
        expandedXOut,
        expandedRowIdxOut,
        expertTokensCountOrCumsumOut,
        expandedScaleOut,
        **kwargs,
    ):
        x_np = _torch_to_numpy(x)
        expert_idx_np = _torch_to_numpy(expertIdx)
        scale_np = _torch_to_numpy(scaleOptional)
        offset_np = _torch_to_numpy(offsetOptional)
        active_expert_range = (
            _to_list(activeExpertRangeOptional)
            if activeExpertRangeOptional is not None
            else [0, int(expertNum)]
        )

        results = _moe_init_routing_v4_numpy(
            x_np,
            expert_idx_np,
            scale_np,
            offset_np,
            None,
            int(activeNum),
            int(expertCapacity),
            int(expertNum),
            int(dropPadMode),
            int(expertTokensNumType),
            bool(expertTokensNumFlag),
            int(quantMode),
            active_expert_range,
            int(rowIdxType),
        )

        templates = (
            expandedXOut,
            expandedRowIdxOut,
            expertTokensCountOrCumsumOut,
            expandedScaleOut,
        )
        return [_numpy_to_torch(arr, tpl) for arr, tpl in zip(results, templates)]

    @staticmethod
    def compare(*outputs, **kwargs):
        results = []
        half = len(outputs) // 2
        for i in range(half):
            npu_out = outputs[i]
            golden_out = outputs[half + i]
            if golden_out is None:
                results.append({"pass": True, "precision": 100.0})
                continue
            if npu_out is None:
                results.append(
                    {
                        "pass": False,
                        "precision": 0.0,
                        "error_info": f"output[{i}] npu is None",
                    }
                )
                continue
            npu_arr = _torch_to_numpy(npu_out)
            golden_arr = _torch_to_numpy(golden_out)
            npu_flat = numpy.asarray(npu_arr).reshape(-1)
            golden_flat = numpy.asarray(golden_arr).reshape(-1)
            min_len = min(npu_flat.size, golden_flat.size)
            # SCATTER 的 expanded_row_idx：golden 为完整置换，只比头部有效区
            prefix_len = _drop_rowidx_tail_len(
                npu_arr, golden_arr, outputs[half], min_len
            )
            if prefix_len < min_len:
                npu_flat = npu_flat[:prefix_len]
                golden_flat = golden_flat[:prefix_len]
                min_len = prefix_len
            npu_cmp = npu_flat[:min_len]
            golden_cmp = golden_flat[:min_len]
            dtype_name = str(golden_arr.dtype).lower()
            all_custom_keywords = [
                "e2m1",
                "e1m2",
                "float4",
                "hifloat8",
                "e4m3",
                "e5m2",
                "e8m0",
                "hif8",
                "int4",
                "uint1",
                "int2",
                "uint2",
            ]
            is_custom_dtype = any(
                x in dtype_name or x in str(npu_arr.dtype).lower()
                for x in all_custom_keywords
            )
            if is_custom_dtype:
                # 自定义 dtype（float4/float8/hifloat8/int4 等）走 raw uint8 字节级比对
                is_hifloat8 = (
                    "hifloat8" in dtype_name
                    or "hif8" in dtype_name
                    or "hifloat8" in str(npu_arr.dtype).lower()
                    or "hif8" in str(npu_arr.dtype).lower()
                )

                def _to_raw_uint8(arr):
                    if arr.dtype.kind in ("i", "u"):
                        vals = arr.reshape(-1).astype(numpy.uint8)
                        if vals.size > 0 and numpy.max(vals) <= 15:
                            n = vals.size - vals.size % 2
                            lo = vals[:n:2]
                            hi = vals[1:n:2]
                            return ((hi << 4) | lo).astype(numpy.uint8)
                        return vals
                    try:
                        raw = arr.view(numpy.uint8).reshape(-1).astype(numpy.uint8)
                    except (ValueError, TypeError):
                        raw = numpy.frombuffer(arr.tobytes(), dtype=numpy.uint8)
                    if raw.size > 0 and numpy.max(raw) <= 15:
                        n = raw.size - raw.size % 2
                        lo = raw[:n:2]
                        hi = raw[1:n:2]
                        return ((hi << 4) | lo).astype(numpy.uint8)
                    return raw

                npu_raw = _to_raw_uint8(numpy.asarray(npu_arr))
                golden_raw = _to_raw_uint8(numpy.asarray(golden_arr))
                min_len_raw = min(npu_raw.size, golden_raw.size)
                if is_hifloat8:
                    npu_i8 = numpy.asarray(npu_raw[:min_len_raw]).astype(numpy.int8)
                    golden_i8 = numpy.asarray(golden_raw[:min_len_raw]).astype(
                        numpy.int8
                    )
                    diff = numpy.abs(numpy.subtract(npu_i8, golden_i8))
                    npu_nan = numpy.isnan(npu_flat[:min_len].astype(numpy.float32))
                    golden_nan = numpy.isnan(
                        golden_flat[:min_len].astype(numpy.float32)
                    )
                    both_nan = numpy.logical_and(npu_nan, golden_nan)
                    diff[both_nan[: diff.size]] = 0
                    diff_count = int(numpy.sum(diff > 1))
                    precision = (
                        float(min_len_raw - diff_count) / min_len_raw * 100.0
                        if min_len_raw > 0
                        else 100.0
                    )
                    results.append(
                        {
                            "pass": (1 - precision / 100.0) <= 0.001,
                            "precision": precision,
                            "error_info": None
                            if diff_count == 0
                            else f"ulp mismatch {diff_count}/{min_len_raw}",
                        }
                    )
                else:
                    is_int4_quant = (
                        "int4" in dtype_name or "int4" in str(npu_arr.dtype).lower()
                    )
                    if is_int4_quant:
                        # int4: 先解包 packed nibble，再用 ULP 判据（差 > 1 计为错误）
                        npu_nib = _unpack_int4(npu_raw[:min_len_raw])
                        golden_nib = _unpack_int4(golden_raw[:min_len_raw])
                        results.append(_quant_ulp_result(npu_nib, golden_nib))
                    else:
                        diff_count = int(
                            numpy.sum(npu_raw[:min_len_raw] != golden_raw[:min_len_raw])
                        )
                        precision = (
                            float(min_len_raw - diff_count) / min_len_raw * 100.0
                            if min_len_raw > 0
                            else 100.0
                        )
                        results.append(
                            {
                                "pass": diff_count == 0,
                                "precision": precision,
                                "error_info": None
                                if diff_count == 0
                                else f"byte mismatch {diff_count}/{min_len_raw}",
                            }
                        )
                continue
            is_int = golden_arr.dtype.kind in ("i", "u")
            is_float = golden_arr.dtype.kind == "f" or any(
                x in dtype_name
                for x in [
                    "float",
                    "bfloat",
                    "hifloat",
                    "e4m3",
                    "e5m2",
                    "e8m0",
                    "e2m1",
                    "e1m2",
                ]
            )
            if is_int and not is_float:
                # int8 量化输出用 ULP 判据（差 > 1 计为错误）；int32/int64 等精确匹配
                is_quant_output = "int8" in dtype_name
                if is_quant_output:
                    npu_i = npu_cmp.astype(numpy.int32)
                    golden_i = golden_cmp.astype(numpy.int32)
                    results.append(_quant_ulp_result(npu_i, golden_i))
                else:
                    diff_count = int(numpy.sum(npu_cmp != golden_cmp))
                    precision = (
                        float(min_len - diff_count) / min_len * 100.0
                        if min_len > 0
                        else 100.0
                    )
                    results.append(
                        {
                            "pass": diff_count == 0,
                            "precision": precision,
                            "error_info": None
                            if diff_count == 0
                            else f"mismatch {diff_count}/{min_len}",
                        }
                    )
            else:
                if "bfloat16" in dtype_name or "bf16" in dtype_name:
                    rtol, atol = 0.004, 0.004
                elif "float32" in dtype_name:
                    rtol, atol = 0.0001, 0.0001
                elif "float16" in dtype_name:
                    rtol, atol = 0.001, 0.001
                else:
                    rtol = kwargs.get("rtol", 0.001)
                    atol = kwargs.get("atol", 0.001)
                npu_cmp_f = npu_cmp.astype(numpy.float32)
                golden_cmp_f = golden_cmp.astype(numpy.float32)
                close = numpy.isclose(
                    npu_cmp_f, golden_cmp_f, rtol=rtol, atol=atol, equal_nan=True
                )
                fulfill = (
                    float(numpy.sum(close)) / min_len * 100.0 if min_len > 0 else 100.0
                )
                results.append(
                    {
                        "pass": fulfill >= 99.5,
                        "precision": fulfill,
                        "error_info": None
                        if fulfill >= 99.5
                        else f"fulfill={fulfill:.2f}%",
                    }
                )
        return results[0] if len(results) == 1 else results


class AclnnMoeInitRoutingV4Spec(AclnnMoeInitRoutingBaseSpec):
    """V4 ACLNN reference: common outputs plus optional topk-weight output."""

    @staticmethod
    def golden(
        x,
        expertIdx,
        scaleOptional=None,
        offsetOptional=None,
        activeNumOptional=None,
        topkWeightOptional=None,
        expertCapacity=-1,
        expertNum=-1,
        dropPadMode=0,
        expertTokensNumType=0,
        expertTokensNumFlag=False,
        quantMode=-1,
        activeExpertRangeOptional=None,
        rowIdxType=0,
        expandedXOut=None,
        expandedRowIdxOut=None,
        expertTokensCountOrCumsumOut=None,
        expandedScaleOut=None,
        expandedTopkWeightOut=None,
        **kwargs,
    ):
        def _optional_numpy(tensor):
            value = _torch_to_numpy(tensor)
            # TTK represents a null optional tensor with a scalar marker ().
            return None if value is not None and value.ndim == 0 else value

        def _scalar(value, default):
            if value is None:
                return default
            array = _torch_to_numpy(value)
            if array is None or array.size == 0:
                return default
            return numpy.asarray(array).reshape(-1)[0].item()

        x_np = _torch_to_numpy(x)
        expert_idx_np = _torch_to_numpy(expertIdx)
        scale_np = _optional_numpy(scaleOptional)
        offset_np = _optional_numpy(offsetOptional)
        topk_weight_np = _optional_numpy(topkWeightOptional)
        active_range_np = _torch_to_numpy(activeExpertRangeOptional)
        expert_num = int(_scalar(expertNum, -1))
        active_expert_range = (
            numpy.asarray(active_range_np).reshape(-1).tolist()
            if active_range_np is not None
            else [0, expert_num]
        )

        results = _moe_init_routing_v4_numpy(
            x_np,
            expert_idx_np,
            scale_np,
            offset_np,
            topk_weight_np,
            _normalize_v4_active_num(
                activeNumOptional,
                int(x_np.shape[0]) * int(expert_idx_np.shape[-1]),
            ),
            int(_scalar(expertCapacity, -1)),
            expert_num,
            int(_scalar(dropPadMode, 0)),
            int(_scalar(expertTokensNumType, 0)),
            bool(_scalar(expertTokensNumFlag, False)),
            int(_scalar(quantMode, -1)),
            active_expert_range,
            int(_scalar(rowIdxType, 0)),
        )
        templates = (
            expandedXOut,
            expandedRowIdxOut,
            expertTokensCountOrCumsumOut,
            expandedScaleOut,
            expandedTopkWeightOut,
        )
        return [_numpy_to_torch(arr, tpl) for arr, tpl in zip(results, templates)]
